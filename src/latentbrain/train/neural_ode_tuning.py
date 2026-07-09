from __future__ import annotations

import copy
import itertools
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import torch
import yaml

from latentbrain.eval.neural_ode_tuning import (
    build_neural_ode_result_row,
    rank_neural_ode_results,
    summarize_neural_ode_tuning,
)
from latentbrain.models.neural_sde import NeuralSDE, NeuralSDEConfig
from latentbrain.paths import get_repo_root, resolve_configured_path
from latentbrain.randomness import seed_everything
from latentbrain.torch.datasets import create_dataloaders, create_torch_datasets
from latentbrain.torch.device import resolve_device
from latentbrain.torch.rate_initialization import compute_train_mean_rates_hz
from latentbrain.train.neural_sde_tuning import (
    _behavior_mean_r2,
    _json_default,
    _load_dataset,
    _relative,
    _resolve_processed_path,
    _run_neural_sde_evaluation,
    _split_and_mask,
    _train_neural_sde,
    _trial_mask,
    _validate_cuda,
    _verify_reference_zero,
)

CheckpointPayload = tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    pd.DataFrame,
]


def expand_neural_ode_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    values = [list(grid[key]) for key in keys]
    runs = [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]
    for run in runs:
        if float(run.get("diffusion_scale", 0.0)) != 0.0:
            msg = "deterministic neural-ODE-style tuning requires diffusion_scale == 0.0"
            raise ValueError(msg)
    return runs


def make_neural_ode_run_id(index: int, params: dict[str, Any]) -> str:
    def value(key: str) -> str:
        return str(params[key]).replace(".", "p")

    return (
        f"run_{index:03d}_enc{value('encoder_hidden_dim')}"
        f"_drift{value('drift_hidden_dim')}_lat{value('latent_dim')}"
        f"_fac{value('factor_dim')}_drop{value('input_dropout_rate')}"
        f"_hw{value('heldout_loss_weight')}_kl{value('kl_scale')}"
    )


def build_neural_ode_train_config(
    base_config: dict[str, Any], run_params: dict[str, Any], run_output_dir: Path
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    params = dict(run_params)
    params["diffusion_scale"] = 0.0
    config.setdefault("data", {})
    config["data"].update(
        {
            "batch_size": int(config["model"]["batch_size"]),
            "max_time_bins": int(config.get("_window_bins", 0)) or None,
        }
    )
    config["model"].update(
        {
            "name": "neural_ode",
            "encoder_hidden_dim": int(params["encoder_hidden_dim"]),
            "drift_hidden_dim": int(params["drift_hidden_dim"]),
            "diffusion_hidden_dim": int(params["diffusion_hidden_dim"]),
            "latent_dim": int(params["latent_dim"]),
            "factor_dim": int(params["factor_dim"]),
            "dropout": float(config["model"].get("model_dropout", 0.0)),
            "diffusion_scale": 0.0,
        }
    )
    config["training"] = {
        "device": str(config["runtime"]["device"]),
        "seed": int(config["splits"]["seed"]),
        "epochs": int(params["epochs"]),
        "learning_rate": float(config["model"]["learning_rate"]),
        "weight_decay": float(config["model"]["weight_decay"]),
        "gradient_clip_norm": float(config["model"]["gradient_clip_norm"]),
        "heldin_loss_weight": float(config["model"].get("heldin_loss_weight", 1.0)),
        "heldout_loss_weight": float(params["heldout_loss_weight"]),
        "loss_normalization": str(config["model"].get("loss_normalization", "mean")),
        "kl_warmup_epochs": int(params["kl_warmup_epochs"]),
        "kl_scale": float(params["kl_scale"]),
        "checkpoint_metric": str(config["model"].get("checkpoint_metric", "validation_total_loss")),
        "checkpoint_mode": str(config["model"].get("checkpoint_mode", "min")),
        "save_unified_checkpoints": bool(config["model"].get("save_unified_checkpoints", True)),
        "evaluate_checkpoints_by_unified_metric": bool(
            config["model"].get("evaluate_checkpoints_by_unified_metric", True)
        ),
        "input_dropout": {
            "enabled": float(params["input_dropout_rate"]) > 0.0,
            "rate": float(params["input_dropout_rate"]),
            "apply_to": ["train"],
            "keep_at_least_one_neuron": True,
            "seed": int(config["splits"]["seed"]),
        },
    }
    config["reporting"] = dict(config["reporting"])
    config["reporting"]["output_dir"] = str(run_output_dir)
    return config


def _checkpoint_metric_row(
    checkpoint_path: Path,
    source: str,
    run_config: dict[str, Any],
    dataset: Any,
    split: Any,
    mask: Any,
    device: torch.device,
) -> CheckpointPayload:
    eval_config = copy.deepcopy(run_config)
    eval_config["model"]["checkpoint_path"] = str(checkpoint_path)
    split_metrics, neuron_metrics, behavior_metrics, factor_summary, metadata, diagnostics = (
        _run_neural_sde_evaluation(dataset, split, mask, eval_config, device)
    )
    validation = split_metrics[
        (split_metrics["split"] == "validation")
        & (split_metrics["prediction_source"] == "direct_model")
    ].iloc[0]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metrics = dict(checkpoint.get("metrics", {}))
    row = {
        "checkpoint_source": source,
        "epoch": int(checkpoint.get("epoch", -1)),
        "validation_total_loss": float(metrics.get("validation_total_loss", float("nan"))),
        "validation_unified_bits_per_spike": float(validation["bits_per_spike"]),
        "validation_poisson_nll": float(validation["poisson_nll"]),
        "checkpoint_path": str(checkpoint_path),
        "selected_by_loss": source == "best_validation",
        "selected_by_unified": False,
    }
    return (
        row,
        split_metrics,
        neuron_metrics,
        behavior_metrics,
        factor_summary,
        metadata,
        diagnostics,
    )


def _write_final_evaluation(
    output_dir: Path,
    run_config: dict[str, Any],
    split_metrics: pd.DataFrame,
    neuron_metrics: pd.DataFrame,
    behavior_metrics: pd.DataFrame,
    factor_summary: pd.DataFrame,
    metadata: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    from latentbrain.eval.reporting import write_lfads_gru_evaluation_outputs

    validation = split_metrics[
        (split_metrics["split"] == "validation")
        & (split_metrics["prediction_source"] == "direct_model")
    ].iloc[0]
    factor_rows = split_metrics[
        (split_metrics["split"] == "validation")
        & (split_metrics["prediction_source"] == "factor_decoder")
    ]
    factor_bits = None if factor_rows.empty else float(factor_rows.iloc[0]["bits_per_spike"])
    write_lfads_gru_evaluation_outputs(
        output_dir / "evaluation",
        {
            "dataset_name": run_config["dataset"]["name"],
            "dataset_hash": run_config["dataset"].get("expected_hash"),
            "checkpoint_path": _relative(checkpoint_path, get_repo_root()),
            "checkpoint_epoch": metadata.get("checkpoint_epoch"),
            "model_name": "neural_ode",
            "factor_dim": int(metadata["factor_dim"]),
            "latent_dim": int(metadata["latent_dim"]),
            "primary_split": "validation",
            "primary_bits_per_spike": float(validation["bits_per_spike"]),
            "primary_poisson_nll": float(validation["poisson_nll"]),
            "primary_behavior_mean_r2": _behavior_mean_r2(
                output_dir / "evaluation" / "behavior_metrics.csv"
            ),
            "primary_prediction_source": "direct_model",
            "direct_model_available": True,
            "factor_decoder_evaluated": True,
            "direct_model_validation_bits_per_spike": float(validation["bits_per_spike"]),
            "factor_decoder_validation_bits_per_spike": factor_bits,
            "heldout_decoder_alpha": 1000.0,
            "behavior_decoder_enabled": bool(
                run_config["evaluation"].get("behavior_decoder_enabled", True)
            ),
            "behavior_decoder_alpha": 100.0,
            "fit_policy": "train trials only",
            "official_benchmark_claim": False,
            "full_lfads_claim": False,
        },
        split_metrics,
        neuron_metrics,
        behavior_metrics,
        factor_summary,
        metadata,
    )


def _unified_scores(run_dir: Path) -> pd.DataFrame:
    scores = pd.read_csv(run_dir / "evaluation" / "split_metrics.csv")
    scores.insert(0, "reference_name", "train_heldout_mean_rate")
    scores.insert(0, "valid_model", True)
    scores.to_csv(run_dir / "unified_scores.csv", index=False)
    return scores


def _selected_checkpoint_metrics(run_dir: Path, device: str) -> dict[str, Any]:
    checkpoint_scores = pd.read_csv(run_dir / "checkpoint_scores.csv")
    selected = checkpoint_scores[checkpoint_scores["selected_by_unified"].astype(bool)].iloc[0]
    checkpoint = torch.load(
        run_dir / "checkpoints" / "best_unified.pt", map_location="cpu", weights_only=False
    )
    metrics = dict(checkpoint.get("metrics", {}))
    return {
        "status": "completed",
        "device": device,
        "validation_behavior_mean_r2": _behavior_mean_r2(
            run_dir / "evaluation" / "behavior_metrics.csv"
        ),
        "train_total_loss": metrics.get("train_total_loss", metrics.get("training_batch_loss")),
        "validation_total_loss": metrics.get("validation_total_loss"),
        "validation_heldout_prediction_loss": metrics.get("validation_heldout_prediction_loss"),
        "z0_kl_loss": metrics.get("z0_kl_loss"),
        "drift_norm": metrics.get("drift_norm"),
        "diffusion_mean": metrics.get("diffusion_mean"),
        "best_checkpoint_source": selected.get("checkpoint_source", "best_unified"),
    }


def _train_and_evaluate_run(
    run_config: dict[str, Any],
    run_index: int,
    dataset: Any,
    split: Any,
    mask: Any,
) -> pd.DataFrame:
    seed_everything(int(run_config["training"]["seed"]) + run_index)
    dataloaders = create_dataloaders(
        create_torch_datasets(dataset, split, mask, int(run_config["data"]["max_time_bins"])),
        batch_size=int(run_config["data"]["batch_size"]),
        num_workers=0,
        drop_last=False,
        seed=int(run_config["training"]["seed"]) + run_index,
    )
    input_dim = int(mask.heldin.sum())
    output_dim = int(dataset.spikes.shape[2])
    model = NeuralSDE(
        NeuralSDEConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            encoder_hidden_dim=int(run_config["model"]["encoder_hidden_dim"]),
            drift_hidden_dim=int(run_config["model"]["drift_hidden_dim"]),
            diffusion_hidden_dim=int(run_config["model"]["diffusion_hidden_dim"]),
            latent_dim=int(run_config["model"]["latent_dim"]),
            factor_dim=int(run_config["model"]["factor_dim"]),
            dropout=float(run_config["model"].get("dropout", 0.0)),
            min_rate_hz=float(run_config["model"]["min_rate_hz"]),
            max_rate_hz=float(run_config["model"]["max_rate_hz"]),
            dt_seconds=float(run_config["model"]["dt_seconds"]),
            diffusion_scale=0.0,
        )
    )
    train_spikes = dataset.spikes[_trial_mask(dataset, split.train)]
    mean_rates = compute_train_mean_rates_hz(
        train_spikes,
        dataset.bin_size_ms,
        float(run_config["model"]["min_rate_hz"]),
        float(run_config["model"]["max_rate_hz"]),
    )
    model.initialize_output_bias_from_rates(torch.as_tensor(mean_rates, dtype=torch.float32))
    device = resolve_device(str(run_config["training"]["device"]))
    output_dir = Path(str(run_config["reporting"]["output_dir"]))
    snapshot = copy.deepcopy(run_config)
    snapshot["dataset"]["bin_size_ms"] = dataset.bin_size_ms
    snapshot["model"]["input_dim"] = input_dim
    snapshot["model"]["resolved_output_dim"] = output_dim
    snapshot["model"]["name"] = "neural_ode"
    snapshot["model"]["diffusion_scale"] = 0.0
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8"
    )
    _train_neural_sde(model, dataloaders, snapshot, output_dir, device)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_rows: list[dict[str, Any]] = []
    checkpoint_payloads = []
    for source, checkpoint_path in (
        ("best_validation", checkpoint_dir / "best_validation.pt"),
        ("latest", checkpoint_dir / "latest.pt"),
    ):
        (
            row,
            split_metrics,
            neuron_metrics,
            behavior_metrics,
            factor_summary,
            metadata,
            diagnostics,
        ) = _checkpoint_metric_row(checkpoint_path, source, snapshot, dataset, split, mask, device)
        checkpoint_rows.append(row)
        checkpoint_payloads.append(
            (
                row,
                split_metrics,
                neuron_metrics,
                behavior_metrics,
                factor_summary,
                metadata,
                diagnostics,
            )
        )
    selected_index = max(
        range(len(checkpoint_rows)),
        key=lambda index: (
            float(checkpoint_rows[index]["validation_unified_bits_per_spike"]),
            -float(checkpoint_rows[index]["validation_poisson_nll"]),
        ),
    )
    checkpoint_rows[selected_index]["selected_by_unified"] = True
    (
        selected_row,
        split_metrics,
        neuron_metrics,
        behavior_metrics,
        factor_summary,
        metadata,
        diagnostics,
    ) = checkpoint_payloads[selected_index]
    best_unified = checkpoint_dir / "best_unified.pt"
    shutil.copy2(Path(str(selected_row["checkpoint_path"])), best_unified)
    checkpoint_scores = pd.DataFrame(checkpoint_rows)
    checkpoint_scores.to_csv(output_dir / "checkpoint_scores.csv", index=False)
    _write_final_evaluation(
        output_dir,
        snapshot,
        split_metrics,
        neuron_metrics,
        behavior_metrics,
        factor_summary,
        metadata,
        best_unified,
    )
    diagnostics.to_csv(output_dir / "latent_diagnostics.csv", index=False)
    return checkpoint_scores


def run_neural_ode_tuning(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    _resolve_processed_path(config)
    gpu_name = _validate_cuda(config)
    dataset, dataset_hash, window_bins = _load_dataset(config)
    split, mask = _split_and_mask(dataset, config)
    reference_zero = _verify_reference_zero(dataset, split, mask, config)
    repo_root = get_repo_root()
    output_dir = resolve_configured_path(str(config["reporting"]["output_dir"]), repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    refs = dict(config["references"])
    refs["train_mean_validation_bits_per_spike"] = reference_zero
    grid = expand_neural_ode_grid(dict(config["grid"]))[: int(config["search"]["max_runs"])]
    rows: list[dict[str, Any]] = []
    all_checkpoint_scores: list[pd.DataFrame] = []
    base = copy.deepcopy(config)
    base["_window_bins"] = window_bins
    for run_index, params in enumerate(grid):
        params = dict(params)
        params["diffusion_scale"] = 0.0
        run_id = make_neural_ode_run_id(run_index, params)
        run_dir = output_dir / "runs" / run_id
        run_config = build_neural_ode_train_config(base, params, run_dir)
        checkpoint_scores = _train_and_evaluate_run(run_config, run_index, dataset, split, mask)
        checkpoint_scores.insert(0, "run_id", run_id)
        all_checkpoint_scores.append(checkpoint_scores)
        scores = _unified_scores(run_dir)
        rows.append(
            build_neural_ode_result_row(
                run_id,
                run_index,
                params,
                scores,
                _selected_checkpoint_metrics(run_dir, str(config["runtime"]["device"])),
                checkpoint_scores,
                refs,
                run_dir,
            )
        )
    results = pd.DataFrame(rows)
    leaderboard = rank_neural_ode_results(results)
    summary = summarize_neural_ode_tuning(results, refs)
    summary.update(
        {
            "dataset_name": config["dataset"]["name"],
            "dataset_hash": dataset_hash,
            "cuda_device": gpu_name,
            "bin_size_ms": int(config["binning"]["target_bin_size_ms"]),
            "window_seconds": float(config["window"]["duration_seconds"]),
            "window_bins": window_bins,
            "output_dir": str(output_dir),
        }
    )
    if all_checkpoint_scores:
        pd.concat(all_checkpoint_scores, ignore_index=True).to_csv(
            output_dir / "checkpoint_selection.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(output_dir / "checkpoint_selection.csv", index=False)
    best_config: dict[str, Any] = {}
    if not leaderboard.empty:
        best = leaderboard.iloc[0]
        best_result = results.loc[results["run_id"] == best["run_id"]].iloc[0]
        best_params = {
            key: best_result[key]
            for key in config["grid"]
            if key in best_result and pd.notna(best_result[key])
        }
        best_params["diffusion_scale"] = 0.0
        best_config = build_neural_ode_train_config(
            base, best_params, Path(str(best_result["output_dir"]))
        )
    (output_dir / "best_config.yaml").write_text(
        yaml.safe_dump(json.loads(json.dumps(best_config, default=_json_default)), sort_keys=False),
        encoding="utf-8",
    )
    return results, summary
