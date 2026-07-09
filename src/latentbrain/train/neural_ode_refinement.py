from __future__ import annotations

import copy
import csv
import itertools
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import torch
import yaml
from torch.utils.data import DataLoader

from latentbrain.eval.neural_ode_refinement import (
    build_neural_ode_refinement_result_row,
    rank_neural_ode_refinement_results,
    summarize_neural_ode_refinement,
)
from latentbrain.models.neural_sde import NeuralSDE, NeuralSDEConfig
from latentbrain.paths import get_repo_root, resolve_configured_path
from latentbrain.randomness import seed_everything
from latentbrain.torch.checkpoints import save_checkpoint
from latentbrain.torch.datasets import create_dataloaders, create_torch_datasets
from latentbrain.torch.device import resolve_device
from latentbrain.torch.masking import (
    apply_input_neuron_dropout,
    sample_neuron_dropout_mask,
    summarize_dropout_mask,
)
from latentbrain.torch.rate_initialization import compute_train_mean_rates_hz
from latentbrain.torch.schedules import linear_warmup
from latentbrain.train.neural_sde_tuning import (
    _behavior_mean_r2,
    _ensure_finite,
    _input_dropout_settings,
    _json_default,
    _load_dataset,
    _loss_for_batch,
    _mean,
    _relative,
    _resolve_processed_path,
    _run_neural_sde_evaluation,
    _split_and_mask,
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


def expand_neural_ode_refinement_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    values = [list(grid[key]) for key in keys]
    runs = [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]
    for run in runs:
        if float(run.get("diffusion_scale", 0.0)) != 0.0:
            msg = "deterministic neural-ODE refinement requires diffusion_scale == 0.0"
            raise ValueError(msg)
        if float(run.get("drift_regularization", 0.0)) < 0.0:
            msg = "drift_regularization must be non-negative"
            raise ValueError(msg)
        if str(run.get("scheduler", "none")) not in {"none", "cosine"}:
            msg = "scheduler must be none or cosine"
            raise ValueError(msg)
    return runs


def make_neural_ode_refinement_run_id(index: int, params: dict[str, Any]) -> str:
    def value(key: str) -> str:
        return str(params[key]).replace(".", "p")

    return (
        f"run_{index:03d}_enc{value('encoder_hidden_dim')}"
        f"_drift{value('drift_hidden_dim')}_lat{value('latent_dim')}"
        f"_fac{value('factor_dim')}_drop{value('input_dropout_rate')}"
        f"_hw{value('heldout_loss_weight')}_kl{value('kl_scale')}"
        f"_kw{value('kl_warmup_epochs')}_dr{value('drift_regularization')}"
        f"_{value('scheduler')}"
    )


def build_neural_ode_refinement_train_config(
    base_config: dict[str, Any], run_params: dict[str, Any], run_output_dir: Path
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    params = dict(run_params)
    params["diffusion_scale"] = 0.0
    learning_rate = float(params.get("learning_rate", config["model"]["learning_rate"]))
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
        "learning_rate": learning_rate,
        "weight_decay": float(config["model"]["weight_decay"]),
        "gradient_clip_norm": float(config["model"]["gradient_clip_norm"]),
        "heldin_loss_weight": float(config["model"].get("heldin_loss_weight", 1.0)),
        "heldout_loss_weight": float(params["heldout_loss_weight"]),
        "loss_normalization": str(config["model"].get("loss_normalization", "mean")),
        "kl_warmup_epochs": int(params["kl_warmup_epochs"]),
        "kl_scale": float(params["kl_scale"]),
        "drift_regularization_scale": float(params.get("drift_regularization", 0.0)),
        "scheduler": str(params.get("scheduler", "none")),
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


def _make_dropout_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    generator.manual_seed(seed)
    return generator


def _is_better(value: float, best: float | None, mode: str) -> bool:
    if best is None:
        return True
    if mode == "min":
        return value < best
    if mode == "max":
        return value > best
    msg = "checkpoint_mode must be min or max"
    raise ValueError(msg)


def _write_history(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def _scheduler_factor(scheduler_name: str, epoch: int, epochs: int) -> float:
    if scheduler_name == "none":
        return 1.0
    if scheduler_name == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * float(epoch) / max(float(epochs), 1.0)))
    msg = "scheduler must be none or cosine"
    raise ValueError(msg)


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _split_metrics(
    model: NeuralSDE,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    bin_size_ms: int,
    z0_beta: float,
    training_config: dict[str, Any],
) -> dict[str, float]:
    model.eval()
    losses: dict[str, list[float]] = {
        "total_loss": [],
        "heldin_reconstruction_loss": [],
        "heldout_prediction_loss": [],
        "z0_kl_loss": [],
        "drift_regularization_loss": [],
        "drift_norm": [],
        "diffusion_mean": [],
        "mean_rate_hz": [],
    }
    scale = float(training_config.get("drift_regularization_scale", 0.0))
    with torch.no_grad():
        for batch in loader:
            loss, rates, _ = _loss_for_batch(
                model, batch, device, bin_size_ms, z0_beta, training_config
            )
            _ensure_finite("evaluation loss", loss["loss"])
            losses["total_loss"].append(float(loss["loss"].detach().cpu()))
            losses["heldin_reconstruction_loss"].append(
                float(loss["heldin_reconstruction_loss"].detach().cpu())
            )
            losses["heldout_prediction_loss"].append(
                float(loss["heldout_prediction_loss"].detach().cpu())
            )
            losses["z0_kl_loss"].append(float(loss["z0_kl_loss"].detach().cpu()))
            losses["drift_regularization_loss"].append(
                float((loss["drift_regularization_loss"] * scale).detach().cpu())
            )
            losses["drift_norm"].append(float(loss["drift_norm"].detach().cpu()))
            losses["diffusion_mean"].append(float(loss["diffusion_mean"].detach().cpu()))
            losses["mean_rate_hz"].append(float(rates.mean().detach().cpu()))
    return {key: _mean(value) for key, value in losses.items()}


def _train_refinement(
    model: NeuralSDE,
    dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> list[dict[str, float]]:
    training_config = dict(config["training"])
    base_lr = float(training_config["learning_rate"])
    epochs = int(training_config["epochs"])
    scheduler_name = str(training_config.get("scheduler", "none"))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr, weight_decay=float(training_config["weight_decay"])
    )
    model.to(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    input_dropout = _input_dropout_settings(training_config)
    dropout_generator = _make_dropout_generator(device, int(input_dropout.get("seed", 0)))
    history: list[dict[str, float]] = []
    best_metric: float | None = None
    scale = float(training_config.get("drift_regularization_scale", 0.0))
    for epoch in range(epochs):
        learning_rate = base_lr * _scheduler_factor(scheduler_name, epoch, epochs)
        _set_optimizer_lr(optimizer, learning_rate)
        model.train()
        kl_beta = linear_warmup(epoch, int(training_config["kl_warmup_epochs"])) * float(
            training_config.get("kl_scale", 1.0)
        )
        train_losses: list[float] = []
        dropout_fractions: list[float] = []
        for batch in dataloaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            model_input = None
            if bool(input_dropout["enabled"]):
                heldin = batch["heldin_spikes"].to(device)
                mask = sample_neuron_dropout_mask(
                    heldin.shape[0],
                    heldin.shape[2],
                    float(input_dropout["rate"]),
                    heldin.device,
                    generator=dropout_generator,
                    keep_at_least_one=bool(input_dropout["keep_at_least_one_neuron"]),
                )
                model_input = apply_input_neuron_dropout(heldin, mask)
                dropout_fractions.append(float(summarize_dropout_mask(mask)["dropout_fraction"]))
            loss, _, _ = _loss_for_batch(
                model,
                batch,
                device,
                int(config["dataset"]["bin_size_ms"]),
                kl_beta,
                training_config,
                model_input,
            )
            _ensure_finite("training loss", loss["loss"])
            torch.autograd.backward(loss["loss"])
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training_config["gradient_clip_norm"])
            )
            optimizer.step()
            train_losses.append(float(loss["loss"].detach().cpu()))
        row: dict[str, float] = {
            "epoch": float(epoch),
            "kl_beta": float(kl_beta),
            "learning_rate": float(learning_rate),
            "training_batch_loss": _mean(train_losses),
            "configured_input_dropout_rate": float(input_dropout["rate"]),
            "realized_input_dropout_fraction": _mean(dropout_fractions)
            if dropout_fractions
            else 0.0,
            "configured_drift_regularization": scale,
        }
        for split_name in config["evaluation"]["evaluate_splits"]:
            metrics = _split_metrics(
                model,
                dataloaders[str(split_name)],
                device,
                int(config["dataset"]["bin_size_ms"]),
                kl_beta,
                training_config,
            )
            row[f"{split_name}_total_loss"] = metrics["total_loss"]
            row[f"{split_name}_heldin_reconstruction_loss"] = metrics["heldin_reconstruction_loss"]
            row[f"{split_name}_heldout_prediction_loss"] = metrics["heldout_prediction_loss"]
            row[f"{split_name}_z0_kl_loss"] = metrics["z0_kl_loss"]
            row[f"{split_name}_drift_regularization_loss"] = metrics["drift_regularization_loss"]
            row[f"{split_name}_drift_norm"] = metrics["drift_norm"]
            row[f"{split_name}_diffusion_mean"] = metrics["diffusion_mean"]
            row[f"{split_name}_mean_rate_hz"] = metrics["mean_rate_hz"]
        row["train_total_loss"] = row.get("train_total_loss", row["training_batch_loss"])
        row["z0_kl_loss"] = row.get("validation_z0_kl_loss", float("nan"))
        row["drift_regularization_loss"] = row.get(
            "validation_drift_regularization_loss", float("nan")
        )
        row["drift_norm"] = row.get("validation_drift_norm", float("nan"))
        row["diffusion_mean"] = row.get("validation_diffusion_mean", float("nan"))
        row["mean_predicted_rate"] = row.get("validation_mean_rate_hz", float("nan"))
        history.append(row)
        save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, epoch, row, config)
        metric = float(row[str(training_config.get("checkpoint_metric", "validation_total_loss"))])
        if _is_better(metric, best_metric, str(training_config.get("checkpoint_mode", "min"))):
            best_metric = metric
            save_checkpoint(
                checkpoint_dir / "best_validation.pt", model, optimizer, epoch, row, config
            )
        _write_history(output_dir / "metrics_history.csv", history)
    final = history[-1] | {"best_metric": best_metric}
    (output_dir / "final_metrics.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return history


def select_best_unified_checkpoint_index(checkpoint_rows: list[dict[str, Any]]) -> int:
    return max(
        range(len(checkpoint_rows)),
        key=lambda index: (
            float(checkpoint_rows[index]["validation_unified_bits_per_spike"]),
            -float(checkpoint_rows[index]["validation_poisson_nll"]),
        ),
    )


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
        "epoch_or_source": source,
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
            "model_name": "neural_ode_refinement",
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
    final_metrics = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
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
        "drift_regularization_loss": metrics.get("drift_regularization_loss"),
        "drift_norm": metrics.get("drift_norm"),
        "diffusion_mean": metrics.get("diffusion_mean"),
        "learning_rate": metrics.get("learning_rate"),
        "final_learning_rate": final_metrics.get("learning_rate"),
        "best_checkpoint_source": selected.get("checkpoint_source", "best_unified"),
    }


def _enrich_latent_diagnostics(
    diagnostics: pd.DataFrame, run_id: str, checkpoint_metrics: dict[str, Any]
) -> pd.DataFrame:
    enriched = diagnostics.copy()
    enriched.insert(0, "run_id", run_id)
    if "mean_predicted_rate" not in enriched:
        enriched["mean_predicted_rate"] = float(
            checkpoint_metrics.get("mean_predicted_rate", float("nan"))
        )
    if "z0_kl_loss" not in enriched:
        enriched["z0_kl_loss"] = float(checkpoint_metrics.get("z0_kl_loss", float("nan")))
    return enriched


def _train_and_evaluate_run(
    run_config: dict[str, Any], run_index: int, run_id: str, dataset: Any, split: Any, mask: Any
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    snapshot["model"]["name"] = "neural_ode_refinement"
    snapshot["model"]["diffusion_scale"] = 0.0
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8"
    )
    _train_refinement(model, dataloaders, snapshot, output_dir, device)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_rows: list[dict[str, Any]] = []
    checkpoint_payloads = []
    for source, checkpoint_path in (
        ("best_validation", checkpoint_dir / "best_validation.pt"),
        ("latest", checkpoint_dir / "latest.pt"),
    ):
        payload = _checkpoint_metric_row(
            checkpoint_path, source, snapshot, dataset, split, mask, device
        )
        checkpoint_rows.append(payload[0])
        checkpoint_payloads.append(payload)
    selected_index = select_best_unified_checkpoint_index(checkpoint_rows)
    checkpoint_rows[selected_index]["selected_by_unified"] = True
    selected = checkpoint_payloads[selected_index]
    selected_row = selected[0]
    best_unified = checkpoint_dir / "best_unified.pt"
    shutil.copy2(Path(str(selected_row["checkpoint_path"])), best_unified)
    checkpoint_scores = pd.DataFrame(checkpoint_rows)
    checkpoint_scores.to_csv(output_dir / "checkpoint_scores.csv", index=False)
    _write_final_evaluation(
        output_dir,
        snapshot,
        selected[1],
        selected[2],
        selected[3],
        selected[4],
        selected[5],
        best_unified,
    )
    checkpoint_metrics = _selected_checkpoint_metrics(
        output_dir, str(run_config["training"]["device"])
    )
    diagnostics = _enrich_latent_diagnostics(selected[6], run_id, checkpoint_metrics)
    diagnostics.to_csv(output_dir / "latent_diagnostics.csv", index=False)
    return checkpoint_scores, diagnostics


def run_neural_ode_refinement(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    _resolve_processed_path(config)
    gpu_name = _validate_cuda(config)
    if float(config["model"].get("diffusion_scale", 0.0)) != 0.0:
        msg = "deterministic neural-ODE refinement requires diffusion_scale == 0.0"
        raise ValueError(msg)
    dataset, dataset_hash, window_bins = _load_dataset(config)
    split, mask = _split_and_mask(dataset, config)
    reference_zero = _verify_reference_zero(dataset, split, mask, config)
    repo_root = get_repo_root()
    output_dir = resolve_configured_path(str(config["reporting"]["output_dir"]), repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    refs = dict(config["references"])
    refs["train_mean_validation_bits_per_spike"] = reference_zero
    grid = expand_neural_ode_refinement_grid(dict(config["grid"]))[
        : int(config["search"]["max_runs"])
    ]
    rows: list[dict[str, Any]] = []
    all_checkpoint_scores: list[pd.DataFrame] = []
    all_latent_diagnostics: list[pd.DataFrame] = []
    base = copy.deepcopy(config)
    base["_window_bins"] = window_bins
    for run_index, params in enumerate(grid):
        params = dict(params)
        params["diffusion_scale"] = 0.0
        run_id = make_neural_ode_refinement_run_id(run_index, params)
        run_dir = output_dir / "runs" / run_id
        run_config = build_neural_ode_refinement_train_config(base, params, run_dir)
        checkpoint_scores, latent_diagnostics = _train_and_evaluate_run(
            run_config, run_index, run_id, dataset, split, mask
        )
        checkpoint_scores.insert(0, "run_id", run_id)
        all_checkpoint_scores.append(checkpoint_scores)
        all_latent_diagnostics.append(latent_diagnostics)
        scores = _unified_scores(run_dir)
        rows.append(
            build_neural_ode_refinement_result_row(
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
    leaderboard = rank_neural_ode_refinement_results(results)
    summary = summarize_neural_ode_refinement(results, refs)
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
    if all_latent_diagnostics:
        pd.concat(all_latent_diagnostics, ignore_index=True).to_csv(
            output_dir / "latent_diagnostics.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(output_dir / "latent_diagnostics.csv", index=False)
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
        best_config = build_neural_ode_refinement_train_config(
            base, best_params, Path(str(best_result["output_dir"]))
        )
    (output_dir / "best_config.yaml").write_text(
        yaml.safe_dump(json.loads(json.dumps(best_config, default=_json_default)), sort_keys=False),
        encoding="utf-8",
    )
    return results, summary
