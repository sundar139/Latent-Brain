from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
import yaml

from latentbrain.data.io import compute_dataset_hash, load_neural_dataset
from latentbrain.data.schemas import NeuralDataset
from latentbrain.data.splits import create_neuron_mask, create_trial_split
from latentbrain.data.validation import (
    validate_neural_dataset,
    validate_neuron_mask,
    validate_trial_split,
)
from latentbrain.eval.lfads_eval import run_lfads_gru_evaluation
from latentbrain.eval.reporting import (
    write_lfads_gru_evaluation_outputs,
    write_lfads_gru_training_report,
)
from latentbrain.eval.tuning import (
    TUNING_RESULT_COLUMNS,
    expand_tuning_grid,
    make_run_id,
    rank_tuning_results,
    summarize_tuning_results,
)
from latentbrain.eval.windowing import crop_neural_dataset_time, describe_time_window
from latentbrain.models.lfads_gru import LFADSGRU, LFADSGRUConfig
from latentbrain.paths import get_repo_root, resolve_configured_path
from latentbrain.randomness import seed_everything
from latentbrain.torch.datasets import create_dataloaders, create_torch_datasets
from latentbrain.torch.device import resolve_device
from latentbrain.train.lfads_trainer import train_lfads_gru


class RecoverableRunError(RuntimeError):
    """Expected per-run failure that should be recorded without hiding other errors."""


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default))


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _validate_required_tuning_config(config: dict[str, Any]) -> None:
    runtime = dict(config.get("runtime", {}))
    if runtime.get("device") != "cuda":
        msg = "runtime.device must be cuda for real LFADS tuning"
        raise ValueError(msg)
    search = dict(config.get("search", {}))
    if int(search.get("max_runs", 0)) <= 0:
        msg = "search.max_runs must be positive"
        raise ValueError(msg)
    if search.get("run_order") != "deterministic":
        msg = "search.run_order must be deterministic"
        raise ValueError(msg)
    expand_tuning_grid(dict(config.get("grid", {})))
    if int(config.get("data", {}).get("max_time_bins", 0)) != 256:
        msg = "data.max_time_bins must be 256 for the window-matched tuning workflow"
        raise ValueError(msg)
    if int(config.get("training", {}).get("epochs", 0)) <= 0:
        msg = "training.epochs must be positive"
        raise ValueError(msg)
    refs = dict(config.get("evaluation", {}).get("baseline_references", {}))
    for key in (
        "window_matched_mean_rate_validation_bits_per_spike",
        "window_matched_factor_latent_validation_bits_per_spike",
        "previous_lfads_masked_direct_validation_bits_per_spike",
    ):
        if key not in refs:
            msg = f"evaluation.baseline_references.{key} is required"
            raise ValueError(msg)


def _validate_cuda(config: dict[str, Any]) -> str:
    if not torch.cuda.is_available():
        msg = "CUDA was requested, but torch.cuda.is_available() is False."
        raise RuntimeError(msg)
    return torch.cuda.get_device_name(0)


def _load_windowed_dataset(config: dict[str, Any]) -> tuple[NeuralDataset, str, dict[str, Any]]:
    repo_root = get_repo_root()
    processed_path = resolve_configured_path(str(config["dataset"]["processed_path"]), repo_root)
    if not processed_path.exists():
        msg = f"Processed dataset is missing: {_relative(processed_path, repo_root)}"
        raise FileNotFoundError(msg)
    dataset = load_neural_dataset(processed_path)
    validate_neural_dataset(dataset)
    dataset_hash = compute_dataset_hash(dataset)
    expected_hash = str(config["dataset"].get("expected_hash", ""))
    if expected_hash and dataset_hash != expected_hash:
        msg = f"Dataset hash mismatch: expected {expected_hash}, got {dataset_hash}"
        raise ValueError(msg)
    original_time_bins = int(dataset.spikes.shape[1])
    windowed = crop_neural_dataset_time(dataset, int(config["data"]["max_time_bins"]), "from_start")
    window = describe_time_window(
        original_time_bins, int(windowed.spikes.shape[1]), windowed.bin_size_ms
    )
    return windowed, dataset_hash, window


def _estimate_parameter_count(input_dim: int, output_dim: int, params: dict[str, Any]) -> int:
    model = LFADSGRU(
        LFADSGRUConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            encoder_hidden_dim=int(params["encoder_hidden_dim"]),
            generator_hidden_dim=int(params["generator_hidden_dim"]),
            latent_dim=int(params["latent_dim"]),
            factor_dim=int(params["factor_dim"]),
            dropout=float(params["dropout"]),
            min_rate_hz=1.0e-4,
            max_rate_hz=500.0,
        )
    )
    return int(sum(parameter.numel() for parameter in model.parameters()))


def build_lfads_run_config(
    base_config: dict[str, Any],
    run_params: dict[str, Any],
    run_output_dir: Path,
) -> dict[str, Any]:
    """Build one concrete LFADS-style masked co-smoothing config without mutating base."""
    config = copy.deepcopy(base_config)
    config.setdefault("model", {})
    config.setdefault("training", {})
    config.setdefault("reporting", {})
    model_params = {
        key: run_params[key]
        for key in (
            "encoder_hidden_dim",
            "generator_hidden_dim",
            "latent_dim",
            "factor_dim",
            "dropout",
        )
        if key in run_params
    }
    training_params = {
        key: run_params[key]
        for key in ("learning_rate", "weight_decay", "heldout_loss_weight", "kl_warmup_epochs")
        if key in run_params
    }
    config["model"].update(model_params)
    config["training"].update(training_params)
    config["training"]["device"] = config.get("runtime", {}).get("device", "cuda")
    config["model"]["output_dim"] = "all"
    config["reporting"]["output_dir"] = str(run_output_dir)
    return config


def _evaluation_config(run_config: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    eval_config = copy.deepcopy(run_config)
    eval_config["model"]["checkpoint_path"] = str(checkpoint_path)
    eval_config["evaluation_mode"] = {
        "use_direct_model_rates_for_heldout": bool(
            run_config.get("evaluation", {}).get("direct_model_primary", True)
        ),
        "also_evaluate_factor_decoder": bool(
            run_config.get("evaluation", {}).get("also_evaluate_factor_decoder", True)
        ),
    }
    eval_config["heldout_decoder"] = {
        "name": "ridge",
        "alpha": 1000.0,
        "fit_intercept": True,
        "min_rate_hz": float(run_config["model"].get("min_rate_hz", 1.0e-4)),
        "max_rate_hz": float(run_config["model"].get("max_rate_hz", 500.0)),
        "standardize_factors": True,
        "train_trials_only": True,
    }
    eval_config["behavior_decoder"] = {
        "enabled": bool(run_config.get("evaluation", {}).get("behavior_decoder_enabled", True)),
        "alpha": 100.0,
        "fit_intercept": True,
        "standardize_factors": True,
        "standardize_targets": True,
        "target_prefixes": ["hand_pos", "cursor_pos"],
        "derive_velocity": True,
        "velocity_method": "central_difference",
        "train_trials_only": True,
    }
    eval_config["reference"] = {"name": "train_mean_rate", "fit_train_trials_only": True}
    eval_config["evaluation"] = {
        "evaluate_splits": ["train", "validation", "test"],
        "primary_split": run_config.get("evaluation", {}).get("primary_split", "validation"),
        "baseline_references": run_config.get("evaluation", {}).get("baseline_references", {}),
    }
    return eval_config


def _mean_finite(series: pd.Series) -> float:
    finite = series[np.isfinite(series)]
    return float("nan") if finite.empty else float(finite.mean())


def _train_and_evaluate_run(
    run_config: dict[str, Any], run_index: int, run_id: str, dataset: NeuralDataset
) -> dict[str, Any]:
    seed_everything(int(run_config["training"]["seed"]) + run_index)
    split = create_trial_split(
        dataset.trial_ids,
        float(run_config["splits"]["train_fraction"]),
        float(run_config["splits"]["validation_fraction"]),
        float(run_config["splits"]["test_fraction"]),
        seed=int(run_config["splits"]["seed"]),
    )
    neuron_mask = create_neuron_mask(
        dataset.spikes.shape[2],
        float(run_config["splits"]["heldout_neuron_fraction"]),
        seed=int(run_config["splits"]["seed"]),
    )
    validate_trial_split(split, dataset.trial_ids)
    validate_neuron_mask(neuron_mask, dataset.spikes.shape[2])
    dataloaders = create_dataloaders(
        create_torch_datasets(
            dataset, split, neuron_mask, int(run_config["data"]["max_time_bins"])
        ),
        batch_size=int(run_config["data"]["batch_size"]),
        num_workers=int(run_config["data"].get("num_workers", 0)),
        drop_last=bool(run_config["data"].get("drop_last", False)),
        seed=int(run_config["training"]["seed"]) + run_index,
    )
    input_dim = int(neuron_mask.heldin.sum())
    output_dim = int(dataset.spikes.shape[2])
    model_config = LFADSGRUConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        encoder_hidden_dim=int(run_config["model"]["encoder_hidden_dim"]),
        generator_hidden_dim=int(run_config["model"]["generator_hidden_dim"]),
        latent_dim=int(run_config["model"]["latent_dim"]),
        factor_dim=int(run_config["model"]["factor_dim"]),
        dropout=float(run_config["model"].get("dropout", 0.0)),
        min_rate_hz=float(run_config["model"]["min_rate_hz"]),
        max_rate_hz=float(run_config["model"]["max_rate_hz"]),
    )
    model = LFADSGRU(model_config)
    device = resolve_device(str(run_config["training"]["device"]))
    output_dir = Path(str(run_config["reporting"]["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = copy.deepcopy(run_config)
    config_snapshot["dataset"]["bin_size_ms"] = dataset.bin_size_ms
    config_snapshot["model"]["input_dim"] = input_dim
    config_snapshot["model"]["resolved_output_dim"] = output_dim
    config_snapshot["training"]["training_mode"] = "cosmoothing"
    (output_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config_snapshot, sort_keys=False), encoding="utf-8"
    )
    try:
        state = train_lfads_gru(model, dataloaders, config_snapshot, output_dir, device)
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise RecoverableRunError(f"CUDA out of memory: {exc}") from exc
    final = state.history[-1]
    repo_root = get_repo_root()
    write_lfads_gru_training_report(
        output_dir / "lfads_gru_training_report.md",
        {
            "dataset_name": run_config["dataset"]["name"],
            "dataset_hash": run_config["dataset"].get("expected_hash"),
            "model_name": "lfads_gru",
            "input_dim": input_dim,
            "output_dim": output_dim,
            "encoder_hidden_dim": model_config.encoder_hidden_dim,
            "generator_hidden_dim": model_config.generator_hidden_dim,
            "factor_dim": model_config.factor_dim,
            "latent_dim": model_config.latent_dim,
            "training_mode": "cosmoothing",
            "output_dim_policy": "all",
            "heldin_loss_weight": run_config["training"]["heldin_loss_weight"],
            "heldout_loss_weight": run_config["training"]["heldout_loss_weight"],
            "epochs": run_config["training"]["epochs"],
            "kl_warmup_epochs": run_config["training"]["kl_warmup_epochs"],
            "best_validation_loss": state.best_metric,
            "best_validation_total_loss": state.best_metric,
            "final_validation_loss": final.get("validation_loss"),
            "final_validation_total_loss": final.get("validation_total_loss"),
            "final_validation_heldout_prediction_loss": final.get(
                "validation_heldout_prediction_loss"
            ),
            "latest_checkpoint": _relative(output_dir / "checkpoints" / "latest.pt", repo_root),
            "best_validation_checkpoint": _relative(
                output_dir / "checkpoints" / "best_validation.pt", repo_root
            ),
        },
    )
    eval_config = _evaluation_config(
        config_snapshot, output_dir / "checkpoints" / "best_validation.pt"
    )
    split_metrics, neuron_metrics, behavior_metrics, factor_summary, metadata = (
        run_lfads_gru_evaluation(dataset, split, neuron_mask, eval_config, device)
    )
    validation = split_metrics[
        (split_metrics["split"] == "validation")
        & (split_metrics["prediction_source"] == "direct_model")
    ]
    if validation.empty:
        validation = split_metrics[split_metrics["split"] == "validation"]
    primary = validation.iloc[0]
    behavior_r2 = (
        float("nan")
        if behavior_metrics.empty
        else _mean_finite(behavior_metrics.loc[behavior_metrics["split"] == "validation", "r2"])
    )
    eval_dir = output_dir / "evaluation"
    write_lfads_gru_evaluation_outputs(
        eval_dir,
        {
            "dataset_name": run_config["dataset"]["name"],
            "dataset_hash": run_config["dataset"].get("expected_hash"),
            "checkpoint_path": _relative(
                output_dir / "checkpoints" / "best_validation.pt", repo_root
            ),
            "checkpoint_epoch": metadata.get("checkpoint_epoch"),
            "model_name": "lfads_gru",
            "factor_dim": int(metadata["factor_dim"]),
            "latent_dim": int(metadata["latent_dim"]),
            "max_time_bins": run_config["data"]["max_time_bins"],
            "primary_split": "validation",
            "primary_bits_per_spike": float(primary["bits_per_spike"]),
            "primary_poisson_nll": float(primary["poisson_nll"]),
            "primary_behavior_mean_r2": behavior_r2,
            "primary_prediction_source": str(primary["prediction_source"]),
            "direct_model_available": bool(metadata.get("direct_model_available", False)),
            "factor_decoder_evaluated": bool(metadata.get("factor_decoder_evaluated", False)),
            "direct_model_validation_bits_per_spike": float(primary["bits_per_spike"]),
            "factor_decoder_validation_bits_per_spike": None,
            "mean_rate_validation_bits_per_spike": run_config["evaluation"]["baseline_references"][
                "window_matched_mean_rate_validation_bits_per_spike"
            ],
            "factor_latent_best_validation_bits_per_spike": run_config["evaluation"][
                "baseline_references"
            ]["window_matched_factor_latent_validation_bits_per_spike"],
            "previous_lfads_eval_validation_bits_per_spike": run_config["evaluation"][
                "baseline_references"
            ]["previous_lfads_masked_direct_validation_bits_per_spike"],
            "beats_previous_lfads_eval": float(primary["bits_per_spike"])
            > run_config["evaluation"]["baseline_references"][
                "previous_lfads_masked_direct_validation_bits_per_spike"
            ],
            "beats_mean_rate_reference": float(primary["bits_per_spike"])
            > run_config["evaluation"]["baseline_references"][
                "window_matched_mean_rate_validation_bits_per_spike"
            ],
            "beats_factor_latent_reference": float(primary["bits_per_spike"])
            > run_config["evaluation"]["baseline_references"][
                "window_matched_factor_latent_validation_bits_per_spike"
            ],
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
    return {
        "validation_bits_per_spike": float(primary["bits_per_spike"]),
        "validation_poisson_nll": float(primary["poisson_nll"]),
        "validation_behavior_mean_r2": behavior_r2,
        "validation_total_loss": float(final.get("validation_total_loss", float("nan"))),
        "validation_heldout_prediction_loss": float(
            final.get("validation_heldout_prediction_loss", float("nan"))
        ),
    }


def _base_result_row(
    run_id: str,
    run_index: int,
    params: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    row = dict.fromkeys(TUNING_RESULT_COLUMNS)
    row.update(params)
    row.update(
        {
            "run_id": run_id,
            "run_index": run_index,
            "epochs": int(config["training"]["epochs"]),
            "device": str(config.get("runtime", {}).get("device", "cuda")),
            "output_dir": str(output_dir),
            "notes": "",
        }
    )
    return row


def run_lfads_tuning(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run a small deterministic CUDA grid for LFADS-style masked co-smoothing."""
    _validate_required_tuning_config(config)
    gpu_name = _validate_cuda(config)
    dataset, dataset_hash, window = _load_windowed_dataset(config)
    repo_root = get_repo_root()
    output_dir = resolve_configured_path(str(config["reporting"]["output_dir"]), repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = expand_tuning_grid(dict(config["grid"]))[: int(config["search"]["max_runs"])]
    rows: list[dict[str, Any]] = []
    for run_index, params in enumerate(grid):
        run_id = make_run_id(run_index, params)
        run_dir = output_dir / "runs" / run_id
        run_config = build_lfads_run_config(config, params, run_dir)
        row = _base_result_row(run_id, run_index, params, run_config, run_dir)
        if hasattr(dataset, "spikes"):
            input_dim = int(
                dataset.spikes.shape[2] * (1.0 - float(config["splits"]["heldout_neuron_fraction"]))
            )
            row["parameter_count_estimate"] = _estimate_parameter_count(
                max(input_dim, 1), int(dataset.spikes.shape[2]), params
            )
        else:
            row["parameter_count_estimate"] = 0
        try:
            metrics = _train_and_evaluate_run(run_config, run_index, run_id, dataset)
        except RecoverableRunError as exc:
            row.update({"status": "failed", "notes": str(exc)})
        else:
            refs = config["evaluation"]["baseline_references"]
            bits = float(metrics["validation_bits_per_spike"])
            row.update(metrics)
            row.update(
                {
                    "status": "completed",
                    "beats_window_matched_mean_rate": bits
                    > float(refs["window_matched_mean_rate_validation_bits_per_spike"]),
                    "beats_window_matched_factor_latent": bits
                    > float(refs["window_matched_factor_latent_validation_bits_per_spike"]),
                    "beats_previous_lfads_masked_direct": bits
                    > float(refs["previous_lfads_masked_direct_validation_bits_per_spike"]),
                }
            )
        rows.append(row)
    results = pd.DataFrame(rows)
    summary = summarize_tuning_results(
        results,
        dict(config["evaluation"]["baseline_references"]),
        str(config["search"]["selection_metric"]),
    )
    summary.update(
        {
            "dataset_name": config["dataset"]["name"],
            "dataset_hash": dataset_hash,
            "window_time_bins": int(window["cropped_time_bins"]),
            "window_seconds": float(window["window_seconds"]),
            "cuda_device": gpu_name,
            "output_dir": str(output_dir),
        }
    )
    ranked = rank_tuning_results(results, str(config["search"]["selection_metric"]), "max")
    best_config: dict[str, Any] = {}
    if not ranked.empty:
        best_row = ranked.iloc[0]
        best_params = {key: best_row[key] for key in config["grid"] if key in best_row}
        best_config = build_lfads_run_config(config, best_params, Path(str(best_row["output_dir"])))
    (output_dir / "best_config.yaml").write_text(
        yaml.safe_dump(_plain(best_config), sort_keys=False), encoding="utf-8"
    )
    (output_dir / "tuning_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return results, summary
