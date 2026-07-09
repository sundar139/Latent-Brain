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
from latentbrain.data.rebinning import rebin_neural_dataset
from latentbrain.data.schemas import NeuralDataset, NeuronMask, TrialSplit
from latentbrain.data.splits import create_neuron_mask, create_trial_split
from latentbrain.data.validation import (
    validate_neural_dataset,
    validate_neuron_mask,
    validate_trial_split,
)
from latentbrain.eval.lfads_unified_tuning import (
    RESULT_COLUMNS,
    build_lfads_unified_result_row,
    rank_lfads_unified_results,
    summarize_lfads_unified_tuning,
)
from latentbrain.eval.rebinning import compute_window_bins_for_duration
from latentbrain.eval.scoring import (
    ScoringConfig,
    score_heldout_prediction,
    train_heldout_mean_rate_reference,
)
from latentbrain.eval.tuning import expand_tuning_grid
from latentbrain.eval.windowing import crop_neural_dataset_time
from latentbrain.paths import get_repo_root, resolve_configured_path
from latentbrain.train.lfads_tuning import _train_and_evaluate_run


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def expand_unified_lfads_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    return expand_tuning_grid(grid)


def _slug(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def make_unified_lfads_run_id(index: int, params: dict[str, Any]) -> str:
    if index < 0:
        msg = "index must be non-negative"
        raise ValueError(msg)
    return (
        f"run_{index:03d}_"
        f"enc{_slug(params.get('encoder_hidden_dim', 'na'))}_"
        f"gen{_slug(params.get('generator_hidden_dim', 'na'))}_"
        f"lat{_slug(params.get('latent_dim', 'na'))}_"
        f"fac{_slug(params.get('factor_dim', 'na'))}_"
        f"idr{_slug(params.get('input_dropout_rate', 'na'))}_"
        f"hw{_slug(params.get('heldout_loss_weight', 'na'))}_"
        f"kl{_slug(params.get('kl_scale', 'na'))}"
    )


def build_unified_lfads_train_config(
    base_config: dict[str, Any],
    run_params: dict[str, Any],
    run_output_dir: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    settings = dict(config["lfads_settings"])
    target_bin = int(config["binning"]["target_bin_size_ms"])
    window_bins = int(config.get("_window_bins", 0))
    if window_bins <= 0:
        window_bins = compute_window_bins_for_duration(
            float(config["window"]["duration_seconds"]), target_bin
        )
    input_dropout_rate = float(run_params["input_dropout_rate"])
    config["dataset"]["bin_size_ms"] = target_bin
    config["data"] = {
        "input_neuron_group": "heldin",
        "target_neuron_group": "heldout",
        "max_time_bins": window_bins,
        "batch_size": int(settings["batch_size"]),
        "num_workers": 0,
        "drop_last": False,
    }
    config["model"] = {
        "name": "lfads_gru",
        "input_dim": None,
        "output_dim": "all",
        "encoder_hidden_dim": int(run_params["encoder_hidden_dim"]),
        "generator_hidden_dim": int(run_params["generator_hidden_dim"]),
        "latent_dim": int(run_params["latent_dim"]),
        "factor_dim": int(run_params["factor_dim"]),
        "dropout": float(settings["model_dropout"]),
        "min_rate_hz": float(settings["min_rate_hz"]),
        "max_rate_hz": float(settings["max_rate_hz"]),
    }
    config["training"] = {
        "seed": int(config["splits"]["seed"]),
        "epochs": int(run_params["epochs"]),
        "learning_rate": float(settings["learning_rate"]),
        "weight_decay": float(settings["weight_decay"]),
        "gradient_clip_norm": float(settings["gradient_clip_norm"]),
        "heldin_loss_weight": float(settings["heldin_loss_weight"]),
        "heldout_loss_weight": float(run_params["heldout_loss_weight"]),
        "kl_warmup_epochs": int(run_params["kl_warmup_epochs"]),
        "kl_scale": float(run_params["kl_scale"]),
        "loss_normalization": str(settings["loss_normalization"]),
        "checkpoint_metric": str(settings["checkpoint_metric"]),
        "checkpoint_mode": str(settings["checkpoint_mode"]),
        "device": str(config["runtime"]["device"]),
    }
    if input_dropout_rate > 0.0:
        config["training"]["input_dropout"] = {
            "enabled": True,
            "rate": input_dropout_rate,
            "apply_to": ["train"],
            "keep_at_least_one_neuron": True,
            "seed": int(config["splits"]["seed"]),
        }
    config["evaluation"] = {
        "evaluate_splits": list(base_config["evaluation"]["evaluate_splits"]),
        "primary_split": str(base_config["scoring"]["primary_split"]),
        "direct_model_primary": bool(base_config["evaluation"]["direct_model_primary"]),
        "also_evaluate_factor_decoder": bool(
            base_config["evaluation"]["also_evaluate_factor_decoder"]
        ),
        "behavior_decoder_enabled": bool(base_config["evaluation"]["behavior_decoder_enabled"]),
        "baseline_references": {
            "window_matched_mean_rate_validation_bits_per_spike": float(
                base_config["references"]["train_mean_validation_bits_per_spike"]
            ),
            "window_matched_factor_latent_validation_bits_per_spike": float(
                base_config["references"]["factor_latent_unified_validation_bits_per_spike"]
            ),
            "previous_lfads_masked_direct_validation_bits_per_spike": float(
                base_config["references"]["previous_best_lfads_family_validation_bits_per_spike"]
            ),
        },
    }
    config["reporting"] = {"output_dir": str(run_output_dir)}
    return config


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _resolve_processed_path(config: dict[str, Any]) -> Path:
    repo_root = get_repo_root()
    processed_path = resolve_configured_path(str(config["dataset"]["processed_path"]), repo_root)
    if not processed_path.exists():
        msg = f"Processed dataset is missing: {_relative(processed_path, repo_root)}"
        raise FileNotFoundError(msg)
    return processed_path


def _validate_cuda(config: dict[str, Any]) -> str:
    if str(config["runtime"]["device"]) != "cuda":
        msg = "runtime.device must be cuda for LFADS-family unified tuning"
        raise ValueError(msg)
    fail_without_cuda = bool(config["runtime"].get("fail_if_cuda_unavailable", True))
    if fail_without_cuda and not torch.cuda.is_available():
        msg = "CUDA was requested, but torch.cuda.is_available() is False."
        raise RuntimeError(msg)
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"


def _load_dataset(config: dict[str, Any]) -> tuple[NeuralDataset, str, int]:
    processed_path = _resolve_processed_path(config)
    dataset = load_neural_dataset(processed_path)
    validate_neural_dataset(dataset)
    dataset_hash = compute_dataset_hash(dataset)
    expected = str(config["dataset"].get("expected_hash", ""))
    if expected and dataset_hash != expected:
        msg = f"Dataset hash mismatch: expected {expected}, got {dataset_hash}"
        raise ValueError(msg)
    target_bin = int(config["binning"]["target_bin_size_ms"])
    rebinned = rebin_neural_dataset(dataset, target_bin)
    window_bins = compute_window_bins_for_duration(
        float(config["window"]["duration_seconds"]), target_bin
    )
    windowed = crop_neural_dataset_time(rebinned, window_bins, str(config["window"]["crop_policy"]))
    return windowed, dataset_hash, window_bins


def _split_and_mask(
    dataset: NeuralDataset, config: dict[str, Any]
) -> tuple[TrialSplit, NeuronMask]:
    split = create_trial_split(
        dataset.trial_ids,
        float(config["splits"]["train_fraction"]),
        float(config["splits"]["validation_fraction"]),
        float(config["splits"]["test_fraction"]),
        seed=int(config["splits"]["seed"]),
    )
    mask = create_neuron_mask(
        dataset.spikes.shape[2],
        float(config["splits"]["heldout_neuron_fraction"]),
        seed=int(config["splits"]["seed"]),
    )
    validate_trial_split(split, dataset.trial_ids)
    validate_neuron_mask(mask, dataset.spikes.shape[2])
    return split, mask


def _trial_mask(dataset: NeuralDataset, trial_ids: np.ndarray) -> np.ndarray:
    return np.isin(dataset.trial_ids, trial_ids)


def _verify_reference_zero(
    dataset: NeuralDataset, split: TrialSplit, mask: NeuronMask, config: dict[str, Any]
) -> float:
    scoring = ScoringConfig(
        bin_size_ms=int(config["binning"]["target_bin_size_ms"]),
        include_poisson_constant=bool(config["scoring"]["include_poisson_constant"]),
        min_rate_hz=float(config["scoring"]["min_rate_hz"]),
        max_rate_hz=float(config["scoring"]["max_rate_hz"]),
        reference_name=str(config["scoring"]["reference_model"]),
    )
    train_counts = dataset.spikes[_trial_mask(dataset, split.train)][:, :, mask.heldout]
    validation_counts = dataset.spikes[_trial_mask(dataset, split.validation)][:, :, mask.heldout]
    reference = train_heldout_mean_rate_reference(train_counts, validation_counts.shape, scoring)
    row = score_heldout_prediction(
        validation_counts,
        reference,
        reference,
        scoring,
        "train_heldout_mean_rate",
        "validation",
        "train_mean_reference_as_model",
        True,
    )
    bits = float(row["bits_per_spike"])
    if abs(bits) > 1e-12:
        msg = "train-heldout mean-rate reference did not score 0.0 bits/spike against itself"
        raise RuntimeError(msg)
    return bits


def _unified_scores(run_dir: Path) -> pd.DataFrame:
    split_metrics = pd.read_csv(run_dir / "evaluation" / "split_metrics.csv")
    scores = split_metrics.copy()
    scores.insert(0, "reference_name", "train_heldout_mean_rate")
    scores.insert(0, "valid_model", True)
    scores.to_csv(run_dir / "unified_scores.csv", index=False)
    return scores


def _behavior_mean_r2(run_dir: Path) -> float:
    path = run_dir / "evaluation" / "behavior_metrics.csv"
    if not path.exists():
        return float("nan")
    metrics = pd.read_csv(path)
    rows = metrics[metrics["split"] == "validation"] if "split" in metrics else metrics
    return float("nan") if rows.empty else float(rows["r2"].mean())


def _run_metrics(run_dir: Path, device: str) -> dict[str, Any]:
    final = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
    return {
        "status": "completed",
        "device": device,
        "validation_behavior_mean_r2": _behavior_mean_r2(run_dir),
        "train_total_loss": final.get("train_total_loss", final.get("training_batch_loss")),
        "validation_total_loss": final.get("validation_total_loss"),
        "validation_heldout_prediction_loss": final.get("validation_heldout_prediction_loss"),
    }


def run_lfads_unified_tuning(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
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
    grid = expand_unified_lfads_grid(dict(config["grid"]))[: int(config["search"]["max_runs"])]
    rows: list[dict[str, Any]] = []
    base = copy.deepcopy(config)
    base["_window_bins"] = window_bins
    for run_index, params in enumerate(grid):
        run_id = make_unified_lfads_run_id(run_index, params)
        run_dir = output_dir / "runs" / run_id
        run_config = build_unified_lfads_train_config(base, params, run_dir)
        _train_and_evaluate_run(run_config, run_index, run_id, dataset)
        scores = _unified_scores(run_dir)
        row = build_lfads_unified_result_row(
            run_id,
            run_index,
            params,
            scores,
            _run_metrics(run_dir, str(config["runtime"]["device"])),
            refs,
            run_dir,
        )
        rows.append(row)
    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    leaderboard = rank_lfads_unified_results(results)
    summary = summarize_lfads_unified_tuning(results, refs)
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
    best_config: dict[str, Any] = {}
    if not leaderboard.empty:
        best = leaderboard.iloc[0]
        best_result = results.loc[results["run_id"] == best["run_id"]].iloc[0]
        best_params = {
            key: best_result[key]
            for key in config["grid"]
            if key in best_result and pd.notna(best_result[key])
        }
        best_output_dir = Path(str(best_result["output_dir"]))
        best_config = build_unified_lfads_train_config(base, best_params, best_output_dir)
    (output_dir / "best_config.yaml").write_text(
        yaml.safe_dump(json.loads(json.dumps(best_config, default=_json_default)), sort_keys=False),
        encoding="utf-8",
    )
    return results, summary
