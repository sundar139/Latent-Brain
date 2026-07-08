from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
import yaml
from rich.console import Console

from latentbrain.data.io import compute_dataset_hash, load_neural_dataset
from latentbrain.data.rebinning import rebin_neural_dataset, validate_rebin_factor
from latentbrain.data.schemas import NeuralDataset, NeuronMask, TrialSplit
from latentbrain.data.splits import create_neuron_mask, create_trial_split
from latentbrain.data.validation import (
    validate_neural_dataset,
    validate_neuron_mask,
    validate_trial_split,
)
from latentbrain.eval.cosmoothing import (
    _broadcast_reference,
    evaluate_cosmoothing_predictions,
)
from latentbrain.eval.lfads_eval import extract_lfads_factors, load_lfads_gru_from_checkpoint
from latentbrain.eval.rate_calibration import (
    apply_log_rate_bias,
    apply_multiplicative_rate_scale,
    blend_with_mean_rate,
    choose_best_blend_alpha,
    fit_log_rate_bias,
    fit_multiplicative_rate_scale,
    mean_rates_from_counts,
)
from latentbrain.eval.rebinning import compute_window_bins_for_duration
from latentbrain.eval.reporting import write_lfads_rate_calibration_outputs
from latentbrain.eval.windowing import crop_neural_dataset_time
from latentbrain.paths import get_repo_root, resolve_configured_path
from latentbrain.torch.datasets import create_dataloaders, create_torch_datasets
from latentbrain.train.lfads_tuning import _train_and_evaluate_run
from latentbrain.train.rebinned_lfads import build_rate_initialized_lfads_train_config

console = Console(markup=False)

CALIBRATION_COLUMNS = [
    "method_name",
    "split",
    "bin_size_ms",
    "prediction_source",
    "spike_count",
    "zero_fraction",
    "observed_rate_hz",
    "mean_predicted_rate_hz",
    "poisson_nll",
    "poisson_log_likelihood",
    "reference_log_likelihood",
    "bits_per_spike",
    "mse_rate_hz",
    "mae_rate_hz",
    "scale_mean",
    "scale_min",
    "scale_max",
    "notes",
]
BLEND_COLUMNS = [
    "alpha",
    "split",
    "bin_size_ms",
    "spike_count",
    "poisson_nll",
    "bits_per_spike",
    "mean_predicted_rate_hz",
    "notes",
]
INITIALIZED_COLUMNS = [
    "method_name",
    "split",
    "bin_size_ms",
    "prediction_source",
    "spike_count",
    "zero_fraction",
    "observed_rate_hz",
    "mean_predicted_rate_hz",
    "poisson_nll",
    "poisson_log_likelihood",
    "reference_log_likelihood",
    "bits_per_spike",
    "behavior_mean_r2",
    "notes",
]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local LFADS-style rate calibration diagnostics."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/mc_maze_small_lfads_rate_calibration.yaml")
    )
    return parser.parse_args(argv)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _cuda_diagnostic() -> dict[str, Any]:
    return {
        "torch": torch.__version__,
        "available": bool(torch.cuda.is_available()),
        "torch_cuda": torch.version.cuda,
        "device_count": int(torch.cuda.device_count()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
    }


def _load_config(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"malformed LFADS rate calibration config: {path}"
        raise ValueError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"LFADS rate calibration config must contain a mapping: {path}"
        raise ValueError(msg)
    _validate_config(raw)
    return raw


def _validate_config(config: dict[str, Any]) -> None:
    if str(config["runtime"]["device"]) != "cuda":
        msg = "runtime.device must be cuda for LFADS rate calibration diagnostics"
        raise ValueError(msg)
    original_bin = int(config["dataset"]["original_bin_size_ms"])
    target_bin = int(config["binning"]["target_bin_size_ms"])
    validate_rebin_factor(original_bin, target_bin)
    compute_window_bins_for_duration(float(config["window"]["duration_seconds"]), target_bin)
    if not str(config["existing_lfads"]["checkpoint_path"]):
        msg = "existing_lfads.checkpoint_path is required"
        raise ValueError(msg)
    alphas = [float(value) for value in config["posthoc_calibration"]["blend_alpha_grid"]]
    if not alphas or any(alpha < 0.0 or alpha > 1.0 for alpha in alphas):
        msg = "posthoc_calibration.blend_alpha_grid values must be between 0 and 1"
        raise ValueError(msg)
    for section_name in ("posthoc_calibration", "initialized_lfads"):
        section = dict(config[section_name])
        if float(section["min_rate_hz"]) <= 0.0 or float(section["max_rate_hz"]) <= float(
            section["min_rate_hz"]
        ):
            msg = f"{section_name} min/max rates must be positive and increasing"
            raise ValueError(msg)
    init = dict(config["initialized_lfads"])
    for key in (
        "encoder_hidden_dim",
        "generator_hidden_dim",
        "latent_dim",
        "factor_dim",
        "epochs",
        "batch_size",
    ):
        if int(init[key]) <= 0:
            msg = f"initialized_lfads.{key} must be positive"
            raise ValueError(msg)
    for key in (
        "same_bin_mean_rate_validation_bits_per_spike",
        "same_bin_factor_latent_validation_bits_per_spike",
        "previous_20ms_lfads_validation_bits_per_spike",
    ):
        if key not in config["references"]:
            msg = f"references.{key} is required"
            raise ValueError(msg)


def _prepare_dataset(
    config: dict[str, Any],
) -> tuple[NeuralDataset, str, TrialSplit, NeuronMask, int]:
    repo_root = get_repo_root()
    processed_path = resolve_configured_path(str(config["dataset"]["processed_path"]), repo_root)
    if not processed_path.exists():
        msg = f"Processed dataset is missing: {_relative(processed_path, repo_root)}"
        raise FileNotFoundError(msg)
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
    split = create_trial_split(
        windowed.trial_ids,
        float(config["splits"]["train_fraction"]),
        float(config["splits"]["validation_fraction"]),
        float(config["splits"]["test_fraction"]),
        seed=int(config["splits"]["seed"]),
    )
    mask = create_neuron_mask(
        windowed.spikes.shape[2],
        float(config["splits"]["heldout_neuron_fraction"]),
        seed=int(config["splits"]["seed"]),
    )
    validate_trial_split(split, windowed.trial_ids)
    validate_neuron_mask(mask, windowed.spikes.shape[2])
    return windowed, dataset_hash, split, mask, window_bins


def _prediction_config(
    config: dict[str, Any], checkpoint_path: Path, window_bins: int
) -> dict[str, Any]:
    init = dict(config["initialized_lfads"])
    return {
        "splits": dict(config["splits"]),
        "data": {
            "max_time_bins": int(window_bins),
            "batch_size": int(init["batch_size"]),
            "num_workers": 0,
            "drop_last": False,
        },
        "model": {
            "checkpoint_path": str(checkpoint_path),
            "output_dim": "all",
            "encoder_hidden_dim": int(init["encoder_hidden_dim"]),
            "generator_hidden_dim": int(init["generator_hidden_dim"]),
            "latent_dim": int(init["latent_dim"]),
            "factor_dim": int(init["factor_dim"]),
            "dropout": float(init["dropout"]),
            "min_rate_hz": float(init["min_rate_hz"]),
            "max_rate_hz": float(init["max_rate_hz"]),
        },
    }


def _direct_predictions(
    dataset: NeuralDataset,
    split: TrialSplit,
    mask: NeuronMask,
    config: dict[str, Any],
    checkpoint_path: Path,
    window_bins: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    pred_config = _prediction_config(config, checkpoint_path, window_bins)
    input_dim = int(mask.heldin.sum())
    model = load_lfads_gru_from_checkpoint(
        checkpoint_path, input_dim, dataset.spikes.shape[2], pred_config, device
    )
    loaders = create_dataloaders(
        create_torch_datasets(dataset, split, mask, window_bins),
        batch_size=int(pred_config["data"]["batch_size"]),
        num_workers=0,
        drop_last=False,
        seed=int(config["splits"]["seed"]),
    )
    extracted = extract_lfads_factors(model, loaders, device)
    target_indices = np.flatnonzero(mask.heldout)
    counts = {name: values["heldout_spikes"] for name, values in extracted.items()}
    predictions = {
        name: values["model_rates_hz"][:, :, target_indices] for name, values in extracted.items()
    }
    return counts, predictions


def _metric_row(
    method_name: str,
    split_name: str,
    counts: np.ndarray,
    predicted: np.ndarray,
    reference: np.ndarray,
    bin_size_ms: int,
    prediction_source: str,
    scale: np.ndarray | None,
    notes: str,
) -> dict[str, Any]:
    metrics = evaluate_cosmoothing_predictions(counts, predicted, reference, bin_size_ms)
    return {
        "method_name": method_name,
        "split": split_name,
        "bin_size_ms": int(bin_size_ms),
        "prediction_source": prediction_source,
        "zero_fraction": float(np.mean(counts == 0.0)),
        "observed_rate_hz": float(np.mean(counts) / (bin_size_ms / 1000.0)),
        "scale_mean": float("nan") if scale is None else float(np.mean(scale)),
        "scale_min": float("nan") if scale is None else float(np.min(scale)),
        "scale_max": float("nan") if scale is None else float(np.max(scale)),
        "notes": notes,
        **metrics,
    }


def _evaluate_calibrations(
    counts: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cal = dict(config["posthoc_calibration"])
    bin_size_ms = int(config["binning"]["target_bin_size_ms"])
    min_rate = float(cal["min_rate_hz"])
    max_rate = float(cal["max_rate_hz"])
    train_counts = counts["train"]
    train_pred = predictions["train"]
    train_mean = mean_rates_from_counts(train_counts, bin_size_ms, min_rate, max_rate)
    reference = {split: _broadcast_reference(train_mean, value) for split, value in counts.items()}
    scale = fit_multiplicative_rate_scale(train_counts, train_pred, bin_size_ms)
    bias = fit_log_rate_bias(train_counts, train_pred, bin_size_ms)
    best_alpha, train_blend = choose_best_blend_alpha(
        train_counts,
        train_pred,
        train_mean,
        [float(value) for value in cal["blend_alpha_grid"]],
        bin_size_ms,
    )
    rows = []
    blend_rows = list(train_blend.to_dict(orient="records"))
    for split_name in config["evaluation"]["evaluate_splits"]:
        split_key = str(split_name)
        raw = predictions[split_key]
        rows.append(
            _metric_row(
                "raw_lfads",
                split_key,
                counts[split_key],
                raw,
                reference[split_key],
                bin_size_ms,
                "direct_model",
                None,
                "uncalibrated direct model",
            )
        )
        rows.append(
            _metric_row(
                "multiplicative_per_neuron",
                split_key,
                counts[split_key],
                apply_multiplicative_rate_scale(raw, scale, min_rate, max_rate),
                reference[split_key],
                bin_size_ms,
                "direct_model",
                scale,
                "scale fit on train trials only",
            )
        )
        rows.append(
            _metric_row(
                "log_bias_per_neuron",
                split_key,
                counts[split_key],
                apply_log_rate_bias(raw, bias, min_rate, max_rate),
                reference[split_key],
                bin_size_ms,
                "direct_model",
                np.exp(bias),
                "log-bias fit on train trials only",
            )
        )
        best_blend = blend_with_mean_rate(raw, train_mean, best_alpha)
        rows.append(
            _metric_row(
                "mean_rate_blend",
                split_key,
                counts[split_key],
                best_blend,
                reference[split_key],
                bin_size_ms,
                "direct_model",
                None,
                f"alpha={best_alpha} fit on train trials only",
            )
        )
        if split_key != "train":
            for alpha in [float(value) for value in cal["blend_alpha_grid"]]:
                pred = blend_with_mean_rate(raw, train_mean, alpha)
                metric = evaluate_cosmoothing_predictions(
                    counts[split_key], pred, reference[split_key], bin_size_ms
                )
                blend_rows.append(
                    {
                        "alpha": alpha,
                        "split": split_key,
                        "bin_size_ms": bin_size_ms,
                        "spike_count": metric["spike_count"],
                        "poisson_nll": metric["poisson_nll"],
                        "bits_per_spike": metric["bits_per_spike"],
                        "mean_predicted_rate_hz": metric["mean_predicted_rate_hz"],
                        "notes": "alpha fit on train only; this row is evaluation",
                    }
                )
    return (
        pd.DataFrame(rows, columns=CALIBRATION_COLUMNS),
        pd.DataFrame(blend_rows, columns=BLEND_COLUMNS),
        {"best_alpha": best_alpha, "train_mean_rates_hz": train_mean},
    )


def _validation_value(table: pd.DataFrame, method: str) -> float:
    row = table[(table["split"] == "validation") & (table["method_name"] == method)]
    return float("nan") if row.empty else float(row.iloc[0]["bits_per_spike"])


def _initialized_metrics_table(split_metrics: pd.DataFrame, bin_size_ms: int) -> pd.DataFrame:
    rows = []
    for _, row in split_metrics.iterrows():
        rows.append(
            {
                "method_name": "initialized_lfads",
                "split": str(row["split"]),
                "bin_size_ms": int(bin_size_ms),
                "prediction_source": str(row["prediction_source"]),
                "spike_count": float(row["spike_count"]),
                "zero_fraction": float("nan"),
                "observed_rate_hz": float("nan"),
                "mean_predicted_rate_hz": float(row["mean_predicted_rate_hz"]),
                "poisson_nll": float(row["poisson_nll"]),
                "poisson_log_likelihood": float(row["poisson_log_likelihood"]),
                "reference_log_likelihood": float(row["reference_log_likelihood"]),
                "bits_per_spike": float(row["bits_per_spike"]),
                "behavior_mean_r2": float("nan"),
                "notes": "readout bias initialized from train-only rates",
            }
        )
    return pd.DataFrame(rows, columns=INITIALIZED_COLUMNS)


def _write_figures(output_dir: Path, calibration: pd.DataFrame, blend: pd.DataFrame) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    validation = calibration[calibration["split"] == "validation"]
    plt.figure()
    plt.bar(validation["method_name"], validation["bits_per_spike"])
    plt.ylabel("Validation bits/spike")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(figures / "calibration_bits_comparison.png")
    plt.close()

    curve = blend[blend["split"] == "validation"].sort_values("alpha")
    plt.figure()
    plt.plot(curve["alpha"], curve["bits_per_spike"], marker="o")
    plt.xlabel("Blend alpha")
    plt.ylabel("Validation bits/spike")
    plt.tight_layout()
    plt.savefig(figures / "blend_alpha_curve.png")
    plt.close()

    raw = validation[validation["method_name"] == "raw_lfads"]
    calibrated = validation[validation["method_name"] == "multiplicative_per_neuron"]
    plt.figure()
    plt.bar(
        ["raw", "multiplicative"],
        [
            float(raw.iloc[0]["mean_predicted_rate_hz"]),
            float(calibrated.iloc[0]["mean_predicted_rate_hz"]),
        ],
    )
    plt.ylabel("Mean predicted rate (Hz)")
    plt.tight_layout()
    plt.savefig(figures / "predicted_rate_before_after.png")
    plt.close()


def run_lfads_rate_calibration(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    _validate_config(config)
    cuda = _cuda_diagnostic()
    if bool(config["runtime"].get("fail_if_cuda_unavailable", True)) and not cuda["available"]:
        msg = "CUDA was requested, but torch.cuda.is_available() is False."
        raise RuntimeError(msg)
    repo_root = get_repo_root()
    checkpoint_path = resolve_configured_path(
        str(config["existing_lfads"]["checkpoint_path"]), repo_root
    )
    if not checkpoint_path.exists():
        msg = f"LFADS-style checkpoint is missing: {_relative(checkpoint_path, repo_root)}"
        raise FileNotFoundError(msg)
    output_dir = resolve_configured_path(str(config["reporting"]["output_dir"]), repo_root)
    dataset, dataset_hash, split, mask, window_bins = _prepare_dataset(config)
    device = torch.device("cuda")
    counts, predictions = _direct_predictions(
        dataset, split, mask, config, checkpoint_path, window_bins, device
    )
    calibration, blend, info = _evaluate_calibrations(counts, predictions, config)

    initialized = pd.DataFrame(columns=INITIALIZED_COLUMNS)
    init_calibration_bits = float("nan")
    if bool(config["initialized_lfads"].get("enabled", True)):
        run_dir = output_dir / str(config["initialized_lfads"]["output_dir_name"])
        run_config = build_rate_initialized_lfads_train_config(
            config, dataset.bin_size_ms, window_bins, run_dir
        )
        _train_and_evaluate_run(
            run_config, 0, str(config["initialized_lfads"]["output_dir_name"]), dataset
        )
        split_metrics = pd.read_csv(run_dir / "evaluation" / "split_metrics.csv")
        initialized = _initialized_metrics_table(split_metrics, dataset.bin_size_ms)
        init_counts, init_predictions = _direct_predictions(
            dataset,
            split,
            mask,
            config,
            run_dir / "checkpoints" / "best_validation.pt",
            window_bins,
            device,
        )
        init_cal, _, _ = _evaluate_calibrations(init_counts, init_predictions, config)
        init_best = (
            init_cal[init_cal["split"] == "validation"]
            .sort_values("bits_per_spike", ascending=False)
            .iloc[0]
        )
        init_calibration_bits = float(init_best["bits_per_spike"])
        initialized = pd.concat(
            [
                initialized,
                pd.DataFrame(
                    [
                        {
                            "method_name": f"initialized_{init_best['method_name']}",
                            "split": "validation",
                            "bin_size_ms": dataset.bin_size_ms,
                            "prediction_source": "direct_model",
                            "spike_count": float(init_best["spike_count"]),
                            "zero_fraction": float(init_best["zero_fraction"]),
                            "observed_rate_hz": float(init_best["observed_rate_hz"]),
                            "mean_predicted_rate_hz": float(init_best["mean_predicted_rate_hz"]),
                            "poisson_nll": float(init_best["poisson_nll"]),
                            "poisson_log_likelihood": float(init_best["poisson_log_likelihood"]),
                            "reference_log_likelihood": float(
                                init_best["reference_log_likelihood"]
                            ),
                            "bits_per_spike": init_calibration_bits,
                            "behavior_mean_r2": float("nan"),
                            "notes": "best post-hoc calibration after readout bias initialization",
                        }
                    ],
                    columns=INITIALIZED_COLUMNS,
                ),
            ],
            ignore_index=True,
        )

    refs = dict(config["references"])
    raw_bits = _validation_value(calibration, "raw_lfads")
    mult_bits = _validation_value(calibration, "multiplicative_per_neuron")
    log_bits = _validation_value(calibration, "log_bias_per_neuron")
    blend_bits = _validation_value(calibration, "mean_rate_blend")
    init_direct = initialized[
        (initialized["split"] == "validation")
        & (initialized["method_name"] == "initialized_lfads")
        & (initialized["prediction_source"] == "direct_model")
    ]
    init_bits = float("nan") if init_direct.empty else float(init_direct.iloc[0]["bits_per_spike"])
    candidates = {
        "raw_lfads": raw_bits,
        "multiplicative_per_neuron": mult_bits,
        "log_bias_per_neuron": log_bits,
        "mean_rate_blend": blend_bits,
        "initialized_lfads": init_bits,
        "initialized_calibrated": init_calibration_bits,
    }
    finite_candidates = {key: value for key, value in candidates.items() if np.isfinite(value)}
    best_method = max(finite_candidates, key=finite_candidates.get)
    best_value = finite_candidates[best_method]
    summary = {
        "dataset_name": config["dataset"]["name"],
        "dataset_hash": dataset_hash,
        "bin_size_ms": dataset.bin_size_ms,
        "window_seconds": float(config["window"]["duration_seconds"]),
        "cuda_device": cuda["gpu"],
        "existing_checkpoint_path": _relative(checkpoint_path, repo_root),
        "raw_lfads_validation_bits_per_spike": raw_bits,
        "multiplicative_calibrated_validation_bits_per_spike": mult_bits,
        "log_bias_calibrated_validation_bits_per_spike": log_bits,
        "best_blend_alpha": float(info["best_alpha"]),
        "best_blend_validation_bits_per_spike": blend_bits,
        "initialized_lfads_validation_bits_per_spike": init_bits,
        "initialized_calibrated_validation_bits_per_spike": init_calibration_bits,
        "same_bin_mean_rate_reference": float(refs["same_bin_mean_rate_validation_bits_per_spike"]),
        "same_bin_factor_latent_reference": float(
            refs["same_bin_factor_latent_validation_bits_per_spike"]
        ),
        "previous_20ms_lfads_validation_bits_per_spike": float(
            refs["previous_20ms_lfads_validation_bits_per_spike"]
        ),
        "best_lfads_family_method": best_method,
        "best_lfads_family_validation_bits_per_spike": best_value,
        "calibration_improves_lfads": bool(max(mult_bits, log_bits, blend_bits) > raw_bits),
        "initialization_improves_lfads": bool(init_bits > raw_bits),
        "beats_previous_20ms_lfads": bool(
            best_value > float(refs["previous_20ms_lfads_validation_bits_per_spike"])
        ),
        "beats_same_bin_factor_latent": bool(
            best_value > float(refs["same_bin_factor_latent_validation_bits_per_spike"])
        ),
        "beats_same_bin_mean_rate": bool(
            best_value > float(refs["same_bin_mean_rate_validation_bits_per_spike"])
        ),
        "warnings": [
            "This is local rate-calibration diagnostic work, not an official NLB "
            "leaderboard result.",
            "The model is LFADS-style only, not full LFADS.",
        ],
        "output_dir": str(output_dir),
    }
    return summary, {
        "calibration_metrics": calibration,
        "blend_metrics": blend,
        "initialized_lfads_metrics": initialized,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "dataset_name",
        "cuda_device",
        "bin_size_ms",
        "raw_lfads_validation_bits_per_spike",
        "multiplicative_calibrated_validation_bits_per_spike",
        "log_bias_calibrated_validation_bits_per_spike",
        "best_blend_alpha",
        "best_blend_validation_bits_per_spike",
        "initialized_lfads_validation_bits_per_spike",
        "initialized_calibrated_validation_bits_per_spike",
        "same_bin_mean_rate_reference",
        "same_bin_factor_latent_reference",
        "best_lfads_family_method",
        "beats_previous_20ms_lfads",
        "beats_same_bin_factor_latent",
        "beats_same_bin_mean_rate",
        "output_dir",
    ):
        console.print(f"{key}: {summary.get(key)}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = _load_config(args.config)
        summary, tables = run_lfads_rate_calibration(config)
        output_dir = resolve_configured_path(
            str(config["reporting"]["output_dir"]), get_repo_root()
        )
        write_lfads_rate_calibration_outputs(
            output_dir,
            summary,
            tables["calibration_metrics"],
            tables["blend_metrics"],
            tables["initialized_lfads_metrics"],
        )
        _write_figures(output_dir, tables["calibration_metrics"], tables["blend_metrics"])
        _print_summary(summary)
    except Exception as exc:
        console.print(str(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
