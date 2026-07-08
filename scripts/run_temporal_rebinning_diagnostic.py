from __future__ import annotations

import argparse
import copy
import json
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
from latentbrain.eval.baselines import evaluate_mean_rate_baseline
from latentbrain.eval.latent_baseline import run_factor_latent_baseline
from latentbrain.eval.rebinning import (
    BASELINE_COLUMNS,
    LFADS_COLUMNS,
    build_binning_comparison_row,
    compute_sparsity_summary,
    compute_window_bins_for_duration,
)
from latentbrain.eval.reporting import write_temporal_rebinning_outputs
from latentbrain.eval.windowing import crop_neural_dataset_time
from latentbrain.paths import get_repo_root, resolve_configured_path
from latentbrain.train.lfads_tuning import _train_and_evaluate_run
from latentbrain.train.rebinned_lfads import build_rebinned_lfads_train_config

console = Console(markup=False)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local temporal rebinning diagnostics.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mc_maze_small_temporal_rebinning.yaml"),
    )
    return parser.parse_args(argv)


def _load_config(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"malformed temporal rebinning config: {path}"
        raise ValueError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"temporal rebinning config must contain a mapping: {path}"
        raise ValueError(msg)
    _validate_config(raw)
    return raw


def _validate_config(config: dict[str, Any]) -> None:
    original_bin = int(config["dataset"]["original_bin_size_ms"])
    if original_bin <= 0:
        msg = "dataset.original_bin_size_ms must be positive"
        raise ValueError(msg)
    targets = [int(value) for value in config["binning"]["target_bin_size_ms"]]
    if not targets:
        msg = "binning.target_bin_size_ms must not be empty"
        raise ValueError(msg)
    for target in targets:
        validate_rebin_factor(original_bin, target)
        compute_window_bins_for_duration(float(config["window"]["duration_seconds"]), target)
    train_bins = [int(value) for value in config["binning"].get("train_lfads_for_bin_size_ms", [])]
    if not set(train_bins).issubset(set(targets)):
        msg = "LFADS training bin sizes must be a subset of target bin sizes"
        raise ValueError(msg)
    if str(config["runtime"].get("device")) != "cuda":
        msg = "runtime.device must be cuda for LFADS temporal rebinning diagnostics"
        raise ValueError(msg)
    splits = config["splits"]
    total = (
        float(splits["train_fraction"])
        + float(splits["validation_fraction"])
        + float(splits["test_fraction"])
    )
    if abs(total - 1.0) > 1e-8:
        msg = "split fractions must sum to 1.0"
        raise ValueError(msg)
    settings = config["lfads_settings"]
    for key in (
        "encoder_hidden_dim",
        "generator_hidden_dim",
        "latent_dim",
        "factor_dim",
        "epochs",
        "batch_size",
    ):
        if int(settings[key]) <= 0:
            msg = f"lfads_settings.{key} must be positive"
            raise ValueError(msg)


def _cuda_diagnostic() -> dict[str, Any]:
    return {
        "torch": torch.__version__,
        "available": bool(torch.cuda.is_available()),
        "torch_cuda": torch.version.cuda,
        "device_count": int(torch.cuda.device_count()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
    }


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _sparsity_lookup(sparsity: pd.DataFrame, bin_size_ms: int, split: str) -> dict[str, Any]:
    row = sparsity[(sparsity["bin_size_ms"] == bin_size_ms) & (sparsity["split"] == split)]
    return {} if row.empty else row.iloc[0].to_dict()


def _mean_rate_config(config: dict[str, Any]) -> dict[str, Any]:
    settings = config["lfads_settings"]
    return {
        "baseline": {
            "name": "mean_rate",
            "min_rate_hz": float(settings["min_rate_hz"]),
            "max_rate_hz": float(settings["max_rate_hz"]),
            "use_train_trials_only": True,
        },
        "evaluation": {
            "evaluate_splits": list(config["evaluation"]["evaluate_splits"]),
            "evaluate_neuron_groups": ["heldout"],
        },
    }


def _factor_latent_config(config: dict[str, Any]) -> dict[str, Any]:
    settings = config["baseline_settings"]["factor_latent"]
    lfads = config["lfads_settings"]
    return {
        "features": {
            "input_neuron_group": "heldin",
            "target_neuron_group": "heldout",
            "smoothing": {
                "method": "gaussian",
                "sigma_ms": float(settings["smoothing_sigma_ms"]),
                "truncate": 4.0,
            },
            "convert_to_hz": True,
            "standardize_features": bool(settings["standardize_features"]),
        },
        "latent_model": {
            "name": "factor_analysis",
            "latent_dim": int(settings["latent_dim"]),
            "random_state": int(config["splits"]["seed"]),
            "max_iter": 1000,
            "tol": 1.0e-4,
            "train_trials_only": True,
        },
        "heldout_decoder": {
            "name": "ridge",
            "alpha": float(settings["heldout_decoder_alpha"]),
            "fit_intercept": True,
            "min_rate_hz": float(lfads["min_rate_hz"]),
            "max_rate_hz": float(lfads["max_rate_hz"]),
            "train_trials_only": True,
        },
        "behavior_decoder": {"enabled": False},
        "reference": {"name": "train_mean_rate", "fit_train_trials_only": True},
        "evaluation": {
            "evaluate_splits": list(config["evaluation"]["evaluate_splits"]),
            "primary_split": str(config["evaluation"].get("primary_split", "validation")),
        },
    }


def _baseline_rows(
    dataset: NeuralDataset,
    split: TrialSplit,
    neuron_mask: NeuronMask,
    config: dict[str, Any],
    bin_size_ms: int,
    sparsity: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if bool(config["baseline_settings"]["mean_rate"].get("enabled", True)):
        split_metrics, _, _ = evaluate_mean_rate_baseline(
            dataset, split, neuron_mask, _mean_rate_config(config)
        )
        heldout = split_metrics[split_metrics["neuron_group"] == "heldout"]
        for _, metric in heldout.iterrows():
            split_name = str(metric["split"])
            sparse = _sparsity_lookup(sparsity, bin_size_ms, split_name)
            rows.append(
                build_binning_comparison_row(
                    "mean_rate",
                    bin_size_ms,
                    split_name,
                    metric.to_dict(),
                    {
                        "prediction_source": "constant_rate",
                        "time_bins": int(metric["n_time_bins"]),
                        "window_seconds": float(metric["n_time_bins"] * bin_size_ms / 1000.0),
                        "zero_fraction": sparse.get("zero_fraction", float("nan")),
                        "observed_rate_hz": sparse.get("observed_rate_hz", float("nan")),
                    },
                )
            )
    if bool(config["baseline_settings"]["factor_latent"].get("enabled", True)):
        split_metrics, _, _, _, _ = run_factor_latent_baseline(
            dataset, split, neuron_mask, _factor_latent_config(config)
        )
        for _, metric in split_metrics.iterrows():
            split_name = str(metric["split"])
            sparse = _sparsity_lookup(sparsity, bin_size_ms, split_name)
            rows.append(
                build_binning_comparison_row(
                    "factor_latent",
                    bin_size_ms,
                    split_name,
                    metric.to_dict(),
                    {
                        "prediction_source": "factor_decoder",
                        "time_bins": int(metric["n_time_bins"]),
                        "window_seconds": float(metric["n_time_bins"] * bin_size_ms / 1000.0),
                        "zero_fraction": sparse.get("zero_fraction", float("nan")),
                        "observed_rate_hz": sparse.get("observed_rate_hz", float("nan")),
                    },
                )
            )
    return rows


def _validation_bits(baseline: pd.DataFrame, bin_size_ms: int, method_name: str) -> float:
    row = baseline[
        (baseline["bin_size_ms"] == bin_size_ms)
        & (baseline["method_name"] == method_name)
        & (baseline["split"] == "validation")
    ]
    return float("nan") if row.empty else float(row.iloc[0]["bits_per_spike"])


def _lfads_rows_for_run(
    run_dir: Path,
    run_id: str,
    bin_size_ms: int,
    window_seconds: float,
    sparsity: pd.DataFrame,
) -> list[dict[str, Any]]:
    split_metrics = pd.read_csv(run_dir / "evaluation" / "split_metrics.csv")
    final = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
    checkpoint = run_dir / "checkpoints" / "best_validation.pt"
    rows = []
    for _, metric in split_metrics.iterrows():
        split_name = str(metric["split"])
        sparse = _sparsity_lookup(sparsity, bin_size_ms, split_name)
        rows.append(
            {
                "bin_size_ms": int(bin_size_ms),
                "run_id": run_id,
                "split": split_name,
                "prediction_source": str(metric["prediction_source"]),
                "time_bins": int(metric["n_time_bins"]),
                "window_seconds": window_seconds,
                "spike_count": float(metric["spike_count"]),
                "zero_fraction": float(sparse.get("zero_fraction", float("nan"))),
                "observed_rate_hz": float(sparse.get("observed_rate_hz", float("nan"))),
                "validation_total_loss": float(final.get("validation_total_loss", float("nan"))),
                "heldout_prediction_loss": float(
                    final.get("validation_heldout_prediction_loss", float("nan"))
                ),
                "poisson_nll": float(metric["poisson_nll"]),
                "poisson_log_likelihood": float(metric["poisson_log_likelihood"]),
                "reference_log_likelihood": float(metric["reference_log_likelihood"]),
                "bits_per_spike": float(metric["bits_per_spike"]),
                "behavior_mean_r2": float("nan"),
                "checkpoint_path": str(checkpoint),
            }
        )
    return rows


def _write_figures(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    sparsity = tables["sparsity"]
    baseline = tables["baseline_metrics"]
    lfads = tables["lfads_metrics"]

    validation_sparsity = sparsity[sparsity["split"] == "validation"].sort_values("bin_size_ms")
    if {"bin_size_ms", "zero_fraction"}.issubset(validation_sparsity.columns):
        plt.figure()
        plt.plot(
            validation_sparsity["bin_size_ms"], validation_sparsity["zero_fraction"], marker="o"
        )
        plt.xlabel("Bin size (ms)")
        plt.ylabel("Validation held-out zero fraction")
        plt.tight_layout()
        plt.savefig(figures / "zero_fraction_by_bin_size.png")
        plt.close()

    if {"bin_size_ms", "observed_rate_hz"}.issubset(validation_sparsity.columns):
        plt.figure()
        plt.plot(
            validation_sparsity["bin_size_ms"], validation_sparsity["observed_rate_hz"], marker="o"
        )
        plt.xlabel("Bin size (ms)")
        plt.ylabel("Validation observed rate (Hz)")
        plt.tight_layout()
        plt.savefig(figures / "observed_rate_by_bin_size.png")
        plt.close()

    plt.figure()
    if {"split", "method_name", "bin_size_ms", "bits_per_spike"}.issubset(baseline.columns):
        validation_baselines = baseline[baseline["split"] == "validation"]
        for method, group in validation_baselines.groupby("method_name"):
            group = group.sort_values("bin_size_ms")
            plt.plot(group["bin_size_ms"], group["bits_per_spike"], marker="o", label=str(method))
    if {"split", "prediction_source", "bin_size_ms", "bits_per_spike"}.issubset(lfads.columns):
        validation_lfads = lfads[
            (lfads["split"] == "validation") & (lfads["prediction_source"] == "direct_model")
        ]
        if not validation_lfads.empty:
            validation_lfads = validation_lfads.sort_values("bin_size_ms")
            plt.plot(
                validation_lfads["bin_size_ms"],
                validation_lfads["bits_per_spike"],
                marker="o",
                label="lfads_style",
            )
    if plt.gca().has_data():
        plt.xlabel("Bin size (ms)")
        plt.ylabel("Validation bits/spike")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures / "validation_bits_by_bin_size.png")
    plt.close()


def run_temporal_rebinning_diagnostic(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    _validate_config(config)
    cuda = _cuda_diagnostic()
    if bool(config["runtime"].get("fail_if_cuda_unavailable", True)) and not cuda["available"]:
        msg = "CUDA was requested, but torch.cuda.is_available() is False."
        raise RuntimeError(msg)
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
    split = create_trial_split(
        dataset.trial_ids,
        float(config["splits"]["train_fraction"]),
        float(config["splits"]["validation_fraction"]),
        float(config["splits"]["test_fraction"]),
        seed=int(config["splits"]["seed"]),
    )
    neuron_mask = create_neuron_mask(
        dataset.spikes.shape[2],
        float(config["splits"]["heldout_neuron_fraction"]),
        seed=int(config["splits"]["seed"]),
    )
    validate_trial_split(split, dataset.trial_ids)
    validate_neuron_mask(neuron_mask, dataset.spikes.shape[2])

    output_dir = resolve_configured_path(str(config["reporting"]["output_dir"]), repo_root)
    targets = [int(value) for value in config["binning"]["target_bin_size_ms"]]
    train_bins = [int(value) for value in config["binning"]["train_lfads_for_bin_size_ms"]]
    sparsity_frames: list[pd.DataFrame] = []
    baseline_rows: list[dict[str, Any]] = []
    lfads_rows: list[dict[str, Any]] = []
    datasets_by_bin = {}
    window_bins_by_bin = {}

    for bin_size in targets:
        rebinned = (
            dataset
            if bin_size == int(config["dataset"]["original_bin_size_ms"])
            else rebin_neural_dataset(dataset, bin_size)
        )
        window_bins = compute_window_bins_for_duration(
            float(config["window"]["duration_seconds"]), bin_size
        )
        windowed = crop_neural_dataset_time(
            rebinned, window_bins, str(config["window"].get("crop_policy", "from_start"))
        )
        datasets_by_bin[bin_size] = windowed
        window_bins_by_bin[bin_size] = window_bins
        sparsity = compute_sparsity_summary(windowed, split, neuron_mask, bin_size, window_bins)
        sparsity_frames.append(sparsity)
        baseline_rows.extend(
            _baseline_rows(windowed, split, neuron_mask, config, bin_size, sparsity)
        )

    sparsity_table = pd.concat(sparsity_frames, ignore_index=True)
    baseline_table = pd.DataFrame(baseline_rows, columns=BASELINE_COLUMNS)

    for run_index, bin_size in enumerate(train_bins):
        run_dir = output_dir / "runs" / f"bin_{bin_size}ms"
        run_base = copy.deepcopy(config)
        run_base["evaluation"]["baseline_references"] = {
            "window_matched_mean_rate_validation_bits_per_spike": _validation_bits(
                baseline_table, bin_size, "mean_rate"
            ),
            "window_matched_factor_latent_validation_bits_per_spike": _validation_bits(
                baseline_table, bin_size, "factor_latent"
            ),
            "previous_lfads_masked_direct_validation_bits_per_spike": float(
                config["evaluation"]["references"]["best_tuned_5ms_lfads_validation_bits_per_spike"]
            ),
        }
        run_config = build_rebinned_lfads_train_config(
            run_base, bin_size, int(window_bins_by_bin[bin_size]), run_dir
        )
        _train_and_evaluate_run(
            run_config, run_index, f"bin_{bin_size}ms", datasets_by_bin[bin_size]
        )
        lfads_rows.extend(
            _lfads_rows_for_run(
                run_dir,
                f"bin_{bin_size}ms",
                bin_size,
                float(config["window"]["duration_seconds"]),
                sparsity_table,
            )
        )

    lfads_table = pd.DataFrame(lfads_rows, columns=LFADS_COLUMNS)
    validation_sparsity = sparsity_table[sparsity_table["split"] == "validation"]
    validation_lfads = lfads_table[
        (lfads_table["split"] == "validation")
        & (lfads_table["prediction_source"] == "direct_model")
    ]
    best_lfads_bin = None
    if not validation_lfads.empty:
        best_lfads_bin = int(
            validation_lfads.sort_values("bits_per_spike", ascending=False).iloc[0]["bin_size_ms"]
        )
    zero_by_bin = {
        str(int(row["bin_size_ms"])): float(row["zero_fraction"])
        for _, row in validation_sparsity.iterrows()
    }
    mean_bits = {
        str(bin_size): _validation_bits(baseline_table, bin_size, "mean_rate")
        for bin_size in targets
    }
    factor_bits = {
        str(bin_size): _validation_bits(baseline_table, bin_size, "factor_latent")
        for bin_size in targets
    }
    lfads_bits = {
        str(int(row["bin_size_ms"])): float(row["bits_per_spike"])
        for _, row in validation_lfads.iterrows()
    }
    lfads_beats_mean = any(
        lfads_bits.get(str(bin_size), -np.inf) > mean_bits[str(bin_size)] for bin_size in train_bins
    )
    best_5ms = float(
        config["evaluation"]["references"]["best_tuned_5ms_lfads_validation_bits_per_spike"]
    )
    summary = {
        "dataset_name": config["dataset"]["name"],
        "dataset_hash": dataset_hash,
        "original_bin_size_ms": int(config["dataset"]["original_bin_size_ms"]),
        "target_bin_sizes_ms": targets,
        "trained_lfads_bin_sizes_ms": train_bins,
        "window_seconds": float(config["window"]["duration_seconds"]),
        "cuda_device": cuda["gpu"],
        "validation_zero_fraction_by_bin_size": zero_by_bin,
        "mean_rate_validation_bits_per_spike_by_bin_size": mean_bits,
        "factor_latent_validation_bits_per_spike_by_bin_size": factor_bits,
        "lfads_validation_bits_per_spike_by_bin_size": lfads_bits,
        "best_lfads_bin_size_ms": best_lfads_bin,
        "lfads_beat_same_bin_mean_rate": bool(lfads_beats_mean),
        "coarser_lfads_improved_over_5ms": bool(
            any(value > best_5ms for value in lfads_bits.values())
        ),
        "lfads_improves_at_coarser_bins": bool(
            any(value > best_5ms for value in lfads_bits.values())
        ),
        "coarser_bins_reduce_zero_fraction": bool(
            any(
                float(zero_by_bin[str(bin_size)]) < float(zero_by_bin[str(targets[0])])
                for bin_size in targets[1:]
            )
        ),
        "warnings": [
            (
                "Bits/spike values across different bin sizes are diagnostic and should not be "
                "treated as direct benchmark comparisons."
            ),
            (
                "This is local temporal-binning diagnostic work, not an official NLB "
                "leaderboard result."
            ),
            "The model is LFADS-style only, not full LFADS.",
        ],
        "output_dir": str(output_dir),
    }
    return summary, {
        "sparsity": sparsity_table,
        "baseline_metrics": baseline_table,
        "lfads_metrics": lfads_table,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    console.print(f"dataset: {summary.get('dataset_name')}")
    console.print(f"gpu: {summary.get('cuda_device')}")
    console.print(f"bin_sizes_evaluated: {summary.get('target_bin_sizes_ms')}")
    console.print(
        "validation_zero_fraction_by_bin_size: "
        f"{summary.get('validation_zero_fraction_by_bin_size')}"
    )
    console.print(
        "mean_rate_validation_bits_per_spike_by_bin_size: "
        f"{summary.get('mean_rate_validation_bits_per_spike_by_bin_size')}"
    )
    console.print(
        "factor_latent_validation_bits_per_spike_by_bin_size: "
        f"{summary.get('factor_latent_validation_bits_per_spike_by_bin_size')}"
    )
    console.print(
        "lfads_validation_bits_per_spike_by_bin_size: "
        f"{summary.get('lfads_validation_bits_per_spike_by_bin_size')}"
    )
    console.print(f"best_lfads_bin_size_ms: {summary.get('best_lfads_bin_size_ms')}")
    console.print(f"lfads_beat_same_bin_mean_rate: {summary.get('lfads_beat_same_bin_mean_rate')}")
    console.print(
        f"coarser_lfads_improved_over_5ms: {summary.get('coarser_lfads_improved_over_5ms')}"
    )
    console.print(f"output_dir: {summary.get('output_dir')}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = _load_config(args.config)
        repo_root = get_repo_root()
        processed_path = resolve_configured_path(
            str(config["dataset"]["processed_path"]), repo_root
        )
        if not processed_path.exists():
            msg = f"Processed dataset is missing: {_relative(processed_path, repo_root)}"
            raise FileNotFoundError(msg)
        cuda = _cuda_diagnostic()
        if bool(config["runtime"].get("fail_if_cuda_unavailable", True)) and not cuda["available"]:
            msg = "CUDA was requested, but torch.cuda.is_available() is False."
            raise RuntimeError(msg)
        summary, tables = run_temporal_rebinning_diagnostic(config)
        output_dir = resolve_configured_path(str(config["reporting"]["output_dir"]), repo_root)
        write_temporal_rebinning_outputs(
            output_dir,
            summary,
            tables["sparsity"],
            tables["baseline_metrics"],
            tables["lfads_metrics"],
        )
        _write_figures(output_dir, tables)
        _print_summary(summary)
    except Exception as exc:
        console.print(str(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
