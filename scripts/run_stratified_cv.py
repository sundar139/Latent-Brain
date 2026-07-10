from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml
from rich.console import Console

from latentbrain.data.io import compute_dataset_hash, load_neural_dataset
from latentbrain.data.rebinning import rebin_neural_dataset, validate_rebin_factor
from latentbrain.data.schemas import NeuralDataset
from latentbrain.data.splits import create_neuron_mask
from latentbrain.data.validation import validate_neural_dataset
from latentbrain.eval.fold_balance import (
    compare_fold_balance,
    compute_fold_balance_statistics,
    summarize_fold_balance,
)
from latentbrain.eval.rebinning import compute_window_bins_for_duration
from latentbrain.eval.reporting import write_stratified_cv_outputs
from latentbrain.eval.stratified_cv import (
    ASSIGNMENT_METHODS,
    BEHAVIOR_FALLBACKS,
    FACTOR_LATENT,
    SPLIT_MEAN_RATE_INVALID,
    build_random_folds,
    build_repeated_stratified_folds,
    build_trial_features,
    compare_random_and_stratified,
    score_folds,
    summarize_methods,
    summarize_stratified_cv,
)
from latentbrain.eval.windowing import crop_neural_dataset_time
from latentbrain.paths import get_repo_root, resolve_configured_path

console = Console(markup=False)

_BIN_KEYS = (
    "endpoint_direction_bins",
    "endpoint_distance_bins",
    "mean_speed_bins",
    "population_rate_bins",
    "heldout_rate_bins",
)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local behavior-stratified cross-validation for MC_Maze Small."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/mc_maze_small_stratified_cv.yaml")
    )
    return parser.parse_args(argv)


def _load_config(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"malformed stratified cv config: {path}"
        raise ValueError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"stratified cv config must contain a mapping: {path}"
        raise ValueError(msg)
    _validate_config(raw)
    return raw


def _validate_config(config: dict[str, Any]) -> None:
    validate_rebin_factor(
        int(config["dataset"]["original_bin_size_ms"]),
        int(config["binning"]["target_bin_size_ms"]),
    )
    window_bins = compute_window_bins_for_duration(
        float(config["window"]["duration_seconds"]),
        int(config["binning"]["target_bin_size_ms"]),
    )
    if window_bins <= 0:
        msg = "window duration must convert to positive integer bins"
        raise ValueError(msg)
    cross_validation = dict(config["cross_validation"])
    if int(cross_validation["fold_count"]) < 3:
        msg = "cross_validation.fold_count must be at least 3"
        raise ValueError(msg)
    if int(cross_validation["repeats"]) < 2:
        msg = "cross_validation.repeats must be at least 2"
        raise ValueError(msg)
    if str(cross_validation["assignment_method"]) not in ASSIGNMENT_METHODS:
        msg = f"cross_validation.assignment_method must be one of {ASSIGNMENT_METHODS}"
        raise ValueError(msg)
    if str(cross_validation["fallback_when_behavior_missing"]) not in BEHAVIOR_FALLBACKS:
        msg = f"cross_validation.fallback_when_behavior_missing must be one of {BEHAVIOR_FALLBACKS}"
        raise ValueError(msg)
    stratification = dict(cross_validation["stratification"])
    for key in _BIN_KEYS:
        if int(stratification[key]) <= 0:
            msg = f"cross_validation.stratification.{key} must be positive"
            raise ValueError(msg)
    if not any(value for key, value in stratification.items() if key.startswith("use_")):
        msg = "at least one stratification variable must be enabled"
        raise ValueError(msg)
    if int(config["statistics"]["bootstrap_repeats"]) <= 0:
        msg = "statistics.bootstrap_repeats must be positive"
        raise ValueError(msg)
    if str(config["scoring"]["reference_model"]) != "train_heldout_mean_rate":
        msg = "scoring.reference_model must be train_heldout_mean_rate"
        raise ValueError(msg)
    methods = {str(method["name"]): dict(method) for method in config["methods"]}
    for name, method in methods.items():
        if not bool(method.get("valid_model", False)) and bool(
            method.get("reportable_as_model_performance", False)
        ):
            msg = f"invalid or reference method must not be reportable as performance: {name}"
            raise ValueError(msg)
    if bool(methods.get("train_mean_rate", {}).get("reportable_as_model_performance", False)):
        msg = "train_mean_rate must not be reportable as model performance"
        raise ValueError(msg)
    if not bool(methods.get(FACTOR_LATENT, {}).get("valid_model", False)):
        msg = "factor_latent must be marked a valid model"
        raise ValueError(msg)
    if bool(methods.get(SPLIT_MEAN_RATE_INVALID, {}).get("valid_model", False)):
        msg = "split_mean_rate_invalid must not be marked a valid model"
        raise ValueError(msg)


def _prepare_dataset(config: dict[str, Any]) -> tuple[NeuralDataset, str]:
    repo_root = get_repo_root()
    processed_path = resolve_configured_path(str(config["dataset"]["processed_path"]), repo_root)
    if not processed_path.exists():
        msg = f"Processed dataset is missing: {processed_path}"
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
    return windowed, dataset_hash


def _write_figures(
    output_dir: Path,
    scores: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    fold_balance: pd.DataFrame,
    random_scores: pd.DataFrame,
) -> None:
    import matplotlib  # type: ignore[import-untyped]  # noqa: PLC0415

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]  # noqa: PLC0415

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    if not scores.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        for method_name, group in scores.groupby("method_name", sort=True):
            ax.hist(group["unified_bits_per_spike"], bins=12, alpha=0.55, label=str(method_name))
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Unified bits/spike (evaluation fold)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / "stratified_cv_score_distribution.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    if fold_assignments["endpoint_angle_rad"].notna().any():
        first = fold_assignments[fold_assignments["repeat_index"] == 0]
        for fold_index, group in first.groupby("fold_index", sort=True):
            ax.hist(
                group["endpoint_angle_rad"],
                bins=8,
                range=(-np.pi, np.pi),
                histtype="step",
                label=f"fold {fold_index}",
            )
        ax.set_xlabel("Endpoint direction (rad)")
        ax.set_ylabel("Trials")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "behavior unavailable", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(figures / "fold_balance_endpoint_direction.png", dpi=150)
    plt.close(fig)

    if not fold_balance.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(
            fold_balance["fold_index"],
            fold_balance["mean_population_rate_hz"],
            "o",
            label="population rate",
        )
        ax.plot(
            fold_balance["fold_index"],
            fold_balance["mean_heldout_rate_hz"],
            "s",
            label="held-out rate",
        )
        ax.set_xlabel("Fold index")
        ax.set_ylabel("Mean rate (Hz)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / "fold_balance_rate_distributions.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(
            fold_balance["fold_index"],
            fold_balance["mean_endpoint_distance"],
            "o",
            label="distance",
        )
        ax.plot(fold_balance["fold_index"], fold_balance["mean_speed"], "s", label="mean speed")
        ax.set_xlabel("Fold index")
        ax.set_ylabel("Behavior summary")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / "fold_balance_distance_speed.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    stratified = scores[scores["method_name"] == FACTOR_LATENT]["unified_bits_per_spike"]
    random_values = random_scores[random_scores["method_name"] == FACTOR_LATENT][
        "unified_bits_per_spike"
    ]
    if not stratified.empty and not random_values.empty:
        ax.boxplot(
            [random_values.to_numpy(), stratified.to_numpy()], tick_labels=["random", "stratified"]
        )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("Factor-latent unified bits/spike")
    else:
        ax.text(0.5, 0.5, "comparison unavailable", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(figures / "random_vs_stratified_variance.png", dpi=150)
    plt.close(fig)


def run_stratified_cv(config: dict[str, Any]) -> dict[str, Any]:
    dataset, dataset_hash = _prepare_dataset(config)
    cross_validation = dict(config["cross_validation"])
    statistics = dict(config["statistics"])

    # Trial features use one reference neuron mask so strata are fixed across repeats; each
    # repeat still resamples its own held-in/held-out mask for scoring.
    reference_mask = create_neuron_mask(
        dataset.spikes.shape[2],
        float(cross_validation["heldout_neuron_fraction"]),
        seed=int(cross_validation["base_seed"]),
    )
    trial_features = build_trial_features(
        dataset.spikes,
        dataset.behavior,
        list(dataset.behavior_names) if dataset.behavior_names is not None else None,
        int(config["binning"]["target_bin_size_ms"]),
        np.flatnonzero(reference_mask.heldout),
    )
    fold_assignments = build_repeated_stratified_folds(trial_features, config)
    fold_balance = compute_fold_balance_statistics(fold_assignments)
    comparisons = compare_fold_balance(fold_balance)
    balance_summary = summarize_fold_balance(fold_balance, comparisons)

    scores = score_folds(dataset, fold_assignments, config)
    method_summary = summarize_methods(
        scores,
        int(statistics["bootstrap_repeats"]),
        float(statistics["confidence_interval"]),
        int(statistics["bootstrap_seed"]),
    )

    random_scores = pd.DataFrame(columns=scores.columns)
    comparison: dict[str, Any] = {
        "stratified_factor_latent_mean": float("nan"),
        "stratified_factor_latent_std": float("nan"),
        "stratification_reduces_variance": False,
        "variance_reduction_fraction": float("nan"),
    }
    if bool(cross_validation.get("compare_random_splits", False)):
        random_folds = build_random_folds(trial_features, config)
        random_scores = score_folds(dataset, random_folds, config)
        comparison = compare_random_and_stratified(scores, random_scores)

    summary = summarize_stratified_cv(
        scores,
        method_summary,
        balance_summary,
        comparison,
        dict(config["references"]),
        config,
    )
    summary.update(
        {
            "dataset_name": config["dataset"]["name"],
            "dataset_hash": dataset_hash,
            "bin_size_ms": int(config["binning"]["target_bin_size_ms"]),
            "window_seconds": float(config["window"]["duration_seconds"]),
        }
    )

    output_dir = resolve_configured_path(str(config["reporting"]["output_dir"]), get_repo_root())
    write_stratified_cv_outputs(
        output_dir, summary, scores, fold_assignments, fold_balance, comparisons, method_summary
    )
    _write_figures(output_dir, scores, fold_assignments, fold_balance, random_scores)
    summary["output_dir"] = str(output_dir)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = get_repo_root()
    args = _parse_args(argv)
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    if not config_path.exists():
        console.print(f"Config file is missing: {config_path}")
        return 2
    try:
        config = _load_config(config_path)
        summary = run_stratified_cv(config)
    except (OSError, ValueError, FileNotFoundError) as exc:
        console.print(f"Stratified cross-validation failed: {exc}")
        return 2
    for key in (
        "dataset_name",
        "bin_size_ms",
        "window_seconds",
        "fold_count",
        "repeats",
        "total_folds",
        "factor_latent_mean_unified_bits_per_spike",
        "factor_latent_std_unified_bits_per_spike",
        "factor_latent_ci95_low",
        "factor_latent_ci95_high",
        "factor_latent_positive_fraction",
        "split_mean_rate_invalid_mean_unified_bits_per_spike",
        "invalid_controls_excluded_from_valid_model_selection",
        "stratification_reduces_variance",
        "variance_reduction_fraction",
        "fold_balance_warning",
        "recommended_reporting_mode",
        "carried_forward_method",
        "output_dir",
    ):
        console.print(f"{key}: {summary.get(key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
