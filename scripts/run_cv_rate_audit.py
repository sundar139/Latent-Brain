from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import yaml
from rich.console import Console

from latentbrain.data.io import compute_dataset_hash, load_neural_dataset
from latentbrain.data.rebinning import rebin_neural_dataset, validate_rebin_factor
from latentbrain.data.schemas import NeuralDataset
from latentbrain.data.validation import validate_neural_dataset
from latentbrain.eval.cv_rate_audit import (
    build_reporting_recommendations,
    decompose_rate_offset,
    run_factor_analysis_random_state_sensitivity,
    run_rate_control_audit,
    run_repeated_split_factor_latent,
    summarize_cv_rate_audit,
    summarize_methods,
)
from latentbrain.eval.rate_controls import (
    KNOWN_CONTROLS,
)
from latentbrain.eval.rebinning import compute_window_bins_for_duration
from latentbrain.eval.reporting import write_cv_rate_audit_outputs
from latentbrain.eval.windowing import crop_neural_dataset_time
from latentbrain.paths import get_repo_root, resolve_configured_path

console = Console(markup=False)

REQUIRED_REFERENCES = (
    "accepted_split_seed",
    "accepted_factor_latent_validation_mean",
    "accepted_factor_latent_test_mean",
    "repeated_split_factor_latent_test_mean",
    "split_mean_rate_validation_reference",
    "split_mean_rate_test_reference",
    "train_mean_validation_bits_per_spike",
)

RATE_CONTROL_FLAGS = {
    "include_train_mean_rate",
    "include_split_mean_rate_invalid",
    "include_train_per_neuron_mean_rate",
    "include_train_population_scaled_mean_rate",
    "include_train_split_rate_calibrated_factor_latent",
    "include_oracle_split_scaled_factor_latent_invalid",
}


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local cross-validated rate-offset audit.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/mc_maze_small_cv_rate_audit.yaml")
    )
    return parser.parse_args(argv)


def _load_config(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"malformed cv rate audit config: {path}"
        raise ValueError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"cv rate audit config must contain a mapping: {path}"
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
    splits = config["splits"]
    seeds = [int(seed) for seed in splits["split_seeds"]]
    if len(set(seeds)) != len(seeds):
        msg = "splits.split_seeds must be unique"
        raise ValueError(msg)
    if len(seeds) < 10:
        msg = "at least ten split seeds are required"
        raise ValueError(msg)
    states = [int(state) for state in splits["factor_analysis_random_states"]]
    if len(set(states)) != len(states):
        msg = "splits.factor_analysis_random_states must be unique"
        raise ValueError(msg)
    if len(states) < 3:
        msg = "at least three FactorAnalysis random states are required"
        raise ValueError(msg)
    unknown = set(config["rate_controls"]) - RATE_CONTROL_FLAGS
    if unknown:
        msg = f"unrecognized rate controls: {sorted(unknown)}"
        raise ValueError(msg)
    if int(config["statistics"]["bootstrap_repeats"]) <= 0:
        msg = "statistics.bootstrap_repeats must be positive"
        raise ValueError(msg)
    if str(config["scoring"]["reference_model"]) != "train_heldout_mean_rate":
        msg = "scoring.reference_model must be train_heldout_mean_rate"
        raise ValueError(msg)
    references = dict(config["references"])
    for key in REQUIRED_REFERENCES:
        if key not in references:
            msg = f"references.{key} is required"
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
    repeated_scores: pd.DataFrame,
    fa_sensitivity: pd.DataFrame,
    method_summary: pd.DataFrame,
    decomposition: pd.DataFrame,
) -> None:
    import matplotlib  # type: ignore[import-untyped]  # noqa: PLC0415

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]  # noqa: PLC0415

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    if not repeated_scores.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(
            repeated_scores["validation_unified_bits_per_spike"],
            bins=15,
            alpha=0.6,
            label="validation",
        )
        ax.hist(repeated_scores["test_unified_bits_per_spike"], bins=15, alpha=0.6, label="test")
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Unified bits/spike")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "repeated_split_score_distribution.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        for state, group in repeated_scores.groupby("factor_analysis_random_state", sort=True):
            ordered = group.sort_values("split_seed")
            ax.plot(
                ordered["split_seed"],
                ordered["test_unified_bits_per_spike"],
                marker="o",
                label=f"rs={state}",
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Split seed")
        ax.set_ylabel("Test unified bits/spike")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / "validation_test_by_split.png", dpi=150)
        plt.close(fig)

    if not fa_sensitivity.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        states = [str(value) for value in fa_sensitivity["factor_analysis_random_state"]]
        ax.plot(
            states,
            fa_sensitivity["validation_unified_bits_per_spike"],
            marker="o",
            label="validation",
        )
        ax.plot(states, fa_sensitivity["test_unified_bits_per_spike"], marker="s", label="test")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("FactorAnalysis random_state")
        ax.set_ylabel("Unified bits/spike")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "factor_analysis_random_state_sensitivity.png", dpi=150)
        plt.close(fig)

    if not method_summary.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        ordered = method_summary.sort_values("mean_test_unified_bits_per_spike")
        colors = ["#4C78A8" if bool(valid) else "#E45756" for valid in ordered["valid_model"]]
        ax.barh(ordered["method_name"], ordered["mean_test_unified_bits_per_spike"], color=colors)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Mean test unified bits/spike (red = invalid control)")
        fig.tight_layout()
        fig.savefig(figures / "rate_control_comparison.png", dpi=150)
        plt.close(fig)

    if not decomposition.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        test_rows = decomposition[decomposition["split"] == "test"].sort_values("split_seed")
        ax.plot(
            test_rows["split_seed"],
            test_rows["valid_calibration_gain"],
            marker="o",
            label="valid calibration gain",
        )
        ax.plot(
            test_rows["split_seed"],
            test_rows["invalid_oracle_gain"],
            marker="s",
            label="invalid oracle gain",
        )
        ax.plot(
            test_rows["split_seed"],
            test_rows["split_mean_advantage_over_factor_latent"],
            marker="^",
            label="invalid split-mean advantage",
        )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Split seed")
        ax.set_ylabel("Test bits/spike gain over factor-latent")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / "rate_offset_decomposition.png", dpi=150)
        plt.close(fig)


def run_cv_rate_audit(config: dict[str, Any]) -> dict[str, Any]:
    dataset, dataset_hash = _prepare_dataset(config)
    statistics = dict(config["statistics"])
    references = dict(config["references"])

    repeated_scores = run_repeated_split_factor_latent(config, dataset)
    fa_sensitivity = run_factor_analysis_random_state_sensitivity(
        repeated_scores, int(references["accepted_split_seed"])
    )
    rate_controls = run_rate_control_audit(config, dataset)
    decomposition = decompose_rate_offset(rate_controls)
    method_summary = summarize_methods(
        rate_controls,
        int(statistics["bootstrap_repeats"]),
        float(statistics["confidence_interval"]),
        int(statistics["bootstrap_seed"]),
    )
    summary = summarize_cv_rate_audit(
        repeated_scores, fa_sensitivity, rate_controls, decomposition, method_summary, references
    )
    summary.update(
        {
            "dataset_name": config["dataset"]["name"],
            "dataset_hash": dataset_hash,
            "bin_size_ms": int(config["binning"]["target_bin_size_ms"]),
            "window_seconds": float(config["window"]["duration_seconds"]),
            "split_seeds": [int(seed) for seed in config["splits"]["split_seeds"]],
            "factor_analysis_random_states": [
                int(state) for state in config["splits"]["factor_analysis_random_states"]
            ],
            "known_controls": list(KNOWN_CONTROLS),
        }
    )
    recommendations = build_reporting_recommendations(summary)

    output_dir = resolve_configured_path(str(config["reporting"]["output_dir"]), get_repo_root())
    write_cv_rate_audit_outputs(
        output_dir,
        summary,
        repeated_scores,
        fa_sensitivity,
        rate_controls,
        decomposition,
        method_summary,
        recommendations,
    )
    _write_figures(output_dir, repeated_scores, fa_sensitivity, method_summary, decomposition)
    summary["output_dir"] = str(output_dir)
    summary["reporting_recommendation"] = recommendations["recommended_reporting_mode"]
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
        summary = run_cv_rate_audit(config)
    except (OSError, ValueError, FileNotFoundError) as exc:
        console.print(f"Cross-validated rate audit failed: {exc}")
        return 2
    for key in (
        "dataset_name",
        "bin_size_ms",
        "window_seconds",
        "split_seeds",
        "factor_analysis_random_states",
        "factor_latent_repeated_split_validation_mean",
        "factor_latent_repeated_split_test_mean",
        "factor_latent_test_positive_fraction",
        "factor_analysis_random_state_validation_range",
        "factor_analysis_random_state_test_range",
        "best_valid_rate_control_method",
        "best_valid_rate_control_test_mean",
        "split_mean_rate_invalid_test_mean",
        "invalid_split_mean_advantage_over_factor_latent",
        "train_only_rate_calibration_helps",
        "invalid_controls_excluded_from_best_valid_model",
        "reporting_recommendation",
        "output_dir",
    ):
        console.print(f"{key}: {summary.get(key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
