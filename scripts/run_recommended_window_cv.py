from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from latentbrain.eval.recommended_window_cv import (  # noqa: E402
    RECOMMENDED_WINDOW_NAME,
    TRIAL_AWARE_SOURCE,
    build_recommended_window_protocol,
    evaluate_large_recommended_window_cv,
    evaluate_recommended_window_cv,
)
from latentbrain.eval.reporting import (  # noqa: E402
    write_large_recommended_window_cv_outputs,
    write_recommended_window_cv_outputs,
)
from latentbrain.eval.stratified_cv import ASSIGNMENT_METHODS  # noqa: E402
from latentbrain.paths import get_repo_root, resolve_configured_path  # noqa: E402

_STRATIFICATION_VARIABLES = {
    "use_endpoint_direction",
    "use_endpoint_distance",
    "use_mean_speed",
    "use_population_rate",
    "use_heldout_rate",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MC_Maze Small recommended-window CV.")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration.")
    return parser.parse_args(argv)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        msg = "Configuration root must be a mapping"
        raise ValueError(msg)
    config = cast(dict[str, Any], loaded)
    _validate_config(config)
    return config


def _is_trial_aware_config(config: dict[str, Any]) -> bool:
    return "trial_source" in config


def _validate_trial_aware_config(config: dict[str, Any]) -> None:
    trial_source = config["trial_source"]
    if str(trial_source["type"]) != TRIAL_AWARE_SOURCE:
        msg = f"trial_source.type must be {TRIAL_AWARE_SOURCE}"
        raise ValueError(msg)
    if bool(trial_source["allow_global_crop_to_min"]):
        msg = "trial_source.allow_global_crop_to_min must be false"
        raise ValueError(msg)
    if not bool(config["binning"]["extract_before_rebin"]):
        msg = "binning.extract_before_rebin must be true"
        raise ValueError(msg)
    if str(config["cross_validation"]["heldout_mask_policy"]) != "fixed_within_repeat":
        msg = "cross_validation.heldout_mask_policy must be fixed_within_repeat"
        raise ValueError(msg)
    sensitivity = config.get("factor_analysis_sensitivity", {})
    states = [int(state) for state in sensitivity.get("random_states", [])]
    if len(set(states)) != len(states):
        msg = "factor_analysis_sensitivity.random_states must be unique"
        raise ValueError(msg)
    if not config["dataset"].get("expected_hash"):
        msg = "dataset.expected_hash is required for trial-aware cross-validation"
        raise ValueError(msg)


def _validate_config(config: dict[str, Any]) -> None:
    dataset = config["dataset"]
    target = int(config["binning"]["target_bin_size_ms"])
    original = int(dataset["original_bin_size_ms"])
    if target <= 0 or target % original != 0:
        msg = "target_bin_size_ms must be a positive multiple of original_bin_size_ms"
        raise ValueError(msg)
    window = config["window"]
    if window["name"] != RECOMMENDED_WINDOW_NAME:
        msg = f"window.name must be {RECOMMENDED_WINDOW_NAME}"
        raise ValueError(msg)
    if float(window["duration_seconds"]) <= 0.0:
        msg = "window.duration_seconds must be positive"
        raise ValueError(msg)
    cv = config["cross_validation"]
    if int(cv["fold_count"]) < 3:
        msg = "cross_validation.fold_count must be at least 3"
        raise ValueError(msg)
    if int(cv["repeats"]) < 2:
        msg = "cross_validation.repeats must be at least 2"
        raise ValueError(msg)
    if cv["assignment_method"] not in ASSIGNMENT_METHODS:
        msg = f"Unknown assignment_method: {cv['assignment_method']}"
        raise ValueError(msg)

    stratification = config["stratification"]
    unknown = {
        key
        for key in stratification
        if key.startswith("use_") and key not in _STRATIFICATION_VARIABLES
    }
    if unknown:
        msg = f"Unknown stratification variables: {sorted(unknown)}"
        raise ValueError(msg)
    if stratification.get("fallback_when_behavior_missing") != "fail":
        msg = "fallback_when_behavior_missing must be fail"
        raise ValueError(msg)

    methods = {str(method["name"]): method for method in config["methods"]}
    factor = methods.get("factor_latent")
    if factor is None or not factor.get("valid_model", False):
        msg = "factor_latent must be configured as a valid model"
        raise ValueError(msg)
    train_mean = methods.get("train_mean_rate")
    if train_mean is None or train_mean.get("reportable_as_model_performance", False):
        msg = "train_mean_rate must not be reportable as model performance"
        raise ValueError(msg)
    for method in methods.values():
        if method.get("type") == "invalid_control" and method.get(
            "reportable_as_model_performance", False
        ):
            msg = f"Invalid control {method['name']} cannot be reportable as model performance"
            raise ValueError(msg)
    if int(config["statistics"]["bootstrap_repeats"]) <= 0:
        msg = "statistics.bootstrap_repeats must be positive"
        raise ValueError(msg)
    if _is_trial_aware_config(config):
        _validate_trial_aware_config(config)


def _write_figures(
    output_dir: Path,
    scores: pd.DataFrame,
    behavior_statistics: pd.DataFrame,
    fold_balance: pd.DataFrame,
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(7, 4))
    for method_name, group in scores.groupby("method_name", sort=False):
        axis.hist(
            group["unified_bits_per_spike"].to_numpy(dtype=np.float64),
            bins=10,
            alpha=0.55,
            label=str(method_name),
        )
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set(xlabel="Unified bits/spike", ylabel="Fold count", title="Recommended-window scores")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "recommended_window_score_distribution.png", dpi=160)
    plt.close(fig)

    compared = scores[scores["method_name"].isin(["factor_latent", "split_mean_rate_invalid"])]
    pivot = compared.pivot(
        index=["fold_repeat", "fold_index"],
        columns="method_name",
        values="unified_bits_per_spike",
    )
    fig, axis = plt.subplots(figsize=(7, 4))
    x = np.arange(len(pivot))
    axis.plot(x, pivot["factor_latent"], marker="o", label="factor_latent")
    axis.plot(x, pivot["split_mean_rate_invalid"], marker="o", label="invalid split mean")
    axis.set(xlabel="Repeated fold", ylabel="Unified bits/spike", title="Valid vs invalid by fold")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "valid_vs_invalid_by_fold.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 4))
    axis.hist(behavior_statistics["moving_bin_fraction"], bins=10, color="#4c78a8")
    axis.axvline(
        float(behavior_statistics["moving_bin_fraction"].mean()),
        color="black",
        linestyle="--",
        label="mean",
    )
    axis.set(xlabel="Moving-bin fraction", ylabel="Trial count", title="Movement coverage")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "movement_coverage_summary.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(np.arange(len(fold_balance)), fold_balance["n_trials"])
    axes[0].set(xlabel="Repeated fold", ylabel="Trials", title="Trial balance")
    axes[1].bar(np.arange(len(fold_balance)), fold_balance["endpoint_direction_entropy"])
    axes[1].set(xlabel="Repeated fold", ylabel="Entropy", title="Direction coverage")
    fig.tight_layout()
    fig.savefig(figure_dir / "fold_balance_summary.png", dpi=160)
    plt.close(fig)


def _write_large_figures(
    output_dir: Path,
    scores: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    sensitivity = tables["factor_analysis_sensitivity"]
    if not sensitivity.empty:
        means = sensitivity.groupby("factor_analysis_random_state")["unified_bits_per_spike"].mean()
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.bar([str(state) for state in means.index], means.to_numpy(dtype=np.float64))
        axis.set(
            xlabel="FactorAnalysis random state",
            ylabel="Mean unified bits/spike",
            title="Random-state sensitivity",
        )
        fig.tight_layout()
        fig.savefig(figure_dir / "factor_analysis_random_state_sensitivity.png", dpi=160)
        plt.close(fig)

    comparison = tables["small_large_comparison"]
    fig, axis = plt.subplots(figsize=(7, 4))
    datasets = comparison["dataset"].astype(str).to_numpy()
    means = comparison["factor_latent_mean"].to_numpy(dtype=np.float64)
    lows = means - comparison["factor_latent_ci95_low"].to_numpy(dtype=np.float64)
    highs = comparison["factor_latent_ci95_high"].to_numpy(dtype=np.float64) - means
    axis.errorbar(datasets, means, yerr=[lows, highs], fmt="o", capsize=5)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set(
        ylabel="Factor-latent unified bits/spike (95% CI)",
        title="Protocol stability, not a performance comparison",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "small_large_stability_comparison.png", dpi=160)
    plt.close(fig)


def _run_large(config: dict[str, Any]) -> int:
    scores, tables, summary = evaluate_large_recommended_window_cv(config)
    protocol = build_recommended_window_protocol(config, summary)
    output_dir = resolve_configured_path(str(config["reporting"]["output_dir"]), get_repo_root())
    write_large_recommended_window_cv_outputs(output_dir, summary, scores, tables, protocol)
    _write_figures(
        output_dir,
        scores.rename(columns={"repeat_index": "fold_repeat"}),
        tables["behavior_statistics"],
        tables["fold_balance"],
    )
    _write_large_figures(output_dir, scores, tables)
    for key in (
        "dataset_name",
        "dataset_hash",
        "window_name",
        "trial_source",
        "target_bin_size_ms",
        "trial_count",
        "time_bins",
        "neuron_count",
        "fold_count",
        "repeats",
        "total_folds",
        "train_trials_per_fold",
        "eval_trials_per_fold",
        "factor_latent_mean",
        "factor_latent_std",
        "factor_latent_positive_fraction",
        "split_mean_invalid_mean",
        "factor_latent_minus_split_mean_invalid",
        "factor_latent_beats_invalid_control_mean",
        "factor_latent_beats_invalid_control_fraction",
        "leakage_dominance_persists",
        "factor_analysis_random_state_range",
        "factor_analysis_random_state_warning",
        "fold_balance_warning",
        "recommended_reporting_mode",
        "invalid_controls_excluded_from_model_selection",
        "protocol_frozen",
    ):
        print(f"{key}: {summary.get(key)}")
    print(
        "factor_latent_ci95: "
        f"[{summary['factor_latent_ci95_low']:.6f}, {summary['factor_latent_ci95_high']:.6f}]"
    )
    print(f"output_directory: {output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config_path = resolve_configured_path(args.config, get_repo_root())
        config = _load_config(config_path)
        if _is_trial_aware_config(config):
            return _run_large(config)
        scores, tables, summary = evaluate_recommended_window_cv(config)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    protocol = build_recommended_window_protocol(config, summary)
    output_dir = resolve_configured_path(str(config["reporting"]["output_dir"]), get_repo_root())
    write_recommended_window_cv_outputs(
        output_dir,
        summary,
        scores,
        tables["method_summary"],
        tables["fold_assignments"],
        tables["behavior_statistics"],
        tables["fold_balance"],
        tables["leakage_diagnostics"],
        protocol,
    )
    _write_figures(output_dir, scores, tables["behavior_statistics"], tables["fold_balance"])

    print(f"dataset: {summary['dataset_name']}")
    print(f"bin_size_ms: {summary['bin_size_ms']}")
    print(f"recommended_window: {summary['recommended_window_name']}")
    print(f"fold_count: {summary['fold_count']}")
    print(f"repeats: {summary['repeats']}")
    print(f"total_folds: {summary['total_folds']}")
    print(f"factor_latent_mean: {summary['factor_latent_mean']:.6f}")
    print(
        "factor_latent_ci95: "
        f"[{summary['factor_latent_ci95_low']:.6f}, {summary['factor_latent_ci95_high']:.6f}]"
    )
    print(f"factor_latent_positive_fraction: {summary['factor_latent_positive_fraction']:.4f}")
    print(f"split_mean_invalid_mean: {summary['split_mean_invalid_mean']:.6f}")
    print(
        "factor_latent_minus_split_mean_invalid: "
        f"{summary['factor_latent_minus_split_mean_invalid']:+.6f}"
    )
    print(f"leakage_dominance_persists: {summary['leakage_dominance_persists']}")
    print(f"moving_bin_fraction_mean: {summary['moving_bin_fraction_mean']:.4f}")
    print(f"endpoint_direction_entropy_mean: {summary['endpoint_direction_entropy_mean']:.4f}")
    print(f"fold_balance_warning: {summary['fold_balance_warning']}")
    print(f"recommended_reporting_mode: {summary['recommended_reporting_mode']}")
    print(
        "invalid_controls_excluded_from_model_selection: "
        f"{summary['invalid_controls_excluded_from_model_selection']}"
    )
    print(f"output_dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
