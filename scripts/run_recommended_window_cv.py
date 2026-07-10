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
    build_recommended_window_protocol,
    evaluate_recommended_window_cv,
)
from latentbrain.eval.reporting import write_recommended_window_cv_outputs  # noqa: E402
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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config_path = resolve_configured_path(args.config, get_repo_root())
        config = _load_config(config_path)
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
