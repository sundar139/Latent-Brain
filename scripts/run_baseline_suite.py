from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib
import pandas as pd  # type: ignore[import-untyped]
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from latentbrain.eval.baseline_suite import (  # noqa: E402
    FACTOR_LATENT_FIXED,
    build_baseline_protocol,
    build_method_summary,
    build_paired_comparisons,
    build_readiness,
    build_repeat_level_scores,
    choose_baseline_to_beat,
    run_baseline_suite,
    summarize_baseline_suite,
)
from latentbrain.eval.reporting import write_baseline_suite_outputs  # noqa: E402
from latentbrain.paths import get_repo_root, resolve_configured_path  # noqa: E402


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MC_Maze Large valid baseline suite.")
    parser.add_argument("--config", required=True)
    return parser.parse_args(argv)


def _load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        msg = "Configuration root must be a mapping"
        raise ValueError(msg)
    config = cast(dict[str, Any], loaded)
    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    if str(config["trial_source"]["type"]) != "trial_aware_raw":
        msg = "trial_source.type must be trial_aware_raw"
        raise ValueError(msg)
    if bool(config["trial_source"]["allow_global_crop_to_min"]):
        msg = "trial_source.allow_global_crop_to_min must be false"
        raise ValueError(msg)
    if not bool(config["window"]["extract_before_rebin"]):
        msg = "window.extract_before_rebin must be true"
        raise ValueError(msg)
    outer = config["outer_cross_validation"]
    if not bool(outer["reuse_exact_assignments"]) or not bool(outer["reuse_exact_neuron_masks"]):
        msg = "the accepted outer assignments and neuron masks must be reused"
        raise ValueError(msg)
    if int(config["inner_selection"]["fold_count"]) < 2:
        msg = "inner_selection.fold_count must be at least 2"
        raise ValueError(msg)
    if str(config["statistics"]["comparison_unit"]) != "repeat":
        msg = "statistics.comparison_unit must be repeat; folds inside a repeat are correlated"
        raise ValueError(msg)

    methods = {str(method["name"]): method for method in config["methods"]}
    for name, method in methods.items():
        family = str(method["family"])
        if family in ("invalid_control", "reference") and bool(
            method["reportable_as_model_performance"]
        ):
            msg = f"{name} must not be reportable as model performance"
            raise ValueError(msg)
        if bool(method.get("valid_model", False)) and "search" in method:
            for key, values in method["search"].items():
                if not isinstance(values, list) or not values:
                    msg = f"{name} search entry {key} must be a non-empty list"
                    raise ValueError(msg)
    baseline = str(config["selection"]["baseline_to_beat"])
    if baseline not in methods or not bool(methods[baseline]["reportable_as_model_performance"]):
        msg = "selection.baseline_to_beat must be a reportable valid model"
        raise ValueError(msg)


def _write_figures(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    outer = tables["outer_fold_scores"]

    fig, axis = plt.subplots(figsize=(8, 4))
    names = sorted(outer["method_name"].unique())
    axis.boxplot(
        [outer[outer["method_name"] == name]["unified_bits_per_spike"] for name in names],
        tick_labels=names,
    )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set(ylabel="Unified bits/spike", title="Outer-fold score distribution")
    axis.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(figures / "baseline_score_distribution.png", dpi=160)
    plt.close(fig)

    repeats = tables["repeat_level_scores"]
    baseline = repeats[repeats["method_name"] == FACTOR_LATENT_FIXED].set_index("repeat_index")[
        "mean_unified_bits_per_spike"
    ]
    fig, axis = plt.subplots(figsize=(8, 4))
    for name in sorted(repeats["method_name"].unique()):
        if name == FACTOR_LATENT_FIXED:
            continue
        values = repeats[repeats["method_name"] == name].set_index("repeat_index")[
            "mean_unified_bits_per_spike"
        ]
        axis.plot(values.index, (values - baseline).to_numpy(dtype=float), marker="o", label=name)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set(xlabel="Repeat", ylabel="Paired difference", title="Paired differences by repeat")
    axis.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(figures / "paired_differences_by_repeat.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4))
    for name in sorted(repeats["method_name"].unique()):
        values = repeats[repeats["method_name"] == name]
        axis.plot(
            values["repeat_index"],
            values["mean_unified_bits_per_spike"],
            marker="o",
            label=name,
        )
    axis.set(
        xlabel="Repeat (one held-out neuron mask each)",
        ylabel="Mean unified bits/spike",
        title="Held-out-mask variability",
    )
    axis.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(figures / "heldout_mask_variability.png", dpi=160)
    plt.close(fig)

    selected = tables["selected_hyperparameters"]
    fig, axis = plt.subplots(figsize=(9, 4))
    if not selected.empty:
        counts = selected.groupby(["method_name", "selected_configuration_id"]).size()
        labels = [f"{method}#{index}" for method, index in counts.index]
        axis.bar(labels, counts.to_numpy(dtype=float))
        axis.set(ylabel="Outer folds selecting it", title="Selected configurations")
        axis.tick_params(axis="x", rotation=70, labelsize=6)
    else:
        axis.text(0.5, 0.5, "no train-selected methods", ha="center", va="center")
        axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(figures / "selected_hyperparameters.png", dpi=160)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config_path = resolve_configured_path(args.config, get_repo_root())
        config = _load_config(config_path)
        result = run_baseline_suite(config)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    outer_scores = result["outer_scores"]
    repeat_scores = build_repeat_level_scores(outer_scores)
    comparisons = build_paired_comparisons(outer_scores, repeat_scores, config)
    baseline_choice = choose_baseline_to_beat(comparisons, config)
    method_summary = build_method_summary(outer_scores, baseline_choice, config)
    summary = summarize_baseline_suite(result, repeat_scores, comparisons, baseline_choice, config)
    readiness = build_readiness(summary, config, method_summary)
    summary["neural_reevaluation_ready"] = readiness["ready"]
    summary["readiness_blockers"] = readiness["blockers"]
    summary["baseline_to_beat_ci95_low"] = readiness["baseline_ci95_low"]
    summary["baseline_to_beat_ci95_high"] = readiness["baseline_ci95_high"]
    protocol = build_baseline_protocol(config, summary)

    tables = {
        "outer_fold_scores": outer_scores,
        "inner_selection": result["inner_selection"],
        "selected_hyperparameters": result["selected_hyperparameters"],
        "method_summary": method_summary,
        "paired_method_comparisons": comparisons,
        "repeat_level_scores": repeat_scores,
    }
    output_dir = resolve_configured_path(str(config["reporting"]["output_dir"]), get_repo_root())
    write_baseline_suite_outputs(output_dir, summary, tables, protocol, readiness)
    _write_figures(output_dir, tables)

    for key in (
        "dataset_name",
        "dataset_hash",
        "window_name",
        "outer_fold_count",
        "outer_repeats",
        "total_outer_evaluations",
        "inner_fold_count",
        "factor_latent_fixed_mean",
        "factor_latent_reproduction_difference",
        "valid_methods",
        "best_valid_method",
        "best_valid_method_mean",
        "baseline_to_beat",
        "baseline_replaced",
        "baseline_replacement_supported",
        "paired_difference_against_factor_latent",
        "paired_ci_against_factor_latent",
        "positive_repeat_fraction_against_factor_latent",
        "invalid_controls_excluded",
        "neural_reevaluation_ready",
        "readiness_blockers",
    ):
        print(f"{key}: {summary.get(key)}")
    print(f"output_directory: {output_dir}")
    print(json.dumps({"status": "baseline_suite_complete"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
