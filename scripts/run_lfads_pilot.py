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

from latentbrain.paths import get_repo_root, resolve_configured_path  # noqa: E402
from latentbrain.train.lfads_pilot import (  # noqa: E402
    run_lfads_pilot,
    validate_lfads_pilot_config,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MC_Maze Large LFADS feasibility pilot.")
    parser.add_argument("--config", required=True)
    return parser.parse_args(argv)


def _load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        msg = "LFADS pilot configuration root must be a mapping"
        raise ValueError(msg)
    config = cast(dict[str, Any], loaded)
    validate_lfads_pilot_config(config)
    return config


def _write_figures(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    scores = tables["fold_seed_scores"]
    resources = tables["training_resource_summary"]

    fig, axis = plt.subplots(figsize=(7, 4))
    for seed, group in scores.groupby("initialization_seed", sort=True):
        axis.scatter(group["fold_index"], group["outer_unified_bits_per_spike"], label=str(seed))
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set(xlabel="Outer fold", ylabel="Unified bits/spike", title="Fold-seed score distribution")
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "fold_seed_score_distribution.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.scatter(scores["fold_index"], scores["paired_difference_vs_baseline"])
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set(xlabel="Outer fold", ylabel="LFADS - baseline", title="Descriptive paired differences")
    fig.tight_layout()
    fig.savefig(figures / "paired_baseline_difference.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    histories = sorted((output_dir / "runs").glob("**/metrics_history.csv"))
    for path in histories:
        history = pd.read_csv(path)
        axis.plot(history["epoch"], history["inner_validation_unified_bits_per_spike"], alpha=0.3)
    axis.set(xlabel="Epoch", ylabel="Inner-validation unified bits/spike", title="Training curves")
    fig.tight_layout()
    fig.savefig(figures / "training_curves.png", dpi=160)
    plt.close(fig)

    seed_summary = tables["seed_summary"]
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.errorbar(
        seed_summary["initialization_seed"].astype(str),
        seed_summary["mean_outer_unified_bits_per_spike"],
        yerr=seed_summary["std_outer_unified_bits_per_spike"],
        fmt="o",
    )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set(
        xlabel="Initialization seed",
        ylabel="Mean unified bits/spike",
        title="Seed variability",
    )
    fig.tight_layout()
    fig.savefig(figures / "seed_variability.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].boxplot(resources["training_seconds"])
    axes[0].set(title="Runtime", ylabel="Seconds")
    axes[1].boxplot(resources["peak_cuda_memory_mb"])
    axes[1].set(title="Peak CUDA memory", ylabel="MB")
    fig.tight_layout()
    fig.savefig(figures / "runtime_memory_summary.png", dpi=160)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config_path = resolve_configured_path(args.config, get_repo_root())
        config = _load_config(config_path)
        result = run_lfads_pilot(config)
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 2

    _write_figures(result["output_dir"], result["tables"])
    summary = result["summary"]
    recommendation = result["recommendation"]
    for key in (
        "dataset_name",
        "dataset_hash",
        "data_shape",
        "repeat_index",
        "fold_indices",
        "initialization_seeds",
        "completed_runs",
        "failed_runs",
        "mean_unified_bits_per_spike",
        "score_std",
        "seed_level_std",
        "positive_seed_fraction",
        "pilot_repeat_baseline_mean",
        "mean_paired_difference_vs_baseline",
        "checkpoint_selection_split",
        "checkpoint_selection_valid",
        "leakage_checks_passed",
        "full_evaluation_recommended",
        "pilot_final_claim_allowed",
    ):
        print(f"{key}: {summary.get(key)}")
    print(
        "runtime_estimate_full_evaluation_hours: "
        f"{recommendation['runtime_estimate_full_evaluation_hours']}"
    )
    print(f"estimated_peak_cuda_memory_mb: {recommendation['estimated_peak_cuda_memory_mb']}")
    print(f"gate_reasons: {recommendation['reasons']}")
    print(f"output_directory: {result['output_dir']}")
    print(json.dumps({"status": "lfads_pilot_complete"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
