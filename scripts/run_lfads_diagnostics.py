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

from latentbrain.eval.lfads_diagnostics import run_lfads_diagnostics  # noqa: E402
from latentbrain.paths import get_repo_root, resolve_configured_path  # noqa: E402


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit accepted MC_Maze Large LFADS checkpoints.")
    parser.add_argument("--config", required=True)
    return parser.parse_args(argv)


def _load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("LFADS diagnostics configuration root must be a mapping")
    return cast(dict[str, Any], loaded)


def _save(figure: Any, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_figures(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    runs = tables["run_diagnostics"]
    neurons = tables["neuron_diagnostics"]
    times = tables["time_bin_diagnostics"]
    rates = tables["rate_diagnostics"]
    latents = tables["latent_diagnostics"]
    objectives = tables["objective_diagnostics"]

    figure, axis = plt.subplots(figsize=(7, 4))
    runs.boxplot(column="unified_bits_per_spike", by="split_name", ax=axis)
    axis.set(title="Train, inner, and outer metrics", xlabel="Split", ylabel="Unified bits/spike")
    figure.suptitle("")
    _save(figure, figures / "train_inner_outer_metrics.png")

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(neurons["unified_bits_per_spike"], bins=30)
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set(title="Per-neuron LFADS scores", xlabel="Unified bits/spike", ylabel="Count")
    _save(figure, figures / "per_neuron_bits_per_spike.png")

    figure, axis = plt.subplots(figsize=(7, 4))
    grouped = times.groupby("relative_time_seconds", sort=True)["unified_bits_per_spike"].mean()
    axis.plot(grouped.index, grouped.values)
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set(
        title="Score by time bin", xlabel="Seconds from peak speed", ylabel="Unified bits/spike"
    )
    _save(figure, figures / "score_by_time_bin.png")

    figure, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(rates["observed_mean_rate_hz"], rates["predicted_mean_rate_hz"])
    limit = float(max(rates["observed_mean_rate_hz"].max(), rates["predicted_mean_rate_hz"].max()))
    axis.plot([0.0, limit], [0.0, limit], color="black", linewidth=1)
    axis.set(title="Predicted versus observed rates", xlabel="Observed Hz", ylabel="Predicted Hz")
    _save(figure, figures / "predicted_vs_observed_rates.png")

    figure, axis = plt.subplots(figsize=(7, 4))
    spectrum = (
        latents[latents["representation"] == "factor"]
        .groupby("dimension", sort=True)["covariance_eigenvalue"]
        .mean()
    )
    axis.semilogy(spectrum.index, spectrum.values)
    axis.set(title="Factor covariance spectrum", xlabel="Dimension", ylabel="Eigenvalue")
    _save(figure, figures / "latent_variance_spectrum.png")

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.scatter(objectives["reconstruction_loss"], objectives["weighted_kl_contribution"])
    axis.set(title="KL and reconstruction", xlabel="Held-in reconstruction", ylabel="Weighted KL")
    _save(figure, figures / "kl_and_reconstruction_summary.png")

    figure, axis = plt.subplots(figsize=(7, 4))
    outer = runs[runs["split_name"] == "outer_evaluation"]
    axis.scatter(outer["fold_index"], outer["unified_bits_per_spike"])
    axis.set(title="Outer-fold LFADS scores", xlabel="Fold", ylabel="Unified bits/spike")
    _save(figure, figures / "fold_baseline_gap.png")

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.scatter(rates["fold_index"], rates["first_difference_variance_ratio"])
    axis.axhline(1.0, color="black", linewidth=1)
    axis.set(title="Temporal smoothness comparison", xlabel="Fold", ylabel="First-difference ratio")
    _save(figure, figures / "temporal_smoothness_comparison.png")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config_path = resolve_configured_path(args.config, get_repo_root())
        result = run_lfads_diagnostics(_load_config(config_path))
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    _write_figures(result["output_dir"], result["tables"])
    summary = result["summary"]
    recommendation = result["recommendation"]
    for key in (
        "integrity_checks_passed",
        "accepted_checkpoints",
        "accepted_outer_scores_reproduced",
        "train_mean_unified_bits_per_spike",
        "inner_mean_unified_bits_per_spike",
        "outer_mean_unified_bits_per_spike",
        "mean_train_to_inner_gap",
        "mean_inner_to_outer_gap",
        "positive_neuron_fraction",
        "negative_neuron_fraction",
        "mean_effective_rank",
        "posterior_collapse_detected",
        "dominant_failure_mode",
        "estimated_recoverable_gap",
    ):
        print(f"{key}: {summary.get(key)}")
    print(f"recommended_next_action: {recommendation['recommended_next_action']}")
    print(f"full_lfads_evaluation_allowed: {recommendation['full_lfads_evaluation_allowed']}")
    print(f"output_directory: {result['output_dir']}")
    print(json.dumps({"status": "lfads_diagnostics_complete"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
