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

from latentbrain.eval.neural_ode_diagnostics import run_neural_ode_diagnostics  # noqa: E402
from latentbrain.paths import get_repo_root, resolve_configured_path  # noqa: E402


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit accepted MC_Maze Large deterministic neural-ODE checkpoints."
    )
    parser.add_argument("--config", required=True)
    return parser.parse_args(argv)


def _load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        msg = "neural-ODE diagnostics configuration root must be a mapping"
        raise ValueError(msg)
    return cast(dict[str, Any], loaded)


def _save(figure: Any, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_figures(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    splits = tables["split_diagnostics"]
    neurons = tables["neuron_diagnostics"]
    times = tables["time_bin_diagnostics"]
    latents = tables["latent_diagnostics"]
    decoders = tables["decoder_diagnostics"]
    counterfactuals = tables["counterfactual_diagnostics"]

    figure, axis = plt.subplots(figsize=(7, 4))
    splits.boxplot(column="unified_bits_per_spike", by="split_name", ax=axis)
    axis.set(title="Train, inner, and outer scores", xlabel="Split", ylabel="Unified bits/spike")
    figure.suptitle("")
    _save(figure, figures / "train_inner_outer_scores.png")

    figure, axis = plt.subplots(figsize=(7, 4))
    outer_neurons = neurons[neurons["split_name"] == "outer_evaluation"]
    axis.hist(outer_neurons["unified_bits_per_spike"], bins=30)
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set(title="Per-neuron scores", xlabel="Unified bits/spike", ylabel="Count")
    _save(figure, figures / "neuron_score_distribution.png")

    figure, axis = plt.subplots(figsize=(7, 4))
    grouped = times.groupby("relative_time_seconds", sort=True)["unified_bits_per_spike"].mean()
    axis.plot(grouped.index, grouped.values)
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set(
        title="Score by time bin", xlabel="Seconds from peak speed", ylabel="Unified bits/spike"
    )
    _save(figure, figures / "time_resolved_scores.png")

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
    axis.scatter(
        tables["dynamics_diagnostics"]["fold_index"],
        tables["dynamics_diagnostics"]["mean_drift_norm"],
    )
    axis.set(title="Drift norm by fold", xlabel="Fold", ylabel="Mean drift norm")
    _save(figure, figures / "drift_norm_over_time.png")

    figure, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(
        splits[splits["split_name"] == "outer_evaluation"]["mean_observed_rate_hz"],
        splits[splits["split_name"] == "outer_evaluation"]["mean_predicted_rate_hz"],
    )
    limit = float(
        max(splits["mean_observed_rate_hz"].max(), splits["mean_predicted_rate_hz"].max(), 1.0)
    )
    axis.plot([0.0, limit], [0.0, limit], color="black", linewidth=1)
    axis.set(title="Predicted versus observed rates", xlabel="Observed Hz", ylabel="Predicted Hz")
    _save(figure, figures / "observed_predicted_population_rate.png")

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.scatter(decoders["fold_index"], decoders["decoder_condition_number"])
    axis.set(title="Decoder condition number", xlabel="Fold", ylabel="Condition number")
    axis.set_yscale("log")
    _save(figure, figures / "decoder_singular_values.png")

    figure, axis = plt.subplots(figsize=(7, 4))
    for method, group in counterfactuals.groupby("method"):
        axis.scatter(group["fold_index"], group["recovery_vs_accepted"], label=method)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.legend(fontsize=6)
    axis.set(
        title="Counterfactual recovery vs accepted", xlabel="Fold", ylabel="Recovery (bits/spike)"
    )
    _save(figure, figures / "counterfactual_gap_recovery.png")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config_path = resolve_configured_path(args.config, get_repo_root())
        result = run_neural_ode_diagnostics(_load_config(config_path))
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
        "outer_training_mean_unified_bits_per_spike",
        "inner_validation_mean_unified_bits_per_spike",
        "outer_evaluation_mean_unified_bits_per_spike",
        "mean_train_to_inner_gap",
        "mean_inner_to_outer_gap",
        "positive_neuron_fraction",
        "negative_neuron_fraction",
        "mean_effective_rank",
        "mean_decoder_condition_number",
        "exact_required_recovery",
        "estimated_recoverable_gap",
        "dominant_failure_mode",
        "targeted_repair_available",
        "proposed_single_repair",
        "recommended_next_action",
        "full_evaluation_allowed",
        "broad_sweep_allowed",
        "lfads_remains_retired",
    ):
        print(f"{key}: {summary.get(key)}")
    print(f"rationale: {recommendation['rationale']}")
    print(f"output_directory: {result['output_dir']}")
    print(json.dumps({"status": "neural_ode_diagnostics_complete"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
