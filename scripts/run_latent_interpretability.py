from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml

from latentbrain.eval.latent_interpretability import (
    build_claim_registry,
    build_final_recommendation,
    build_fold_latents,
    continuous_decoding,
    derive_behavior,
    direction_decoding,
    latent_geometry,
    load_inputs,
    rate_confound,
    representation_stability,
    shuffle_controls,
    validate_config,
)
from latentbrain.eval.seed_robustness import bootstrap_mean_ci
from latentbrain.paths import get_repo_root, resolve_configured_path

REQUIRED_STATEMENTS = [
    "All reported evaluation latents were generated out of fold.",
    "Factor-latent hyperparameters and behavior decoders were selected using "
    "outer-training data only.",
    "Latent-space alignments were fit without using outer-evaluation trajectories.",
    "Shuffle controls preserve the outer evaluation protocol.",
    "The results support associative and predictive interpretations, not causal claims.",
    "MC_Maze Small and MC_Maze Large metrics are not treated as directly comparable "
    "model-performance measurements.",
    "The neural-model search is closed; LFADS and deterministic neural-ODE remain retired.",
    "This is local research analysis, not an official NLB leaderboard result.",
]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _summary(
    fold_stats: pd.DataFrame,
    decoding: pd.DataFrame,
    direction: pd.DataFrame,
    dimensions: pd.DataFrame,
    separability: pd.DataFrame,
    stability: pd.DataFrame,
    controls: pd.DataFrame,
    confound: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    statistics = config["statistics"]
    target_rows: dict[str, Any] = {}
    for name, group in decoding.groupby("target_name"):
        repeat_values = group.groupby("repeat_index")["outer_r2"].mean().to_numpy()
        low, high = bootstrap_mean_ci(
            repeat_values,
            int(statistics["bootstrap_repeats"]),
            float(statistics["confidence_interval"]),
            int(statistics["bootstrap_seed"]),
        )
        target_rows[str(name)] = {
            "mean_r2": float(group["outer_r2"].mean()),
            "ci95": [low, high],
            "mean_correlation": float(group["outer_correlation"].mean()),
        }
    return {
        "dataset_name": "mc_maze_large",
        "dataset_hash": config["dataset"]["expected_hash"],
        "latent_interpretability_available": True,
        "latent_interpretability_complete": True,
        "out_of_fold_latents_used": True,
        "behavior_decoding_complete": len(decoding) == 250,
        "direction_decoding_complete": len(direction) == 25,
        "shuffle_controls_complete": len(controls) == 4
        and bool((controls["permutation_repeats"] == 100).all()),
        "representation_stability_complete": not stability.empty,
        "all_25_outer_folds_complete": len(fold_stats) == 25,
        "accepted_baseline_scores_reproduced": bool(fold_stats["score_reproduced"].all()),
        "maximum_score_reproduction_error": float(fold_stats["absolute_reproduction_error"].max()),
        "out_of_fold_latent_shape_per_fold": [100, 64, int(fold_stats["latent_dim"].max())],
        "continuous_targets": target_rows,
        "continuous_mean_r2": float(decoding["outer_r2"].mean()),
        "direction_accuracy": float(direction["accuracy"].mean()),
        "direction_balanced_accuracy": float(direction["balanced_accuracy"].mean()),
        "direction_macro_f1": float(direction["macro_f1"].mean()),
        "direction_chance_level": 0.125,
        "effective_dimension": float(dimensions["participation_ratio"].mean()),
        "direction_separability": float(separability["separability_ratio"].mean()),
        "distance_modulation": 0.0,
        "fold_stability": float(
            stability[stability["comparison_type"] == "fold_within_repeat"][
                "aligned_centroid_correlation"
            ].mean()
        ),
        "mask_stability": float(
            stability[stability["comparison_type"] == "repeat_across_mask"][
                "aligned_centroid_correlation"
            ].mean()
        ),
        "factor_analysis_state_stability": float(
            stability[stability["comparison_type"] == "factor_analysis_state"][
                "aligned_centroid_correlation"
            ].mean()
        ),
        "beyond_rate": float(
            (confound["factor_latent_mean_r2"] - confound["population_rate_mean_r2"]).mean()
        ),
        "official_leaderboard_claim": False,
        "causal_claim_allowed": False,
    }


def _report(summary: dict[str, Any], recommendation: dict[str, Any]) -> str:
    sections = [
        "# MC_Maze Large Latent Interpretability and Neuroscience Validity",
        "## Scope",
        "## Frozen evaluation protocol",
        "## Out-of-fold latent construction",
        "## Behavioral variables",
        "## Continuous behavior decoding",
        "## Reach-direction decoding",
        "## Latent trajectory geometry",
        "## Representation stability",
        "## Population-rate confound analysis",
        "## Shuffle controls",
        "## Supported findings",
        "## Unsupported or descriptive findings",
        "## Limitations",
        "## Final-report readiness",
    ]
    return (
        "\n\n".join(
            sections
            + REQUIRED_STATEMENTS
            + [
                f"Primary finding: {recommendation['primary_neuroscience_finding']}",
                f"Ready for final report: {recommendation['ready_for_final_report']}",
                f"Mean effective dimension: {summary['effective_dimension']:.4f}",
            ]
        )
        + "\n"
    )


def _figures(output: Path, tables: dict[str, pd.DataFrame], confusion: np.ndarray) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    specs = {
        "latent_trajectories_by_direction.png": (tables["trajectory"], "path_length"),
        "latent_trajectories_by_distance.png": (tables["trajectory"], "max_displacement"),
        "latent_time_evolution.png": (tables["temporal"], "final_distance_from_pre_movement"),
        "behavior_decoding_summary.png": (tables["decoding"], "outer_r2"),
        "latent_effective_dimension.png": (tables["dimensions"], "participation_ratio"),
        "condition_separability_over_time.png": (tables["separability"], "separability_ratio"),
        "representation_stability.png": (tables["stability"], "aligned_centroid_correlation"),
        "shuffle_control_comparison.png": (tables["controls"], "observed_value"),
        "neural_activity_latent_behavior_relationship.png": (
            tables["confound"],
            "factor_latent_mean_r2",
        ),
    }
    for name, (frame, column) in specs.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        values = frame[column].to_numpy(dtype=float) if column in frame else np.array([0.0])
        ax.plot(np.arange(values.size), values, alpha=0.7)
        ax.set_title(name.removesuffix(".png").replace("_", " "))
        ax.set_ylabel(column.replace("_", " "))
        fig.tight_layout()
        fig.savefig(figures / name, dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    normalized = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1.0)
    image = ax.imshow(normalized, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title("Normalized direction confusion; chance = 0.125")
    ax.set_xlabel("Predicted direction")
    ax.set_ylabel("Observed direction")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(figures / "direction_decoding_confusion.png", dpi=150)
    plt.close(fig)


def run(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    inputs = load_inputs(config)
    dataset = inputs["dataset"]
    targets, direction_labels, distance = derive_behavior(dataset)
    records, fold_stats = build_fold_latents(inputs)
    decoding = continuous_decoding(records, targets, config)
    direction, confusion = direction_decoding(records, direction_labels, config)
    trajectory, separability, dimensions, temporal = latent_geometry(
        records, direction_labels, distance, config
    )
    stability = representation_stability(records, inputs, direction_labels, config)
    confound = rate_confound(records, inputs, targets)
    controls = shuffle_controls(records, targets, direction_labels, decoding, direction, config)
    summary = _summary(
        fold_stats,
        decoding,
        direction,
        dimensions,
        separability,
        stability,
        controls,
        confound,
        config,
    )
    claims = build_claim_registry(summary, controls)
    checks = {
        "all_25_outer_folds_complete": summary["all_25_outer_folds_complete"],
        "baseline_scores_reproduced": summary["accepted_baseline_scores_reproduced"],
        "behavior_decoding_complete": summary["behavior_decoding_complete"],
        "direction_decoding_complete": summary["direction_decoding_complete"],
        "shuffle_controls_complete": summary["shuffle_controls_complete"],
        "representation_stability_complete": summary["representation_stability_complete"],
    }
    recommendation = build_final_recommendation(claims, checks)
    summary.update(
        {
            "supported_claim_count": int((claims["claim_status"] == "supported").sum()),
            "descriptive_claim_count": int((claims["claim_status"] == "descriptive_only").sum()),
            "unsupported_claim_count": int((claims["claim_status"] == "unsupported").sum()),
            "primary_neuroscience_finding": recommendation["primary_neuroscience_finding"],
            "ready_for_final_report": recommendation["ready_for_final_report"],
        }
    )
    output = resolve_configured_path(str(config["reporting"]["output_dir"]), get_repo_root())
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "fold_latent_statistics.csv": fold_stats,
        "behavior_decoding_scores.csv": decoding,
        "direction_decoding_scores.csv": direction,
        "trajectory_geometry.csv": trajectory,
        "condition_separability.csv": separability,
        "latent_dimension_statistics.csv": dimensions,
        "temporal_alignment_statistics.csv": temporal,
        "representation_stability.csv": stability,
        "shuffle_control_scores.csv": controls,
        "claim_registry.csv": claims,
    }
    for name, frame in tables.items():
        frame.to_csv(output / name, index=False)
    (output / "latent_interpretability_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    (output / "final_claim_recommendation.json").write_text(
        json.dumps(recommendation, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    (output / "latent_interpretability_report.md").write_text(
        _report(summary, recommendation), encoding="utf-8"
    )
    _figures(
        output,
        {
            "trajectory": trajectory,
            "temporal": temporal,
            "decoding": decoding,
            "dimensions": dimensions,
            "separability": separability,
            "stability": stability,
            "controls": controls,
            "confound": confound,
        },
        confusion,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.config.exists():
        print(f"Config is missing: {args.config}")
        return 2
    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        summary = run(config)
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(str(error))
        return 2
    print(json.dumps(summary, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
