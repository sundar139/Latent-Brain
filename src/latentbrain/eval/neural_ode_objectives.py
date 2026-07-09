from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

RESULT_COLUMNS = [
    "run_id",
    "run_index",
    "status",
    "objective_name",
    "heldin_loss_weight",
    "heldout_loss_weight",
    "zero_count_weight",
    "positive_count_weight",
    "rate_calibration_loss_weight",
    "kl_warmup_epochs",
    "kl_scale",
    "drift_regularization",
    "scheduler",
    "input_dropout_rate",
    "epochs",
    "device",
    "validation_unified_bits_per_spike",
    "validation_poisson_nll",
    "validation_behavior_mean_r2",
    "validation_factor_decoder_unified_bits_per_spike",
    "train_total_loss",
    "validation_total_loss",
    "validation_heldout_prediction_loss",
    "z0_kl_loss",
    "drift_regularization_loss",
    "rate_calibration_loss",
    "drift_norm",
    "diffusion_mean",
    "best_checkpoint_source",
    "beats_train_mean_reference",
    "beats_factor_latent_unified",
    "beats_previous_neural_ode_refinement",
    "beats_switching_ode",
    "output_dir",
    "notes",
]

LEADERBOARD_COLUMNS = [
    "rank",
    "run_id",
    "objective_name",
    "validation_unified_bits_per_spike",
    "validation_poisson_nll",
    "validation_factor_decoder_unified_bits_per_spike",
    "heldout_loss_weight",
    "zero_count_weight",
    "positive_count_weight",
    "rate_calibration_loss_weight",
    "kl_scale",
    "drift_regularization",
    "input_dropout_rate",
    "best_checkpoint_source",
    "beats_factor_latent_unified",
    "beats_previous_neural_ode_refinement",
    "notes",
]

DIAGNOSTIC_COLUMNS = [
    "run_id",
    "objective_name",
    "heldin_loss_weight",
    "heldout_loss_weight",
    "zero_count_weight",
    "positive_count_weight",
    "rate_calibration_loss_weight",
    "drift_regularization",
    "validation_unified_bits_per_spike",
    "validation_heldout_prediction_loss",
    "rate_calibration_loss",
    "drift_regularization_loss",
    "drift_norm",
    "mean_predicted_rate",
    "mean_observed_rate",
]

_VARIANT_COLUMNS = (
    "objective_name",
    "heldin_loss_weight",
    "heldout_loss_weight",
    "zero_count_weight",
    "positive_count_weight",
    "rate_calibration_loss_weight",
    "kl_warmup_epochs",
    "kl_scale",
    "drift_regularization",
    "scheduler",
    "input_dropout_rate",
)


def _validation_metric(unified_scores: pd.DataFrame, prediction_source: str, column: str) -> float:
    rows = unified_scores[
        (unified_scores["split"] == "validation")
        & (unified_scores["prediction_source"] == prediction_source)
    ]
    if rows.empty or column not in rows:
        return float("nan")
    return float(rows.iloc[0][column])


def _best_checkpoint_source(checkpoint_scores: pd.DataFrame) -> str:
    if checkpoint_scores.empty or "selected_by_unified" not in checkpoint_scores:
        return "best_unified"
    selected = checkpoint_scores[checkpoint_scores["selected_by_unified"].astype(bool)]
    if selected.empty:
        return "best_unified"
    return str(selected.iloc[0].get("checkpoint_source", "best_unified"))


def _objective_simplicity(row: pd.Series) -> float:
    """Lower is simpler: no rate calibration, count weights closest to 1."""
    return (
        abs(float(row.get("rate_calibration_loss_weight", 0.0)))
        + abs(float(row.get("zero_count_weight", 1.0)) - 1.0)
        + abs(float(row.get("positive_count_weight", 1.0)) - 1.0)
    )


def build_neural_ode_objective_result_row(
    run_id: str,
    run_index: int,
    variant: dict[str, Any],
    unified_scores: pd.DataFrame,
    training_metrics: dict[str, Any],
    checkpoint_scores: pd.DataFrame,
    references: dict[str, float],
    output_dir: Path,
) -> dict[str, Any]:
    bits = _validation_metric(unified_scores, "direct_model", "bits_per_spike")
    poisson_nll = _validation_metric(unified_scores, "direct_model", "poisson_nll")
    factor_decoder_bits = _validation_metric(unified_scores, "factor_decoder", "bits_per_spike")
    row = dict.fromkeys(RESULT_COLUMNS)
    row.update({key: value for key, value in variant.items() if key in RESULT_COLUMNS})
    row.update(
        {
            "run_id": run_id,
            "run_index": run_index,
            "status": str(training_metrics.get("status", "completed")),
            "objective_name": str(variant["name"]),
            "epochs": int(variant.get("epochs", training_metrics.get("epochs", 0))),
            "device": str(training_metrics.get("device", "cuda")),
            "validation_unified_bits_per_spike": bits,
            "validation_poisson_nll": poisson_nll,
            "validation_behavior_mean_r2": float(
                training_metrics.get("validation_behavior_mean_r2", float("nan"))
            ),
            "validation_factor_decoder_unified_bits_per_spike": factor_decoder_bits,
            "train_total_loss": training_metrics.get("train_total_loss"),
            "validation_total_loss": training_metrics.get("validation_total_loss"),
            "validation_heldout_prediction_loss": training_metrics.get(
                "validation_heldout_prediction_loss"
            ),
            "z0_kl_loss": training_metrics.get("z0_kl_loss"),
            "drift_regularization_loss": training_metrics.get("drift_regularization_loss"),
            "rate_calibration_loss": training_metrics.get("rate_calibration_loss"),
            "drift_norm": training_metrics.get("drift_norm"),
            "diffusion_mean": training_metrics.get("diffusion_mean"),
            "best_checkpoint_source": _best_checkpoint_source(checkpoint_scores),
            "beats_train_mean_reference": bits
            > float(references["train_mean_validation_bits_per_spike"]),
            "beats_factor_latent_unified": bits
            > float(references["factor_latent_unified_validation_bits_per_spike"]),
            "beats_previous_neural_ode_refinement": bits
            > float(references["previous_neural_ode_refinement_validation_bits_per_spike"]),
            "beats_switching_ode": bits
            > float(references["switching_ode_validation_bits_per_spike"]),
            "output_dir": str(output_dir),
            "notes": str(variant.get("notes", training_metrics.get("notes", ""))),
        }
    )
    return row


def rank_neural_ode_objective_results(
    results: pd.DataFrame,
    metric: str = "validation_unified_bits_per_spike",
) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    completed = results[results["status"] == "completed"].copy()
    if completed.empty:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    completed["_simplicity"] = completed.apply(_objective_simplicity, axis=1)
    ranked = completed.sort_values(
        [
            metric,
            "validation_poisson_nll",
            "validation_behavior_mean_r2",
            "_simplicity",
            "run_index",
        ],
        ascending=[False, True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked[LEADERBOARD_COLUMNS]


def build_neural_ode_objective_diagnostics(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=DIAGNOSTIC_COLUMNS)
    diagnostics = results.reindex(columns=DIAGNOSTIC_COLUMNS)
    return diagnostics


def summarize_neural_ode_objectives(
    results: pd.DataFrame,
    references: dict[str, float],
) -> dict[str, Any]:
    leaderboard = rank_neural_ode_objective_results(results)
    successful = int((results.get("status", pd.Series(dtype=str)) == "completed").sum())
    summary: dict[str, Any] = {
        "runs_attempted": int(len(results)),
        "successful_runs": successful,
        "selection_metric": "validation_unified_bits_per_spike",
        "selection_mode": "max",
        "checkpoint_selection_method": "post_training_unified_rerank",
        "reference_model": "train_heldout_mean_rate",
        "evaluation_metric_is_unweighted": True,
        "train_mean_validation_bits_per_spike": float(
            references["train_mean_validation_bits_per_spike"]
        ),
        "factor_latent_unified_reference": float(
            references["factor_latent_unified_validation_bits_per_spike"]
        ),
        "previous_neural_ode_refinement_reference": float(
            references["previous_neural_ode_refinement_validation_bits_per_spike"]
        ),
        "switching_ode_reference": float(references["switching_ode_validation_bits_per_spike"]),
        "oracle_validation_bits_per_spike": float(references["oracle_validation_bits_per_spike"]),
        "old_incompatible_mean_rate_values_used_as_targets": False,
        "official_benchmark_claim": False,
        "complete_neural_ode_research_claim": False,
    }
    if leaderboard.empty:
        return summary | {
            "best_run_id": None,
            "best_objective_name": None,
            "best_validation_unified_bits_per_spike": None,
            "best_validation_poisson_nll": None,
            "best_factor_decoder_unified_bits_per_spike": None,
            "best_heldout_loss_weight": None,
            "best_zero_count_weight": None,
            "best_positive_count_weight": None,
            "best_rate_calibration_loss_weight": None,
            "best_rate_calibration_loss": None,
            "best_drift_norm": None,
            "best_drift_regularization_loss": None,
            "best_diffusion_mean": None,
            "best_checkpoint_source": None,
            "best_run_params": {},
            "beats_factor_latent_unified": None,
            "beats_previous_neural_ode_refinement": None,
            "beats_switching_ode": None,
        }
    best = leaderboard.iloc[0]
    best_result = results.loc[results["run_id"] == best["run_id"]].iloc[0]
    return summary | {
        "best_run_id": str(best["run_id"]),
        "best_objective_name": str(best["objective_name"]),
        "best_validation_unified_bits_per_spike": float(best["validation_unified_bits_per_spike"]),
        "best_validation_poisson_nll": float(best["validation_poisson_nll"]),
        "best_factor_decoder_unified_bits_per_spike": float(
            best["validation_factor_decoder_unified_bits_per_spike"]
        ),
        "best_heldout_loss_weight": float(best_result["heldout_loss_weight"]),
        "best_zero_count_weight": float(best_result["zero_count_weight"]),
        "best_positive_count_weight": float(best_result["positive_count_weight"]),
        "best_rate_calibration_loss_weight": float(best_result["rate_calibration_loss_weight"]),
        "best_rate_calibration_loss": float(best_result["rate_calibration_loss"]),
        "best_drift_norm": float(best_result["drift_norm"]),
        "best_drift_regularization_loss": float(best_result["drift_regularization_loss"]),
        "best_diffusion_mean": float(best_result["diffusion_mean"]),
        "best_checkpoint_source": str(best_result["best_checkpoint_source"]),
        "beats_factor_latent_unified": bool(best["beats_factor_latent_unified"]),
        "beats_previous_neural_ode_refinement": bool(best["beats_previous_neural_ode_refinement"]),
        "beats_switching_ode": bool(best_result["beats_switching_ode"]),
        "best_run_params": {
            key: best_result[key] for key in _VARIANT_COLUMNS if key in best_result
        },
    }
