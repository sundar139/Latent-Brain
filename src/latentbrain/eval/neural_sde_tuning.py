from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

RESULT_COLUMNS = [
    "run_id",
    "run_index",
    "status",
    "encoder_hidden_dim",
    "drift_hidden_dim",
    "diffusion_hidden_dim",
    "latent_dim",
    "factor_dim",
    "input_dropout_rate",
    "heldout_loss_weight",
    "kl_scale",
    "diffusion_scale",
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
    "drift_norm",
    "diffusion_mean",
    "beats_train_mean_reference",
    "beats_factor_latent_unified",
    "beats_previous_best_lfads_family",
    "output_dir",
    "notes",
]

LEADERBOARD_COLUMNS = [
    "rank",
    "run_id",
    "validation_unified_bits_per_spike",
    "validation_poisson_nll",
    "validation_factor_decoder_unified_bits_per_spike",
    "input_dropout_rate",
    "heldout_loss_weight",
    "kl_scale",
    "diffusion_scale",
    "latent_dim",
    "factor_dim",
    "beats_factor_latent_unified",
    "beats_previous_best_lfads_family",
    "notes",
]


def _validation_metric(unified_scores: pd.DataFrame, prediction_source: str, column: str) -> float:
    rows = unified_scores[
        (unified_scores["split"] == "validation")
        & (unified_scores["prediction_source"] == prediction_source)
    ]
    if rows.empty or column not in rows:
        return float("nan")
    return float(rows.iloc[0][column])


def build_neural_sde_result_row(
    run_id: str,
    run_index: int,
    params: dict[str, Any],
    unified_scores: pd.DataFrame,
    training_metrics: dict[str, Any],
    references: dict[str, float],
    output_dir: Path,
) -> dict[str, Any]:
    bits = _validation_metric(unified_scores, "direct_model", "bits_per_spike")
    poisson_nll = _validation_metric(unified_scores, "direct_model", "poisson_nll")
    factor_decoder_bits = _validation_metric(unified_scores, "factor_decoder", "bits_per_spike")
    row = dict.fromkeys(RESULT_COLUMNS)
    row.update(params)
    row.update(
        {
            "run_id": run_id,
            "run_index": run_index,
            "status": str(training_metrics.get("status", "completed")),
            "epochs": int(params.get("epochs", training_metrics.get("epochs", 0))),
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
            "drift_norm": training_metrics.get("drift_norm"),
            "diffusion_mean": training_metrics.get("diffusion_mean"),
            "beats_train_mean_reference": bits
            > float(references["train_mean_validation_bits_per_spike"]),
            "beats_factor_latent_unified": bits
            > float(references["factor_latent_unified_validation_bits_per_spike"]),
            "beats_previous_best_lfads_family": bits
            > float(references["previous_best_lfads_family_validation_bits_per_spike"]),
            "output_dir": str(output_dir),
            "notes": str(training_metrics.get("notes", "")),
        }
    )
    return row


def rank_neural_sde_results(
    results: pd.DataFrame,
    metric: str = "validation_unified_bits_per_spike",
) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    completed = results[results["status"] == "completed"].copy()
    if completed.empty:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    ranked = completed.sort_values(
        [
            metric,
            "validation_poisson_nll",
            "validation_behavior_mean_r2",
            "diffusion_scale",
            "run_index",
        ],
        ascending=[False, True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked[LEADERBOARD_COLUMNS]


def summarize_neural_sde_tuning(
    results: pd.DataFrame,
    references: dict[str, float],
) -> dict[str, Any]:
    leaderboard = rank_neural_sde_results(results)
    successful = int((results.get("status", pd.Series(dtype=str)) == "completed").sum())
    summary: dict[str, Any] = {
        "runs_attempted": int(len(results)),
        "successful_runs": successful,
        "selection_metric": "validation_unified_bits_per_spike",
        "selection_mode": "max",
        "reference_model": "train_heldout_mean_rate",
        "train_mean_validation_bits_per_spike": float(
            references["train_mean_validation_bits_per_spike"]
        ),
        "factor_latent_unified_reference": float(
            references["factor_latent_unified_validation_bits_per_spike"]
        ),
        "previous_best_lfads_family_reference": float(
            references["previous_best_lfads_family_validation_bits_per_spike"]
        ),
        "oracle_validation_bits_per_spike": float(references["oracle_validation_bits_per_spike"]),
        "old_incompatible_mean_rate_values_used_as_targets": False,
        "official_benchmark_claim": False,
        "full_neural_sde_claim": False,
    }
    if leaderboard.empty:
        return summary | {
            "best_run_id": None,
            "best_validation_unified_bits_per_spike": None,
            "best_validation_poisson_nll": None,
            "best_factor_decoder_unified_bits_per_spike": None,
            "best_drift_norm": None,
            "best_diffusion_mean": None,
            "beats_factor_latent_unified": None,
            "beats_previous_best_lfads_family": None,
        }
    best = leaderboard.iloc[0]
    best_result = results.loc[results["run_id"] == best["run_id"]].iloc[0]
    return summary | {
        "best_run_id": str(best["run_id"]),
        "best_validation_unified_bits_per_spike": float(best["validation_unified_bits_per_spike"]),
        "best_validation_poisson_nll": float(best["validation_poisson_nll"]),
        "best_factor_decoder_unified_bits_per_spike": float(
            best["validation_factor_decoder_unified_bits_per_spike"]
        ),
        "best_drift_norm": float(best_result["drift_norm"]),
        "best_diffusion_mean": float(best_result["diffusion_mean"]),
        "beats_factor_latent_unified": bool(best["beats_factor_latent_unified"]),
        "beats_previous_best_lfads_family": bool(best["beats_previous_best_lfads_family"]),
        "best_run_params": {
            key: best_result[key]
            for key in (
                "encoder_hidden_dim",
                "drift_hidden_dim",
                "diffusion_hidden_dim",
                "latent_dim",
                "factor_dim",
                "input_dropout_rate",
                "heldout_loss_weight",
                "kl_scale",
                "diffusion_scale",
            )
            if key in best_result
        },
    }
