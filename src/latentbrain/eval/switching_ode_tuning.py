from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

RESULT_COLUMNS = [
    "run_id",
    "run_index",
    "status",
    "encoder_hidden_dim",
    "drift_hidden_dim",
    "latent_dim",
    "factor_dim",
    "n_regimes",
    "regime_temperature",
    "input_dropout_rate",
    "heldout_loss_weight",
    "kl_scale",
    "entropy_regularization",
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
    "mean_regime_entropy",
    "active_regime_count",
    "max_regime_occupancy",
    "min_regime_occupancy",
    "best_checkpoint_source",
    "beats_train_mean_reference",
    "beats_factor_latent_unified",
    "beats_previous_neural_ode",
    "beats_previous_neural_sde",
    "output_dir",
    "notes",
]

LEADERBOARD_COLUMNS = [
    "rank",
    "run_id",
    "validation_unified_bits_per_spike",
    "validation_poisson_nll",
    "validation_factor_decoder_unified_bits_per_spike",
    "n_regimes",
    "regime_temperature",
    "entropy_regularization",
    "active_regime_count",
    "mean_regime_entropy",
    "best_checkpoint_source",
    "beats_factor_latent_unified",
    "beats_previous_neural_ode",
    "notes",
]


def compute_regime_diagnostics(regime_probs: np.ndarray, split: str) -> pd.DataFrame:
    if regime_probs.ndim != 3:
        msg = "regime_probs must have shape [trials, time, regimes]"
        raise ValueError(msg)
    clipped = np.clip(regime_probs.astype(float), 1.0e-12, 1.0)
    occupancy = np.round(clipped.mean(axis=(0, 1)), 12)
    std = clipped.reshape(-1, clipped.shape[-1]).std(axis=0)
    entropy = float((-(clipped * np.log(clipped)).sum(axis=-1)).mean())
    rows = [
        {
            "split": split,
            "regime_index": index,
            "mean_occupancy": float(value),
            "std_occupancy": float(std[index]),
            "min_probability": float(clipped[:, :, index].min()),
            "max_probability": float(clipped[:, :, index].max()),
            "entropy": entropy,
            "active": bool(value > 0.05),
        }
        for index, value in enumerate(occupancy)
    ]
    return pd.DataFrame(rows)


def summarize_regime_diagnostics(
    diagnostics: pd.DataFrame, split: str = "validation"
) -> dict[str, Any]:
    rows = diagnostics[diagnostics["split"] == split] if not diagnostics.empty else diagnostics
    if rows.empty:
        return {
            "mean_regime_entropy": float("nan"),
            "active_regime_count": 0,
            "max_regime_occupancy": float("nan"),
            "min_regime_occupancy": float("nan"),
        }
    return {
        "mean_regime_entropy": float(rows["entropy"].mean()),
        "active_regime_count": int(rows["active"].astype(bool).sum()),
        "max_regime_occupancy": float(rows["mean_occupancy"].max()),
        "min_regime_occupancy": float(rows["mean_occupancy"].min()),
    }


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


def build_switching_ode_result_row(
    run_id: str,
    run_index: int,
    params: dict[str, Any],
    unified_scores: pd.DataFrame,
    training_metrics: dict[str, Any],
    checkpoint_scores: pd.DataFrame,
    regime_summary: dict[str, Any],
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
            "mean_regime_entropy": regime_summary.get("mean_regime_entropy"),
            "active_regime_count": regime_summary.get("active_regime_count"),
            "max_regime_occupancy": regime_summary.get("max_regime_occupancy"),
            "min_regime_occupancy": regime_summary.get("min_regime_occupancy"),
            "best_checkpoint_source": _best_checkpoint_source(checkpoint_scores),
            "beats_train_mean_reference": bits
            > float(references["train_mean_validation_bits_per_spike"]),
            "beats_factor_latent_unified": bits
            > float(references["factor_latent_unified_validation_bits_per_spike"]),
            "beats_previous_neural_ode": bits
            > float(references["previous_neural_ode_validation_bits_per_spike"]),
            "beats_previous_neural_sde": bits
            > float(references["previous_neural_sde_validation_bits_per_spike"]),
            "output_dir": str(output_dir),
            "notes": str(training_metrics.get("notes", "")),
        }
    )
    return row


def rank_switching_ode_results(
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
            "active_regime_count",
            "run_index",
        ],
        ascending=[False, True, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked[LEADERBOARD_COLUMNS]


def summarize_switching_ode_tuning(
    results: pd.DataFrame,
    references: dict[str, float],
) -> dict[str, Any]:
    leaderboard = rank_switching_ode_results(results)
    successful = int((results.get("status", pd.Series(dtype=str)) == "completed").sum())
    summary: dict[str, Any] = {
        "runs_attempted": int(len(results)),
        "successful_runs": successful,
        "selection_metric": "validation_unified_bits_per_spike",
        "selection_mode": "max",
        "checkpoint_selection_method": "post_training_unified_rerank",
        "reference_model": "train_heldout_mean_rate",
        "train_mean_validation_bits_per_spike": float(
            references["train_mean_validation_bits_per_spike"]
        ),
        "factor_latent_unified_reference": float(
            references["factor_latent_unified_validation_bits_per_spike"]
        ),
        "previous_neural_ode_reference": float(
            references["previous_neural_ode_validation_bits_per_spike"]
        ),
        "previous_neural_sde_reference": float(
            references["previous_neural_sde_validation_bits_per_spike"]
        ),
        "previous_best_lfads_family_reference": float(
            references["previous_best_lfads_family_validation_bits_per_spike"]
        ),
        "oracle_validation_bits_per_spike": float(references["oracle_validation_bits_per_spike"]),
        "old_incompatible_mean_rate_values_used_as_targets": False,
        "official_benchmark_claim": False,
        "full_rslds_claim": False,
    }
    if leaderboard.empty:
        return summary | {
            "best_run_id": None,
            "best_validation_unified_bits_per_spike": None,
            "best_validation_poisson_nll": None,
            "best_factor_decoder_unified_bits_per_spike": None,
            "best_drift_norm": None,
            "best_checkpoint_source": None,
            "best_mean_regime_entropy": None,
            "best_active_regime_count": None,
            "best_max_regime_occupancy": None,
            "beats_factor_latent_unified": None,
            "beats_previous_neural_ode": None,
            "beats_previous_neural_sde": None,
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
        "best_checkpoint_source": str(best_result["best_checkpoint_source"]),
        "best_mean_regime_entropy": float(best_result["mean_regime_entropy"]),
        "best_active_regime_count": int(best_result["active_regime_count"]),
        "best_max_regime_occupancy": float(best_result["max_regime_occupancy"]),
        "best_min_regime_occupancy": float(best_result["min_regime_occupancy"]),
        "beats_factor_latent_unified": bool(best["beats_factor_latent_unified"]),
        "beats_previous_neural_ode": bool(best["beats_previous_neural_ode"]),
        "beats_previous_neural_sde": bool(best_result["beats_previous_neural_sde"]),
        "best_run_params": {
            key: best_result[key]
            for key in (
                "encoder_hidden_dim",
                "drift_hidden_dim",
                "latent_dim",
                "factor_dim",
                "n_regimes",
                "regime_temperature",
                "input_dropout_rate",
                "heldout_loss_weight",
                "kl_scale",
                "entropy_regularization",
            )
            if key in best_result
        },
    }
