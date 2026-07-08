from __future__ import annotations

from typing import Any

import pandas as pd  # type: ignore[import-untyped]

EVALUATION_COLUMNS = [
    "run_id",
    "dropout_rate",
    "status",
    "validation_bits_per_spike",
    "validation_poisson_nll",
    "validation_behavior_mean_r2",
    "validation_factor_decoder_bits_per_spike",
    "train_total_loss",
    "validation_total_loss",
    "validation_heldout_prediction_loss",
    "beats_previous_20ms_lfads",
    "beats_same_bin_factor_latent",
    "beats_same_bin_mean_rate",
    "output_dir",
    "notes",
]


def _ref(references: dict[str, float], key: str) -> float:
    return float(references[key])


def build_dropout_result_row(
    dropout_rate: float,
    run_id: str,
    metrics: dict[str, Any],
    references: dict[str, float],
) -> dict[str, Any]:
    bits = float(metrics.get("validation_bits_per_spike", float("nan")))
    status = str(metrics.get("status", "completed"))
    row = {
        "run_id": run_id,
        "dropout_rate": float(dropout_rate),
        "status": status,
        "validation_bits_per_spike": bits,
        "validation_poisson_nll": float(metrics.get("validation_poisson_nll", float("nan"))),
        "validation_behavior_mean_r2": float(
            metrics.get("validation_behavior_mean_r2", float("nan"))
        ),
        "validation_factor_decoder_bits_per_spike": float(
            metrics.get("validation_factor_decoder_bits_per_spike", float("nan"))
        ),
        "train_total_loss": float(metrics.get("train_total_loss", float("nan"))),
        "validation_total_loss": float(metrics.get("validation_total_loss", float("nan"))),
        "validation_heldout_prediction_loss": float(
            metrics.get("validation_heldout_prediction_loss", float("nan"))
        ),
        "beats_previous_20ms_lfads": bits
        > _ref(references, "previous_20ms_lfads_validation_bits_per_spike"),
        "beats_same_bin_factor_latent": bits
        > _ref(references, "same_bin_factor_latent_validation_bits_per_spike"),
        "beats_same_bin_mean_rate": bits
        > _ref(references, "same_bin_mean_rate_validation_bits_per_spike"),
        "output_dir": str(metrics.get("output_dir", "")),
        "notes": str(metrics.get("notes", "")),
    }
    return {column: row.get(column) for column in EVALUATION_COLUMNS}


def rank_dropout_runs(
    results: pd.DataFrame,
    primary_metric: str = "validation_bits_per_spike",
) -> pd.DataFrame:
    if primary_metric not in results.columns:
        msg = f"missing primary metric column: {primary_metric}"
        raise ValueError(msg)
    completed = results[results.get("status") == "completed"].copy()
    if completed.empty:
        return completed.assign(rank=pd.Series(dtype="int64"))
    ranked = completed.sort_values(
        [primary_metric, "validation_poisson_nll", "dropout_rate"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def summarize_dropout_runs(
    results: pd.DataFrame,
    references: dict[str, float],
) -> dict[str, Any]:
    ranked = rank_dropout_runs(results)
    summary: dict[str, Any] = {
        "runs_attempted": int(len(results)),
        "successful_runs": int((results.get("status") == "completed").sum()),
        "references": dict(references),
        "official_benchmark_claim": False,
        "full_lfads_claim": False,
    }
    if ranked.empty:
        return summary | {
            "best_run_id": None,
            "best_dropout_rate": None,
            "best_validation_bits_per_spike": None,
            "best_validation_poisson_nll": None,
            "best_validation_factor_decoder_bits_per_spike": None,
            "coordinated_dropout_improves_lfads": None,
            "beats_same_bin_factor_latent": None,
            "beats_same_bin_mean_rate": None,
        }
    best = ranked.iloc[0]
    bits = float(best["validation_bits_per_spike"])
    return summary | {
        "best_run_id": str(best["run_id"]),
        "best_dropout_rate": float(best["dropout_rate"]),
        "best_validation_bits_per_spike": bits,
        "best_validation_poisson_nll": float(best["validation_poisson_nll"]),
        "best_validation_behavior_mean_r2": float(best["validation_behavior_mean_r2"]),
        "best_validation_factor_decoder_bits_per_spike": float(
            best["validation_factor_decoder_bits_per_spike"]
        ),
        "coordinated_dropout_improves_lfads": bits
        > _ref(references, "previous_20ms_lfads_validation_bits_per_spike"),
        "beats_same_bin_factor_latent": bits
        > _ref(references, "same_bin_factor_latent_validation_bits_per_spike"),
        "beats_same_bin_mean_rate": bits
        > _ref(references, "same_bin_mean_rate_validation_bits_per_spike"),
        "best_output_dir": str(best.get("output_dir", "")),
    }
