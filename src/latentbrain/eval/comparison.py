from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

COMPARISON_COLUMNS = [
    "method_name",
    "split",
    "prediction_source",
    "time_bins",
    "window_seconds",
    "n_trials",
    "n_target_neurons",
    "spike_count",
    "poisson_nll",
    "poisson_log_likelihood",
    "reference_log_likelihood",
    "bits_per_spike",
    "mse_rate_hz",
    "mae_rate_hz",
    "behavior_mean_r2",
    "uses_neural_network",
    "uses_train_only_fit",
    "official_benchmark_claim",
    "notes",
]


def _float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _int_or_zero(value: Any) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    return int(value)


def build_comparison_row(
    method_name: str,
    split: str,
    prediction_source: str,
    metrics: dict[str, float],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build one normalized comparison row."""
    row = {
        "method_name": method_name,
        "split": split,
        "prediction_source": prediction_source,
        "time_bins": _int_or_zero(metadata.get("time_bins", metadata.get("n_time_bins"))),
        "window_seconds": _float_or_nan(metadata.get("window_seconds")),
        "n_trials": _int_or_zero(metadata.get("n_trials")),
        "n_target_neurons": _int_or_zero(
            metadata.get("n_target_neurons", metadata.get("n_neurons"))
        ),
        "spike_count": _float_or_nan(metrics.get("spike_count")),
        "poisson_nll": _float_or_nan(metrics.get("poisson_nll")),
        "poisson_log_likelihood": _float_or_nan(metrics.get("poisson_log_likelihood")),
        "reference_log_likelihood": _float_or_nan(metrics.get("reference_log_likelihood")),
        "bits_per_spike": _float_or_nan(metrics.get("bits_per_spike")),
        "mse_rate_hz": _float_or_nan(metrics.get("mse_rate_hz")),
        "mae_rate_hz": _float_or_nan(metrics.get("mae_rate_hz")),
        "behavior_mean_r2": _float_or_nan(metrics.get("behavior_mean_r2")),
        "uses_neural_network": bool(metadata.get("uses_neural_network", False)),
        "uses_train_only_fit": bool(metadata.get("uses_train_only_fit", True)),
        "official_benchmark_claim": False,
        "notes": str(metadata.get("notes", "")),
    }
    return {column: row[column] for column in COMPARISON_COLUMNS}


def rank_validation_methods(
    metrics: pd.DataFrame,
    primary_split: str,
    primary_metric: str,
) -> pd.DataFrame:
    """Rank validation rows by the configured metric and conservative tie breakers."""
    if primary_metric != "bits_per_spike":
        msg = "only bits_per_spike ranking is supported"
        raise ValueError(msg)
    validation = metrics[metrics["split"] == primary_split].copy()
    if validation.empty:
        return pd.DataFrame(columns=list(metrics.columns))
    validation["_primary"] = pd.to_numeric(validation[primary_metric], errors="coerce")
    validation["_poisson_nll"] = pd.to_numeric(validation["poisson_nll"], errors="coerce")
    validation["_behavior"] = pd.to_numeric(validation["behavior_mean_r2"], errors="coerce")
    validation["_neural"] = validation["uses_neural_network"].astype(bool)
    ranked = validation.sort_values(
        by=["_primary", "_poisson_nll", "_behavior", "_neural"],
        ascending=[False, True, False, True],
        na_position="last",
        kind="mergesort",
    )
    return ranked.drop(columns=["_primary", "_poisson_nll", "_behavior", "_neural"]).reset_index(
        drop=True
    )


def _metric_value(row: pd.Series, name: str) -> float | None:
    value = row.get(name)
    if value is None or pd.isna(value):
        return None
    return float(value)


def _method_split_metric(
    metrics: pd.DataFrame,
    method_name: str,
    split: str,
    metric: str,
    prediction_source: str | None = None,
) -> float | None:
    rows = metrics[(metrics["method_name"] == method_name) & (metrics["split"] == split)]
    if prediction_source is not None:
        rows = rows[rows["prediction_source"] == prediction_source]
    if rows.empty:
        return None
    value = rows.iloc[0].get(metric)
    if pd.isna(value):
        return None
    return float(value)


def summarize_comparison(
    comparison_metrics: pd.DataFrame,
    primary_split: str,
    primary_metric: str,
) -> dict[str, Any]:
    """Summarize normalized comparison metrics."""
    ranked = rank_validation_methods(comparison_metrics, primary_split, primary_metric)
    best = ranked.iloc[0] if not ranked.empty else None
    summary: dict[str, Any] = {
        "primary_split": primary_split,
        "primary_metric": primary_metric,
        "best_method_name": None if best is None else str(best["method_name"]),
        "best_prediction_source": None if best is None else str(best["prediction_source"]),
        "best_validation_bits_per_spike": None
        if best is None
        else _metric_value(best, "bits_per_spike"),
        "best_validation_poisson_nll": None if best is None else _metric_value(best, "poisson_nll"),
        "best_behavior_mean_r2": None if best is None else _metric_value(best, "behavior_mean_r2"),
        "official_benchmark_claim": False,
    }
    summary.update(
        {
            "mean_rate_windowed_validation_bits_per_spike": _method_split_metric(
                comparison_metrics, "mean_rate_windowed", primary_split, "bits_per_spike"
            ),
            "factor_latent_windowed_validation_bits_per_spike": _method_split_metric(
                comparison_metrics, "factor_latent_windowed", primary_split, "bits_per_spike"
            ),
            "lfads_cosmoothing_direct_validation_bits_per_spike": _method_split_metric(
                comparison_metrics,
                "lfads_gru_cosmoothing_direct",
                primary_split,
                "bits_per_spike",
                "direct_model",
            ),
            "lfads_cosmoothing_factor_decoder_validation_bits_per_spike": _method_split_metric(
                comparison_metrics,
                "lfads_gru_cosmoothing_factor_decoder",
                primary_split,
                "bits_per_spike",
                "factor_decoder",
            ),
        }
    )
    return summary
