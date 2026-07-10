from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

FOLD_BALANCE_COLUMNS = [
    "repeat_index",
    "fold_index",
    "n_trials",
    "mean_population_rate_hz",
    "std_population_rate_hz",
    "mean_heldout_rate_hz",
    "std_heldout_rate_hz",
    "mean_endpoint_distance",
    "std_endpoint_distance",
    "mean_speed",
    "std_mean_speed",
    "endpoint_direction_entropy",
    "stratum_count",
]

COMPARISON_COLUMNS = [
    "repeat_index",
    "metric",
    "min_value",
    "max_value",
    "range",
    "mean_value",
    "std_value",
    "coefficient_of_variation",
]

COMPARISON_METRICS = (
    "n_trials",
    "mean_population_rate_hz",
    "mean_heldout_rate_hz",
    "mean_endpoint_distance",
    "mean_speed",
    "endpoint_direction_entropy",
)

DIRECTION_BINS = 8

# A fold whose trial count deviates from the mean by more than this fraction, or whose mean
# held-out rate spans more than this fraction of the overall mean, is flagged.
TRIAL_COUNT_TOLERANCE = 0.25
RATE_RANGE_TOLERANCE = 0.20


def _mean_std(values: pd.Series) -> tuple[float, float]:
    finite = values.dropna()
    if finite.empty:
        return (float("nan"), float("nan"))
    std = float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
    return (float(finite.mean()), std)


def endpoint_direction_entropy(angles: pd.Series, bins: int = DIRECTION_BINS) -> float:
    """Shannon entropy (nats) of reach directions binned into equal-width sectors."""
    finite = angles.dropna().to_numpy(dtype=np.float64)
    if finite.size == 0:
        return float("nan")
    shifted = (finite + np.pi) % (2.0 * np.pi)
    indices = np.clip((shifted / (2.0 * np.pi / bins)).astype(int), 0, bins - 1)
    counts = np.bincount(indices, minlength=bins).astype(np.float64)
    probabilities = counts[counts > 0.0] / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def compute_fold_balance_statistics(fold_assignments: pd.DataFrame) -> pd.DataFrame:
    if fold_assignments.empty:
        return pd.DataFrame(columns=FOLD_BALANCE_COLUMNS)
    rows: list[dict[str, Any]] = []
    for (repeat_index, fold_index), group in fold_assignments.groupby(
        ["repeat_index", "fold_index"], sort=True
    ):
        population_mean, population_std = _mean_std(group["population_rate_hz"])
        heldout_mean, heldout_std = _mean_std(group["heldout_rate_hz"])
        distance_mean, distance_std = _mean_std(group["endpoint_distance"])
        speed_mean, speed_std = _mean_std(group["mean_speed"])
        rows.append(
            {
                "repeat_index": int(repeat_index),
                "fold_index": int(fold_index),
                "n_trials": int(len(group)),
                "mean_population_rate_hz": population_mean,
                "std_population_rate_hz": population_std,
                "mean_heldout_rate_hz": heldout_mean,
                "std_heldout_rate_hz": heldout_std,
                "mean_endpoint_distance": distance_mean,
                "std_endpoint_distance": distance_std,
                "mean_speed": speed_mean,
                "std_mean_speed": speed_std,
                "endpoint_direction_entropy": endpoint_direction_entropy(
                    group["endpoint_angle_rad"]
                ),
                "stratum_count": int(group["stratum"].nunique()),
            }
        )
    return pd.DataFrame(rows, columns=FOLD_BALANCE_COLUMNS)


def compare_fold_balance(fold_balance: pd.DataFrame) -> pd.DataFrame:
    if fold_balance.empty:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)
    rows: list[dict[str, Any]] = []
    for repeat_index, group in fold_balance.groupby("repeat_index", sort=True):
        for metric in COMPARISON_METRICS:
            values = group[metric].dropna().to_numpy(dtype=np.float64)
            if values.size == 0:
                rows.append(
                    {
                        "repeat_index": int(repeat_index),
                        "metric": metric,
                        "min_value": float("nan"),
                        "max_value": float("nan"),
                        "range": float("nan"),
                        "mean_value": float("nan"),
                        "std_value": float("nan"),
                        "coefficient_of_variation": float("nan"),
                    }
                )
                continue
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            rows.append(
                {
                    "repeat_index": int(repeat_index),
                    "metric": metric,
                    "min_value": float(np.min(values)),
                    "max_value": float(np.max(values)),
                    "range": float(np.max(values) - np.min(values)),
                    "mean_value": mean,
                    "std_value": std,
                    "coefficient_of_variation": float(std / mean) if mean != 0.0 else float("nan"),
                }
            )
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def _mean_range(comparisons: pd.DataFrame, metric: str) -> float:
    rows = comparisons[comparisons["metric"] == metric]
    return float("nan") if rows.empty else float(rows["range"].mean())


def summarize_fold_balance(fold_balance: pd.DataFrame, comparisons: pd.DataFrame) -> dict[str, Any]:
    if fold_balance.empty:
        return {
            "mean_population_rate_fold_range": float("nan"),
            "mean_heldout_rate_fold_range": float("nan"),
            "mean_endpoint_distance_fold_range": float("nan"),
            "mean_speed_fold_range": float("nan"),
            "mean_endpoint_direction_entropy": float("nan"),
            "endpoint_direction_entropy_max": float(np.log(DIRECTION_BINS)),
            "endpoint_direction_concentrated": False,
            "fold_balance_warning": "no folds were assigned",
        }
    warnings: list[str] = []
    trial_counts = fold_balance["n_trials"].to_numpy(dtype=np.float64)
    mean_trials = float(np.mean(trial_counts))
    if mean_trials > 0.0:
        deviation = float(np.max(np.abs(trial_counts - mean_trials)) / mean_trials)
        if deviation > TRIAL_COUNT_TOLERANCE:
            warnings.append(
                f"fold trial counts are imbalanced (max deviation {deviation:.3f} of the mean)"
            )
    heldout_range = _mean_range(comparisons, "mean_heldout_rate_hz")
    heldout_mean = float(fold_balance["mean_heldout_rate_hz"].mean())
    if np.isfinite(heldout_range) and heldout_mean > 0.0:
        relative = heldout_range / heldout_mean
        if relative > RATE_RANGE_TOLERANCE:
            warnings.append(
                "held-out rate differs across folds by "
                f"{relative:.3f} of the mean, indicating distribution shift"
            )
    population_range = _mean_range(comparisons, "mean_population_rate_hz")
    population_mean = float(fold_balance["mean_population_rate_hz"].mean())
    if np.isfinite(population_range) and population_mean > 0.0:
        relative = population_range / population_mean
        if relative > RATE_RANGE_TOLERANCE:
            warnings.append(
                "population rate differs across folds by "
                f"{relative:.3f} of the mean, indicating distribution shift"
            )
    mean_entropy = float(fold_balance["endpoint_direction_entropy"].mean())
    max_entropy = float(np.log(DIRECTION_BINS))
    # Low entropy is a property of the dataset and window, not of the fold assignment. It means
    # direction stratification has little left to balance, so say so rather than imply success.
    concentrated = bool(np.isfinite(mean_entropy) and mean_entropy < 0.5 * max_entropy)
    return {
        "mean_population_rate_fold_range": population_range,
        "mean_heldout_rate_fold_range": heldout_range,
        "mean_endpoint_distance_fold_range": _mean_range(comparisons, "mean_endpoint_distance"),
        "mean_speed_fold_range": _mean_range(comparisons, "mean_speed"),
        "endpoint_direction_entropy_max": max_entropy,
        "endpoint_direction_concentrated": concentrated,
        "mean_endpoint_direction_entropy": float(fold_balance["endpoint_direction_entropy"].mean()),
        "fold_balance_warning": "; ".join(warnings) if warnings else "none",
    }
