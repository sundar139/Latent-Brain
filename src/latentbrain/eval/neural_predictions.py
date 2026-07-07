from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


def _as_finite_array(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must be finite"
        raise ValueError(msg)
    return array


def flatten_batch_time(values: np.ndarray) -> np.ndarray:
    """Flatten [trials, time, features] into [trials * time, features]."""
    array = _as_finite_array("values", values)
    if array.ndim != 3:
        msg = f"values must have rank 3; got shape {array.shape}"
        raise ValueError(msg)
    return array.reshape(array.shape[0] * array.shape[1], array.shape[2])


def reshape_flat_predictions(
    flat_values: np.ndarray,
    n_trials: int,
    n_time_bins: int,
    n_outputs: int,
) -> np.ndarray:
    """Reshape [trials * time, outputs] predictions back to trial-major form."""
    array = _as_finite_array("flat_values", flat_values)
    if array.ndim != 2:
        msg = f"flat_values must have rank 2; got shape {array.shape}"
        raise ValueError(msg)
    if min(n_trials, n_time_bins, n_outputs) <= 0:
        msg = "n_trials, n_time_bins, and n_outputs must be positive"
        raise ValueError(msg)
    expected = n_trials * n_time_bins
    if array.shape != (expected, n_outputs):
        msg = f"cannot reshape {array.shape} to ({n_trials}, {n_time_bins}, {n_outputs})"
        raise ValueError(msg)
    return array.reshape(n_trials, n_time_bins, n_outputs)


def summarize_factor_activity(factors: np.ndarray, split_name: str) -> pd.DataFrame:
    """Summarize LFADS-style factor activity per split."""
    flat = flatten_batch_time(factors)
    rows = []
    for factor_index in range(flat.shape[1]):
        values = flat[:, factor_index]
        rows.append(
            {
                "split": split_name,
                "factor_index": factor_index,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "variance": float(np.var(values)),
            }
        )
    return pd.DataFrame(
        rows, columns=["split", "factor_index", "mean", "std", "min", "max", "variance"]
    )


def summarize_rate_predictions(rates_hz: np.ndarray) -> dict[str, float]:
    """Summarize finite positive rate predictions without making benchmark claims."""
    rates = _as_finite_array("rates_hz", rates_hz)
    if rates.size == 0:
        msg = "rates_hz must not be empty"
        raise ValueError(msg)
    if np.any(rates <= 0.0):
        msg = "rates_hz must be positive"
        raise ValueError(msg)
    return {
        "mean_rate_hz": float(np.mean(rates)),
        "min_rate_hz": float(np.min(rates)),
        "max_rate_hz": float(np.max(rates)),
    }
