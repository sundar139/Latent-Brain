from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

CALIBRATION_COLUMNS = [
    "rate_bin",
    "n_observations",
    "mean_predicted_rate_hz",
    "mean_reference_rate_hz",
    "observed_rate_hz",
    "mean_count",
    "spike_count",
]


def _aligned_arrays(
    counts: np.ndarray, predicted_rates_hz: np.ndarray, reference_rates_hz: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts_array = np.asarray(counts, dtype=np.float64)
    predicted = np.asarray(predicted_rates_hz, dtype=np.float64)
    reference = np.asarray(reference_rates_hz, dtype=np.float64)
    if counts_array.shape != predicted.shape or counts_array.shape != reference.shape:
        msg = (
            "counts, predicted_rates_hz, and reference_rates_hz must have the same shape; "
            f"got {counts_array.shape}, {predicted.shape}, and {reference.shape}"
        )
        raise ValueError(msg)
    if counts_array.ndim != 3:
        msg = (
            "counts, predicted rates, and reference rates must have rank 3; "
            f"got {counts_array.shape}"
        )
        raise ValueError(msg)
    if not (
        np.all(np.isfinite(counts_array))
        and np.all(np.isfinite(predicted))
        and np.all(np.isfinite(reference))
    ):
        msg = "counts and rates must contain only finite values"
        raise ValueError(msg)
    if np.any(counts_array < 0.0):
        msg = "counts must be non-negative"
        raise ValueError(msg)
    return counts_array, predicted, reference


def _safe_mean(values: np.ndarray) -> float:
    return float("nan") if values.size == 0 else float(np.mean(values))


def compute_rate_calibration_table(
    counts: np.ndarray,
    predicted_rates_hz: np.ndarray,
    reference_rates_hz: np.ndarray,
    bin_size_ms: int,
    n_bins: int,
) -> pd.DataFrame:
    """Bin predictions by predicted rate and compare predicted/reference/observed rates."""
    if bin_size_ms <= 0:
        msg = "bin_size_ms must be positive"
        raise ValueError(msg)
    if n_bins <= 0:
        msg = "n_bins must be positive"
        raise ValueError(msg)
    counts_array, predicted, reference = _aligned_arrays(
        counts, predicted_rates_hz, reference_rates_hz
    )
    flat_counts = counts_array.reshape(-1)
    flat_predicted = predicted.reshape(-1)
    flat_reference = reference.reshape(-1)
    if flat_predicted.size == 0:
        return pd.DataFrame(columns=CALIBRATION_COLUMNS)

    minimum = float(np.min(flat_predicted))
    maximum = float(np.max(flat_predicted))
    if minimum == maximum:
        edges = np.linspace(minimum - 0.5, maximum + 0.5, n_bins + 1)
    else:
        edges = np.linspace(minimum, maximum, n_bins + 1)
    assignments = np.clip(np.digitize(flat_predicted, edges[1:-1], right=False), 0, n_bins - 1)

    rows = []
    seconds_per_bin = bin_size_ms / 1000.0
    for rate_bin in range(n_bins):
        mask = assignments == rate_bin
        bin_counts = flat_counts[mask]
        spike_count = float(np.sum(bin_counts)) if bin_counts.size else 0.0
        n_observations = int(np.sum(mask))
        observed_rate = (
            float(spike_count / (n_observations * seconds_per_bin))
            if n_observations > 0
            else float("nan")
        )
        rows.append(
            {
                "rate_bin": rate_bin,
                "n_observations": n_observations,
                "mean_predicted_rate_hz": _safe_mean(flat_predicted[mask]),
                "mean_reference_rate_hz": _safe_mean(flat_reference[mask]),
                "observed_rate_hz": observed_rate,
                "mean_count": _safe_mean(bin_counts),
                "spike_count": spike_count,
            }
        )
    return pd.DataFrame(rows, columns=CALIBRATION_COLUMNS)


def summarize_rate_distribution(
    predicted_rates_hz: np.ndarray,
    reference_rates_hz: np.ndarray,
    counts: np.ndarray,
) -> dict[str, float]:
    """Summarize broad rate-scale and clipping-like symptoms."""
    counts_array, predicted, reference = _aligned_arrays(
        counts, predicted_rates_hz, reference_rates_hz
    )
    return {
        "spike_count": float(np.sum(counts_array)),
        "mean_predicted_rate_hz": float(np.mean(predicted)),
        "median_predicted_rate_hz": float(np.median(predicted)),
        "min_predicted_rate_hz": float(np.min(predicted)),
        "max_predicted_rate_hz": float(np.max(predicted)),
        "mean_reference_rate_hz": float(np.mean(reference)),
        "median_reference_rate_hz": float(np.median(reference)),
        "min_reference_rate_hz": float(np.min(reference)),
        "max_reference_rate_hz": float(np.max(reference)),
        "mean_count": float(np.mean(counts_array)),
        "zero_fraction": float(np.mean(counts_array == 0.0)),
    }


def compute_prediction_reference_correlation(
    predicted_rates_hz: np.ndarray,
    reference_rates_hz: np.ndarray,
) -> float:
    """Return Pearson correlation between predicted and reference rates, or NaN if undefined."""
    predicted = np.asarray(predicted_rates_hz, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference_rates_hz, dtype=np.float64).reshape(-1)
    if predicted.shape != reference.shape:
        msg = (
            "predicted_rates_hz and reference_rates_hz must have the same shape; "
            f"got {predicted.shape} and {reference.shape}"
        )
        raise ValueError(msg)
    if predicted.size < 2 or np.std(predicted) == 0.0 or np.std(reference) == 0.0:
        return float("nan")
    return float(np.corrcoef(predicted, reference)[0, 1])
