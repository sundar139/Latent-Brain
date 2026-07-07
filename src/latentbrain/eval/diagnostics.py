from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from latentbrain.eval.calibration import compute_prediction_reference_correlation
from latentbrain.eval.metrics import poisson_log_likelihood

FACTOR_USAGE_COLUMNS = ["split", "factor_index", "mean", "std", "variance", "min", "max", "active"]
NEURON_DIAGNOSTIC_COLUMNS = [
    "split",
    "neuron_index",
    "spike_count",
    "zero_fraction",
    "bits_per_spike",
    "model_poisson_nll",
    "reference_poisson_nll",
    "mean_predicted_rate_hz",
    "mean_reference_rate_hz",
    "observed_rate_hz",
    "rate_correlation",
]


def _validate_aligned_3d(
    counts: np.ndarray, predicted_rates_hz: np.ndarray, reference_rates_hz: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts_array = np.asarray(counts, dtype=np.float64)
    predicted = np.asarray(predicted_rates_hz, dtype=np.float64)
    reference = np.asarray(reference_rates_hz, dtype=np.float64)
    if counts_array.ndim != 3:
        msg = f"counts must have rank 3; got shape {counts_array.shape}"
        raise ValueError(msg)
    if counts_array.shape != predicted.shape or counts_array.shape != reference.shape:
        msg = (
            "counts, predicted_rates_hz, and reference_rates_hz must have matching shapes; "
            f"got {counts_array.shape}, {predicted.shape}, and {reference.shape}"
        )
        raise ValueError(msg)
    if not (
        np.all(np.isfinite(counts_array))
        and np.all(np.isfinite(predicted))
        and np.all(np.isfinite(reference))
    ):
        msg = "counts and rates must be finite"
        raise ValueError(msg)
    return counts_array, predicted, reference


def _observed_rate_hz(counts: np.ndarray, bin_size_ms: int) -> float:
    seconds = counts.size * (bin_size_ms / 1000.0)
    return float(np.sum(counts) / seconds) if seconds > 0.0 else float("nan")


def _bits(model_ll: float, reference_ll: float, spike_count: float) -> float:
    if spike_count <= 0.0:
        return float("nan")
    return float((model_ll - reference_ll) / (np.log(2.0) * spike_count))


def compute_loss_scale_diagnostics(
    counts: np.ndarray,
    predicted_rates_hz: np.ndarray,
    reference_rates_hz: np.ndarray,
    bin_size_ms: int,
) -> dict[str, float]:
    """Compute likelihood scale diagnostics for one split/source."""
    if bin_size_ms <= 0:
        msg = "bin_size_ms must be positive"
        raise ValueError(msg)
    counts_array, predicted, reference = _validate_aligned_3d(
        counts, predicted_rates_hz, reference_rates_hz
    )
    model_ll = poisson_log_likelihood(counts_array, predicted, bin_size_ms)
    reference_ll = poisson_log_likelihood(counts_array, reference, bin_size_ms)
    spike_count = float(np.sum(counts_array))
    model_nll = -model_ll
    return {
        "spike_count": spike_count,
        "total_observations": float(counts_array.size),
        "zero_fraction": float(np.mean(counts_array == 0.0)),
        "model_poisson_nll": model_nll,
        "reference_poisson_nll": -reference_ll,
        "model_log_likelihood": model_ll,
        "reference_log_likelihood": reference_ll,
        "bits_per_spike": _bits(model_ll, reference_ll, spike_count),
        "nll_per_observation": float(model_nll / counts_array.size),
        "nll_per_spike": float(model_nll / spike_count) if spike_count > 0.0 else float("nan"),
        "mean_predicted_rate_hz": float(np.mean(predicted)),
        "mean_reference_rate_hz": float(np.mean(reference)),
        "observed_rate_hz": _observed_rate_hz(counts_array, bin_size_ms),
    }


def compute_factor_usage(
    factors: np.ndarray,
    split_name: str,
    active_variance_threshold: float = 1.0e-6,
) -> pd.DataFrame:
    """Summarize factor activity by latent factor dimension."""
    array = np.asarray(factors, dtype=np.float64)
    if array.ndim != 3:
        msg = f"factors must have rank 3; got shape {array.shape}"
        raise ValueError(msg)
    rows = []
    for factor_index in range(array.shape[2]):
        values = array[:, :, factor_index].reshape(-1)
        variance = float(np.var(values))
        rows.append(
            {
                "split": split_name,
                "factor_index": factor_index,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "variance": variance,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "active": bool(variance > active_variance_threshold),
            }
        )
    return pd.DataFrame(rows, columns=FACTOR_USAGE_COLUMNS)


def compute_neuron_prediction_diagnostics(
    counts: np.ndarray,
    predicted_rates_hz: np.ndarray,
    reference_rates_hz: np.ndarray,
    bin_size_ms: int,
    neuron_indices: np.ndarray,
    split_name: str,
) -> pd.DataFrame:
    """Compute held-out prediction diagnostics per target neuron."""
    counts_array, predicted, reference = _validate_aligned_3d(
        counts, predicted_rates_hz, reference_rates_hz
    )
    indices = np.asarray(neuron_indices, dtype=np.int64)
    if indices.shape[0] != counts_array.shape[2]:
        msg = "neuron_indices length must match the neuron dimension"
        raise ValueError(msg)
    rows = []
    for rank, neuron_index in enumerate(indices):
        c = counts_array[:, :, rank : rank + 1]
        p = predicted[:, :, rank : rank + 1]
        r = reference[:, :, rank : rank + 1]
        model_ll = poisson_log_likelihood(c, p, bin_size_ms)
        reference_ll = poisson_log_likelihood(c, r, bin_size_ms)
        spike_count = float(np.sum(c))
        rows.append(
            {
                "split": split_name,
                "neuron_index": int(neuron_index),
                "spike_count": spike_count,
                "zero_fraction": float(np.mean(c == 0.0)),
                "bits_per_spike": _bits(model_ll, reference_ll, spike_count),
                "model_poisson_nll": -model_ll,
                "reference_poisson_nll": -reference_ll,
                "mean_predicted_rate_hz": float(np.mean(p)),
                "mean_reference_rate_hz": float(np.mean(r)),
                "observed_rate_hz": _observed_rate_hz(c, bin_size_ms),
                "rate_correlation": compute_prediction_reference_correlation(p, r),
            }
        )
    return pd.DataFrame(rows, columns=NEURON_DIAGNOSTIC_COLUMNS)
