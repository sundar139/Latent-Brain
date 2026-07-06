from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln  # type: ignore[import-untyped]


def _as_float_array(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    return array


def _validate_counts(counts: np.ndarray) -> np.ndarray:
    array = _as_float_array("counts", counts)
    if np.any(array < 0):
        msg = "counts must be non-negative"
        raise ValueError(msg)
    if not np.all(np.equal(array, np.floor(array))):
        msg = "counts must be integer-valued"
        raise ValueError(msg)
    return array


def _validate_rates(rates_hz: np.ndarray) -> np.ndarray:
    array = _as_float_array("rates_hz", rates_hz)
    if np.any(array <= 0):
        msg = "rates_hz must be positive before Poisson metric evaluation"
        raise ValueError(msg)
    return array


def _broadcast_counts_and_rates(
    counts: np.ndarray,
    rates_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    counts_array = _validate_counts(counts)
    rates_array = _validate_rates(rates_hz)
    try:
        broadcast_counts, broadcast_rates = np.broadcast_arrays(counts_array, rates_array)
        return broadcast_counts, broadcast_rates
    except ValueError as exc:
        msg = (
            "counts and rates_hz must be broadcast-compatible; "
            f"got {counts.shape} and {rates_hz.shape}"
        )
        raise ValueError(msg) from exc


def safe_clip_rates(rates_hz: np.ndarray, min_rate_hz: float, max_rate_hz: float) -> np.ndarray:
    """Clip finite firing rates to a positive configured interval."""
    if min_rate_hz <= 0:
        msg = "min_rate_hz must be positive"
        raise ValueError(msg)
    if max_rate_hz <= min_rate_hz:
        msg = "max_rate_hz must exceed min_rate_hz"
        raise ValueError(msg)
    rates = _as_float_array("rates_hz", rates_hz)
    return np.asarray(np.clip(rates, min_rate_hz, max_rate_hz), dtype=np.float64)


def poisson_log_likelihood(
    counts: np.ndarray,
    rates_hz: np.ndarray,
    bin_size_ms: int,
    include_constant: bool = True,
) -> float:
    """Compute summed Poisson log-likelihood for spike counts and rates in Hz."""
    if bin_size_ms <= 0:
        msg = "bin_size_ms must be positive"
        raise ValueError(msg)
    counts_array, rates_array = _broadcast_counts_and_rates(counts, rates_hz)
    expected_counts = rates_array * (bin_size_ms / 1000.0)
    log_likelihood = counts_array * np.log(expected_counts) - expected_counts
    if include_constant:
        log_likelihood = log_likelihood - gammaln(counts_array + 1.0)
    return float(np.sum(log_likelihood))


def poisson_nll(
    counts: np.ndarray,
    rates_hz: np.ndarray,
    bin_size_ms: int,
    include_constant: bool = True,
) -> float:
    """Compute summed Poisson negative log-likelihood."""
    return -poisson_log_likelihood(counts, rates_hz, bin_size_ms, include_constant)


def bits_per_spike(
    model_log_likelihood: float,
    reference_log_likelihood: float,
    spike_count: float,
) -> float:
    """Compute bits/spike improvement over a reference log-likelihood."""
    if not np.isfinite(model_log_likelihood) or not np.isfinite(reference_log_likelihood):
        msg = "log-likelihood values must be finite"
        raise ValueError(msg)
    if spike_count <= 0:
        msg = "spike_count must be positive for bits_per_spike"
        raise ValueError(msg)
    return float((model_log_likelihood - reference_log_likelihood) / (math.log(2.0) * spike_count))


def summarize_poisson_metrics(
    counts: np.ndarray,
    predicted_rates_hz: np.ndarray,
    reference_rates_hz: np.ndarray,
    bin_size_ms: int,
) -> dict[str, float]:
    """Summarize Poisson likelihood and bits/spike for one evaluation slice."""
    counts_array = _validate_counts(counts)
    predicted = _validate_rates(predicted_rates_hz)
    reference = _validate_rates(reference_rates_hz)
    model_ll = poisson_log_likelihood(counts_array, predicted, bin_size_ms)
    reference_ll = poisson_log_likelihood(counts_array, reference, bin_size_ms)
    spike_count = float(np.sum(counts_array))
    return {
        "spike_count": spike_count,
        "poisson_nll": -model_ll,
        "poisson_log_likelihood": model_ll,
        "reference_log_likelihood": reference_ll,
        "bits_per_spike": bits_per_spike(model_ll, reference_ll, spike_count),
        "mean_predicted_rate_hz": float(np.mean(predicted)),
    }
