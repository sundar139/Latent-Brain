from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d  # type: ignore[import-untyped]


def _validate_spikes(spikes: np.ndarray) -> np.ndarray:
    array = np.asarray(spikes, dtype=np.float64)
    if array.ndim != 3:
        msg = f"spikes must have rank 3; got shape {array.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = "spikes must be finite"
        raise ValueError(msg)
    if np.any(array < 0):
        msg = "spikes must be non-negative"
        raise ValueError(msg)
    return array


def smooth_spike_counts(
    spikes: np.ndarray,
    bin_size_ms: int,
    method: str = "gaussian",
    sigma_ms: float = 50.0,
    truncate: float = 4.0,
) -> np.ndarray:
    """Smooth spike counts along time within each trial; boundary mode is reflect."""
    if bin_size_ms <= 0:
        msg = "bin_size_ms must be positive"
        raise ValueError(msg)
    if sigma_ms <= 0:
        msg = "sigma_ms must be positive"
        raise ValueError(msg)
    if method != "gaussian":
        msg = "only gaussian smoothing is supported"
        raise ValueError(msg)
    spikes_array = _validate_spikes(spikes)
    sigma_bins = sigma_ms / bin_size_ms
    smoothed = gaussian_filter1d(
        spikes_array,
        sigma=sigma_bins,
        axis=1,
        mode="reflect",
        truncate=truncate,
    )
    return np.asarray(smoothed, dtype=np.float64)


def spike_counts_to_rates_hz(smoothed_counts: np.ndarray, bin_size_ms: int) -> np.ndarray:
    """Convert spike counts per bin to rates in Hz."""
    if bin_size_ms <= 0:
        msg = "bin_size_ms must be positive"
        raise ValueError(msg)
    counts = np.asarray(smoothed_counts, dtype=np.float64)
    if not np.all(np.isfinite(counts)):
        msg = "smoothed_counts must be finite"
        raise ValueError(msg)
    return counts / (bin_size_ms / 1000.0)
