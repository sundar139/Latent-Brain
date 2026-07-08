from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from latentbrain.eval.metrics import poisson_log_likelihood, safe_clip_rates


def _validate_pair(
    counts: np.ndarray, predicted_rates_hz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    counts_array = np.asarray(counts, dtype=np.float64)
    rates = np.asarray(predicted_rates_hz, dtype=np.float64)
    if counts_array.shape != rates.shape:
        msg = (
            "counts and predicted rates must have matching shapes, "
            f"got {counts_array.shape} and {rates.shape}"
        )
        raise ValueError(msg)
    if counts_array.ndim != 3:
        msg = (
            "counts and predicted rates must have shape [trials, time, neurons], "
            f"got {counts_array.shape}"
        )
        raise ValueError(msg)
    if np.any(counts_array < 0.0) or not np.all(np.isfinite(counts_array)):
        msg = "counts must be finite and non-negative"
        raise ValueError(msg)
    if np.any(rates < 0.0) or not np.all(np.isfinite(rates)):
        msg = "predicted rates must be finite and non-negative"
        raise ValueError(msg)
    return counts_array, rates


def _train_mean_rates(train_counts: np.ndarray, bin_size_ms: int) -> np.ndarray:
    seconds = train_counts.shape[0] * train_counts.shape[1] * (bin_size_ms / 1000.0)
    if seconds <= 0.0:
        msg = "training counts must contain at least one time bin"
        raise ValueError(msg)
    return np.asarray(train_counts.sum(axis=(0, 1)) / seconds, dtype=np.float64)


def fit_multiplicative_rate_scale(
    train_counts: np.ndarray,
    train_predicted_rates_hz: np.ndarray,
    bin_size_ms: int,
    min_scale: float = 1.0e-3,
    max_scale: float = 1.0e3,
) -> np.ndarray:
    """Fit per-neuron train-only MLE scale for predicted firing rates."""
    counts, rates = _validate_pair(train_counts, train_predicted_rates_hz)
    if bin_size_ms <= 0:
        msg = "bin_size_ms must be positive"
        raise ValueError(msg)
    if min_scale <= 0.0 or max_scale <= min_scale:
        msg = "scale bounds must be positive and increasing"
        raise ValueError(msg)
    expected_counts = rates.sum(axis=(0, 1)) * (bin_size_ms / 1000.0)
    observed_counts = counts.sum(axis=(0, 1))
    raw = np.divide(
        observed_counts,
        expected_counts,
        out=np.full_like(observed_counts, min_scale, dtype=np.float64),
        where=expected_counts > 0.0,
    )
    return np.asarray(np.clip(raw, min_scale, max_scale), dtype=np.float64)


def apply_multiplicative_rate_scale(
    predicted_rates_hz: np.ndarray,
    scale: np.ndarray,
    min_rate_hz: float,
    max_rate_hz: float,
) -> np.ndarray:
    rates = np.asarray(predicted_rates_hz, dtype=np.float64)
    scaled = rates * np.asarray(scale, dtype=np.float64).reshape((1, 1, -1))
    return safe_clip_rates(scaled, min_rate_hz, max_rate_hz)


def fit_log_rate_bias(
    train_counts: np.ndarray,
    train_predicted_rates_hz: np.ndarray,
    bin_size_ms: int,
) -> np.ndarray:
    """Fit per-neuron log-rate bias; equivalent to log MLE scale."""
    scale = fit_multiplicative_rate_scale(train_counts, train_predicted_rates_hz, bin_size_ms)
    return np.asarray(np.log(scale), dtype=np.float64)


def apply_log_rate_bias(
    predicted_rates_hz: np.ndarray,
    bias: np.ndarray,
    min_rate_hz: float,
    max_rate_hz: float,
) -> np.ndarray:
    rates = np.asarray(predicted_rates_hz, dtype=np.float64)
    adjusted = rates * np.exp(np.asarray(bias, dtype=np.float64).reshape((1, 1, -1)))
    return safe_clip_rates(adjusted, min_rate_hz, max_rate_hz)


def blend_with_mean_rate(
    predicted_rates_hz: np.ndarray,
    train_mean_rates_hz: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if alpha < 0.0 or alpha > 1.0:
        msg = "alpha must be between 0 and 1"
        raise ValueError(msg)
    rates = np.asarray(predicted_rates_hz, dtype=np.float64)
    mean = np.asarray(train_mean_rates_hz, dtype=np.float64).reshape((1, 1, -1))
    return np.asarray(alpha * rates + (1.0 - alpha) * mean, dtype=np.float64)


def choose_best_blend_alpha(
    train_counts: np.ndarray,
    train_predicted_rates_hz: np.ndarray,
    train_mean_rates_hz: np.ndarray,
    alpha_grid: Sequence[float],
    bin_size_ms: int,
) -> tuple[float, pd.DataFrame]:
    """Select blend alpha on train trials only by Poisson bits/spike."""
    counts, predicted = _validate_pair(train_counts, train_predicted_rates_hz)
    if not alpha_grid:
        msg = "alpha_grid must not be empty"
        raise ValueError(msg)
    reference = np.broadcast_to(np.asarray(train_mean_rates_hz, dtype=np.float64), counts.shape)
    reference_ll = poisson_log_likelihood(counts, reference, bin_size_ms)
    spike_count = float(counts.sum())
    rows = []
    best_alpha = float(alpha_grid[0])
    best_bits = -np.inf
    for alpha_value in alpha_grid:
        alpha = float(alpha_value)
        blended = blend_with_mean_rate(predicted, train_mean_rates_hz, alpha)
        model_ll = poisson_log_likelihood(counts, blended, bin_size_ms)
        bits = (
            float((model_ll - reference_ll) / (np.log(2.0) * spike_count))
            if spike_count > 0.0
            else float("nan")
        )
        rows.append(
            {
                "alpha": alpha,
                "split": "train",
                "bin_size_ms": int(bin_size_ms),
                "spike_count": spike_count,
                "poisson_nll": -model_ll,
                "bits_per_spike": bits,
                "mean_predicted_rate_hz": float(np.mean(blended)),
                "notes": "fit on train trials only",
            }
        )
        if np.isfinite(bits) and bits > best_bits:
            best_alpha = alpha
            best_bits = bits
    return best_alpha, pd.DataFrame(rows)


def mean_rates_from_counts(
    counts: np.ndarray,
    bin_size_ms: int,
    min_rate_hz: float,
    max_rate_hz: float,
) -> np.ndarray:
    return safe_clip_rates(
        _train_mean_rates(np.asarray(counts, dtype=np.float64), bin_size_ms),
        min_rate_hz,
        max_rate_hz,
    )
