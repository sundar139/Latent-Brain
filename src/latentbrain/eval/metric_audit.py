from __future__ import annotations

import math
from typing import Any

import numpy as np

from latentbrain.eval.metrics import poisson_log_likelihood, safe_clip_rates

REQUIRED_SCORE_COLUMNS = [
    "method_name",
    "split",
    "prediction_source",
    "spike_count",
    "total_observations",
    "zero_fraction",
    "model_log_likelihood",
    "reference_log_likelihood",
    "log_likelihood_delta",
    "bits_per_spike",
    "poisson_nll",
    "mean_predicted_rate_hz",
    "mean_reference_rate_hz",
    "observed_rate_hz",
    "reference_name",
]


def poisson_log_likelihood_from_rates(
    counts: np.ndarray,
    rates_hz: np.ndarray,
    bin_size_ms: int,
    include_constant: bool = True,
) -> float:
    return poisson_log_likelihood(counts, rates_hz, bin_size_ms, include_constant)


def bits_per_spike_from_log_likelihoods(
    model_log_likelihood: float,
    reference_log_likelihood: float,
    spike_count: float,
) -> float:
    if spike_count <= 0.0:
        return float("nan")
    return float((model_log_likelihood - reference_log_likelihood) / (math.log(2.0) * spike_count))


def compute_train_heldout_mean_rates(
    train_heldout_counts: np.ndarray,
    bin_size_ms: int,
    min_rate_hz: float,
    max_rate_hz: float,
) -> np.ndarray:
    counts = np.asarray(train_heldout_counts, dtype=np.float64)
    if counts.ndim != 3:
        msg = "train_heldout_counts must have shape [trials, time, neurons]"
        raise ValueError(msg)
    seconds = counts.shape[0] * counts.shape[1] * (bin_size_ms / 1000.0)
    if seconds <= 0.0:
        msg = "train_heldout_counts must have non-empty trial and time axes"
        raise ValueError(msg)
    return safe_clip_rates(counts.sum(axis=(0, 1)) / seconds, min_rate_hz, max_rate_hz)


def score_prediction_against_reference(
    counts: np.ndarray,
    predicted_rates_hz: np.ndarray,
    reference_rates_hz: np.ndarray,
    bin_size_ms: int,
    method_name: str,
    split: str,
    prediction_source: str,
) -> dict[str, Any]:
    counts_array = np.asarray(counts, dtype=np.float64)
    predicted = np.asarray(predicted_rates_hz, dtype=np.float64)
    reference = np.asarray(reference_rates_hz, dtype=np.float64)
    if counts_array.shape != predicted.shape or counts_array.shape != reference.shape:
        msg = "counts, predicted_rates_hz, and reference_rates_hz must have the same shape"
        raise ValueError(msg)
    model_ll = poisson_log_likelihood_from_rates(counts_array, predicted, bin_size_ms)
    reference_ll = poisson_log_likelihood_from_rates(counts_array, reference, bin_size_ms)
    spike_count = float(counts_array.sum())
    delta = model_ll - reference_ll
    observed_rate = spike_count / (counts_array.size * (bin_size_ms / 1000.0))
    return {
        "method_name": method_name,
        "split": split,
        "prediction_source": prediction_source,
        "spike_count": spike_count,
        "total_observations": int(counts_array.size),
        "zero_fraction": float(np.mean(counts_array == 0.0)),
        "model_log_likelihood": model_ll,
        "reference_log_likelihood": reference_ll,
        "log_likelihood_delta": delta,
        "bits_per_spike": bits_per_spike_from_log_likelihoods(model_ll, reference_ll, spike_count),
        "poisson_nll": -model_ll,
        "mean_predicted_rate_hz": float(np.mean(predicted)),
        "mean_reference_rate_hz": float(np.mean(reference)),
        "observed_rate_hz": float(observed_rate),
        "reference_name": "train_heldout_mean_rate",
    }
