from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from latentbrain.eval.metrics import poisson_log_likelihood as _poisson_log_likelihood
from latentbrain.eval.metrics import safe_clip_rates

REQUIRED_HELDOUT_SCORE_COLUMNS = [
    "method_name",
    "split",
    "prediction_source",
    "reference_name",
    "valid_model",
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
    "notes",
]


@dataclass(frozen=True)
class ScoringConfig:
    bin_size_ms: int
    include_poisson_constant: bool
    min_rate_hz: float
    max_rate_hz: float
    reference_name: str = "train_heldout_mean_rate"

    def __post_init__(self) -> None:
        if self.bin_size_ms <= 0:
            msg = "bin_size_ms must be positive"
            raise ValueError(msg)
        if self.min_rate_hz <= 0.0:
            msg = "min_rate_hz must be positive"
            raise ValueError(msg)
        if self.max_rate_hz <= self.min_rate_hz:
            msg = "max_rate_hz must exceed min_rate_hz"
            raise ValueError(msg)
        if not self.reference_name:
            msg = "reference_name must be non-empty"
            raise ValueError(msg)


def _as_counts(counts: np.ndarray) -> np.ndarray:
    counts_array = np.asarray(counts, dtype=np.float64)
    if counts_array.ndim != 3:
        msg = "counts must have shape [trials, time, neurons]"
        raise ValueError(msg)
    return counts_array


def _clip_rates(rates_hz: np.ndarray, config: ScoringConfig) -> np.ndarray:
    rates = np.asarray(rates_hz, dtype=np.float64)
    return safe_clip_rates(rates, config.min_rate_hz, config.max_rate_hz)


def poisson_log_likelihood(
    counts: np.ndarray,
    rates_hz: np.ndarray,
    config: ScoringConfig,
) -> float:
    counts_array = _as_counts(counts)
    rates = _clip_rates(rates_hz, config)
    if counts_array.shape != rates.shape:
        msg = "counts and rates_hz must have the same shape"
        raise ValueError(msg)
    return _poisson_log_likelihood(
        counts_array,
        rates,
        config.bin_size_ms,
        include_constant=config.include_poisson_constant,
    )


def canonical_bits_per_spike(
    model_log_likelihood: float,
    reference_log_likelihood: float,
    spike_count: float,
) -> float:
    if spike_count <= 0.0:
        return float("nan")
    return float((model_log_likelihood - reference_log_likelihood) / (math.log(2.0) * spike_count))


def train_heldout_mean_rate_reference(
    train_heldout_counts: np.ndarray,
    target_shape: tuple[int, int, int],
    config: ScoringConfig,
) -> np.ndarray:
    counts = _as_counts(train_heldout_counts)
    if len(target_shape) != 3:
        msg = "target_shape must be [trials, time, neurons]"
        raise ValueError(msg)
    if counts.shape[2] != target_shape[2]:
        msg = "train_heldout_counts neuron dimension must match target_shape"
        raise ValueError(msg)
    seconds = counts.shape[0] * counts.shape[1] * (config.bin_size_ms / 1000.0)
    if seconds <= 0.0:
        msg = "train_heldout_counts must have non-empty trial and time axes"
        raise ValueError(msg)
    rates = _clip_rates(counts.sum(axis=(0, 1)) / seconds, config)
    return np.broadcast_to(rates.reshape(1, 1, -1), target_shape).copy()


def score_heldout_prediction(
    counts: np.ndarray,
    predicted_rates_hz: np.ndarray,
    reference_rates_hz: np.ndarray,
    config: ScoringConfig,
    method_name: str,
    split: str,
    prediction_source: str,
    valid_model: bool,
    notes: str = "",
) -> dict[str, Any]:
    counts_array = _as_counts(counts)
    predicted = _clip_rates(predicted_rates_hz, config)
    reference = _clip_rates(reference_rates_hz, config)
    if counts_array.shape != predicted.shape or counts_array.shape != reference.shape:
        msg = "counts, predicted_rates_hz, and reference_rates_hz must have the same shape"
        raise ValueError(msg)
    model_ll = poisson_log_likelihood(counts_array, predicted, config)
    reference_ll = poisson_log_likelihood(counts_array, reference, config)
    spike_count = float(counts_array.sum())
    observed_rate = spike_count / (counts_array.size * (config.bin_size_ms / 1000.0))
    delta = model_ll - reference_ll
    return {
        "method_name": method_name,
        "split": split,
        "prediction_source": prediction_source,
        "reference_name": config.reference_name,
        "valid_model": bool(valid_model),
        "spike_count": spike_count,
        "total_observations": int(counts_array.size),
        "zero_fraction": float(np.mean(counts_array == 0.0)),
        "model_log_likelihood": model_ll,
        "reference_log_likelihood": reference_ll,
        "log_likelihood_delta": delta,
        "bits_per_spike": canonical_bits_per_spike(model_ll, reference_ll, spike_count),
        "poisson_nll": -model_ll,
        "mean_predicted_rate_hz": float(np.mean(predicted)),
        "mean_reference_rate_hz": float(np.mean(reference)),
        "observed_rate_hz": float(observed_rate),
        "notes": notes,
    }
