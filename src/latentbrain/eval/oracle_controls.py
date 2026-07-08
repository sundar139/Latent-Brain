from __future__ import annotations

import numpy as np

from latentbrain.eval.metric_audit import compute_train_heldout_mean_rates
from latentbrain.eval.metrics import safe_clip_rates
from latentbrain.eval.smoothing import smooth_spike_counts, spike_counts_to_rates_hz


def make_train_mean_rate_prediction(
    train_counts: np.ndarray,
    target_shape: tuple[int, int, int],
    bin_size_ms: int,
    min_rate_hz: float,
    max_rate_hz: float,
) -> np.ndarray:
    rates = compute_train_heldout_mean_rates(train_counts, bin_size_ms, min_rate_hz, max_rate_hz)
    if rates.shape[0] != target_shape[2]:
        msg = "train_counts neuron dimension must match target_shape"
        raise ValueError(msg)
    return np.broadcast_to(rates, target_shape).copy()


def make_oracle_smoothed_heldout_prediction(
    heldout_counts: np.ndarray,
    bin_size_ms: int,
    smoothing_sigma_ms: float,
    min_rate_hz: float,
    max_rate_hz: float,
) -> np.ndarray:
    smoothed = smooth_spike_counts(heldout_counts, bin_size_ms, sigma_ms=smoothing_sigma_ms)
    return safe_clip_rates(
        spike_counts_to_rates_hz(smoothed, bin_size_ms), min_rate_hz, max_rate_hz
    )


def make_trial_shuffled_heldin_prediction(
    train_heldin_counts: np.ndarray,
    train_heldout_counts: np.ndarray,
    eval_heldin_counts: np.ndarray,
    target_shape: tuple[int, int, int],
    seed: int,
    min_rate_hz: float,
    max_rate_hz: float,
) -> np.ndarray:
    # ponytail: negative control; ignores held-in values except for shape sanity.
    if train_heldin_counts.shape[0] != train_heldout_counts.shape[0]:
        msg = "train held-in and held-out trial counts must match"
        raise ValueError(msg)
    if eval_heldin_counts.shape[:2] != target_shape[:2]:
        msg = "eval_heldin_counts trial/time shape must match target_shape"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    trial_indices = rng.integers(0, train_heldout_counts.shape[0], size=target_shape[0])
    sampled = np.asarray(train_heldout_counts[trial_indices], dtype=np.float64)
    if sampled.shape[1] != target_shape[1]:
        sampled = np.resize(sampled, target_shape)
    return safe_clip_rates(sampled / 0.02, min_rate_hz, max_rate_hz)


def make_random_rate_prediction(
    target_shape: tuple[int, int, int],
    train_mean_rates_hz: np.ndarray,
    seed: int,
    min_rate_hz: float,
    max_rate_hz: float,
) -> np.ndarray:
    rates = np.asarray(train_mean_rates_hz, dtype=np.float64)
    if rates.shape[0] != target_shape[2]:
        msg = "train_mean_rates_hz length must match target_shape neuron dimension"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    noise = rng.lognormal(mean=0.0, sigma=0.5, size=target_shape)
    return safe_clip_rates(rates.reshape(1, 1, -1) * noise, min_rate_hz, max_rate_hz)
