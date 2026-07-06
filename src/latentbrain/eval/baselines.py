from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from latentbrain.data.schemas import NeuralDataset, NeuronMask, TrialSplit
from latentbrain.eval.metrics import safe_clip_rates, summarize_poisson_metrics


def _trial_mask(dataset: NeuralDataset, trial_ids: np.ndarray) -> np.ndarray:
    return np.isin(dataset.trial_ids, trial_ids)


def _group_mask(neuron_mask: NeuronMask, group: str) -> np.ndarray:
    if group == "heldin":
        return np.asarray(neuron_mask.heldin, dtype=bool)
    if group == "heldout":
        return np.asarray(neuron_mask.heldout, dtype=bool)
    if group == "all":
        return np.asarray(neuron_mask.heldin | neuron_mask.heldout, dtype=bool)
    msg = f"unknown neuron group: {group}"
    raise ValueError(msg)


def _split_ids(split: TrialSplit, name: str) -> np.ndarray:
    if name == "train":
        return split.train
    if name == "validation":
        return split.validation
    if name == "test":
        return split.test
    msg = f"unknown split: {name}"
    raise ValueError(msg)


def fit_mean_rate_baseline(
    train_spikes: np.ndarray,
    bin_size_ms: int,
    min_rate_hz: float,
    max_rate_hz: float,
) -> np.ndarray:
    """Fit one constant firing rate per neuron from training spikes only."""
    if train_spikes.ndim != 3:
        msg = f"train_spikes must have rank 3; got shape {train_spikes.shape}"
        raise ValueError(msg)
    if train_spikes.shape[0] == 0 or train_spikes.shape[1] == 0 or train_spikes.shape[2] == 0:
        msg = "train_spikes dimensions must be positive"
        raise ValueError(msg)
    if bin_size_ms <= 0:
        msg = "bin_size_ms must be positive"
        raise ValueError(msg)
    if not np.all(np.isfinite(train_spikes)):
        msg = "train_spikes must be finite"
        raise ValueError(msg)
    if np.any(train_spikes < 0):
        msg = "train_spikes must be non-negative"
        raise ValueError(msg)
    if not np.all(np.equal(train_spikes, np.floor(train_spikes))):
        msg = "train_spikes must be integer-valued"
        raise ValueError(msg)
    bin_seconds = bin_size_ms / 1000.0
    denominator = train_spikes.shape[0] * train_spikes.shape[1] * bin_seconds
    rates = train_spikes.sum(axis=(0, 1)) / denominator
    return safe_clip_rates(rates, min_rate_hz, max_rate_hz)


def predict_mean_rate(fitted_rates_hz: np.ndarray, n_trials: int, n_time_bins: int) -> np.ndarray:
    """Broadcast fitted per-neuron rates to a trial/time/neuron tensor."""
    if fitted_rates_hz.ndim != 1:
        msg = "fitted_rates_hz must be rank 1"
        raise ValueError(msg)
    if n_trials <= 0 or n_time_bins <= 0:
        msg = "n_trials and n_time_bins must be positive"
        raise ValueError(msg)
    return np.broadcast_to(fitted_rates_hz, (n_trials, n_time_bins, fitted_rates_hz.size)).copy()


def _reference_rate_for_group(
    train_spikes: np.ndarray,
    group_mask: np.ndarray,
    bin_size_ms: int,
    min_rate_hz: float,
    max_rate_hz: float,
) -> float:
    group_train = train_spikes[:, :, group_mask]
    bin_seconds = bin_size_ms / 1000.0
    rate = group_train.sum() / (group_train.size * bin_seconds)
    return float(safe_clip_rates(np.array([rate]), min_rate_hz, max_rate_hz)[0])


def _neuron_metrics(
    dataset: NeuralDataset,
    split: TrialSplit,
    neuron_mask: NeuronMask,
    fitted_rates_hz: np.ndarray,
) -> pd.DataFrame:
    validation_mask = _trial_mask(dataset, split.validation)
    test_mask = _trial_mask(dataset, split.test)
    rows: list[dict[str, Any]] = []
    for group in ("heldin", "heldout", "all"):
        group_values = _group_mask(neuron_mask, group)
        for neuron_index in np.flatnonzero(group_values):
            rows.append(
                {
                    "neuron_index": int(neuron_index),
                    "neuron_group": group,
                    "train_mean_rate_hz": float(fitted_rates_hz[neuron_index]),
                    "total_spikes_all_trials": int(dataset.spikes[:, :, neuron_index].sum()),
                    "validation_spikes": int(
                        dataset.spikes[validation_mask, :, neuron_index].sum()
                    ),
                    "test_spikes": int(dataset.spikes[test_mask, :, neuron_index].sum()),
                }
            )
    return pd.DataFrame(rows)


def evaluate_mean_rate_baseline(
    dataset: NeuralDataset,
    split: TrialSplit,
    neuron_mask: NeuronMask,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit and evaluate a train-only mean-rate baseline across splits and neuron groups."""
    baseline_config = config["baseline"]
    evaluation_config = config["evaluation"]
    min_rate_hz = float(baseline_config["min_rate_hz"])
    max_rate_hz = float(baseline_config["max_rate_hz"])
    train_mask = _trial_mask(dataset, split.train)
    train_spikes = dataset.spikes[train_mask]
    fitted_rates_hz = fit_mean_rate_baseline(
        train_spikes,
        dataset.bin_size_ms,
        min_rate_hz,
        max_rate_hz,
    )
    rows: list[dict[str, Any]] = []
    for split_name in evaluation_config["evaluate_splits"]:
        split_trial_ids = _split_ids(split, split_name)
        split_mask = _trial_mask(dataset, split_trial_ids)
        split_spikes = dataset.spikes[split_mask]
        for group in evaluation_config["evaluate_neuron_groups"]:
            group_values = _group_mask(neuron_mask, group)
            counts = split_spikes[:, :, group_values]
            predicted_rates = predict_mean_rate(
                fitted_rates_hz[group_values],
                counts.shape[0],
                counts.shape[1],
            )
            reference_rate = _reference_rate_for_group(
                train_spikes,
                group_values,
                dataset.bin_size_ms,
                min_rate_hz,
                max_rate_hz,
            )
            reference_rates = np.full_like(predicted_rates, reference_rate, dtype=np.float64)
            metrics = summarize_poisson_metrics(
                counts,
                predicted_rates,
                reference_rates,
                dataset.bin_size_ms,
            )
            rows.append(
                {
                    "split": split_name,
                    "neuron_group": group,
                    "n_trials": int(counts.shape[0]),
                    "n_neurons": int(counts.shape[2]),
                    "n_time_bins": int(counts.shape[1]),
                    **metrics,
                }
            )
    metadata = {
        "baseline_name": str(baseline_config.get("name", "mean_rate")),
        "fit_trials": [int(value) for value in split.train.tolist()],
        "bin_size_ms": int(dataset.bin_size_ms),
        "min_rate_hz": min_rate_hz,
        "max_rate_hz": max_rate_hz,
        "train_only_fit": bool(baseline_config.get("use_train_trials_only", True)),
        "fitted_rates_hz": [float(value) for value in fitted_rates_hz.tolist()],
    }
    neuron_metrics = _neuron_metrics(dataset, split, neuron_mask, fitted_rates_hz)
    return pd.DataFrame(rows), neuron_metrics, metadata
