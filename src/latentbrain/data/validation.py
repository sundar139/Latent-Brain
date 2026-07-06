from __future__ import annotations

import numpy as np

from latentbrain.data.schemas import NeuralDataset, NeuronMask, TrialSplit


def _require_rank(name: str, array: np.ndarray, rank: int) -> None:
    if array.ndim != rank:
        msg = f"{name} must have rank {rank}; got shape {array.shape}"
        raise ValueError(msg)


def validate_neural_dataset(dataset: NeuralDataset) -> None:
    """Validate neural population arrays and metadata-independent invariants."""
    _require_rank("spikes", dataset.spikes, 3)
    if not np.all(np.isfinite(dataset.spikes)):
        msg = "spikes must be finite"
        raise ValueError(msg)
    if not np.issubdtype(dataset.spikes.dtype, np.integer):
        msg = "spikes must have an integer dtype"
        raise ValueError(msg)
    if np.any(dataset.spikes < 0):
        msg = "spikes must be non-negative"
        raise ValueError(msg)

    n_trials, n_time_bins, n_neurons = dataset.spikes.shape
    if n_trials == 0 or n_time_bins == 0 or n_neurons == 0:
        msg = "spikes dimensions must be positive"
        raise ValueError(msg)

    if dataset.rates is not None:
        _require_rank("rates", dataset.rates, 3)
        if dataset.rates.shape != dataset.spikes.shape:
            msg = "rates must match spikes shape"
            raise ValueError(msg)
        if not np.all(np.isfinite(dataset.rates)):
            msg = "rates must be finite"
            raise ValueError(msg)
        if np.any(dataset.rates <= 0):
            msg = "rates must be positive"
            raise ValueError(msg)

    if dataset.latents is not None:
        _require_rank("latents", dataset.latents, 3)
        if dataset.latents.shape[:2] != (n_trials, n_time_bins):
            msg = "latents must match spike trial and time dimensions"
            raise ValueError(msg)
        if dataset.latents.shape[2] == 0:
            msg = "latent dimension must be positive"
            raise ValueError(msg)
        if not np.all(np.isfinite(dataset.latents)):
            msg = "latents must be finite"
            raise ValueError(msg)

    if dataset.behavior is not None:
        _require_rank("behavior", dataset.behavior, 3)
        if dataset.behavior.shape[:2] != (n_trials, n_time_bins):
            msg = "behavior must match spike trial and time dimensions"
            raise ValueError(msg)
        if dataset.behavior.shape[2] == 0 or dataset.behavior.size == 0:
            msg = "behavior must not be empty"
            raise ValueError(msg)
        if dataset.behavior_names is None:
            msg = "behavior_names must exist when behavior exists"
            raise ValueError(msg)
        if len(dataset.behavior_names) != dataset.behavior.shape[2]:
            msg = "behavior_names length must match behavior dimension"
            raise ValueError(msg)
        if any(not isinstance(name, str) or not name.strip() for name in dataset.behavior_names):
            msg = "behavior_names must be non-empty strings"
            raise ValueError(msg)
        if len(set(dataset.behavior_names)) != len(dataset.behavior_names):
            msg = "behavior_names must be unique"
            raise ValueError(msg)
        if np.isnan(dataset.behavior).all():
            msg = "behavior must not be all NaN"
            raise ValueError(msg)
        if not np.all(np.isfinite(dataset.behavior)):
            msg = "behavior must be finite"
            raise ValueError(msg)
    elif dataset.behavior_names is not None:
        msg = "behavior_names require behavior"
        raise ValueError(msg)

    _require_rank("trial_ids", dataset.trial_ids, 1)
    if len(dataset.trial_ids) != n_trials:
        msg = "trial_ids length must match number of trials"
        raise ValueError(msg)
    if len(np.unique(dataset.trial_ids)) != len(dataset.trial_ids):
        msg = "trial_ids must be unique"
        raise ValueError(msg)

    _require_rank("time_ms", dataset.time_ms, 1)
    if len(dataset.time_ms) != n_time_bins:
        msg = "time_ms length must match number of time bins"
        raise ValueError(msg)
    if not np.all(np.isfinite(dataset.time_ms)):
        msg = "time_ms must be finite"
        raise ValueError(msg)
    if np.any(np.diff(dataset.time_ms) <= 0):
        msg = "time_ms must be strictly increasing"
        raise ValueError(msg)
    if dataset.bin_size_ms <= 0:
        msg = "bin_size_ms must be positive"
        raise ValueError(msg)


def validate_neural_dataset_minimums(
    dataset: NeuralDataset,
    min_trials: int,
    min_neurons: int,
    min_time_bins: int,
) -> None:
    """Validate minimum dimensions expected for a real-data ingestion contract."""
    validate_neural_dataset(dataset)
    n_trials, n_time_bins, n_neurons = dataset.spikes.shape
    if n_trials < min_trials:
        msg = f"dataset has {n_trials} trials, fewer than required minimum {min_trials}"
        raise ValueError(msg)
    if n_time_bins < min_time_bins:
        msg = f"dataset has {n_time_bins} time bins, fewer than required minimum {min_time_bins}"
        raise ValueError(msg)
    if n_neurons < min_neurons:
        msg = f"dataset has {n_neurons} neurons, fewer than required minimum {min_neurons}"
        raise ValueError(msg)


def validate_trial_split(split: TrialSplit, trial_ids: np.ndarray) -> None:
    """Validate split coverage and leakage against known trial identifiers."""
    for name, values in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        _require_rank(name, values, 1)
        if len(np.unique(values)) != len(values):
            msg = f"{name} split contains duplicate trial IDs"
            raise ValueError(msg)

    from latentbrain.data.splits import assert_no_trial_leakage

    assert_no_trial_leakage(split)
    combined = np.concatenate([split.train, split.validation, split.test])
    if set(combined.tolist()) != set(trial_ids.tolist()) or len(combined) != len(trial_ids):
        msg = "trial split must contain every trial exactly once"
        raise ValueError(msg)


def validate_neuron_mask(mask: NeuronMask, n_neurons: int) -> None:
    """Validate held-in and held-out masks."""
    for name, values in (("heldin", mask.heldin), ("heldout", mask.heldout)):
        _require_rank(name, values, 1)
        if values.dtype != np.bool_:
            msg = f"{name} mask must be boolean"
            raise ValueError(msg)
        if len(values) != n_neurons:
            msg = f"{name} mask length must equal n_neurons"
            raise ValueError(msg)

    if not mask.heldin.any():
        msg = "at least one held-in neuron is required"
        raise ValueError(msg)
    if not mask.heldout.any():
        msg = "at least one held-out neuron is required"
        raise ValueError(msg)
    if np.any(mask.heldin & mask.heldout):
        msg = "held-in and held-out masks must not overlap"
        raise ValueError(msg)
    if not np.all(mask.heldin | mask.heldout):
        msg = "held-in and held-out masks must cover all neurons"
        raise ValueError(msg)
