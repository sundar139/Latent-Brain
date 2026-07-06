from __future__ import annotations

import numpy as np

from latentbrain.data.schemas import NeuronMask, TrialSplit


def _validate_split_fractions(
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> None:
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(fraction <= 0.0 or fraction >= 1.0 for fraction in fractions):
        msg = "split fractions must be greater than 0 and less than 1"
        raise ValueError(msg)
    if abs(sum(fractions) - 1.0) > 1e-8:
        msg = "split fractions must sum to 1.0"
        raise ValueError(msg)


def create_trial_split(
    trial_ids: np.ndarray,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> TrialSplit:
    """Create a deterministic train/validation/test split over trial IDs."""
    _validate_split_fractions(train_fraction, validation_fraction, test_fraction)
    if trial_ids.ndim != 1:
        msg = "trial_ids must be a rank-1 array"
        raise ValueError(msg)
    if len(np.unique(trial_ids)) != len(trial_ids):
        msg = "trial_ids must be unique before splitting"
        raise ValueError(msg)
    if len(trial_ids) < 3:
        msg = "at least three trials are required for train/validation/test splitting"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    shuffled = np.array(trial_ids, copy=True)
    rng.shuffle(shuffled)

    n_trials = len(shuffled)
    n_train = int(round(n_trials * train_fraction))
    n_validation = int(round(n_trials * validation_fraction))
    n_test = n_trials - n_train - n_validation
    if min(n_train, n_validation, n_test) <= 0:
        msg = "split fractions produce an empty train, validation, or test split"
        raise ValueError(msg)

    split = TrialSplit(
        train=shuffled[:n_train],
        validation=shuffled[n_train : n_train + n_validation],
        test=shuffled[n_train + n_validation :],
    )
    assert_no_trial_leakage(split)
    return split


def assert_no_trial_leakage(split: TrialSplit) -> None:
    """Raise if trial IDs overlap between train, validation, and test splits."""
    sections = (split.train, split.validation, split.test)
    combined = np.concatenate(sections)
    if len(np.unique(combined)) != len(combined):
        msg = "trial leakage detected across train, validation, and test splits"
        raise ValueError(msg)


def create_neuron_mask(n_neurons: int, heldout_fraction: float, seed: int) -> NeuronMask:
    """Create deterministic held-in and held-out boolean neuron masks."""
    if n_neurons < 2:
        msg = "at least two neurons are required for held-in and held-out masks"
        raise ValueError(msg)
    if heldout_fraction <= 0.0 or heldout_fraction >= 1.0:
        msg = "heldout_fraction must be greater than 0 and less than 1"
        raise ValueError(msg)

    heldout_count = int(round(n_neurons * heldout_fraction))
    heldout_count = min(max(heldout_count, 1), n_neurons - 1)
    rng = np.random.default_rng(seed)
    heldout_indices = rng.choice(n_neurons, size=heldout_count, replace=False)

    heldout = np.zeros(n_neurons, dtype=bool)
    heldout[heldout_indices] = True
    heldin = ~heldout
    return NeuronMask(heldin=heldin, heldout=heldout)
