from __future__ import annotations

import numpy as np
import pytest

from latentbrain.data.splits import (
    assert_no_trial_leakage,
    create_neuron_mask,
    create_trial_split,
)
from latentbrain.data.validation import validate_neuron_mask, validate_trial_split


def test_create_trial_split_is_deterministic_and_covers_trials() -> None:
    trial_ids = np.arange(64)
    first = create_trial_split(trial_ids, 0.7, 0.15, 0.15, seed=2027)
    second = create_trial_split(trial_ids, 0.7, 0.15, 0.15, seed=2027)

    np.testing.assert_array_equal(first.train, second.train)
    np.testing.assert_array_equal(first.validation, second.validation)
    np.testing.assert_array_equal(first.test, second.test)
    validate_trial_split(first, trial_ids)
    assert len(first.train) == 45
    assert len(first.validation) == 10
    assert len(first.test) == 9


def test_assert_no_trial_leakage_rejects_overlap() -> None:
    split = create_trial_split(np.arange(6), 0.5, 0.25, 0.25, seed=1)
    split.validation[0] = split.train[0]

    with pytest.raises(ValueError, match="leakage"):
        assert_no_trial_leakage(split)


def test_create_neuron_mask_is_valid_and_deterministic() -> None:
    first = create_neuron_mask(32, 0.25, seed=2027)
    second = create_neuron_mask(32, 0.25, seed=2027)

    validate_neuron_mask(first, 32)
    np.testing.assert_array_equal(first.heldin, second.heldin)
    np.testing.assert_array_equal(first.heldout, second.heldout)
    assert int(first.heldout.sum()) == 8
    assert int(first.heldin.sum()) == 24


def test_create_neuron_mask_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError, match="greater than 0 and less than 1"):
        create_neuron_mask(32, 1.0, seed=2027)
