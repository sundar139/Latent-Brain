from __future__ import annotations

import numpy as np
import pytest

from latentbrain.data.schemas import NeuralDataset, NeuronMask, TrialSplit
from latentbrain.data.validation import (
    validate_neural_dataset,
    validate_neuron_mask,
    validate_trial_split,
)


def _dataset() -> NeuralDataset:
    return NeuralDataset(
        spikes=np.zeros((2, 3, 4), dtype=np.int64),
        rates=np.ones((2, 3, 4), dtype=np.float64),
        latents=np.zeros((2, 3, 2), dtype=np.float64),
        trial_ids=np.array([0, 1], dtype=np.int64),
        time_ms=np.array([0.0, 20.0, 40.0], dtype=np.float64),
        bin_size_ms=20,
        metadata={},
    )


def test_validate_neural_dataset_rejects_duplicate_trial_ids() -> None:
    dataset = _dataset()
    dataset.trial_ids[1] = dataset.trial_ids[0]

    with pytest.raises(ValueError, match="unique"):
        validate_neural_dataset(dataset)


def test_validate_neural_dataset_rejects_nonfinite_rates() -> None:
    dataset = _dataset()
    assert dataset.rates is not None
    dataset.rates[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        validate_neural_dataset(dataset)


def test_validate_trial_split_rejects_missing_trial() -> None:
    split = TrialSplit(
        train=np.array([0], dtype=np.int64),
        validation=np.array([], dtype=np.int64),
        test=np.array([], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="exactly once"):
        validate_trial_split(split, np.array([0, 1], dtype=np.int64))


def test_validate_neuron_mask_rejects_overlap() -> None:
    mask = NeuronMask(
        heldin=np.array([True, True], dtype=bool),
        heldout=np.array([True, False], dtype=bool),
    )

    with pytest.raises(ValueError, match="overlap"):
        validate_neuron_mask(mask, 2)
