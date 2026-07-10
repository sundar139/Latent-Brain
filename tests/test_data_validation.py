from __future__ import annotations

import numpy as np
import pytest

from latentbrain.data.schemas import NeuralDataset, NeuronMask, TrialSequences, TrialSplit
from latentbrain.data.validation import (
    validate_neural_dataset,
    validate_neuron_mask,
    validate_source_bin_size,
    validate_trial_sequences,
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


def _dataset_with_behavior() -> NeuralDataset:
    dataset = _dataset()
    dataset.behavior = np.arange(12, dtype=np.float64).reshape(2, 3, 2)
    dataset.behavior_names = ["hand_pos_x", "hand_pos_y"]
    return dataset


def test_validate_neural_dataset_accepts_valid_behavior() -> None:
    validate_neural_dataset(_dataset_with_behavior())


def test_validate_neural_dataset_rejects_wrong_behavior_shape() -> None:
    dataset = _dataset_with_behavior()
    dataset.behavior = np.zeros((2, 2, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="behavior.*trial and time"):
        validate_neural_dataset(dataset)


def test_validate_neural_dataset_rejects_behavior_name_mismatch() -> None:
    dataset = _dataset_with_behavior()
    dataset.behavior_names = ["hand_pos_x"]

    with pytest.raises(ValueError, match="behavior_names length"):
        validate_neural_dataset(dataset)


def test_validate_neural_dataset_rejects_duplicate_behavior_names() -> None:
    dataset = _dataset_with_behavior()
    dataset.behavior_names = ["hand_pos_x", "hand_pos_x"]

    with pytest.raises(ValueError, match="unique"):
        validate_neural_dataset(dataset)


def test_validate_neural_dataset_rejects_nan_behavior() -> None:
    dataset = _dataset_with_behavior()
    dataset.behavior[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        validate_neural_dataset(dataset)


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


def test_validate_source_bin_size_accepts_matching_bin_size() -> None:
    validate_source_bin_size(_dataset(), 20)


def test_validate_source_bin_size_rejects_mismatched_bin_size() -> None:
    with pytest.raises(ValueError, match="config expects 5"):
        validate_source_bin_size(_dataset(), 5)


def test_validate_source_bin_size_rejects_inconsistent_time_axis() -> None:
    dataset = _dataset()
    dataset.time_ms = np.array([0.0, 20.0, 100.0], dtype=np.float64)

    with pytest.raises(ValueError, match="time_ms spacing does not match"):
        validate_source_bin_size(dataset, 20)


def _sequences() -> TrialSequences:
    return TrialSequences(
        spikes=[np.zeros((4, 3), dtype=np.int64), np.zeros((6, 3), dtype=np.int64)],
        behavior=[np.zeros((4, 2)), np.zeros((6, 2))],
        behavior_names=["hand_pos_x", "hand_pos_y"],
        trial_ids=np.array([0, 1], dtype=np.int64),
        trial_lengths=np.array([4, 6], dtype=np.int64),
        bin_size_ms=5,
        metadata={},
    )


def test_validate_trial_sequences_accepts_variable_lengths() -> None:
    validate_trial_sequences(_sequences())


def test_validate_trial_sequences_rejects_inconsistent_neuron_count() -> None:
    sequences = _sequences()
    sequences.spikes[1] = np.zeros((6, 4), dtype=np.int64)

    with pytest.raises(ValueError, match="same neuron count"):
        validate_trial_sequences(sequences)


def test_validate_trial_sequences_rejects_length_mismatch() -> None:
    sequences = _sequences()
    sequences.trial_lengths = np.array([4, 5], dtype=np.int64)

    with pytest.raises(ValueError, match="does not match trial_lengths"):
        validate_trial_sequences(sequences)


def test_validate_trial_sequences_rejects_negative_spikes() -> None:
    sequences = _sequences()
    sequences.spikes[0][0, 0] = -1

    with pytest.raises(ValueError, match="non-negative"):
        validate_trial_sequences(sequences)
