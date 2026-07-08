from __future__ import annotations

import numpy as np
import pytest

from latentbrain.data.rebinning import (
    rebin_behavior,
    rebin_neural_dataset,
    rebin_spike_counts,
    validate_rebin_factor,
)
from latentbrain.data.schemas import NeuralDataset


def _dataset() -> NeuralDataset:
    spikes = np.arange(2 * 5 * 3).reshape(2, 5, 3)
    behavior = np.arange(2 * 5 * 2, dtype=float).reshape(2, 5, 2)
    return NeuralDataset(
        spikes=spikes,
        rates=None,
        latents=None,
        trial_ids=np.array([10, 11]),
        time_ms=np.arange(5) * 5,
        bin_size_ms=5,
        metadata={"name": "toy"},
        behavior=behavior,
        behavior_names=["x", "y"],
    )


def test_rebin_factor_validation_works() -> None:
    assert validate_rebin_factor(5, 20) == 4


def test_invalid_target_bin_raises() -> None:
    with pytest.raises(ValueError, match="multiple"):
        validate_rebin_factor(5, 12)


def test_spike_rebin_sums_counts_and_trims() -> None:
    spikes = np.ones((1, 5, 2), dtype=int)
    rebinned = rebin_spike_counts(spikes, 2, trim=True)
    assert rebinned.shape == (1, 2, 2)
    assert np.all(rebinned == 2)
    assert rebinned.sum() == spikes[:, :4].sum()


def test_non_divisible_length_with_trim_false_raises() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        rebin_spike_counts(np.ones((1, 5, 1)), 2, trim=False)


def test_behavior_rebin_keeps_expected_shape_and_average() -> None:
    behavior = np.array([[[1.0], [3.0], [5.0], [7.0]]])
    rebinned = rebin_behavior(behavior, 2)
    assert rebinned.shape == (1, 2, 1)
    assert np.allclose(rebinned[:, :, 0], [[2.0, 6.0]])


def test_dataset_rebin_updates_bin_size_and_preserves_counts() -> None:
    dataset = _dataset()
    rebinned = rebin_neural_dataset(dataset, 10)
    assert rebinned.bin_size_ms == 10
    assert rebinned.spikes.shape == (2, 2, 3)
    assert rebinned.trial_ids.tolist() == [10, 11]
    assert rebinned.behavior_names == ["x", "y"]
    assert rebinned.spikes.sum() == dataset.spikes[:, :4].sum()
    assert rebinned.behavior is not None
    assert rebinned.behavior.shape == (2, 2, 2)
