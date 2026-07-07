from __future__ import annotations

import numpy as np
import pytest

from latentbrain.data.schemas import NeuralDataset
from latentbrain.eval.windowing import (
    crop_neural_dataset_time,
    crop_time_window,
    describe_time_window,
)


def _dataset() -> NeuralDataset:
    spikes = np.arange(2 * 5 * 3, dtype=np.int64).reshape(2, 5, 3)
    behavior = np.arange(2 * 5 * 2, dtype=np.float64).reshape(2, 5, 2)
    return NeuralDataset(
        spikes=spikes,
        rates=spikes.astype(np.float64) + 0.5,
        latents=np.ones((2, 5, 2), dtype=np.float64),
        trial_ids=np.array([10, 11], dtype=np.int64),
        time_ms=np.arange(5, dtype=np.float64) * 5.0,
        bin_size_ms=5,
        metadata={"source": "unit"},
        behavior=behavior,
        behavior_names=["x", "y"],
    )


def test_crop_time_window_crops_trial_time_arrays() -> None:
    values = np.arange(2 * 5 * 3).reshape(2, 5, 3)

    cropped = crop_time_window(values, 3)

    assert cropped.shape == (2, 3, 3)
    np.testing.assert_array_equal(cropped, values[:, :3, :])


def test_crop_time_window_keeps_valid_result_when_limit_is_large() -> None:
    values = np.arange(2 * 3).reshape(1, 2, 3)

    cropped = crop_time_window(values, 10)

    np.testing.assert_array_equal(cropped, values)
    assert cropped is not values


def test_crop_time_window_rejects_unsupported_policy() -> None:
    with pytest.raises(ValueError, match="unsupported crop policy"):
        crop_time_window(np.zeros((1, 2, 3)), 1, policy="center")


def test_crop_neural_dataset_preserves_trials_neurons_and_aligns_behavior() -> None:
    dataset = _dataset()

    cropped = crop_neural_dataset_time(dataset, 3)

    assert cropped.spikes.shape == (2, 3, 3)
    assert cropped.behavior is not None
    assert cropped.behavior.shape == (2, 3, 2)
    assert cropped.trial_ids.tolist() == [10, 11]
    assert cropped.metadata["time_window"]["cropped_time_bins"] == 3
    np.testing.assert_array_equal(cropped.spikes, dataset.spikes[:, :3, :])
    np.testing.assert_array_equal(cropped.behavior, dataset.behavior[:, :3, :])


def test_describe_time_window_reports_seconds() -> None:
    summary = describe_time_window(100, 25, bin_size_ms=20)

    assert summary["original_time_bins"] == 100
    assert summary["cropped_time_bins"] == 25
    assert summary["window_seconds"] == 0.5
