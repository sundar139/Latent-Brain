from __future__ import annotations

import numpy as np

from latentbrain.analysis.quality import (
    compute_dataset_summary,
    compute_neuron_activity,
    compute_quality_flags,
    compute_split_activity_summary,
    compute_time_activity,
    compute_trial_activity,
)
from latentbrain.data.schemas import NeuralDataset


def _dataset(spikes: np.ndarray | None = None) -> NeuralDataset:
    if spikes is None:
        spikes = np.array(
            [
                [[1, 0], [0, 0], [2, 1]],
                [[0, 1], [0, 0], [1, 0]],
            ],
            dtype=np.int64,
        )
    return NeuralDataset(
        spikes=spikes,
        rates=None,
        latents=None,
        trial_ids=np.array([10, 20], dtype=np.int64),
        time_ms=np.array([0.0, 5.0, 10.0], dtype=np.float64),
        bin_size_ms=5,
        metadata={},
    )


def test_dataset_summary_has_expected_dimensions_and_rates() -> None:
    summary = compute_dataset_summary(_dataset(), dataset_hash="abc")

    assert summary["n_trials"] == 2
    assert summary["n_time_bins"] == 3
    assert summary["n_neurons"] == 2
    assert summary["duration_seconds"] == 0.015
    assert summary["total_spikes"] == 6
    assert summary["mean_population_rate_hz"] == 200.0
    assert summary["dataset_hash"] == "abc"
    assert summary["has_rates"] is False
    assert summary["has_latents"] is False


def test_neuron_activity_uses_bin_size_for_rate_and_zero_fraction() -> None:
    activity = compute_neuron_activity(_dataset())

    assert len(activity) == 2
    assert activity.loc[activity["neuron_index"] == 0, "mean_rate_hz"].item() == 4 / 0.03
    assert activity.loc[activity["neuron_index"] == 1, "zero_fraction"].item() == 4 / 6
    assert activity["activity_rank"].tolist() == [1, 2]


def test_trial_and_time_activity_shapes_and_zero_fraction() -> None:
    dataset = _dataset()
    trial_activity = compute_trial_activity(dataset)
    time_activity = compute_time_activity(dataset)

    assert len(trial_activity) == 2
    assert trial_activity["trial_id"].tolist() == [10, 20]
    assert len(time_activity) == 3
    assert time_activity["time_ms"].tolist() == [0.0, 5.0, 10.0]
    assert time_activity.loc[time_activity["time_bin"] == 1, "zero_fraction"].item() == 1.0


def test_quality_flags_detect_bad_values_and_empty_activity() -> None:
    thresholds = {
        "max_nan_count": 0,
        "max_inf_count": 0,
        "min_total_spikes": 1,
        "max_zero_fraction_warning": 0.99,
        "inactive_neuron_rate_hz_threshold": 0.01,
        "high_rate_warning_hz": 200.0,
    }
    bad = _dataset(np.array([[[np.nan, np.inf]]], dtype=np.float64))
    flags = compute_quality_flags(bad, {"total_spikes": 0, "zero_fraction": 1.0}, thresholds)

    assert {flag["code"] for flag in flags if flag["severity"] == "error"} == {
        "nan_spikes",
        "inf_spikes",
        "zero_total_spikes",
    }


def test_split_activity_summary_counts_trials() -> None:
    summary = compute_split_activity_summary(
        _dataset(),
        train_ids=np.array([10], dtype=np.int64),
        validation_ids=np.array([20], dtype=np.int64),
        test_ids=np.array([], dtype=np.int64),
    )

    assert summary["split"].tolist() == ["train", "validation", "test"]
    assert summary["trial_count"].tolist() == [1, 1, 0]
