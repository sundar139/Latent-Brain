from __future__ import annotations

import numpy as np

from latentbrain.data.schemas import NeuralDataset, NeuronMask, TrialSplit
from latentbrain.eval.rebinning import (
    build_binning_comparison_row,
    compute_sparsity_summary,
    compute_window_bins_for_duration,
)


def _dataset() -> NeuralDataset:
    return NeuralDataset(
        spikes=np.array([[[0, 1], [0, 0]], [[2, 0], [0, 0]]]),
        rates=None,
        latents=None,
        trial_ids=np.array([0, 1]),
        time_ms=np.array([0, 5]),
        bin_size_ms=5,
        metadata={},
    )


def test_window_bins_computed_correctly() -> None:
    assert compute_window_bins_for_duration(1.28, 10) == 128


def test_sparsity_summary_has_expected_columns_and_zero_fraction() -> None:
    summary = compute_sparsity_summary(
        _dataset(),
        TrialSplit(train=np.array([0]), validation=np.array([1]), test=np.array([], dtype=int)),
        NeuronMask(heldin=np.array([True, False]), heldout=np.array([False, True])),
        5,
        2,
    )
    expected = {
        "bin_size_ms",
        "split",
        "time_bins",
        "window_seconds",
        "n_trials",
        "n_heldout_neurons",
        "spike_count",
        "total_observations",
        "zero_fraction",
        "observed_rate_hz",
        "mean_spikes_per_bin",
    }
    assert set(summary.columns) == expected
    validation = summary[summary["split"] == "validation"].iloc[0]
    assert validation["zero_fraction"] == 1.0


def test_comparison_row_has_required_columns() -> None:
    row = build_binning_comparison_row(
        "mean_rate",
        10,
        "validation",
        {"bits_per_spike": 0.1, "spike_count": 2.0},
        {"prediction_source": "constant", "time_bins": 128, "window_seconds": 1.28},
    )
    for key in (
        "bin_size_ms",
        "method_name",
        "split",
        "prediction_source",
        "time_bins",
        "window_seconds",
        "spike_count",
        "bits_per_spike",
    ):
        assert key in row
