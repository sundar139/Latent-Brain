from __future__ import annotations

import numpy as np
import pytest

from latentbrain.eval.smoothing import smooth_spike_counts, spike_counts_to_rates_hz


def test_smoothing_preserves_shape() -> None:
    spikes = np.zeros((2, 9, 3), dtype=np.int64)
    spikes[:, 4, :] = 1

    smoothed = smooth_spike_counts(spikes, bin_size_ms=5, sigma_ms=10.0)

    assert smoothed.shape == spikes.shape
    assert np.all(np.isfinite(smoothed))


def test_smoothing_does_not_mix_trials() -> None:
    spikes = np.zeros((2, 9, 1), dtype=np.int64)
    spikes[0, 4, 0] = 1

    smoothed = smooth_spike_counts(spikes, bin_size_ms=5, sigma_ms=10.0)

    assert smoothed[0].sum() > 0
    assert smoothed[1].sum() == 0


def test_counts_to_hz_conversion_is_correct() -> None:
    counts = np.array([[[0.5, 1.0]]], dtype=np.float64)

    rates = spike_counts_to_rates_hz(counts, bin_size_ms=5)

    np.testing.assert_allclose(rates, np.array([[[100.0, 200.0]]]))


def test_negative_spikes_raise_error() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        smooth_spike_counts(np.array([[[-1]]], dtype=np.int64), bin_size_ms=5)


def test_nonfinite_spikes_raise_error() -> None:
    with pytest.raises(ValueError, match="finite"):
        smooth_spike_counts(np.array([[[np.nan]]], dtype=np.float64), bin_size_ms=5)
