from __future__ import annotations

import math

import numpy as np
import pytest

from latentbrain.eval.metrics import (
    bits_per_spike,
    poisson_log_likelihood,
    poisson_nll,
    safe_clip_rates,
    summarize_poisson_metrics,
)


def test_poisson_log_likelihood_matches_hand_computed_value() -> None:
    counts = np.array([0, 1, 2], dtype=np.int64)
    rates_hz = np.array([10.0, 20.0, 30.0])
    bin_size_ms = 100
    expected_counts = rates_hz * (bin_size_ms / 1000.0)
    expected = float(
        np.sum(counts * np.log(expected_counts) - expected_counts)
        - sum(math.lgamma(float(count) + 1.0) for count in counts)
    )

    assert poisson_log_likelihood(counts, rates_hz, bin_size_ms) == pytest.approx(expected)
    assert poisson_nll(counts, rates_hz, bin_size_ms) == pytest.approx(-expected)


def test_rate_clipping_enforces_minimum_and_maximum() -> None:
    clipped = safe_clip_rates(np.array([0.0, 1.0, 1000.0]), min_rate_hz=0.1, max_rate_hz=500.0)

    np.testing.assert_allclose(clipped, np.array([0.1, 1.0, 500.0]))


@pytest.mark.parametrize(
    ("counts", "rates", "message"),
    [
        (np.array([-1]), np.array([1.0]), "non-negative"),
        (np.array([0.5]), np.array([1.0]), "integer"),
        (np.array([1]), np.array([0.0]), "positive"),
        (np.array([1]), np.array([-1.0]), "positive"),
    ],
)
def test_poisson_metrics_reject_invalid_inputs(
    counts: np.ndarray,
    rates: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        poisson_log_likelihood(counts, rates, bin_size_ms=5)


def test_bits_per_spike_matches_formula_and_rejects_zero_spikes() -> None:
    assert bits_per_spike(-8.0, -10.0, 4.0) == pytest.approx(2.0 / (math.log(2.0) * 4.0))

    with pytest.raises(ValueError, match="spike_count"):
        bits_per_spike(-1.0, -2.0, 0.0)


def test_summarize_poisson_metrics_returns_expected_keys() -> None:
    counts = np.array([[0, 1], [2, 0]], dtype=np.int64)
    predicted = np.full_like(counts, 10.0, dtype=np.float64)
    reference = np.full_like(counts, 5.0, dtype=np.float64)

    summary = summarize_poisson_metrics(counts, predicted, reference, bin_size_ms=100)

    assert set(summary) == {
        "spike_count",
        "poisson_nll",
        "poisson_log_likelihood",
        "reference_log_likelihood",
        "bits_per_spike",
        "mean_predicted_rate_hz",
    }
    assert summary["spike_count"] == 3.0
