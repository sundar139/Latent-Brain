from __future__ import annotations

import math

import numpy as np

from latentbrain.eval.metric_audit import (
    REQUIRED_SCORE_COLUMNS,
    bits_per_spike_from_log_likelihoods,
    compute_train_heldout_mean_rates,
    score_prediction_against_reference,
)
from latentbrain.eval.oracle_controls import make_train_mean_rate_prediction


def test_predicted_equals_reference_gives_zero_bits_per_spike() -> None:
    counts = np.array([[[0.0, 1.0], [2.0, 0.0]]])
    rates = np.full_like(counts, 5.0)

    row = score_prediction_against_reference(counts, rates, rates, 20, "same", "validation", "unit")

    assert row["bits_per_spike"] == 0.0


def test_better_prediction_gives_positive_bits_per_spike() -> None:
    counts = np.array([[[3.0], [2.0], [3.0]]])
    reference = np.full_like(counts, 1.0)
    predicted = np.full_like(counts, 120.0)

    row = score_prediction_against_reference(
        counts, predicted, reference, 20, "better", "validation", "unit"
    )

    assert row["bits_per_spike"] > 0.0


def test_worse_prediction_gives_negative_bits_per_spike() -> None:
    counts = np.array([[[3.0], [2.0], [3.0]]])
    reference = np.full_like(counts, 120.0)
    predicted = np.full_like(counts, 1.0)

    row = score_prediction_against_reference(
        counts, predicted, reference, 20, "worse", "validation", "unit"
    )

    assert row["bits_per_spike"] < 0.0


def test_zero_spike_bits_per_spike_returns_nan() -> None:
    assert math.isnan(bits_per_spike_from_log_likelihoods(-1.0, -1.0, 0.0))


def test_score_row_contains_required_columns() -> None:
    counts = np.array([[[0.0, 1.0]]])
    rates = np.full_like(counts, 10.0)

    row = score_prediction_against_reference(counts, rates, rates, 20, "same", "test", "unit")

    assert set(REQUIRED_SCORE_COLUMNS).issubset(row)


def test_train_heldout_mean_rate_prediction_shape_is_correct() -> None:
    train = np.ones((2, 3, 4))
    rates = compute_train_heldout_mean_rates(train, 20, 1.0e-4, 500.0)
    prediction = make_train_mean_rate_prediction(train, (5, 7, 4), 20, 1.0e-4, 500.0)

    assert rates.shape == (4,)
    assert prediction.shape == (5, 7, 4)
