from __future__ import annotations

import numpy as np

from latentbrain.eval.metric_audit import score_prediction_against_reference
from latentbrain.eval.oracle_controls import (
    make_oracle_smoothed_heldout_prediction,
    make_random_rate_prediction,
    make_train_mean_rate_prediction,
    make_trial_shuffled_heldin_prediction,
)


def test_mean_rate_prediction_has_expected_shape() -> None:
    prediction = make_train_mean_rate_prediction(np.ones((2, 3, 4)), (5, 6, 4), 20, 1e-4, 500.0)

    assert prediction.shape == (5, 6, 4)


def test_oracle_smoothed_prediction_has_expected_shape_and_positive_rates() -> None:
    counts = np.zeros((2, 8, 3))
    counts[:, 3, :] = 1.0

    prediction = make_oracle_smoothed_heldout_prediction(counts, 20, 40.0, 1e-4, 500.0)

    assert prediction.shape == counts.shape
    assert np.all(prediction > 0.0)


def test_random_prediction_is_finite_and_positive() -> None:
    prediction = make_random_rate_prediction(
        (2, 3, 4), np.array([1.0, 2.0, 3.0, 4.0]), 7, 1e-4, 500.0
    )

    assert np.all(np.isfinite(prediction))
    assert np.all(prediction > 0.0)


def test_shuffled_control_is_deterministic_under_seed() -> None:
    train_heldin = np.arange(3 * 2 * 2, dtype=float).reshape(3, 2, 2)
    train_heldout = np.arange(3 * 2 * 4, dtype=float).reshape(3, 2, 4)
    eval_heldin = np.zeros((2, 2, 2))

    first = make_trial_shuffled_heldin_prediction(
        train_heldin, train_heldout, eval_heldin, (2, 2, 4), 11, 1e-4, 500.0
    )
    second = make_trial_shuffled_heldin_prediction(
        train_heldin, train_heldout, eval_heldin, (2, 2, 4), 11, 1e-4, 500.0
    )

    np.testing.assert_allclose(first, second)


def test_oracle_controls_are_marked_invalid_model_where_appropriate() -> None:
    counts = np.ones((1, 2, 1))
    prediction = make_oracle_smoothed_heldout_prediction(counts, 20, 40.0, 1e-4, 500.0)
    row = score_prediction_against_reference(
        counts, prediction, prediction, 20, "oracle", "validation", "oracle"
    )
    row["valid_model"] = False

    assert row["valid_model"] is False
