from __future__ import annotations

import numpy as np
import pytest

from latentbrain.eval.neural_predictions import (
    flatten_batch_time,
    reshape_flat_predictions,
    summarize_factor_activity,
    summarize_rate_predictions,
)


def test_summarize_rate_predictions_returns_safe_summary() -> None:
    summary = summarize_rate_predictions(np.array([1.0, 2.0, 3.0]))

    assert summary == {"mean_rate_hz": 2.0, "min_rate_hz": 1.0, "max_rate_hz": 3.0}


def test_flatten_batch_time_and_reshape_round_trip() -> None:
    values = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    flat = flatten_batch_time(values)
    restored = reshape_flat_predictions(flat, n_trials=2, n_time_bins=3, n_outputs=4)

    assert flat.shape == (6, 4)
    np.testing.assert_array_equal(restored, values)


def test_invalid_neural_prediction_shapes_raise_clear_errors() -> None:
    with pytest.raises(ValueError, match="rank 3"):
        flatten_batch_time(np.ones((2, 3)))
    with pytest.raises(ValueError, match="rank 2"):
        reshape_flat_predictions(np.ones((2, 3, 4)), 2, 3, 4)
    with pytest.raises(ValueError, match="cannot reshape"):
        reshape_flat_predictions(np.ones((5, 4)), 2, 3, 4)


def test_factor_summary_has_expected_rows_and_columns() -> None:
    factors = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)

    summary = summarize_factor_activity(factors, split_name="validation")

    assert list(summary.columns) == [
        "split",
        "factor_index",
        "mean",
        "std",
        "min",
        "max",
        "variance",
    ]
    assert len(summary) == 4
    assert set(summary["split"]) == {"validation"}
    assert summary["factor_index"].tolist() == [0, 1, 2, 3]
