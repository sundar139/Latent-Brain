from __future__ import annotations

import numpy as np

from latentbrain.eval.calibration import (
    compute_prediction_reference_correlation,
    compute_rate_calibration_table,
    summarize_rate_distribution,
)

EXPECTED_COLUMNS = [
    "rate_bin",
    "n_observations",
    "mean_predicted_rate_hz",
    "mean_reference_rate_hz",
    "observed_rate_hz",
    "mean_count",
    "spike_count",
]


def test_calibration_table_has_expected_columns_and_observed_rate() -> None:
    counts = np.array([[[0.0], [1.0]]])
    predicted = np.array([[[10.0], [20.0]]])
    reference = np.array([[[15.0], [15.0]]])

    table = compute_rate_calibration_table(counts, predicted, reference, 5, 2)

    assert table.columns.tolist() == EXPECTED_COLUMNS
    assert table["spike_count"].sum() == 1.0
    assert np.isclose(table["observed_rate_hz"].mean(), 100.0)


def test_zero_spike_case_returns_finite_counts_and_nan_safe_rates() -> None:
    counts = np.zeros((2, 3, 1))
    rates = np.ones_like(counts) * 5.0

    table = compute_rate_calibration_table(counts, rates, rates, 10, 3)
    summary = summarize_rate_distribution(rates, rates, counts)

    assert table["spike_count"].sum() == 0.0
    assert np.all(np.isfinite(table["observed_rate_hz"].dropna()))
    assert summary["spike_count"] == 0.0


def test_constant_prediction_correlation_is_nan_not_crash() -> None:
    predicted = np.ones((2, 2, 1)) * 3.0
    reference = np.ones((2, 2, 1)) * 4.0

    assert np.isnan(compute_prediction_reference_correlation(predicted, reference))


def test_shape_mismatch_raises_clear_error() -> None:
    counts = np.zeros((1, 2, 1))
    predicted = np.zeros((1, 2, 2)) + 1.0
    reference = np.zeros((1, 2, 1)) + 1.0

    try:
        compute_rate_calibration_table(counts, predicted, reference, 5, 2)
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
