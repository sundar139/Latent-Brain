from __future__ import annotations

import numpy as np

from latentbrain.eval.decoding import (
    apply_standardization,
    fit_ridge_decoder,
    predict_ridge_decoder,
    r2_score_numpy,
    regression_metrics,
    standardize_train_apply,
)


def test_ridge_decoder_recovers_simple_linear_relationship() -> None:
    x = np.arange(20, dtype=np.float64).reshape(-1, 1)
    y = 2.0 * x + 1.0

    model = fit_ridge_decoder(x, y, alpha=0.0, fit_intercept=True)
    pred = predict_ridge_decoder(x, model)

    np.testing.assert_allclose(pred, y, atol=1e-10)


def test_ridge_prediction_shape_is_correct() -> None:
    model = fit_ridge_decoder(np.ones((5, 2)), np.ones((5, 3)), alpha=1.0)

    pred = predict_ridge_decoder(np.ones((7, 2)), model)

    assert pred.shape == (7, 3)


def test_standardization_uses_train_statistics_only() -> None:
    train = np.array([[1.0], [3.0]])
    values = np.array([[1.0], [3.0], [101.0]])

    standardized, stats = standardize_train_apply(train, values)
    reapplied = apply_standardization(values, stats)

    np.testing.assert_allclose(standardized, reapplied)
    np.testing.assert_allclose(stats["mean"], np.array([2.0]))


def test_r2_equals_one_for_perfect_prediction() -> None:
    y = np.array([[1.0, 2.0], [3.0, 4.0]])

    r2 = r2_score_numpy(y, y)

    np.testing.assert_allclose(r2, np.array([1.0, 1.0]))


def test_r2_negative_for_poor_prediction() -> None:
    y = np.array([[0.0], [1.0], [2.0]])
    pred = np.array([[10.0], [10.0], [10.0]])

    r2 = r2_score_numpy(y, pred)

    assert float(r2[0]) < 0


def test_constant_targets_return_nan_r2() -> None:
    y = np.ones((3, 1), dtype=np.float64)

    r2 = r2_score_numpy(y, y)

    assert np.isnan(r2[0])


def test_regression_metrics_have_expected_columns() -> None:
    y = np.array([[0.0, 1.0], [1.0, 2.0]])

    metrics = regression_metrics(y, y, ["a", "b"])

    assert {"target_name", "r2", "mse", "mae", "target_variance"}.issubset(metrics.columns)
    assert metrics["target_name"].tolist() == ["a", "b"]
