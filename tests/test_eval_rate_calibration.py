from __future__ import annotations

import numpy as np

from latentbrain.eval.rate_calibration import (
    apply_log_rate_bias,
    apply_multiplicative_rate_scale,
    blend_with_mean_rate,
    choose_best_blend_alpha,
    fit_log_rate_bias,
    fit_multiplicative_rate_scale,
)


def test_multiplicative_scale_recovers_known_factor() -> None:
    counts = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
    observed_rates = counts / 0.02
    predicted = observed_rates / np.array([2.0, 0.5])

    scale = fit_multiplicative_rate_scale(counts, predicted, bin_size_ms=20)

    assert np.allclose(scale, [2.0, 0.5])


def test_zero_count_neuron_is_handled_safely() -> None:
    counts = np.array([[[0.0, 1.0]], [[0.0, 2.0]]])
    predicted = np.ones_like(counts) * 10.0

    scale = fit_multiplicative_rate_scale(counts, predicted, bin_size_ms=20)
    calibrated = apply_multiplicative_rate_scale(predicted, scale, 1.0e-4, 500.0)

    assert np.isfinite(scale).all()
    assert scale[0] == 1.0e-3
    assert np.isfinite(calibrated).all()
    assert np.all(calibrated > 0.0)


def test_log_bias_calibration_returns_finite_values() -> None:
    counts = np.array([[[1.0, 0.0]], [[2.0, 1.0]]])
    predicted = np.ones_like(counts) * 20.0

    bias = fit_log_rate_bias(counts, predicted, bin_size_ms=20)
    calibrated = apply_log_rate_bias(predicted, bias, 1.0e-4, 500.0)

    assert np.isfinite(bias).all()
    assert np.isfinite(calibrated).all()
    assert np.all(calibrated > 0.0)


def test_mean_rate_blend_alpha_endpoints() -> None:
    predicted = np.array([[[2.0, 4.0]]])
    mean = np.array([10.0, 20.0])

    assert np.allclose(blend_with_mean_rate(predicted, mean, 0.0), [[[10.0, 20.0]]])
    assert np.allclose(blend_with_mean_rate(predicted, mean, 1.0), predicted)


def test_best_alpha_selection_uses_train_data_and_chooses_expected_alpha() -> None:
    counts = np.array([[[1.0]], [[2.0]], [[1.0]]])
    train_mean = np.array([counts.sum() / (counts.shape[0] * counts.shape[1] * 0.02)])
    predicted = np.full_like(counts, train_mean[0] * 10.0)

    alpha, table = choose_best_blend_alpha(
        counts,
        predicted,
        train_mean,
        alpha_grid=[0.0, 0.5, 1.0],
        bin_size_ms=20,
    )

    assert alpha == 0.0
    assert list(table["alpha"]) == [0.0, 0.5, 1.0]
    assert table.loc[table["alpha"] == 0.0, "bits_per_spike"].iloc[0] == 0.0


def test_calibrated_rates_remain_positive_and_finite() -> None:
    predicted = np.array([[[0.0, 1.0e9]]])
    scale = np.array([2.0, 2.0])

    calibrated = apply_multiplicative_rate_scale(predicted, scale, 1.0e-4, 500.0)

    assert np.isfinite(calibrated).all()
    assert calibrated.min() >= 1.0e-4
    assert calibrated.max() <= 500.0
