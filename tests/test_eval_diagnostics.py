from __future__ import annotations

import numpy as np

from latentbrain.eval.diagnostics import (
    compute_factor_usage,
    compute_loss_scale_diagnostics,
    compute_neuron_prediction_diagnostics,
)

REQUIRED_LOSS_KEYS = {
    "spike_count",
    "total_observations",
    "zero_fraction",
    "model_poisson_nll",
    "reference_poisson_nll",
    "model_log_likelihood",
    "reference_log_likelihood",
    "bits_per_spike",
    "nll_per_observation",
    "nll_per_spike",
    "mean_predicted_rate_hz",
    "mean_reference_rate_hz",
    "observed_rate_hz",
}


def test_loss_scale_diagnostics_include_required_keys() -> None:
    counts = np.array([[[0.0, 1.0], [1.0, 0.0]]])
    predicted = np.ones_like(counts) * 20.0
    reference = np.ones_like(counts) * 10.0

    summary = compute_loss_scale_diagnostics(counts, predicted, reference, 5)

    assert set(summary) >= REQUIRED_LOSS_KEYS
    assert summary["total_observations"] == 4.0


def test_factor_usage_marks_active_and_inactive_dimensions() -> None:
    factors = np.zeros((2, 3, 2))
    factors[:, :, 1] = np.arange(6).reshape(2, 3)

    usage = compute_factor_usage(factors, "validation")

    assert usage["active"].tolist() == [False, True]
    assert usage["split"].unique().tolist() == ["validation"]


def test_neuron_diagnostics_handles_zero_spike_neurons_with_nan_bits() -> None:
    counts = np.zeros((1, 2, 2))
    counts[:, :, 1] = 1.0
    predicted = np.ones_like(counts) * 5.0
    reference = np.ones_like(counts) * 5.0

    diagnostics = compute_neuron_prediction_diagnostics(
        counts, predicted, reference, 10, np.array([7, 8]), "validation"
    )

    assert np.isnan(diagnostics.loc[0, "bits_per_spike"])
    assert diagnostics.loc[0, "neuron_index"] == 7
    assert np.isfinite(diagnostics.loc[1, "bits_per_spike"])


def test_invalid_shape_raises_clear_error() -> None:
    counts = np.zeros((1, 2))
    predicted = np.ones((1, 2))
    reference = np.ones((1, 2))

    try:
        compute_loss_scale_diagnostics(counts, predicted, reference, 5)
    except ValueError as exc:
        assert "rank 3" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
