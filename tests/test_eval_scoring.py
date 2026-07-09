from __future__ import annotations

import math

import numpy as np
import pytest

from latentbrain.eval.scoring import (
    REQUIRED_HELDOUT_SCORE_COLUMNS,
    ScoringConfig,
    canonical_bits_per_spike,
    score_heldout_prediction,
    train_heldout_mean_rate_reference,
)


def _config() -> ScoringConfig:
    return ScoringConfig(
        bin_size_ms=20,
        include_poisson_constant=True,
        min_rate_hz=1.0e-4,
        max_rate_hz=500.0,
    )


def test_scoring_config_stores_parameters() -> None:
    config = _config()

    assert config.bin_size_ms == 20
    assert config.include_poisson_constant is True
    assert config.reference_name == "train_heldout_mean_rate"


def test_reference_as_model_gives_zero_bits_per_spike() -> None:
    counts = np.array([[[0.0, 1.0], [2.0, 0.0]]])
    rates = np.full_like(counts, 5.0)

    row = score_heldout_prediction(
        counts, rates, rates, _config(), "same", "validation", "unit", True
    )

    assert row["bits_per_spike"] == 0.0
    assert set(REQUIRED_HELDOUT_SCORE_COLUMNS).issubset(row)


def test_better_model_scores_positive() -> None:
    counts = np.array([[[3.0], [2.0], [3.0]]])
    row = score_heldout_prediction(
        counts,
        np.full_like(counts, 120.0),
        np.full_like(counts, 1.0),
        _config(),
        "better",
        "validation",
        "unit",
        True,
    )

    assert row["bits_per_spike"] > 0.0


def test_worse_model_scores_negative() -> None:
    counts = np.array([[[3.0], [2.0], [3.0]]])
    row = score_heldout_prediction(
        counts,
        np.full_like(counts, 1.0),
        np.full_like(counts, 120.0),
        _config(),
        "worse",
        "validation",
        "unit",
        True,
    )

    assert row["bits_per_spike"] < 0.0


def test_zero_spike_count_gives_nan() -> None:
    assert math.isnan(canonical_bits_per_spike(-1.0, -1.0, 0.0))


def test_shape_mismatch_raises_clear_error() -> None:
    counts = np.zeros((1, 2, 1))

    with pytest.raises(ValueError, match="same shape"):
        score_heldout_prediction(
            counts,
            np.ones((1, 2, 2)),
            np.ones_like(counts),
            _config(),
            "bad",
            "validation",
            "unit",
            True,
        )


def test_clipping_makes_rates_positive_and_finite() -> None:
    counts = np.array([[[0.0], [1.0]]])
    row = score_heldout_prediction(
        counts,
        np.array([[[0.0], [-5.0]]]),
        np.array([[[0.0], [0.0]]]),
        _config(),
        "clipped",
        "validation",
        "unit",
        True,
    )

    assert math.isfinite(row["model_log_likelihood"])
    assert row["mean_predicted_rate_hz"] == _config().min_rate_hz


def test_train_heldout_reference_matches_target_shape() -> None:
    reference = train_heldout_mean_rate_reference(np.ones((2, 3, 4)), (5, 7, 4), _config())

    assert reference.shape == (5, 7, 4)
    assert np.all(reference > 0.0)
