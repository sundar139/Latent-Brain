from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from latentbrain.eval.rate_controls import (
    FACTOR_LATENT,
    INVALID_CONTROLS,
    ORACLE_SPLIT_SCALED_FACTOR_LATENT_INVALID,
    SPLIT_MEAN_RATE_INVALID,
    TRAIN_MEAN_RATE,
    TRAIN_PER_NEURON_MEAN_RATE,
    VALID_CONTROLS,
    apply_rate_calibration,
    compute_oracle_split_scaled_factor_latent_invalid_control,
    compute_split_mean_rate_invalid_control,
    compute_train_mean_rate_control,
    compute_train_per_neuron_mean_rate_control,
    compute_train_population_scaled_mean_rate_control,
    compute_train_rate_calibration,
    invalid_reason,
    is_valid_control,
    select_best_valid_method,
)
from latentbrain.eval.scoring import (
    ScoringConfig,
    score_heldout_prediction,
    train_heldout_mean_rate_reference,
)


def _scoring() -> ScoringConfig:
    return ScoringConfig(
        bin_size_ms=20, include_poisson_constant=True, min_rate_hz=1.0e-4, max_rate_hz=500.0
    )


def _counts(trials: int = 6, time: int = 8, neurons: int = 4, seed: int = 0) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.poisson(0.4, size=(trials, time, neurons)).astype(np.float64)


def test_train_mean_rate_control_scores_exactly_zero() -> None:
    scoring = _scoring()
    train = _counts(seed=1)
    evaluation = _counts(seed=2)
    control = compute_train_mean_rate_control(train, evaluation.shape, scoring)
    reference = train_heldout_mean_rate_reference(train, evaluation.shape, scoring)

    scored = score_heldout_prediction(
        evaluation,
        control["predicted_rates_hz"],
        reference,
        scoring,
        TRAIN_MEAN_RATE,
        "test",
        "c",
        True,
    )

    assert scored["bits_per_spike"] == pytest.approx(0.0, abs=1e-12)
    assert control["valid_model"] is True


def test_train_per_neuron_mean_rate_equals_the_canonical_reference() -> None:
    """The canonical reference is the per-neuron train mean, so this control is degenerate."""
    scoring = _scoring()
    train = _counts(seed=1)
    evaluation = _counts(seed=2)

    control = compute_train_per_neuron_mean_rate_control(train, evaluation.shape, scoring)
    reference = train_heldout_mean_rate_reference(train, evaluation.shape, scoring)

    assert np.allclose(control["predicted_rates_hz"], reference)
    scored = score_heldout_prediction(
        evaluation,
        control["predicted_rates_hz"],
        reference,
        scoring,
        TRAIN_PER_NEURON_MEAN_RATE,
        "test",
        "c",
        True,
    )
    assert scored["bits_per_spike"] == pytest.approx(0.0, abs=1e-12)
    assert "degeneracy" in control["notes"]


def test_split_mean_rate_control_is_marked_invalid() -> None:
    control = compute_split_mean_rate_invalid_control(_counts(), _scoring())

    assert control["valid_model"] is False
    assert control["invalid_reason"]
    assert "leaks evaluation targets" in control["invalid_reason"]
    assert is_valid_control(SPLIT_MEAN_RATE_INVALID) is False


def test_oracle_split_scaled_control_is_marked_invalid() -> None:
    scoring = _scoring()
    evaluation = _counts(seed=3)
    predicted = np.full_like(evaluation, 0.5)

    control = compute_oracle_split_scaled_factor_latent_invalid_control(
        predicted, evaluation, scoring
    )

    assert control["valid_model"] is False
    assert "leaks evaluation targets" in control["invalid_reason"]
    assert is_valid_control(ORACLE_SPLIT_SCALED_FACTOR_LATENT_INVALID) is False
    # After oracle rescaling the mean predicted rate matches the split's own observed rate.
    seconds_per_bin = scoring.bin_size_ms / 1000.0
    observed = float(evaluation.sum()) / (evaluation.size * seconds_per_bin)
    assert float(control["predicted_rates_hz"].mean()) == pytest.approx(observed, rel=1e-9)


def test_oracle_reason_is_reported_for_invalid_methods_only() -> None:
    assert invalid_reason(SPLIT_MEAN_RATE_INVALID)
    assert invalid_reason(ORACLE_SPLIT_SCALED_FACTOR_LATENT_INVALID)
    assert invalid_reason(FACTOR_LATENT) == ""
    assert set(INVALID_CONTROLS).isdisjoint(VALID_CONTROLS)


def test_train_calibration_does_not_use_evaluation_counts() -> None:
    scoring = _scoring()
    train_counts = _counts(seed=1)
    train_predicted = np.full_like(train_counts, 0.5)

    calibration = compute_train_rate_calibration(train_counts, train_predicted, scoring)

    assert calibration["fit_on"] == "train_heldout_counts_only"
    seconds_per_bin = scoring.bin_size_ms / 1000.0
    expected_observed = float(train_counts.sum()) / (train_counts.size * seconds_per_bin)
    assert calibration["train_observed_rate_hz"] == pytest.approx(expected_observed)
    assert calibration["scale"] == pytest.approx(expected_observed / 0.5)

    # Changing only the evaluation counts must not change the calibration.
    other = compute_train_rate_calibration(train_counts, train_predicted, scoring)
    assert other["scale"] == calibration["scale"]


def test_calibration_rejects_shape_mismatch_and_unknown_method() -> None:
    scoring = _scoring()
    with pytest.raises(ValueError, match="same shape"):
        compute_train_rate_calibration(_counts(), np.zeros((1, 1, 1)), scoring)
    with pytest.raises(ValueError, match="calibration method"):
        compute_train_rate_calibration(_counts(), np.full(_counts().shape, 0.5), scoring, "bogus")


def test_multiplicative_and_log_bias_calibration_agree() -> None:
    scoring = _scoring()
    rates = np.full((2, 3, 4), 0.7)
    calibration = {"method": "multiplicative", "scale": 1.6, "log_bias": float(np.log(1.6))}
    log_calibration = dict(calibration) | {"method": "log_bias"}

    multiplicative = apply_rate_calibration(rates, calibration, scoring)
    log_bias = apply_rate_calibration(rates, log_calibration, scoring)

    assert np.allclose(multiplicative, log_bias)
    assert np.allclose(multiplicative, rates * 1.6)


def test_apply_calibration_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="calibration method"):
        apply_rate_calibration(np.ones((1, 1, 1)), {"method": "bogus"}, _scoring())


def test_population_scaled_control_uses_heldin_spikes_only() -> None:
    scoring = _scoring()
    train_heldout = _counts(seed=1)
    train_heldin = _counts(seed=2)
    split_heldin = train_heldin * 2.0

    control = compute_train_population_scaled_mean_rate_control(
        train_heldout, train_heldin, split_heldin, train_heldout.shape, scoring
    )

    assert control["valid_model"] is True
    assert control["population_scale"] == pytest.approx(2.0)


def test_invalid_controls_are_excluded_from_best_valid_model_selection() -> None:
    scores = pd.DataFrame(
        [
            {"method_name": FACTOR_LATENT, "valid_model": True, "unified_bits_per_spike": 0.01},
            {
                "method_name": SPLIT_MEAN_RATE_INVALID,
                "valid_model": False,
                "unified_bits_per_spike": 0.09,
            },
            {
                "method_name": ORACLE_SPLIT_SCALED_FACTOR_LATENT_INVALID,
                "valid_model": False,
                "unified_bits_per_spike": 0.08,
            },
        ]
    )

    assert select_best_valid_method(scores) == FACTOR_LATENT


def test_best_valid_method_is_none_when_no_valid_rows_exist() -> None:
    scores = pd.DataFrame(
        [
            {
                "method_name": SPLIT_MEAN_RATE_INVALID,
                "valid_model": False,
                "unified_bits_per_spike": 1.0,
            }
        ]
    )

    assert select_best_valid_method(scores) is None
    assert select_best_valid_method(pd.DataFrame()) is None
