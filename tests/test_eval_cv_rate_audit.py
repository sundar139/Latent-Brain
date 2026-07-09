from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from latentbrain.eval.cv_rate_audit import (
    DECOMPOSITION_COLUMNS,
    METHOD_SUMMARY_COLUMNS,
    RATE_CONTROL_COLUMNS,
    REPEATED_SPLIT_COLUMNS,
    build_reporting_recommendations,
    decompose_rate_offset,
    run_factor_analysis_random_state_sensitivity,
    summarize_cv_rate_audit,
    summarize_methods,
)
from latentbrain.eval.rate_controls import (
    FACTOR_LATENT,
    ORACLE_SPLIT_SCALED_FACTOR_LATENT_INVALID,
    SPLIT_MEAN_RATE_INVALID,
    TRAIN_MEAN_RATE,
    TRAIN_RATE_CALIBRATED_FACTOR_LATENT,
)


def _repeated(validation: list[float], test: list[float], states: list[int]) -> pd.DataFrame:
    rows = []
    for state, val, tst in zip(states, validation, test, strict=True):
        rows.append(
            {
                "split_seed": 2027,
                "factor_analysis_random_state": state,
                "method_name": FACTOR_LATENT,
                "valid_model": True,
                "validation_unified_bits_per_spike": val,
                "test_unified_bits_per_spike": tst,
                "validation_poisson_nll": 2000.0,
                "test_poisson_nll": 2100.0,
                "validation_heldout_rate_hz": 0.59,
                "test_heldout_rate_hz": 0.57,
                "validation_trial_count": 15,
                "test_trial_count": 15,
                "notes": "",
            }
        )
    return pd.DataFrame(rows, columns=REPEATED_SPLIT_COLUMNS)


def _rate_control_row(
    split_seed: int, split: str, method: str, valid: bool, bits: float
) -> dict[str, object]:
    return {
        "split_seed": split_seed,
        "split": split,
        "method_name": method,
        "valid_model": valid,
        "invalid_reason": "" if valid else "leaks evaluation targets",
        "unified_bits_per_spike": bits,
        "poisson_nll": 2000.0,
        "heldout_rate_hz": 0.57,
        "predicted_rate_hz": 0.55,
        "rate_error_hz": -0.02,
        "notes": "note",
    }


def _rate_controls() -> pd.DataFrame:
    rows = []
    for seed, factor, calibrated, split_mean, oracle in (
        (2027, 0.01, 0.012, 0.09, 0.05),
        (2028, 0.02, 0.021, 0.10, 0.06),
    ):
        for split in ("validation", "test"):
            rows.append(_rate_control_row(seed, split, FACTOR_LATENT, True, factor))
            rows.append(
                _rate_control_row(
                    seed, split, TRAIN_RATE_CALIBRATED_FACTOR_LATENT, True, calibrated
                )
            )
            rows.append(_rate_control_row(seed, split, TRAIN_MEAN_RATE, True, 0.0))
            rows.append(_rate_control_row(seed, split, SPLIT_MEAN_RATE_INVALID, False, split_mean))
            rows.append(
                _rate_control_row(
                    seed, split, ORACLE_SPLIT_SCALED_FACTOR_LATENT_INVALID, False, oracle
                )
            )
    return pd.DataFrame(rows, columns=RATE_CONTROL_COLUMNS)


def test_repeated_split_scores_have_required_columns() -> None:
    frame = _repeated([0.03, 0.02, 0.04], [0.01, -0.01, 0.02], [0, 2027, 2028])

    assert list(frame.columns) == REPEATED_SPLIT_COLUMNS


def test_factor_analysis_sensitivity_computes_differences_from_random_state_zero() -> None:
    repeated = _repeated([0.030, 0.025, 0.040], [0.010, -0.005, 0.020], [0, 2027, 2028])

    sensitivity = run_factor_analysis_random_state_sensitivity(repeated, 2027)

    assert sensitivity.iloc[0]["factor_analysis_random_state"] == 0
    assert sensitivity.iloc[0]["difference_from_random_state_0_validation"] == pytest.approx(0.0)
    assert sensitivity.iloc[1]["difference_from_random_state_0_validation"] == pytest.approx(-0.005)
    assert sensitivity.iloc[2]["difference_from_random_state_0_test"] == pytest.approx(0.010)


def test_factor_analysis_sensitivity_of_unknown_split_seed_is_empty() -> None:
    repeated = _repeated([0.03], [0.01], [0])

    assert run_factor_analysis_random_state_sensitivity(repeated, 9999).empty


def test_rate_offset_decomposition_computes_valid_and_invalid_gains() -> None:
    decomposition = decompose_rate_offset(_rate_controls())

    assert list(decomposition.columns) == DECOMPOSITION_COLUMNS
    row = decomposition[
        (decomposition["split_seed"] == 2027) & (decomposition["split"] == "test")
    ].iloc[0]
    assert row["valid_calibration_gain"] == pytest.approx(0.002)
    assert row["invalid_oracle_gain"] == pytest.approx(0.04)
    assert row["split_mean_advantage_over_factor_latent"] == pytest.approx(0.08)
    # 0.04 >= 0.5 * 0.08, so pure rescaling explains at least half the invalid advantage.
    assert bool(row["rate_offset_explains_gap"]) is True


def test_rate_offset_does_not_explain_gap_when_rescaling_recovers_little() -> None:
    controls = _rate_controls()
    mask = controls["method_name"] == ORACLE_SPLIT_SCALED_FACTOR_LATENT_INVALID
    controls.loc[mask, "unified_bits_per_spike"] = 0.011

    decomposition = decompose_rate_offset(controls)
    row = decomposition[decomposition["split_seed"] == 2027].iloc[0]

    assert bool(row["rate_offset_explains_gap"]) is False


def test_method_summary_keeps_invalid_flag_and_is_deterministic() -> None:
    summary = summarize_methods(_rate_controls(), 200, 0.95, 1337)
    again = summarize_methods(_rate_controls(), 200, 0.95, 1337)

    assert list(summary.columns) == METHOD_SUMMARY_COLUMNS
    assert summary.equals(again)
    invalid = summary[summary["method_name"] == SPLIT_MEAN_RATE_INVALID].iloc[0]
    assert bool(invalid["valid_model"]) is False
    train_mean = summary[summary["method_name"] == TRAIN_MEAN_RATE].iloc[0]
    assert train_mean["mean_test_unified_bits_per_spike"] == pytest.approx(0.0)


def test_summary_excludes_invalid_controls_from_best_valid_model() -> None:
    controls = _rate_controls()
    summary = summarize_cv_rate_audit(
        _repeated([0.03, 0.02], [0.01, -0.01], [0, 2027]),
        run_factor_analysis_random_state_sensitivity(
            _repeated([0.03, 0.02], [0.01, -0.01], [0, 2027]), 2027
        ),
        controls,
        decompose_rate_offset(controls),
        summarize_methods(controls, 200, 0.95, 1337),
        {"accepted_split_seed": 2027},
    )

    assert summary["best_valid_rate_control_method"] != SPLIT_MEAN_RATE_INVALID
    assert summary["best_valid_rate_control_method"] != ORACLE_SPLIT_SCALED_FACTOR_LATENT_INVALID
    assert summary["invalid_controls_excluded_from_best_valid_model"] is True
    assert summary["invalid_controls_dominate_valid_models"] is True
    assert set(summary["invalid_control_methods"]) == {
        SPLIT_MEAN_RATE_INVALID,
        ORACLE_SPLIT_SCALED_FACTOR_LATENT_INVALID,
    }


def test_summary_reports_calibration_help_and_split_mean_advantage() -> None:
    controls = _rate_controls()
    summary = summarize_cv_rate_audit(
        _repeated([0.03, 0.02], [0.01, -0.01], [0, 2027]),
        pd.DataFrame(),
        controls,
        decompose_rate_offset(controls),
        summarize_methods(controls, 200, 0.95, 1337),
        {"accepted_split_seed": 2027},
    )

    assert summary["train_only_rate_calibration_helps"] is True
    assert summary["invalid_split_mean_advantage_over_factor_latent"] == pytest.approx(0.08)
    assert summary["single_split_results_reportable"] is False
    assert summary["recommended_reporting_mode"] == "repeated_split"
    assert summary["carried_forward_for_reporting"] == FACTOR_LATENT


def test_summary_separates_split_variance_from_random_state_variance() -> None:
    rows = []
    for split_seed, base in ((2027, 0.00), (2028, 0.05)):
        for state, jitter in ((0, 0.0), (2027, 0.001)):
            rows.append(
                {
                    "split_seed": split_seed,
                    "factor_analysis_random_state": state,
                    "method_name": FACTOR_LATENT,
                    "valid_model": True,
                    "validation_unified_bits_per_spike": base + jitter,
                    "test_unified_bits_per_spike": base + jitter,
                    "validation_poisson_nll": 1.0,
                    "test_poisson_nll": 1.0,
                    "validation_heldout_rate_hz": 0.5,
                    "test_heldout_rate_hz": 0.5,
                    "validation_trial_count": 15,
                    "test_trial_count": 15,
                    "notes": "",
                }
            )
    repeated = pd.DataFrame(rows, columns=REPEATED_SPLIT_COLUMNS)
    controls = _rate_controls()

    summary = summarize_cv_rate_audit(
        repeated,
        pd.DataFrame(),
        controls,
        decompose_rate_offset(controls),
        summarize_methods(controls, 200, 0.95, 1337),
        {"accepted_split_seed": 2027},
    )

    assert (
        summary["between_split_test_variance"] > summary["within_split_random_state_test_variance"]
    )
    assert summary["split_variance_exceeds_random_state_variance"] is True


def test_reporting_recommendation_is_repeated_split_and_flags_rate_offset() -> None:
    controls = _rate_controls()
    summary = summarize_cv_rate_audit(
        _repeated([0.03, 0.02], [0.01, -0.01], [0, 2027]),
        pd.DataFrame(),
        controls,
        decompose_rate_offset(controls),
        summarize_methods(controls, 200, 0.95, 1337),
        {"accepted_split_seed": 2027},
    )

    recommendations = build_reporting_recommendations(summary)

    assert recommendations["single_split_results_reportable"] is False
    assert recommendations["recommended_reporting_mode"] == "repeated_split"
    assert recommendations["neural_models_carried_forward"] is False
    assert recommendations["carried_forward_for_reporting"] == FACTOR_LATENT
    assert set(recommendations["must_label_invalid"]) == {
        SPLIT_MEAN_RATE_INVALID,
        ORACLE_SPLIT_SCALED_FACTOR_LATENT_INVALID,
    }
    assert "split-level rate offset is modeled" in recommendations["rate_offset_warning"]
    assert "beats every valid model" in recommendations["rate_offset_warning"]


def test_bootstrap_ci_in_method_summary_is_deterministic() -> None:
    first = summarize_methods(_rate_controls(), 300, 0.95, 7)
    second = summarize_methods(_rate_controls(), 300, 0.95, 7)

    assert np.allclose(first["ci95_low_test"], second["ci95_low_test"])
    assert np.allclose(first["ci95_high_test"], second["ci95_high_test"])


def test_empty_inputs_return_empty_frames() -> None:
    assert decompose_rate_offset(pd.DataFrame()).empty
    assert summarize_methods(pd.DataFrame()).empty
    assert run_factor_analysis_random_state_sensitivity(pd.DataFrame(), 2027).empty
