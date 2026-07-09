from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from latentbrain.eval.generalization import (
    GAP_COLUMNS,
    GAP_SUMMARY_COLUMNS,
    RISK_HIGH,
    RISK_LOW,
    RISK_MODERATE,
    RISK_UNRESOLVED,
    bootstrap_gap_ci,
    flag_generalization_risk,
    overall_generalization_risk,
    summarize_gap_dictionary,
    summarize_validation_test_gap,
    validation_test_gap_table,
)


def _results(
    validation: list[float], test: list[float], method: str = "factor_latent"
) -> pd.DataFrame:
    seeds = list(range(2027, 2027 + len(validation)))
    return pd.DataFrame(
        {
            "method_name": [method] * len(seeds),
            "seed": seeds,
            "status": ["completed"] * len(seeds),
            "validation_unified_bits_per_spike": validation,
            "test_unified_bits_per_spike": test,
        }
    )


def test_gap_table_computes_validation_minus_test() -> None:
    table = validation_test_gap_table(_results([0.03, 0.02], [-0.01, 0.00]))

    assert list(table.columns) == GAP_COLUMNS
    assert table["gap_validation_minus_test"].tolist() == pytest.approx([0.04, 0.02])


def test_gap_table_excludes_failed_runs() -> None:
    results = _results([0.03, 0.02], [-0.01, 0.0])
    results.loc[1, "status"] = "failed"

    table = validation_test_gap_table(results)

    assert len(table) == 1
    assert table.iloc[0]["seed"] == 2027


def test_gap_table_of_empty_results_is_empty() -> None:
    assert validation_test_gap_table(pd.DataFrame()).empty


def test_bootstrap_gap_ci_is_deterministic_under_seed() -> None:
    a = np.array([0.03, 0.028, 0.031, 0.029, 0.03])
    b = np.array([-0.01, -0.008, -0.009, -0.007, -0.011])

    first = bootstrap_gap_ci(a, b, 500, 0.95, 1337)
    second = bootstrap_gap_ci(a, b, 500, 0.95, 1337)

    assert first == second
    assert first[0] < float(np.mean(a - b)) < first[1]


def test_bootstrap_gap_ci_rejects_bad_arguments() -> None:
    a = np.array([0.01, 0.02])
    b = np.array([0.0, 0.0])

    with pytest.raises(ValueError, match="same shape"):
        bootstrap_gap_ci(a, np.array([0.0]), 10, 0.95, 1)
    with pytest.raises(ValueError, match="repeats must be positive"):
        bootstrap_gap_ci(a, b, 0, 0.95, 1)
    with pytest.raises(ValueError, match="confidence must be in"):
        bootstrap_gap_ci(a, b, 10, 1.0, 1)


def test_high_risk_when_validation_positive_and_test_negative() -> None:
    assert flag_generalization_risk(0.03, -0.008, 0.02, 0.05) == RISK_HIGH


def test_moderate_risk_when_gap_ci_excludes_zero_but_test_is_positive() -> None:
    assert flag_generalization_risk(0.03, 0.01, 0.005, 0.03) == RISK_MODERATE


def test_low_risk_when_gap_ci_contains_zero_and_test_positive() -> None:
    assert flag_generalization_risk(0.03, 0.029, -0.005, 0.006) == RISK_LOW


def test_unresolved_risk_when_values_are_missing() -> None:
    assert flag_generalization_risk(float("nan"), 0.01, 0.0, 0.1) == RISK_UNRESOLVED
    assert flag_generalization_risk(0.03, -0.01, float("nan"), float("nan")) == RISK_UNRESOLVED


def test_summary_reports_required_columns_and_high_risk() -> None:
    table = validation_test_gap_table(_results([0.030, 0.028, 0.031], [-0.008, -0.009, -0.007]))

    summary = summarize_validation_test_gap(table, 500, 0.95, 1337)

    assert list(summary.columns) == GAP_SUMMARY_COLUMNS
    row = summary.iloc[0]
    assert row["generalization_risk"] == RISK_HIGH
    assert row["test_positive_fraction"] == 0.0
    assert row["mean_gap"] == pytest.approx(row["mean_validation"] - row["mean_test"])


def test_summary_reports_test_positive_fraction() -> None:
    table = validation_test_gap_table(_results([0.03, 0.03, 0.03], [0.01, -0.01, 0.02]))

    summary = summarize_validation_test_gap(table, 500, 0.95, 1337)

    assert summary.iloc[0]["test_positive_fraction"] == pytest.approx(2.0 / 3.0)


def test_overall_risk_takes_the_worst_method() -> None:
    summary = pd.DataFrame(
        {
            "method_name": ["a", "b"],
            "mean_validation": [0.03, 0.03],
            "mean_test": [0.02, -0.01],
            "mean_gap": [0.01, 0.04],
            "gap_ci95_low": [-0.01, 0.02],
            "gap_ci95_high": [0.03, 0.06],
            "test_positive_fraction": [1.0, 0.0],
            "generalization_risk": [RISK_LOW, RISK_HIGH],
        }
    )

    assert overall_generalization_risk(summary) == RISK_HIGH


def test_missing_robustness_data_returns_unresolved_risk() -> None:
    empty = summarize_validation_test_gap(pd.DataFrame(columns=GAP_COLUMNS))

    assert empty.empty
    assert overall_generalization_risk(empty) == RISK_UNRESOLVED
    fields = summarize_gap_dictionary(empty)
    assert fields["generalization_risk"] == RISK_UNRESOLVED
    assert fields["model_gap_diagnostics_available"] is False
    assert fields["validation_test_instability_detected"] is False


def test_gap_dictionary_lists_methods_with_negative_test_mean() -> None:
    table = validation_test_gap_table(_results([0.03, 0.03], [-0.01, -0.01]))
    fields = summarize_gap_dictionary(summarize_validation_test_gap(table, 200, 0.95, 1337))

    assert fields["generalization_risk"] == RISK_HIGH
    assert fields["validation_test_instability_detected"] is True
    assert fields["methods_with_negative_test_mean"] == ["factor_latent"]
