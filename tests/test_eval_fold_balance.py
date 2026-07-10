from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from latentbrain.eval.fold_balance import (
    COMPARISON_COLUMNS,
    FOLD_BALANCE_COLUMNS,
    compare_fold_balance,
    compute_fold_balance_statistics,
    endpoint_direction_entropy,
    summarize_fold_balance,
)


def _assignments(
    fold_count: int = 4,
    trials_per_fold: int = 5,
    rate_offsets: list[float] | None = None,
) -> pd.DataFrame:
    offsets = rate_offsets or [0.0] * fold_count
    rows = []
    trial_index = 0
    for fold_index in range(fold_count):
        for position in range(trials_per_fold):
            angle = -np.pi + (2.0 * np.pi) * (position / trials_per_fold)
            rows.append(
                {
                    "repeat_index": 0,
                    "fold_index": fold_index,
                    "trial_index": trial_index,
                    "stratum": f"s{position % 3}",
                    "total_spikes": 500.0,
                    "population_rate_hz": 0.6 + offsets[fold_index],
                    "heldout_rate_hz": 0.5 + offsets[fold_index],
                    "endpoint_angle_rad": angle,
                    "endpoint_distance": 2.0 + 0.1 * position,
                    "mean_speed": 3.0 + 0.1 * position,
                    "assignment_method": "greedy_balanced",
                    "seed": 2027,
                }
            )
            trial_index += 1
    return pd.DataFrame(rows)


def test_fold_balance_statistics_have_required_columns() -> None:
    balance = compute_fold_balance_statistics(_assignments())

    assert list(balance.columns) == FOLD_BALANCE_COLUMNS
    assert len(balance) == 4
    assert (balance["n_trials"] == 5).all()
    assert (balance["stratum_count"] == 3).all()
    assert np.isfinite(balance["endpoint_direction_entropy"]).all()


def test_empty_assignments_return_empty_statistics() -> None:
    assert compute_fold_balance_statistics(pd.DataFrame()).empty
    assert compare_fold_balance(pd.DataFrame()).empty


def test_endpoint_direction_entropy_is_finite_when_directions_exist() -> None:
    uniform = pd.Series(np.linspace(-np.pi, np.pi, 64, endpoint=False))
    single = pd.Series([0.1, 0.1, 0.1])

    uniform_entropy = endpoint_direction_entropy(uniform)
    single_entropy = endpoint_direction_entropy(single)

    assert np.isfinite(uniform_entropy)
    assert uniform_entropy == pytest.approx(np.log(8.0))
    assert single_entropy == pytest.approx(0.0)


def test_endpoint_direction_entropy_of_missing_behavior_is_nan() -> None:
    assert np.isnan(endpoint_direction_entropy(pd.Series([np.nan, np.nan])))


def test_comparison_computes_ranges_and_coefficients_of_variation() -> None:
    balance = compute_fold_balance_statistics(_assignments(rate_offsets=[0.0, 0.1, 0.2, 0.3]))

    comparisons = compare_fold_balance(balance)

    assert list(comparisons.columns) == COMPARISON_COLUMNS
    row = comparisons[comparisons["metric"] == "mean_population_rate_hz"].iloc[0]
    assert row["min_value"] == pytest.approx(0.6)
    assert row["max_value"] == pytest.approx(0.9)
    assert row["range"] == pytest.approx(0.3)
    assert row["coefficient_of_variation"] == pytest.approx(row["std_value"] / row["mean_value"])
    trials = comparisons[comparisons["metric"] == "n_trials"].iloc[0]
    assert trials["range"] == pytest.approx(0.0)


def test_summary_reports_no_warning_for_balanced_folds() -> None:
    balance = compute_fold_balance_statistics(_assignments())
    summary = summarize_fold_balance(balance, compare_fold_balance(balance))

    assert summary["fold_balance_warning"] == "none"
    assert summary["mean_population_rate_fold_range"] == pytest.approx(0.0)
    assert np.isfinite(summary["mean_endpoint_direction_entropy"])


def test_summary_warns_on_severe_rate_imbalance() -> None:
    balance = compute_fold_balance_statistics(_assignments(rate_offsets=[0.0, 0.3, 0.6, 0.9]))
    summary = summarize_fold_balance(balance, compare_fold_balance(balance))

    assert "distribution shift" in summary["fold_balance_warning"]
    assert "held-out rate" in summary["fold_balance_warning"]


def test_summary_warns_on_imbalanced_trial_counts() -> None:
    assignments = _assignments()
    # Drop most of one fold so the trial counts diverge well past tolerance.
    trimmed = assignments[
        ~((assignments["fold_index"] == 0) & (assignments["trial_index"] % 5 != 0))
    ]
    balance = compute_fold_balance_statistics(trimmed)

    summary = summarize_fold_balance(balance, compare_fold_balance(balance))

    assert "fold trial counts are imbalanced" in summary["fold_balance_warning"]


def test_summary_of_empty_balance_reports_no_folds() -> None:
    summary = summarize_fold_balance(pd.DataFrame(), pd.DataFrame())

    assert summary["fold_balance_warning"] == "no folds were assigned"
    assert np.isnan(summary["mean_population_rate_fold_range"])


def test_summary_flags_concentrated_endpoint_directions() -> None:
    # All reaches in one sector: entropy is 0, far below the ln(8) maximum.
    assignments = _assignments()
    assignments["endpoint_angle_rad"] = 0.1
    balance = compute_fold_balance_statistics(assignments)

    summary = summarize_fold_balance(balance, compare_fold_balance(balance))

    assert summary["endpoint_direction_concentrated"] is True
    assert summary["endpoint_direction_entropy_max"] == pytest.approx(np.log(8.0))
    # Concentration is a dataset property, not a fold-assignment failure.
    assert summary["fold_balance_warning"] == "none"


def test_summary_does_not_flag_uniform_endpoint_directions() -> None:
    balance = compute_fold_balance_statistics(_assignments(trials_per_fold=8))

    summary = summarize_fold_balance(balance, compare_fold_balance(balance))

    assert summary["endpoint_direction_concentrated"] is False
