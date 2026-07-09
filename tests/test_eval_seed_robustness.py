from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from latentbrain.eval.seed_robustness import (
    LEADERBOARD_COLUMNS,
    METHOD_SUMMARY_COLUMNS,
    SEED_EFFECT_COLUMNS,
    bootstrap_mean_ci,
    build_seed_robustness_leaderboard,
    paired_seed_differences,
    summarize_method_scores,
    summarize_seed_robustness,
)


def _row(method: str, method_type: str, seed: int, bits: float, **overrides: object) -> dict:
    row = {
        "method_name": method,
        "method_type": method_type,
        "seed": seed,
        "split_seed": 2027,
        "initialization_seed": seed,
        "config_hash": "abc123",
        "valid_model": True,
        "status": "completed",
        "validation_unified_bits_per_spike": bits,
        "validation_poisson_nll": 2000.0,
        "validation_behavior_mean_r2": 0.0,
        "validation_factor_decoder_unified_bits_per_spike": bits,
        "train_unified_bits_per_spike": bits + 0.01,
        "test_unified_bits_per_spike": bits - 0.001,
        "beats_train_mean_reference": bits > 0.0,
        "beats_factor_latent_single_seed_reference": False,
        "beats_neural_ode_refinement_single_seed_reference": False,
        "output_dir": "out",
        "notes": "",
    }
    return row | overrides


def _results(factor: list[float], neural: list[float]) -> pd.DataFrame:
    seeds = [2027, 2028, 2029]
    rows = [
        _row("factor_latent", "factor_latent", s, b) for s, b in zip(seeds, factor, strict=True)
    ]
    rows += [
        _row("neural_ode_refinement", "neural_ode", s, b)
        for s, b in zip(seeds, neural, strict=True)
    ]
    return pd.DataFrame(rows)


def _refs() -> dict[str, float]:
    return {
        "train_mean_validation_bits_per_spike": 0.0,
        "factor_latent_single_seed_reference": 0.0316,
        "neural_ode_refinement_single_seed_reference": 0.0283,
        "neural_ode_objective_single_seed_reference": 0.0115,
        "oracle_validation_bits_per_spike": 3.54,
    }


def test_method_summary_computes_mean_std_min_max() -> None:
    summary = summarize_method_scores(_results([0.03, 0.04, 0.05], [0.01, 0.02, 0.03]))

    assert list(summary.columns) == METHOD_SUMMARY_COLUMNS
    factor = summary[summary["method_name"] == "factor_latent"].iloc[0]
    assert factor["mean_validation_unified_bits_per_spike"] == pytest.approx(0.04)
    assert factor["min_validation_unified_bits_per_spike"] == pytest.approx(0.03)
    assert factor["max_validation_unified_bits_per_spike"] == pytest.approx(0.05)
    assert factor["median_validation_unified_bits_per_spike"] == pytest.approx(0.04)
    assert factor["std_validation_unified_bits_per_spike"] == pytest.approx(0.01)
    assert factor["n_seeds"] == 3


def test_bootstrap_ci_is_deterministic_under_seed() -> None:
    values = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

    first = bootstrap_mean_ci(values, 500, 0.95, 1337)
    second = bootstrap_mean_ci(values, 500, 0.95, 1337)

    assert first == second
    assert first[0] < float(values.mean()) < first[1]


def test_bootstrap_ci_of_a_single_value_is_that_value() -> None:
    assert bootstrap_mean_ci(np.array([0.03]), 100, 0.95, 1337) == (0.03, 0.03)


def test_bootstrap_ci_ignores_non_finite_values() -> None:
    low, high = bootstrap_mean_ci(np.array([0.02, np.nan, 0.02]), 100, 0.95, 1337)

    assert (low, high) == (0.02, 0.02)


def test_bootstrap_ci_rejects_bad_arguments() -> None:
    values = np.array([0.01, 0.02])

    with pytest.raises(ValueError, match="repeats must be positive"):
        bootstrap_mean_ci(values, 0, 0.95, 1)
    with pytest.raises(ValueError, match="confidence must be in"):
        bootstrap_mean_ci(values, 10, 1.0, 1)


def test_paired_seed_differences_are_computed_per_seed() -> None:
    differences = paired_seed_differences(
        _results([0.03, 0.04, 0.05], [0.01, 0.02, 0.03]),
        "neural_ode_refinement",
        "factor_latent",
    )

    assert list(differences.columns) == SEED_EFFECT_COLUMNS
    assert differences["seed"].tolist() == [2027, 2028, 2029]
    assert differences["difference"].tolist() == pytest.approx([-0.02, -0.02, -0.02])


def test_leaderboard_ranks_by_mean_then_ci_lower_bound() -> None:
    summary = pd.DataFrame(
        [
            {
                "method_name": "low_mean",
                "method_type": "neural_ode",
                "valid_model": True,
                "n_seeds": 3,
                "mean_validation_unified_bits_per_spike": 0.01,
                "std_validation_unified_bits_per_spike": 0.001,
                "median_validation_unified_bits_per_spike": 0.01,
                "min_validation_unified_bits_per_spike": 0.009,
                "max_validation_unified_bits_per_spike": 0.011,
                "ci95_low": 0.009,
                "ci95_high": 0.011,
                "mean_validation_poisson_nll": 2000.0,
                "mean_test_unified_bits_per_spike": 0.01,
                "beats_factor_latent_mean": False,
                "beats_factor_latent_lower_ci": False,
                "notes": "",
            },
            {
                "method_name": "high_mean",
                "method_type": "factor_latent",
                "valid_model": True,
                "n_seeds": 3,
                "mean_validation_unified_bits_per_spike": 0.03,
                "std_validation_unified_bits_per_spike": 0.002,
                "median_validation_unified_bits_per_spike": 0.03,
                "min_validation_unified_bits_per_spike": 0.028,
                "max_validation_unified_bits_per_spike": 0.032,
                "ci95_low": 0.028,
                "ci95_high": 0.032,
                "mean_validation_poisson_nll": 1900.0,
                "mean_test_unified_bits_per_spike": 0.03,
                "beats_factor_latent_mean": False,
                "beats_factor_latent_lower_ci": False,
                "notes": "",
            },
        ]
    )

    leaderboard = build_seed_robustness_leaderboard(summary)

    assert list(leaderboard.columns) == LEADERBOARD_COLUMNS
    assert leaderboard.iloc[0]["method_name"] == "high_mean"


def test_leaderboard_breaks_mean_ties_by_ci_lower_bound() -> None:
    base = {
        "method_type": "neural_ode",
        "valid_model": True,
        "n_seeds": 3,
        "mean_validation_unified_bits_per_spike": 0.02,
        "std_validation_unified_bits_per_spike": 0.001,
        "median_validation_unified_bits_per_spike": 0.02,
        "min_validation_unified_bits_per_spike": 0.019,
        "max_validation_unified_bits_per_spike": 0.021,
        "ci95_high": 0.021,
        "mean_validation_poisson_nll": 2000.0,
        "mean_test_unified_bits_per_spike": 0.02,
        "beats_factor_latent_mean": False,
        "beats_factor_latent_lower_ci": False,
        "notes": "",
    }
    summary = pd.DataFrame(
        [
            base | {"method_name": "wide", "ci95_low": 0.010},
            base | {"method_name": "tight", "ci95_low": 0.019},
        ]
    )

    leaderboard = build_seed_robustness_leaderboard(summary)

    assert leaderboard.iloc[0]["method_name"] == "tight"


def test_beating_factor_latent_by_mean_and_lower_ci_is_computed() -> None:
    summary = summarize_method_scores(_results([0.010, 0.010, 0.010], [0.030, 0.031, 0.032]))
    neural = summary[summary["method_name"] == "neural_ode_refinement"].iloc[0]

    assert bool(neural["beats_factor_latent_mean"]) is True
    assert bool(neural["beats_factor_latent_lower_ci"]) is True


def test_mean_win_without_lower_ci_win_is_reported_separately() -> None:
    # Neural mean edges out factor-latent, but its spread drags the CI lower bound below.
    summary = summarize_method_scores(_results([0.020, 0.020, 0.020], [-0.02, 0.02, 0.07]))
    neural = summary[summary["method_name"] == "neural_ode_refinement"].iloc[0]

    assert bool(neural["beats_factor_latent_mean"]) is True
    assert bool(neural["beats_factor_latent_lower_ci"]) is False


def test_failed_runs_are_excluded_from_method_summary() -> None:
    results = _results([0.03, 0.04, 0.05], [0.01, 0.02, 0.03])
    results = pd.concat(
        [
            results,
            pd.DataFrame(
                [
                    _row(
                        "neural_ode_refinement",
                        "neural_ode",
                        2030,
                        9.0,
                        status="failed",
                    )
                ]
            ),
        ],
        ignore_index=True,
    )

    summary = summarize_method_scores(results)
    neural = summary[summary["method_name"] == "neural_ode_refinement"].iloc[0]

    assert neural["n_seeds"] == 3
    assert neural["max_validation_unified_bits_per_spike"] == pytest.approx(0.03)


def test_summary_recommends_factor_latent_when_no_neural_method_wins() -> None:
    results = _results([0.030, 0.031, 0.032], [0.010, 0.011, 0.012])
    summary = summarize_seed_robustness(results, summarize_method_scores(results), _refs())

    assert summary["best_mean_method"] == "factor_latent"
    assert summary["any_neural_beats_factor_latent_mean"] is False
    assert summary["any_neural_beats_factor_latent_lower_ci"] is False
    assert summary["carried_forward_method"] == "factor_latent"
    assert "stop adding architecture" in summary["carried_forward_reason"]
    assert summary["paired_mean_difference_best_neural_minus_factor_latent"] == pytest.approx(-0.02)
    assert summary["single_seed_leaderboards_are_insufficient"] is True


def test_summary_recommends_more_seeds_when_mean_wins_but_ci_does_not() -> None:
    results = _results([0.020, 0.020, 0.020], [-0.02, 0.02, 0.07])
    summary = summarize_seed_robustness(results, summarize_method_scores(results), _refs())

    assert summary["any_neural_beats_factor_latent_mean"] is True
    assert summary["any_neural_beats_factor_latent_lower_ci"] is False
    assert "more seeds" in summary["carried_forward_reason"]


def test_summary_promotes_neural_method_when_lower_ci_wins() -> None:
    results = _results([0.010, 0.010, 0.010], [0.030, 0.031, 0.032])
    summary = summarize_seed_robustness(results, summarize_method_scores(results), _refs())

    assert summary["any_neural_beats_factor_latent_lower_ci"] is True
    assert summary["carried_forward_method"] == "neural_ode_refinement"
    assert "held-out test reporting" in summary["carried_forward_reason"]
