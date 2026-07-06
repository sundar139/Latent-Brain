from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]
import pytest

from latentbrain.eval.sweeps import expand_grid, rank_sweep_results, select_best_config


def test_expand_grid_returns_expected_count_and_order() -> None:
    grid = {
        "smoothing_sigma_ms": [10.0, 25.0],
        "ridge_alpha": [1.0, 10.0],
        "standardize_features": [True, False],
    }

    configs = expand_grid(grid)

    assert len(configs) == 8
    assert configs[:3] == [
        {"smoothing_sigma_ms": 10.0, "ridge_alpha": 1.0, "standardize_features": True},
        {"smoothing_sigma_ms": 10.0, "ridge_alpha": 1.0, "standardize_features": False},
        {"smoothing_sigma_ms": 10.0, "ridge_alpha": 10.0, "standardize_features": True},
    ]


def test_rank_sweep_results_selects_higher_bits_per_spike() -> None:
    results = pd.DataFrame(
        {
            "run_id": ["a", "b"],
            "split": ["validation", "validation"],
            "bits_per_spike": [0.1, 0.2],
            "poisson_nll": [10.0, 20.0],
            "ridge_alpha": [1.0, 1.0],
            "smoothing_sigma_ms": [10.0, 10.0],
        }
    )

    ranked = rank_sweep_results(results, "validation", "bits_per_spike")

    assert ranked.iloc[0]["run_id"] == "b"
    assert ranked.iloc[0]["rank"] == 1


def test_rank_sweep_results_tie_breaks_by_lower_poisson_nll_alpha_and_sigma() -> None:
    results = pd.DataFrame(
        {
            "run_id": ["nll", "alpha", "sigma", "winner"],
            "split": ["validation"] * 4,
            "bits_per_spike": [0.1, 0.1, 0.1, 0.1],
            "poisson_nll": [9.0, 8.0, 8.0, 8.0],
            "ridge_alpha": [0.1, 10.0, 1.0, 1.0],
            "smoothing_sigma_ms": [10.0, 10.0, 25.0, 10.0],
        }
    )

    ranked = rank_sweep_results(results, "validation", "bits_per_spike")

    assert ranked["run_id"].tolist() == ["winner", "sigma", "alpha", "nll"]
    assert ranked["rank"].tolist() == [1, 2, 3, 4]


def test_rank_sweep_results_missing_primary_metric_raises_clear_error() -> None:
    results = pd.DataFrame({"run_id": ["a"], "split": ["validation"]})

    with pytest.raises(ValueError, match="bits_per_spike"):
        rank_sweep_results(results, "validation", "bits_per_spike")


def test_select_best_config_contains_expected_fields() -> None:
    ranked = pd.DataFrame(
        {
            "rank": [1],
            "run_id": ["run_000"],
            "split": ["validation"],
            "smoothing_sigma_ms": [25.0],
            "ridge_alpha": [10.0],
            "standardize_features": [False],
            "fit_intercept": [True],
            "bits_per_spike": [0.3],
        }
    )

    best = select_best_config(ranked)

    assert best == {
        "run_id": "run_000",
        "smoothing_sigma_ms": 25.0,
        "ridge_alpha": 10.0,
        "standardize_features": False,
        "fit_intercept": True,
    }
