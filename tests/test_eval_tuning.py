from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]

from latentbrain.eval.tuning import (
    expand_tuning_grid,
    make_run_id,
    rank_tuning_results,
    summarize_tuning_results,
)


def _results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": ["run_000_a", "run_001_b", "run_002_c"],
            "run_index": [0, 1, 2],
            "status": ["completed", "completed", "failed"],
            "validation_bits_per_spike": [0.1, 0.2, 9.0],
            "validation_poisson_nll": [5.0, 6.0, 1.0],
            "validation_behavior_mean_r2": [0.0, 0.1, 1.0],
            "parameter_count_estimate": [100, 200, 1],
            "beats_window_matched_mean_rate": [False, False, True],
            "beats_window_matched_factor_latent": [True, True, True],
            "beats_previous_lfads_masked_direct": [True, True, True],
        }
    )


def test_grid_expansion_is_deterministic_and_truncates_by_max_runs() -> None:
    grid = {"encoder_hidden_dim": [64, 96], "latent_dim": [16, 32], "dropout": [0.0]}

    expanded = expand_tuning_grid(grid)

    assert expanded[:3] == [
        {"encoder_hidden_dim": 64, "latent_dim": 16, "dropout": 0.0},
        {"encoder_hidden_dim": 64, "latent_dim": 32, "dropout": 0.0},
        {"encoder_hidden_dim": 96, "latent_dim": 16, "dropout": 0.0},
    ]
    assert expanded[:2] == expanded[0:2]


def test_run_ids_are_stable_and_include_index() -> None:
    run_id = make_run_id(3, {"encoder_hidden_dim": 64, "latent_dim": 16, "dropout": 0.1})

    assert run_id == "run_003_enc64_genna_lat16_facna_drop0p1_hwna"


def test_ranking_chooses_higher_bits_per_spike() -> None:
    ranked = rank_tuning_results(_results(), "validation_bits_per_spike", "max")

    assert ranked.iloc[0]["run_id"] == "run_001_b"
    assert ranked.iloc[0]["rank"] == 1


def test_ranking_tie_breaks_by_lower_poisson_nll() -> None:
    results = pd.DataFrame(
        {
            "run_id": ["worse", "better"],
            "run_index": [0, 1],
            "status": ["completed", "completed"],
            "validation_bits_per_spike": [0.2, 0.2],
            "validation_poisson_nll": [6.0, 5.0],
            "validation_behavior_mean_r2": [0.9, 0.1],
            "parameter_count_estimate": [1, 100],
        }
    )

    ranked = rank_tuning_results(results, "validation_bits_per_spike", "max")

    assert ranked["run_id"].tolist() == ["better", "worse"]


def test_summary_computes_baseline_comparisons_and_excludes_failed_best() -> None:
    summary = summarize_tuning_results(
        _results(),
        {
            "window_matched_mean_rate_validation_bits_per_spike": 0.15,
            "window_matched_factor_latent_validation_bits_per_spike": 0.03,
            "previous_lfads_masked_direct_validation_bits_per_spike": -0.04,
        },
        "validation_bits_per_spike",
    )

    assert summary["runs_attempted"] == 3
    assert summary["successful_runs"] == 2
    assert summary["best_run_id"] == "run_001_b"
    assert summary["best_validation_bits_per_spike"] == 0.2
    assert summary["beats_window_matched_mean_rate"] is True
    assert summary["beats_window_matched_factor_latent"] is True
    assert summary["beats_previous_lfads_masked_direct"] is True
