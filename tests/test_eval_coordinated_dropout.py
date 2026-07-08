from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]

from latentbrain.eval.coordinated_dropout import (
    build_dropout_result_row,
    rank_dropout_runs,
    summarize_dropout_runs,
)


def _refs() -> dict[str, float]:
    return {
        "same_bin_mean_rate_validation_bits_per_spike": 0.7,
        "same_bin_factor_latent_validation_bits_per_spike": 0.03,
        "previous_20ms_lfads_validation_bits_per_spike": 0.01,
    }


def test_result_row_includes_reference_flags() -> None:
    row = build_dropout_result_row(
        0.25,
        "dropout_0p25",
        {"validation_bits_per_spike": 0.04, "validation_poisson_nll": 12.0},
        _refs(),
    )

    assert row["beats_previous_20ms_lfads"] is True
    assert row["beats_same_bin_factor_latent"] is True
    assert row["beats_same_bin_mean_rate"] is False


def test_ranking_chooses_highest_validation_bits() -> None:
    ranked = rank_dropout_runs(
        pd.DataFrame(
            {
                "run_id": ["a", "b"],
                "dropout_rate": [0.1, 0.25],
                "status": ["completed", "completed"],
                "validation_bits_per_spike": [0.01, 0.03],
                "validation_poisson_nll": [2.0, 3.0],
            }
        )
    )

    assert ranked.iloc[0]["run_id"] == "b"


def test_summary_identifies_best_run() -> None:
    summary = summarize_dropout_runs(
        pd.DataFrame(
            {
                "run_id": ["a", "b"],
                "dropout_rate": [0.1, 0.25],
                "status": ["completed", "completed"],
                "validation_bits_per_spike": [0.01, 0.04],
                "validation_poisson_nll": [2.0, 3.0],
                "validation_behavior_mean_r2": [0.0, 0.1],
                "validation_factor_decoder_bits_per_spike": [0.0, 0.02],
                "output_dir": ["a", "b"],
            }
        ),
        _refs(),
    )

    assert summary["best_run_id"] == "b"
    assert summary["coordinated_dropout_improves_lfads"] is True


def test_failed_runs_are_handled_safely() -> None:
    summary = summarize_dropout_runs(
        pd.DataFrame(
            {
                "run_id": ["a"],
                "dropout_rate": [0.1],
                "status": ["failed"],
                "validation_bits_per_spike": [float("nan")],
                "validation_poisson_nll": [float("nan")],
            }
        ),
        _refs(),
    )

    assert summary["best_run_id"] is None
    assert summary["successful_runs"] == 0
