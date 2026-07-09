from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from latentbrain.eval.neural_ode_tuning import (
    LEADERBOARD_COLUMNS,
    RESULT_COLUMNS,
    build_neural_ode_result_row,
    rank_neural_ode_results,
    summarize_neural_ode_tuning,
)


def _scores(bits: float, nll: float = 2.0, factor_bits: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["validation", "validation"],
            "prediction_source": ["direct_model", "factor_decoder"],
            "bits_per_spike": [bits, factor_bits],
            "poisson_nll": [nll, nll + 1.0],
        }
    )


def _checkpoints(source: str = "latest") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "checkpoint_source": ["best_validation", "latest"],
            "validation_total_loss": [1.0, 2.0],
            "validation_unified_bits_per_spike": [0.01, 0.02],
            "validation_poisson_nll": [3.0, 2.0],
            "selected_by_loss": [True, False],
            "selected_by_unified": [source == "best_validation", source == "latest"],
        }
    )


def _refs() -> dict[str, float]:
    return {
        "train_mean_validation_bits_per_spike": 0.0,
        "factor_latent_unified_validation_bits_per_spike": 0.03,
        "previous_neural_sde_validation_bits_per_spike": 0.018,
        "previous_best_lfads_family_validation_bits_per_spike": 0.01,
        "oracle_validation_bits_per_spike": 3.0,
    }


def test_result_row_has_required_columns(tmp_path: Path) -> None:
    row = build_neural_ode_result_row(
        "run_000",
        0,
        {"latent_dim": 32, "factor_dim": 32, "kl_scale": 0.1, "epochs": 2},
        _scores(0.02, 1.0, 0.01),
        {"validation_total_loss": 3.0, "drift_norm": 0.4, "diffusion_mean": 0.0},
        _checkpoints(),
        _refs(),
        tmp_path,
    )

    assert set(RESULT_COLUMNS).issubset(row)
    assert row["validation_unified_bits_per_spike"] == 0.02
    assert row["beats_previous_neural_sde"] is True
    assert row["best_checkpoint_source"] == "latest"


def test_ranking_uses_unified_bits_and_tie_breakers() -> None:
    results = pd.DataFrame(
        [
            {
                "run_id": "a",
                "status": "completed",
                "validation_unified_bits_per_spike": 0.02,
                "validation_poisson_nll": 3.0,
                "validation_behavior_mean_r2": 0.0,
                "kl_scale": 0.05,
                "run_index": 0,
                "validation_factor_decoder_unified_bits_per_spike": 0.0,
                "input_dropout_rate": 0.25,
                "heldout_loss_weight": 4.0,
                "latent_dim": 32,
                "factor_dim": 32,
                "best_checkpoint_source": "best_validation",
                "beats_factor_latent_unified": False,
                "beats_previous_neural_sde": True,
                "notes": "",
            },
            {
                "run_id": "b",
                "status": "completed",
                "validation_unified_bits_per_spike": 0.02,
                "validation_poisson_nll": 2.0,
                "validation_behavior_mean_r2": 0.0,
                "kl_scale": 0.1,
                "run_index": 1,
                "validation_factor_decoder_unified_bits_per_spike": 0.0,
                "input_dropout_rate": 0.25,
                "heldout_loss_weight": 4.0,
                "latent_dim": 32,
                "factor_dim": 32,
                "best_checkpoint_source": "latest",
                "beats_factor_latent_unified": False,
                "beats_previous_neural_sde": True,
                "notes": "",
            },
            {
                "run_id": "failed",
                "status": "failed",
                "validation_unified_bits_per_spike": 9.0,
                "validation_poisson_nll": 1.0,
                "validation_behavior_mean_r2": 0.0,
                "kl_scale": 0.0,
                "run_index": 2,
            },
        ]
    )

    ranked = rank_neural_ode_results(results)

    assert list(ranked.columns) == LEADERBOARD_COLUMNS
    assert ranked.iloc[0]["run_id"] == "b"
    assert "failed" not in set(ranked["run_id"])


def test_summary_reports_best_and_reference_comparisons() -> None:
    results = pd.DataFrame(
        [
            {
                "run_id": "a",
                "status": "completed",
                "validation_unified_bits_per_spike": 0.02,
                "validation_poisson_nll": 2.0,
                "validation_behavior_mean_r2": 0.0,
                "validation_factor_decoder_unified_bits_per_spike": 0.01,
                "run_index": 0,
                "input_dropout_rate": 0.25,
                "heldout_loss_weight": 4.0,
                "kl_scale": 0.1,
                "latent_dim": 32,
                "factor_dim": 32,
                "drift_norm": 0.4,
                "diffusion_mean": 0.0,
                "best_checkpoint_source": "latest",
                "beats_factor_latent_unified": False,
                "beats_previous_neural_sde": True,
                "beats_previous_best_lfads_family": True,
                "notes": "",
            }
        ]
    )

    summary = summarize_neural_ode_tuning(results, _refs())

    assert summary["best_run_id"] == "a"
    assert summary["beats_factor_latent_unified"] is False
    assert summary["beats_previous_neural_sde"] is True
    assert summary["best_checkpoint_source"] == "latest"
