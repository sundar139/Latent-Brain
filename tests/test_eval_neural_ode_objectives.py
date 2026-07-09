from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from latentbrain.eval.neural_ode_objectives import (
    LEADERBOARD_COLUMNS,
    RESULT_COLUMNS,
    build_neural_ode_objective_diagnostics,
    build_neural_ode_objective_result_row,
    rank_neural_ode_objective_results,
    summarize_neural_ode_objectives,
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
        "previous_neural_ode_refinement_validation_bits_per_spike": 0.018,
        "switching_ode_validation_bits_per_spike": 0.006,
        "oracle_validation_bits_per_spike": 3.0,
    }


def _variant(name: str = "heldout_heavy") -> dict[str, object]:
    return {
        "name": name,
        "heldin_loss_weight": 0.5,
        "heldout_loss_weight": 10.0,
        "zero_count_weight": 1.0,
        "positive_count_weight": 1.0,
        "rate_calibration_loss_weight": 0.0,
        "kl_warmup_epochs": 10,
        "kl_scale": 0.01,
        "drift_regularization": 1.0e-4,
        "scheduler": "cosine",
        "input_dropout_rate": 0.1,
        "epochs": 50,
        "notes": "held-out emphasis",
    }


def _training_metrics() -> dict[str, object]:
    return {
        "validation_total_loss": 3.0,
        "validation_heldout_prediction_loss": 1.5,
        "z0_kl_loss": 0.2,
        "drift_norm": 0.4,
        "diffusion_mean": 0.0,
        "drift_regularization_loss": 0.001,
        "rate_calibration_loss": 0.0,
        "mean_predicted_rate": 0.56,
        "mean_observed_rate": 0.58,
    }


def _result_row(**overrides: object) -> dict[str, object]:
    row = _variant() | {
        "run_id": "a",
        "run_index": 0,
        "status": "completed",
        "objective_name": "heldout_heavy",
        "validation_unified_bits_per_spike": 0.02,
        "validation_poisson_nll": 2.0,
        "validation_behavior_mean_r2": 0.0,
        "validation_factor_decoder_unified_bits_per_spike": 0.01,
        "validation_heldout_prediction_loss": 1.5,
        "z0_kl_loss": 0.2,
        "drift_norm": 0.4,
        "diffusion_mean": 0.0,
        "drift_regularization_loss": 0.001,
        "rate_calibration_loss": 0.0,
        "best_checkpoint_source": "latest",
        "beats_factor_latent_unified": False,
        "beats_previous_neural_ode_refinement": True,
        "beats_switching_ode": True,
        "notes": "",
    }
    return row | overrides


def test_result_row_has_required_columns(tmp_path: Path) -> None:
    row = build_neural_ode_objective_result_row(
        "run_000_heldout_heavy",
        0,
        _variant(),
        _scores(0.02, 1.0, 0.01),
        _training_metrics(),
        _checkpoints(),
        _refs(),
        tmp_path,
    )

    assert set(RESULT_COLUMNS).issubset(row)
    assert row["objective_name"] == "heldout_heavy"
    assert row["validation_unified_bits_per_spike"] == 0.02
    assert row["beats_previous_neural_ode_refinement"] is True
    assert row["beats_factor_latent_unified"] is False
    assert row["beats_switching_ode"] is True
    assert row["best_checkpoint_source"] == "latest"


def test_ranking_uses_unified_bits_then_poisson_nll() -> None:
    results = pd.DataFrame(
        [
            _result_row(run_id="a", run_index=0, validation_poisson_nll=3.0),
            _result_row(run_id="b", run_index=1, validation_poisson_nll=2.0),
            _result_row(
                run_id="failed",
                run_index=2,
                status="failed",
                validation_unified_bits_per_spike=9.0,
                validation_poisson_nll=1.0,
            ),
        ]
    )

    ranked = rank_neural_ode_objective_results(results)

    assert list(ranked.columns) == LEADERBOARD_COLUMNS
    assert ranked.iloc[0]["run_id"] == "b"
    assert "failed" not in set(ranked["run_id"])


def test_ranking_prefers_simpler_objective_on_exact_tie() -> None:
    results = pd.DataFrame(
        [
            _result_row(
                run_id="complex",
                run_index=0,
                rate_calibration_loss_weight=0.05,
                zero_count_weight=0.5,
                positive_count_weight=1.5,
            ),
            _result_row(run_id="simple", run_index=1),
        ]
    )

    ranked = rank_neural_ode_objective_results(results)

    assert ranked.iloc[0]["run_id"] == "simple"


def test_failed_runs_are_excluded_from_best_selection() -> None:
    results = pd.DataFrame(
        [
            _result_row(run_id="ok", run_index=0),
            _result_row(
                run_id="failed",
                run_index=1,
                status="failed",
                validation_unified_bits_per_spike=9.0,
            ),
        ]
    )

    summary = summarize_neural_ode_objectives(results, _refs())

    assert summary["best_run_id"] == "ok"
    assert summary["successful_runs"] == 1
    assert summary["runs_attempted"] == 2


def test_summary_reports_reference_comparisons() -> None:
    summary = summarize_neural_ode_objectives(pd.DataFrame([_result_row()]), _refs())

    assert summary["best_objective_name"] == "heldout_heavy"
    assert summary["beats_factor_latent_unified"] is False
    assert summary["beats_previous_neural_ode_refinement"] is True
    assert summary["evaluation_metric_is_unweighted"] is True
    assert summary["old_incompatible_mean_rate_values_used_as_targets"] is False


def test_summary_can_report_beating_factor_latent() -> None:
    results = pd.DataFrame(
        [
            _result_row(
                validation_unified_bits_per_spike=0.04,
                beats_factor_latent_unified=True,
            )
        ]
    )

    summary = summarize_neural_ode_objectives(results, _refs())

    assert summary["beats_factor_latent_unified"] is True


def test_diagnostics_table_exposes_objective_controls() -> None:
    diagnostics = build_neural_ode_objective_diagnostics(
        pd.DataFrame([_result_row() | {"mean_predicted_rate": 0.56, "mean_observed_rate": 0.58}])
    )

    assert diagnostics.iloc[0]["zero_count_weight"] == 1.0
    assert diagnostics.iloc[0]["rate_calibration_loss"] == 0.0
    assert diagnostics.iloc[0]["mean_observed_rate"] == 0.58
