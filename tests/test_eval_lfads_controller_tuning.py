from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from latentbrain.eval.lfads_controller_tuning import (
    RESULT_COLUMNS,
    build_controller_result_row,
    rank_controller_results,
    summarize_controller_tuning,
)


def _scores(bits: float = 0.02, nll: float = 5.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["validation", "validation"],
            "prediction_source": ["direct_model", "factor_decoder"],
            "bits_per_spike": [bits, bits / 2.0],
            "poisson_nll": [nll, nll + 1.0],
        }
    )


def _refs() -> dict[str, float]:
    return {
        "train_mean_validation_bits_per_spike": 0.0,
        "factor_latent_unified_validation_bits_per_spike": 0.03,
        "previous_best_lfads_family_validation_bits_per_spike": 0.01,
        "oracle_validation_bits_per_spike": 3.0,
    }


def _row(run_id: str, index: int, bits: float, nll: float, **extra: Any) -> dict[str, Any]:
    params = {
        "encoder_hidden_dim": 64,
        "controller_hidden_dim": 64,
        "generator_hidden_dim": 96,
        "latent_dim": extra.get("latent_dim", 16),
        "factor_dim": 32,
        "inferred_input_dim": 4,
        "input_dropout_rate": 0.25,
        "heldout_loss_weight": 4.0,
        "kl_warmup_epochs": 5,
        "kl_scale": extra.get("kl_scale", 0.1),
        "inferred_input_kl_scale": extra.get("inferred_input_kl_scale", 0.01),
        "epochs": 20,
    }
    training = {
        "validation_behavior_mean_r2": extra.get("r2", 0.0),
        "train_total_loss": 1.0,
        "validation_total_loss": extra.get("loss", 1.0),
        "validation_heldout_prediction_loss": 2.0,
        "z0_kl_loss": 0.3,
        "inferred_input_kl_loss": 0.4,
    }
    return build_controller_result_row(
        run_id, index, params, _scores(bits, nll), training, _refs(), Path("out") / run_id
    )


def test_result_row_has_required_columns() -> None:
    row = _row("run_000", 0, 0.02, 5.0)

    assert set(RESULT_COLUMNS).issubset(row)
    assert row["beats_previous_best_lfads_family"] is True


def test_ranking_uses_unified_bits() -> None:
    results = pd.DataFrame([_row("a", 0, 0.01, 1.0), _row("b", 1, 0.02, 9.0)])

    assert rank_controller_results(results).iloc[0]["run_id"] == "b"


def test_tie_breaker_uses_poisson_nll() -> None:
    results = pd.DataFrame([_row("a", 0, 0.02, 2.0), _row("b", 1, 0.02, 1.0)])

    assert rank_controller_results(results).iloc[0]["run_id"] == "b"


def test_summary_identifies_factor_latent_status() -> None:
    results = pd.DataFrame([_row("a", 0, 0.04, 1.0)])

    summary = summarize_controller_tuning(results, _refs())

    assert summary["best_run_id"] == "a"
    assert summary["beats_factor_latent_unified"] is True


def test_failed_runs_are_excluded_from_best_selection() -> None:
    good = _row("good", 1, 0.02, 1.0)
    failed = _row("failed", 0, 0.5, 0.1)
    failed["status"] = "failed"
    results = pd.DataFrame([failed, good])

    assert summarize_controller_tuning(results, _refs())["best_run_id"] == "good"
