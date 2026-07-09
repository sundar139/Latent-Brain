from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from latentbrain.eval.switching_ode_tuning import (
    RESULT_COLUMNS,
    build_switching_ode_result_row,
    compute_regime_diagnostics,
    rank_switching_ode_results,
    summarize_switching_ode_tuning,
)


def _scores(bits: float, nll: float = 2.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split": "validation",
                "prediction_source": "direct_model",
                "bits_per_spike": bits,
                "poisson_nll": nll,
            },
            {
                "split": "validation",
                "prediction_source": "factor_decoder",
                "bits_per_spike": bits - 0.01,
                "poisson_nll": nll + 1.0,
            },
        ]
    )


def _refs() -> dict[str, float]:
    return {
        "train_mean_validation_bits_per_spike": 0.0,
        "factor_latent_unified_validation_bits_per_spike": 0.03,
        "previous_neural_ode_validation_bits_per_spike": 0.02,
        "previous_neural_sde_validation_bits_per_spike": 0.01,
        "previous_best_lfads_family_validation_bits_per_spike": 0.005,
        "oracle_validation_bits_per_spike": 3.0,
    }


def _row(run_id: str, bits: float, nll: float, status: str = "completed") -> dict[str, object]:
    return build_switching_ode_result_row(
        run_id,
        0 if run_id == "a" else 1,
        {
            "encoder_hidden_dim": 4,
            "drift_hidden_dim": 4,
            "latent_dim": 2,
            "factor_dim": 2,
            "n_regimes": 2,
            "regime_temperature": 1.0,
            "input_dropout_rate": 0.0,
            "heldout_loss_weight": 1.0,
            "kl_scale": 0.1,
            "entropy_regularization": 0.0,
            "epochs": 1,
        },
        _scores(bits, nll),
        {"status": status, "validation_behavior_mean_r2": 0.0, "drift_norm": 0.2},
        pd.DataFrame([{"checkpoint_source": "latest", "selected_by_unified": True}]),
        {
            "mean_regime_entropy": 0.5,
            "active_regime_count": 2,
            "max_regime_occupancy": 0.6,
            "min_regime_occupancy": 0.4,
        },
        _refs(),
        Path("out"),
    )


def test_result_row_has_required_columns() -> None:
    row = _row("a", 0.04, 2.0)
    assert set(RESULT_COLUMNS).issubset(row)
    assert row["beats_factor_latent_unified"] is True
    assert row["beats_previous_neural_ode"] is True


def test_ranking_uses_unified_bits_then_poisson_nll_and_excludes_failed() -> None:
    ranked = rank_switching_ode_results(
        pd.DataFrame([_row("a", 0.02, 3.0), _row("b", 0.03, 5.0), _row("c", 0.9, 1.0, "failed")])
    )
    assert ranked.iloc[0]["run_id"] == "b"

    tied = rank_switching_ode_results(pd.DataFrame([_row("a", 0.02, 3.0), _row("b", 0.02, 2.0)]))
    assert tied.iloc[0]["run_id"] == "b"


def test_regime_diagnostics_compute_occupancy_and_entropy() -> None:
    probs = np.array([[[0.8, 0.2], [0.6, 0.4]], [[0.7, 0.3], [0.5, 0.5]]])
    diagnostics = compute_regime_diagnostics(probs, "validation")
    assert list(diagnostics["mean_occupancy"]) == [0.65, 0.35]
    assert diagnostics["entropy"].notna().all()
    assert diagnostics["active"].all()


def test_summary_identifies_reference_beats_and_excludes_failed_best() -> None:
    summary = summarize_switching_ode_tuning(
        pd.DataFrame([_row("a", 0.04, 2.0), _row("c", 0.9, 1.0, "failed")]), _refs()
    )
    assert summary["best_run_id"] == "a"
    assert summary["beats_factor_latent_unified"] is True
    assert summary["beats_previous_neural_ode"] is True
