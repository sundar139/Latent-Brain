from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
import torch

from latentbrain.eval.neural_ode_diagnostics import (
    COUNTERFACTUAL_COLUMNS,
    build_next_action_recommendation,
    decoder_spectrum_diagnostics,
    static_state_rates,
    validate_checkpoint_integrity,
    verify_score_reproduction,
)
from latentbrain.models.neural_sde import NeuralSDE, NeuralSDEConfig

EXPECTED_FOLDS = [0, 1, 2, 3, 4]
EXPECTED_SEEDS = [2027, 2028, 2029, 2030, 2031]


def _config() -> dict[str, Any]:
    return {
        "dataset": {"name": "mc_maze_large", "expected_hash": "0" * 64},
        "protocol": {
            "repeat_index": 0,
            "fold_indices": EXPECTED_FOLDS,
            "initialization_seeds": EXPECTED_SEEDS,
        },
        "thresholds": {
            "train_inner_gap_warning": 0.05,
            "inner_outer_gap_warning": 0.05,
            "low_effective_rank_fraction": 0.10,
            "decoder_condition_warning": 10000.0,
            "severe_temporal_smoothing_ratio": 0.50,
        },
        "decision": {
            "full_evaluation_currently_allowed": False,
            "allow_one_targeted_repair": True,
            "prohibit_broad_sweep": True,
            "default_when_no_clear_repair": "retire_neural_ode_and_close_neural_model_search",
        },
    }


def _checkpoint(path: Path, fold: int, seed: int) -> str:
    payload = {
        "epoch": 5,
        "config": {
            "dataset": {"name": "mc_maze_large", "bin_size_ms": 20},
            "model": {
                "name": "deterministic_neural_ode",
                "input_dim": 122,
                "resolved_output_dim": 162,
                "diffusion_scale": 0.0,
            },
            "pilot": {
                "repeat_index": 0,
                "fold_index": fold,
                "initialization_seed": seed,
                "selection_split": "inner_validation",
                "outer_evaluation_used_for_selection": False,
            },
        },
        "model_state_dict": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _integrity_frames(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest_rows = []
    run_rows = []
    for fold in EXPECTED_FOLDS:
        for seed in EXPECTED_SEEDS:
            path = tmp_path / f"fold_{fold}" / f"seed_{seed}" / "best.pt"
            digest = _checkpoint(path, fold, seed)
            manifest_rows.append(
                {
                    "repeat_index": 0,
                    "fold_index": fold,
                    "initialization_seed": seed,
                    "epoch": 5,
                    "checkpoint_type": "best",
                    "selection_split": "inner_validation",
                    "selection_metric": "inner_validation_unified_bits_per_spike",
                    "selection_metric_value": 0.14,
                    "checkpoint_path": str(path),
                    "checkpoint_sha256": digest,
                    "model_config_digest": "shared_model_digest",
                    "solver_config_digest": "shared_solver_digest",
                }
            )
            run_rows.append(
                {
                    "repeat_index": 0,
                    "fold_index": fold,
                    "initialization_seed": seed,
                    "status": "completed",
                    "checkpoint_source": "inner_validation",
                }
            )
    summary = {
        "dataset_hash": "0" * 64,
        "repeat_index": 0,
        "fold_indices": EXPECTED_FOLDS,
        "initialization_seeds": EXPECTED_SEEDS,
        "completed_runs": 25,
        "failed_runs": 0,
        "diffusion_enabled": False,
        "leakage_checks_passed": True,
        "checkpoint_selection_valid": True,
    }
    return pd.DataFrame(manifest_rows), pd.DataFrame(run_rows), summary


def test_checkpoint_integrity_accepts_exact_completed_schedule(tmp_path: Path) -> None:
    manifest, runs, summary = _integrity_frames(tmp_path)

    result = validate_checkpoint_integrity(manifest, runs, summary, _config())

    assert result["integrity_checks_passed"] is True
    assert result["accepted_checkpoints"] == 25
    assert result["diffusion_disabled_confirmed"] is True


def test_missing_checkpoint_fails(tmp_path: Path) -> None:
    manifest, runs, summary = _integrity_frames(tmp_path)
    Path(str(manifest.loc[0, "checkpoint_path"])).unlink()

    with pytest.raises(FileNotFoundError, match="checkpoint is missing"):
        validate_checkpoint_integrity(manifest, runs, summary, _config())


def test_hash_mismatch_fails(tmp_path: Path) -> None:
    manifest, runs, summary = _integrity_frames(tmp_path)
    manifest.loc[0, "checkpoint_sha256"] = "bad"

    with pytest.raises(ValueError, match="hash mismatch"):
        validate_checkpoint_integrity(manifest, runs, summary, _config())


def test_wrong_repeat_or_seed_fails(tmp_path: Path) -> None:
    manifest, runs, summary = _integrity_frames(tmp_path)
    manifest.loc[0, "initialization_seed"] = 9999

    with pytest.raises(ValueError, match="schedule"):
        validate_checkpoint_integrity(manifest, runs, summary, _config())


def test_outer_evaluation_checkpoint_selection_fails(tmp_path: Path) -> None:
    manifest, runs, summary = _integrity_frames(tmp_path)
    manifest.loc[0, "selection_split"] = "outer_evaluation"

    with pytest.raises(ValueError, match="inner_validation"):
        validate_checkpoint_integrity(manifest, runs, summary, _config())


def test_diffusion_enabled_checkpoint_fails(tmp_path: Path) -> None:
    manifest, runs, summary = _integrity_frames(tmp_path)
    summary["diffusion_enabled"] = True

    with pytest.raises(ValueError, match="diffusion"):
        validate_checkpoint_integrity(manifest, runs, summary, _config())


def test_partial_or_background_runs_are_excluded(tmp_path: Path) -> None:
    manifest, runs, summary = _integrity_frames(tmp_path)
    runs.loc[len(runs)] = {
        "repeat_index": 0,
        "fold_index": 0,
        "initialization_seed": 2027,
        "status": "failed",
        "checkpoint_source": "terminated_preflight",
    }

    result = validate_checkpoint_integrity(manifest, runs, summary, _config())

    assert result["accepted_checkpoints"] == 25
    assert result["excluded_preflight_artifacts"] == 1


def test_score_reproduction_within_tolerance_passes() -> None:
    verify_score_reproduction(0.14129368147360896, 0.14129368147360896, 0, 2027)


def test_score_reproduction_mismatch_fails() -> None:
    with pytest.raises(ValueError, match="not reproduced"):
        verify_score_reproduction(0.10, 0.14, 0, 2027)


def _tiny_model() -> NeuralSDE:
    return NeuralSDE(
        NeuralSDEConfig(
            input_dim=3,
            output_dim=5,
            encoder_hidden_dim=4,
            drift_hidden_dim=4,
            diffusion_hidden_dim=4,
            latent_dim=2,
            factor_dim=2,
            dropout=0.0,
            min_rate_hz=1.0e-4,
            max_rate_hz=500.0,
            dt_seconds=0.02,
            diffusion_scale=0.0,
        )
    )


def test_static_state_rates_ignore_time_and_match_shape() -> None:
    model = _tiny_model()
    z0_mean = np.zeros((2, 2), dtype=np.float32)

    rates = static_state_rates(model, z0_mean, time_bins=6)

    assert rates.shape == (2, 6, 5)
    assert np.isfinite(rates).all()
    assert (rates > 0.0).all()
    np.testing.assert_allclose(rates[:, 0, :], rates[:, -1, :])


def test_decoder_spectrum_reports_rank_and_condition() -> None:
    model = _tiny_model()

    diagnostics = decoder_spectrum_diagnostics(model)

    assert diagnostics["decoder_effective_rank"] > 0.0
    assert diagnostics["decoder_condition_number"] > 0.0
    assert np.isfinite(diagnostics["decoder_mean_output_weight_norm"])


def test_counterfactual_columns_are_fixed() -> None:
    assert COUNTERFACTUAL_COLUMNS == [
        "fold_index",
        "initialization_seed",
        "method",
        "outer_unified_bits_per_spike",
        "accepted_outer_unified_bits_per_spike",
        "recovery_vs_accepted",
        "fit_policy",
        "diagnostic_only",
    ]


def _decomposition(**overrides: float) -> pd.DataFrame:
    base = {
        "trained decoder limitation": 0.0,
        "learned dynamics limitation": 0.0,
        "global rate calibration": 0.0,
        "excessive drift regularization": 0.0,
        "checkpoint-selection mismatch": 0.0,
        "latent dimension bottleneck": 0.0,
        "late-window temporal failure": 0.0,
        "negative-neuron concentration": 0.0,
        "temporal lag misalignment": 0.0,
        "unexplained remainder": 0.05,
    }
    base.update(overrides)
    return pd.DataFrame(
        [
            {"component": name, "estimated_recoverable_bits_per_spike": value}
            for name, value in base.items()
        ]
    )


def test_integrity_failure_blocks_progression() -> None:
    result = build_next_action_recommendation(False, 0.0126, _decomposition(), True, _config())

    assert result["recommended_next_action"] == "block_due_to_integrity_issue"
    assert result["full_evaluation_allowed"] is False
    assert result["broad_sweep_allowed"] is False


def test_sufficient_single_repair_recommends_targeted_repair() -> None:
    decomposition = _decomposition(**{"trained decoder limitation": 0.02})

    result = build_next_action_recommendation(True, 0.0126, decomposition, True, _config())

    assert result["recommended_next_action"] == "run_targeted_neural_ode_repair_pilot"
    assert result["proposed_single_repair"] == "replace_or_retrain_only_the_heldout_readout"
    assert result["targeted_repair_available"] is True


def test_weak_counterfactual_recovery_retires_neural_model_search() -> None:
    decomposition = _decomposition(**{"trained decoder limitation": 0.001})

    result = build_next_action_recommendation(True, 0.0126, decomposition, True, _config())

    assert result["recommended_next_action"] == "retire_neural_ode_and_close_neural_model_search"
    assert result["targeted_repair_available"] is False


def test_broad_sweep_is_always_false() -> None:
    for integrity, positive, decomposition in (
        (True, True, _decomposition()),
        (True, True, _decomposition(**{"trained decoder limitation": 0.05})),
        (False, True, _decomposition()),
    ):
        result = build_next_action_recommendation(
            integrity, 0.0126, decomposition, positive, _config()
        )
        assert result["broad_sweep_allowed"] is False


def test_full_evaluation_cannot_be_recommended_by_relaxing_margin() -> None:
    decomposition = _decomposition(**{"trained decoder limitation": 0.05})

    result = build_next_action_recommendation(True, 0.0126, decomposition, True, _config())

    assert result["recommended_next_action"] != "run_full_neural_ode_evaluation"
    assert result["full_evaluation_allowed"] is False


def test_unstable_pilot_never_recommends_targeted_repair() -> None:
    decomposition = _decomposition(**{"trained decoder limitation": 0.05})

    result = build_next_action_recommendation(True, 0.0126, decomposition, False, _config())

    assert result["recommended_next_action"] == "retire_neural_ode_and_close_neural_model_search"
