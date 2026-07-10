from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

import latentbrain.train.neural_ode_pilot as pilot_module
from latentbrain.train.neural_ode_pilot import (
    build_full_evaluation_recommendation,
    build_next_action_recommendation,
    build_pilot_model,
    build_pilot_run_schedule,
    checkpoint_sha256,
    validate_neural_ode_pilot_config,
)


def _config() -> dict[str, Any]:
    return {
        "dataset": {"name": "mc_maze_large", "expected_hash": "0" * 64},
        "trial_source": {"type": "trial_aware_raw", "allow_global_crop_to_min": False},
        "window": {
            "name": "behavior_speed_peak_centered_1p28s",
            "duration_seconds": 1.28,
            "extract_before_rebin": True,
        },
        "binning": {"target_bin_size_ms": 20},
        "outer_protocol": {
            "assignments_path": "assignments.csv",
            "baseline_scores_path": "scores.csv",
            "baseline_summary_path": "summary.json",
            "readiness_path": "readiness.json",
            "repeat_index": 0,
            "fold_indices": [0, 1, 2, 3, 4],
            "reuse_exact_assignments": True,
            "reuse_exact_neuron_mask": True,
        },
        "initialization": {
            "seeds": [2027, 2028, 2029, 2030, 2031],
            "seed_policy": "exact_declared_seed",
            "deterministic_algorithms": True,
        },
        "model": {
            "name": "deterministic_neural_ode",
            "diffusion_enabled": False,
            "latent_dim": 2,
            "factor_dim": 2,
            "encoder_hidden_dim": 4,
            "drift_hidden_dim": 4,
            "diffusion_hidden_dim": 4,
            "dropout_rate": 0.0,
            "min_rate_hz": 1.0e-4,
            "max_rate_hz": 500.0,
        },
        "dynamics": {
            "solver": "euler",
            "integration_step_seconds": 0.02,
            "integration_horizon_seconds": 1.28,
            "adjoint": False,
            "maximum_state_norm": 100.0,
            "maximum_drift_norm": 100.0,
            "fail_on_nonfinite_state": True,
            "fail_on_solver_error": True,
        },
        "training": {
            "device": "cuda",
            "epochs": 50,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "checkpoint_metric": "inner_validation_unified_bits_per_spike",
            "checkpoint_mode": "max",
            "mixed_precision": True,
        },
        "inner_checkpoint_selection": {
            "enabled": True,
            "validation_fraction": 0.15,
            "split_seed_base": 6061,
            "stratified": True,
            "use_outer_evaluation_for_selection": False,
        },
        "scoring": {"reference_model": "train_heldout_mean_rate"},
        "baseline": {"method": "factor_latent_train_selected", "comparison_unit": "fold"},
        "lfads_reference": {"pilot_mean": 0.02925965290281923},
        "pilot_gates": {
            "require_all_runs_complete": True,
            "maximum_failed_run_fraction": 0.0,
            "require_finite_scores": True,
            "require_finite_losses": True,
            "require_checkpoint_from_inner_validation": True,
            "require_solver_stability": True,
            "maximum_nonfinite_run_fraction": 0.0,
            "minimum_mean_unified_bits_per_spike": 0.0,
            "minimum_positive_seed_fraction": 0.60,
            "maximum_seed_mean_std": 0.05,
            "full_evaluation_margin_over_baseline": -0.02,
        },
        "statistics": {
            "confidence_interval": 0.95,
            "bootstrap_repeats": 100,
            "bootstrap_seed": 1337,
        },
        "reporting": {"output_dir": "out"},
    }


def _runs(score: float = 0.10, paired: float = -0.01) -> pd.DataFrame:
    rows = []
    for fold in range(5):
        for seed in range(2027, 2032):
            rows.append(
                {
                    "repeat_index": 0,
                    "fold_index": fold,
                    "initialization_seed": seed,
                    "status": "completed",
                    "outer_unified_bits_per_spike": score + (seed - 2029) * 0.001,
                    "paired_difference_vs_baseline": paired,
                    "final_train_loss": 1.0,
                    "final_inner_validation_loss": 1.1,
                    "checkpoint_source": "inner_validation",
                    "training_seconds": 10.0,
                    "peak_cuda_memory_mb": 100.0,
                    "solver_failure_count": 0,
                    "nonfinite_state_count": 0,
                }
            )
    return pd.DataFrame(rows)


def test_only_predeclared_repeat_folds_and_seeds_are_allowed() -> None:
    validate_neural_ode_pilot_config(_config())
    for key, value in (("repeat_index", 1), ("fold_indices", [0, 1, 2, 3])):
        broken = _config()
        broken["outer_protocol"][key] = value
        with pytest.raises(ValueError):
            validate_neural_ode_pilot_config(broken)
    broken = _config()
    broken["initialization"]["seeds"] = [2027, 2028, 2029, 2030, 2032]
    with pytest.raises(ValueError):
        validate_neural_ode_pilot_config(broken)


def test_global_crop_and_derived_seed_policy_are_rejected() -> None:
    config = _config()
    config["trial_source"]["allow_global_crop_to_min"] = True
    with pytest.raises(ValueError, match="global crop"):
        validate_neural_ode_pilot_config(config)
    config = _config()
    config["initialization"]["seed_policy"] = "seed_plus_run_index"
    with pytest.raises(ValueError, match="exact_declared_seed"):
        validate_neural_ode_pilot_config(config)


def test_diffusion_and_adjoint_must_be_disabled() -> None:
    config = _config()
    config["model"]["diffusion_enabled"] = True
    with pytest.raises(ValueError, match="deterministic neural ODE"):
        validate_neural_ode_pilot_config(config)
    config = _config()
    config["dynamics"]["adjoint"] = True
    with pytest.raises(ValueError, match="deterministic neural ODE"):
        validate_neural_ode_pilot_config(config)


def test_checkpoint_metric_must_be_inner_validation_unified_max() -> None:
    config = _config()
    config["training"]["checkpoint_metric"] = "validation_total_loss"
    with pytest.raises(ValueError, match="inner_validation_unified_bits_per_spike"):
        validate_neural_ode_pilot_config(config)


def test_schedule_has_25_unique_runs_and_uses_declared_seed_directly() -> None:
    schedule = build_pilot_run_schedule(_config())
    assert len(schedule) == 25
    assert len({run.run_id for run in schedule}) == 25
    assert {run.initialization_seed for run in schedule} == set(range(2027, 2032))
    assert all(run.repeat_index == 0 for run in schedule)
    for seed in range(2027, 2032):
        assert sum(run.initialization_seed == seed for run in schedule) == 5


def test_model_initialization_uses_declared_seed_directly() -> None:
    config = _config()
    first = build_pilot_model(config, input_dim=3, output_dim=5, initialization_seed=2027)
    repeated = build_pilot_model(config, input_dim=3, output_dim=5, initialization_seed=2027)
    different = build_pilot_model(config, input_dim=3, output_dim=5, initialization_seed=2028)
    assert first.config.diffusion_scale == 0.0
    assert all(
        np.array_equal(left.detach().numpy(), right.detach().numpy())
        for left, right in zip(first.parameters(), repeated.parameters(), strict=True)
    )
    assert any(
        not np.array_equal(left.detach().numpy(), right.detach().numpy())
        for left, right in zip(first.parameters(), different.parameters(), strict=True)
    )


def test_stable_competitive_runs_recommend_full_evaluation() -> None:
    recommendation = build_full_evaluation_recommendation(
        _runs(paired=0.01), _config(), leakage_checks_passed=True
    )
    assert recommendation["proceed"] is True
    assert recommendation["solver_stability_passed"] is True
    action = build_next_action_recommendation(recommendation, _config(), {"near_peak": 0.05})
    assert action["recommended_next_action"] == "run_full_neural_ode_evaluation"
    assert action["final_claim_allowed"] is False


def test_large_baseline_deficit_blocks_full_evaluation_and_retires() -> None:
    recommendation = build_full_evaluation_recommendation(
        _runs(paired=-0.15), _config(), leakage_checks_passed=True
    )
    assert recommendation["proceed"] is False
    action = build_next_action_recommendation(recommendation, _config(), {"near_peak": 0.02})
    assert action["recommended_next_action"] == "retire_neural_ode_and_close_neural_model_search"


def test_solver_instability_blocks_all_progression() -> None:
    runs = _runs(paired=0.01)
    runs.loc[0, "nonfinite_state_count"] = 3
    recommendation = build_full_evaluation_recommendation(
        runs, _config(), leakage_checks_passed=True
    )
    assert recommendation["solver_stability_passed"] is False
    action = build_next_action_recommendation(recommendation, _config(), {"near_peak": 0.05})
    assert action["recommended_next_action"] == "block_due_to_integrity_issue"


def test_narrow_actionable_failure_permits_diagnostic() -> None:
    # Stable and positive, only the baseline margin fails and only barely.
    recommendation = build_full_evaluation_recommendation(
        _runs(score=0.10, paired=-0.03), _config(), leakage_checks_passed=True
    )
    assert recommendation["proceed"] is False
    action = build_next_action_recommendation(recommendation, _config(), {"near_peak": 0.05})
    assert action["recommended_next_action"] == "run_targeted_neural_ode_diagnostic"
    assert action["targeted_diagnostic_available"] is True


def test_one_repeat_pilot_never_allows_final_claim() -> None:
    recommendation = build_full_evaluation_recommendation(
        _runs(paired=0.01), _config(), leakage_checks_passed=True
    )
    action = build_next_action_recommendation(recommendation, _config(), {"near_peak": 0.05})
    assert recommendation["pilot_final_claim_allowed"] is False
    assert action["final_claim_allowed"] is False
    assert action["full_evaluation_allowed"] == recommendation["proceed"]


def test_solver_diagnostics_record_norms_and_flag_nonfinite() -> None:
    latents = np.ones((2, 4, 3), dtype=np.float64)
    drift = np.full((2, 4, 3), 2.0, dtype=np.float64)
    diagnostics = pilot_module._solver_diagnostics(
        {"latents": latents, "drift": drift, "rates": np.ones((2, 4, 5))}
    )
    assert diagnostics["integration_steps"] == 4
    assert diagnostics["nonfinite_state_count"] == 0
    assert diagnostics["maximum_state_norm"] == pytest.approx(np.sqrt(3.0))
    assert diagnostics["maximum_drift_norm"] == pytest.approx(2.0 * np.sqrt(3.0))
    bad = latents.copy()
    bad[0, 0, 0] = np.nan
    assert (
        pilot_module._solver_diagnostics({"latents": bad, "drift": drift})["nonfinite_state_count"]
        == 1
    )


def test_latent_diagnostics_report_effective_rank_and_near_zero() -> None:
    rng = np.random.default_rng(0)
    prediction = {
        "factors": rng.normal(size=(3, 4, 2)),
        "latents": rng.normal(size=(3, 4, 2)),
    }
    frame = pilot_module._latent_diagnostic_rows(prediction, fold_index=0, seed=2027)
    assert set(frame["representation"]) == {"factor", "latent"}
    assert {"effective_rank", "near_zero_variance_fraction", "covariance_eigenvalue"}.issubset(
        frame.columns
    )


def test_resume_does_not_duplicate_completed_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    config["reporting"]["output_dir"] = str(tmp_path / "out")
    inputs = {
        "folds": [type("Fold", (), {"fold_index": index})() for index in range(5)],
        "dataset": object(),
        "assignments": pd.DataFrame(),
        "baseline_by_fold": dict.fromkeys(range(5), 0.11),
    }
    calls: list[tuple[int, int]] = []

    def fake_train(
        run: Any,
        _fold: Any,
        _dataset: Any,
        _assignments: pd.DataFrame,
        baseline: float,
        lfads_ref: float,
        _config_value: dict[str, Any],
        output_dir: Path,
        _device: Any,
    ) -> tuple[
        dict[str, Any], dict[str, Any], dict[str, Any], pd.DataFrame, dict[str, Any], dict[str, Any]
    ]:
        calls.append((run.fold_index, run.initialization_seed))
        checkpoint = output_dir / "runs" / run.run_id / "checkpoints" / "best_validation.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(run.run_id.encode())
        row = dict.fromkeys(pilot_module.RUN_COLUMNS, 0)
        row.update(
            {
                "repeat_index": 0,
                "fold_index": run.fold_index,
                "initialization_seed": run.initialization_seed,
                "status": "completed",
                "checkpoint_source": "inner_validation",
                "outer_unified_bits_per_spike": 0.03,
                "paired_difference_vs_baseline": 0.03 - baseline,
                "lfads_outer_mean_reference": lfads_ref,
                "final_train_loss": 1.0,
                "final_inner_validation_loss": 1.0,
                "training_seconds": 1.0,
                "peak_cuda_memory_mb": 10.0,
                "solver_failure_count": 0,
                "nonfinite_state_count": 0,
                "notes": "",
            }
        )
        manifest = {
            "repeat_index": 0,
            "fold_index": run.fold_index,
            "initialization_seed": run.initialization_seed,
            "epoch": 1,
            "checkpoint_type": "best",
            "selection_split": "inner_validation",
            "selection_metric": "inner_validation_unified_bits_per_spike",
            "selection_metric_value": 0.03,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256(checkpoint),
            "model_config_digest": "m",
            "solver_config_digest": "s",
        }
        resource = {
            "repeat_index": 0,
            "fold_index": run.fold_index,
            "initialization_seed": run.initialization_seed,
            "training_seconds": 1.0,
            "integration_seconds": 0.1,
            "best_epoch": 1,
            "epochs_completed": 2,
            "peak_cuda_memory_mb": 10.0,
            "batch_size": 4,
            "mixed_precision_enabled": True,
            "early_stopping_triggered": False,
            "checkpoint_size_bytes": checkpoint.stat().st_size,
        }
        solver = {
            "repeat_index": 0,
            "fold_index": run.fold_index,
            "initialization_seed": run.initialization_seed,
            "solver": "euler",
            "integration_step_seconds": 0.02,
            "integration_steps": 64,
            "solver_failure_count": 0,
            "nonfinite_state_count": 0,
            "maximum_state_norm": 1.0,
            "mean_state_norm": 0.5,
            "maximum_drift_norm": 1.0,
            "mean_drift_norm": 0.5,
            "terminal_state_norm": 0.5,
            "gradient_norm": 0.1,
            "integration_seconds": 0.1,
        }
        latent = pd.DataFrame(
            [
                {
                    "fold_index": run.fold_index,
                    "initialization_seed": run.initialization_seed,
                    "representation": "factor",
                    "dimension": 0,
                    "variance": 1.0,
                    "covariance_eigenvalue": 1.0,
                    "effective_rank": 1.0,
                    "effective_rank_fraction": 0.5,
                    "near_zero_variance_dimensions": 0,
                    "near_zero_variance_fraction": 0.0,
                    "temporal_first_difference_variance": 0.1,
                    "temporal_second_difference_variance": 0.1,
                }
            ]
        )
        near_peak = {
            "fold_index": run.fold_index,
            "initialization_seed": run.initialization_seed,
            "before_peak": 0.04,
            "near_peak": 0.03,
            "after_peak": 0.035,
        }
        return row, manifest, resource, latent, solver, near_peak

    monkeypatch.setattr(pilot_module, "_load_protocol_inputs", lambda _config: inputs)
    monkeypatch.setattr(pilot_module, "_train_one", fake_train)
    monkeypatch.setattr(pilot_module.torch.cuda, "is_available", lambda: True)

    first = pilot_module.run_neural_ode_pilot(config)
    second = pilot_module.run_neural_ode_pilot(config)

    assert len(calls) == 25
    assert first["summary"]["completed_runs"] == 25
    assert second["summary"]["completed_runs"] == 25
    assert first["summary"]["pilot_final_claim_allowed"] is False
    assert len(pd.read_csv(tmp_path / "out" / "neural_ode_pilot_runs.csv")) == 25
    assert (tmp_path / "out" / "next_action_recommendation.json").exists()
    assert (tmp_path / "out" / "solver_diagnostics.csv").exists()
