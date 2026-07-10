from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

import latentbrain.train.lfads_pilot as pilot_module
from latentbrain.train.lfads_pilot import (
    build_full_evaluation_recommendation,
    build_inner_split,
    build_pilot_model,
    build_pilot_run_schedule,
    checkpoint_sha256,
    validate_checkpoint_record,
    validate_input_target_separation,
    validate_lfads_pilot_config,
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
        "model": {"controller_enabled": False},
        "training": {
            "device": "cuda",
            "epochs": 150,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "checkpoint_metric": "inner_validation_unified_bits_per_spike",
            "checkpoint_mode": "max",
        },
        "inner_checkpoint_selection": {
            "enabled": True,
            "validation_fraction": 0.15,
            "split_seed_base": 5051,
            "stratified": True,
            "use_outer_evaluation_for_selection": False,
        },
        "scoring": {"reference_model": "train_heldout_mean_rate"},
        "baseline": {
            "method": "factor_latent_train_selected",
            "comparison_unit": "fold",
        },
        "pilot_gates": {
            "require_all_runs_complete": True,
            "maximum_failed_run_fraction": 0.0,
            "require_finite_scores": True,
            "require_finite_losses": True,
            "require_checkpoint_from_inner_validation": True,
            "minimum_mean_unified_bits_per_spike": 0.0,
            "minimum_positive_seed_fraction": 0.60,
            "maximum_seed_std": 0.05,
            "full_evaluation_margin_over_baseline": -0.02,
        },
        "reporting": {"output_dir": "out"},
    }


def _runs(score: float = 0.10) -> pd.DataFrame:
    rows = []
    for fold in range(5):
        for seed in range(2027, 2032):
            rows.append(
                {
                    "fold_index": fold,
                    "initialization_seed": seed,
                    "status": "completed",
                    "outer_unified_bits_per_spike": score + (seed - 2029) * 0.001,
                    "paired_difference_vs_baseline": -0.01,
                    "final_train_loss": 1.0,
                    "final_inner_validation_loss": 1.1,
                    "checkpoint_source": "inner_validation",
                    "training_seconds": 10.0,
                    "peak_cuda_memory_mb": 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_only_predeclared_repeat_folds_and_seeds_are_allowed() -> None:
    config = _config()
    validate_lfads_pilot_config(config)

    for key, value in (
        ("repeat_index", 1),
        ("fold_indices", [0, 1, 2, 3]),
    ):
        broken = _config()
        broken["outer_protocol"][key] = value
        with pytest.raises(ValueError):
            validate_lfads_pilot_config(broken)

    broken = _config()
    broken["initialization"]["seeds"] = [2027, 2028, 2029, 2030, 2032]
    with pytest.raises(ValueError):
        validate_lfads_pilot_config(broken)


def test_global_crop_and_derived_seed_policy_are_rejected() -> None:
    config = _config()
    config["trial_source"]["allow_global_crop_to_min"] = True
    with pytest.raises(ValueError, match="global crop"):
        validate_lfads_pilot_config(config)

    config = _config()
    config["initialization"]["seed_policy"] = "seed_plus_run_index"
    with pytest.raises(ValueError, match="exact_declared_seed"):
        validate_lfads_pilot_config(config)


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
    config["model"].update(
        {
            "encoder_hidden_dim": 4,
            "generator_hidden_dim": 4,
            "latent_dim": 2,
            "factor_dim": 2,
            "dropout_rate": 0.0,
            "log_rate_min": -12.0,
            "log_rate_max": 8.0,
        }
    )

    first = build_pilot_model(config, input_dim=3, output_dim=5, initialization_seed=2027)
    repeated = build_pilot_model(config, input_dim=3, output_dim=5, initialization_seed=2027)
    different = build_pilot_model(config, input_dim=3, output_dim=5, initialization_seed=2028)

    assert all(
        np.array_equal(left.detach().numpy(), right.detach().numpy())
        for left, right in zip(first.parameters(), repeated.parameters(), strict=True)
    )
    assert any(
        not np.array_equal(left.detach().numpy(), right.detach().numpy())
        for left, right in zip(first.parameters(), different.parameters(), strict=True)
    )


def test_inner_split_is_stratified_and_contains_outer_training_trials_only() -> None:
    outer_train = np.arange(20, dtype=np.int64)
    assignments = pd.DataFrame(
        {"trial_index": np.arange(25), "stratum": [f"s{index % 2}" for index in range(25)]}
    )

    train, validation = build_inner_split(outer_train, assignments, 0.15, 5051)

    assert len(validation) == 3
    assert np.intersect1d(train, validation).size == 0
    assert set(np.concatenate([train, validation])) == set(outer_train)
    assert set(validation).isdisjoint(set(range(20, 25)))
    repeated = build_inner_split(outer_train, assignments, 0.15, 5051)
    np.testing.assert_array_equal(train, repeated[0])
    np.testing.assert_array_equal(validation, repeated[1])


def test_heldout_targets_can_never_enter_input_features() -> None:
    heldin = np.array([0, 1, 3])
    heldout = np.array([2, 4])

    validate_input_target_separation(heldin, heldout, input_dim=3, output_dim=5)

    with pytest.raises(ValueError, match="overlap"):
        validate_input_target_separation(np.array([0, 2]), heldout, input_dim=2, output_dim=5)
    with pytest.raises(ValueError, match="input_dim"):
        validate_input_target_separation(heldin, heldout, input_dim=4, output_dim=5)


def test_selected_checkpoint_must_identify_inner_validation() -> None:
    valid = {
        "selection_split": "inner_validation",
        "selection_metric": "inner_validation_unified_bits_per_spike",
        "checkpoint_type": "best",
    }
    validate_checkpoint_record(valid)

    for forbidden in ("outer_evaluation", "test", "full_dataset"):
        invalid = {**valid, "selection_split": forbidden}
        with pytest.raises(ValueError, match="inner_validation"):
            validate_checkpoint_record(invalid)


def test_checkpoint_hash_is_recorded_from_real_bytes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")

    assert checkpoint_sha256(checkpoint) == hashlib.sha256(b"checkpoint").hexdigest()


def test_stable_successful_runs_recommend_full_evaluation() -> None:
    recommendation = build_full_evaluation_recommendation(
        _runs(), _config(), leakage_checks_passed=True
    )

    assert recommendation["proceed"] is True
    assert recommendation["all_runs_completed"] is True
    assert recommendation["positive_seed_fraction"] == 1.0
    assert recommendation["checkpoint_selection_valid"] is True
    assert recommendation["leakage_checks_passed"] is True


@pytest.mark.parametrize("failure", ["negative", "unstable", "checkpoint", "leakage", "missing"])
def test_gate_blocks_invalid_pilots(failure: str) -> None:
    runs = _runs()
    leakage = True
    if failure == "negative":
        runs["outer_unified_bits_per_spike"] = -0.1
    elif failure == "unstable":
        runs.loc[runs["initialization_seed"] == 2031, "outer_unified_bits_per_spike"] = 0.5
    elif failure == "checkpoint":
        runs.loc[0, "checkpoint_source"] = "outer_evaluation"
    elif failure == "leakage":
        leakage = False
    else:
        runs = runs.iloc[:-1]

    recommendation = build_full_evaluation_recommendation(
        runs, _config(), leakage_checks_passed=leakage
    )

    assert recommendation["proceed"] is False
    assert recommendation["reasons"]


def test_one_repeat_pilot_never_allows_final_claim() -> None:
    recommendation = build_full_evaluation_recommendation(
        _runs(), _config(), leakage_checks_passed=True
    )

    assert recommendation["pilot_final_claim_allowed"] is False


def test_resume_does_not_duplicate_completed_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    config["reporting"]["output_dir"] = str(tmp_path / "out")
    config["statistics"] = {
        "confidence_interval": 0.95,
        "bootstrap_repeats": 100,
        "bootstrap_seed": 1337,
    }
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
        _config_value: dict[str, Any],
        output_dir: Path,
        _device: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        calls.append((run.fold_index, run.initialization_seed))
        checkpoint = output_dir / "runs" / run.run_id / "checkpoints" / "best_validation.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(run.run_id.encode())
        row = {
            "repeat_index": 0,
            "fold_index": run.fold_index,
            "split_seed": 2027,
            "neuron_mask_seed": 2027,
            "initialization_seed": run.initialization_seed,
            "status": "completed",
            "best_epoch": 1,
            "checkpoint_source": "inner_validation",
            "inner_validation_unified_bits_per_spike": 0.1,
            "outer_unified_bits_per_spike": 0.1,
            "outer_poisson_nll": 1.0,
            "outer_behavior_mean_r2": 0.0,
            "baseline_outer_unified_bits_per_spike": baseline,
            "paired_difference_vs_baseline": -0.01,
            "training_seconds": 1.0,
            "peak_cuda_memory_mb": 10.0,
            "final_train_loss": 1.0,
            "final_inner_validation_loss": 1.0,
            "notes": "",
        }
        manifest = {
            "repeat_index": 0,
            "fold_index": run.fold_index,
            "initialization_seed": run.initialization_seed,
            "epoch": 1,
            "checkpoint_type": "best",
            "selection_split": "inner_validation",
            "selection_metric": "inner_validation_unified_bits_per_spike",
            "selection_metric_value": 0.1,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256(checkpoint),
        }
        resource = {
            "repeat_index": 0,
            "fold_index": run.fold_index,
            "initialization_seed": run.initialization_seed,
            "training_seconds": 1.0,
            "best_epoch": 1,
            "epochs_completed": 2,
            "peak_cuda_memory_mb": 10.0,
            "batch_size": 32,
            "mixed_precision_enabled": True,
            "early_stopping_triggered": False,
            "checkpoint_size_bytes": checkpoint.stat().st_size,
        }
        return row, manifest, resource

    monkeypatch.setattr(pilot_module, "_load_protocol_inputs", lambda _config: inputs)
    monkeypatch.setattr(pilot_module, "_train_one", fake_train)
    monkeypatch.setattr(pilot_module.torch.cuda, "is_available", lambda: True)

    first = pilot_module.run_lfads_pilot(config)
    second = pilot_module.run_lfads_pilot(config)

    assert len(calls) == 25
    assert first["summary"]["completed_runs"] == 25
    assert second["summary"]["completed_runs"] == 25
    assert len(pd.read_csv(tmp_path / "out" / "lfads_pilot_runs.csv")) == 25
