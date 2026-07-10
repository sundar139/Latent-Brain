from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
import torch

from latentbrain.eval.lfads_diagnostics import (
    NEURON_DIAGNOSTIC_COLUMNS,
    TIME_BIN_DIAGNOSTIC_COLUMNS,
    detect_posterior_collapse,
    effective_rank,
    per_neuron_diagnostics,
    recommend_next_action,
    split_gap_summary,
    temporal_smoothness_metrics,
    time_bin_diagnostics,
    validate_checkpoint_integrity,
    validate_neuron_partition,
)

EXPECTED_FOLDS = [0, 1, 2, 3, 4]
EXPECTED_SEEDS = [2027, 2028, 2029, 2030, 2031]


def _config() -> dict[str, Any]:
    return {
        "dataset": {
            "name": "mc_maze_large",
            "expected_hash": "0" * 64,
        },
        "protocol": {
            "repeat_index": 0,
            "fold_indices": EXPECTED_FOLDS,
            "initialization_seeds": EXPECTED_SEEDS,
        },
        "thresholds": {
            "baseline_gap_repairable_limit": 0.04,
        },
        "decision": {
            "allow_targeted_lfads_repair_pilot": True,
            "allow_full_lfads_evaluation": False,
            "default_when_no_clear_repair": "retire_lfads_and_start_neural_ode_pilot",
        },
    }


def _checkpoint(path: Path, fold: int, seed: int) -> str:
    payload = {
        "epoch": 5,
        "config": {
            "dataset": {"expected_hash": "0" * 64},
            "model": {"input_dim": 122, "output_dim": "all", "resolved_output_dim": 162},
            "training": {"seed": seed},
            "pilot": {
                "repeat_index": 0,
                "fold_index": fold,
                "initialization_seed": seed,
                "selection_split": "inner_validation",
                "outer_evaluation_used_for_selection": False,
            },
        },
        "model_state_dict": {
            "encoder.weight_ih_l0": torch.zeros(192, 122),
            "rate_readout.weight": torch.zeros(162, 32),
        },
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
                    "selection_metric_value": 0.01,
                    "checkpoint_path": str(path),
                    "checkpoint_sha256": digest,
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
        "scheduled_runs": 25,
        "completed_runs": 25,
        "failed_runs": 0,
        "full_evaluation_recommended": False,
        "leakage_checks_passed": True,
    }
    return pd.DataFrame(manifest_rows), pd.DataFrame(run_rows), summary


def test_checkpoint_integrity_accepts_exact_completed_schedule(tmp_path: Path) -> None:
    manifest, runs, summary = _integrity_frames(tmp_path)

    result = validate_checkpoint_integrity(manifest, runs, summary, _config())

    assert result["integrity_checks_passed"] is True
    assert result["accepted_checkpoints"] == 25
    assert result["excluded_preflight_artifacts"] == 0


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


def test_outer_evaluation_checkpoint_selection_fails(tmp_path: Path) -> None:
    manifest, runs, summary = _integrity_frames(tmp_path)
    manifest.loc[0, "selection_split"] = "outer_evaluation"

    with pytest.raises(ValueError, match="inner_validation"):
        validate_checkpoint_integrity(manifest, runs, summary, _config())


def test_wrong_repeat_or_seed_fails(tmp_path: Path) -> None:
    manifest, runs, summary = _integrity_frames(tmp_path)
    manifest.loc[0, "initialization_seed"] = 9999

    with pytest.raises(ValueError, match="schedule"):
        validate_checkpoint_integrity(manifest, runs, summary, _config())


def test_terminated_preflight_run_is_excluded(tmp_path: Path) -> None:
    manifest, runs, summary = _integrity_frames(tmp_path)
    runs.loc[len(runs)] = {
        "repeat_index": 0,
        "fold_index": 0,
        "initialization_seed": 2027,
        "status": "failed",
        "checkpoint_source": "terminated_preflight:proc_91fdd85abdfb",
    }

    result = validate_checkpoint_integrity(manifest, runs, summary, _config())

    assert result["accepted_checkpoints"] == 25
    assert result["excluded_preflight_artifacts"] == 1


def test_heldout_neuron_in_input_fails() -> None:
    with pytest.raises(ValueError, match="overlap"):
        validate_neuron_partition(np.array([0, 1, 2]), np.array([2, 3]), 4)


def test_split_gap_calculation_uses_labels_without_reselection() -> None:
    rows = pd.DataFrame(
        {
            "fold_index": [0, 0, 0],
            "initialization_seed": [2027, 2027, 2027],
            "split_name": ["outer_training", "inner_validation", "outer_evaluation"],
            "unified_bits_per_spike": [0.10, 0.07, 0.03],
        }
    )

    result = split_gap_summary(rows)

    assert result.loc[0, "train_to_inner_gap"] == pytest.approx(0.03)
    assert result.loc[0, "inner_to_outer_gap"] == pytest.approx(0.04)
    assert set(rows["split_name"]) == {"outer_training", "inner_validation", "outer_evaluation"}


def test_per_neuron_metrics_are_finite_with_zero_spikes() -> None:
    counts = np.zeros((2, 3, 2), dtype=np.float64)
    predicted = np.full_like(counts, 1.0)
    reference = np.full_like(counts, 0.5)

    frame = per_neuron_diagnostics(counts, predicted, reference, np.array([4, 7]), 20, 0, 2027)

    assert list(frame.columns) == NEURON_DIAGNOSTIC_COLUMNS
    assert np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all()
    assert len(frame) == 2


def test_time_resolved_metrics_have_64_finite_bins() -> None:
    counts = np.zeros((3, 64, 2), dtype=np.float64)
    counts[:, 32, :] = 1.0
    predicted = np.full_like(counts, 1.0)
    reference = np.full_like(counts, 0.5)
    speed = np.linspace(0.0, 1.0, 64)

    frame = time_bin_diagnostics(counts, predicted, reference, speed, 20, 0, 2027)

    assert list(frame.columns) == TIME_BIN_DIAGNOSTIC_COLUMNS
    assert frame["time_bin"].tolist() == list(range(64))
    assert np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all()


def test_smoothness_ratios_handle_constant_arrays() -> None:
    observed = np.ones((4, 64, 3), dtype=np.float64)
    predicted = np.ones_like(observed)

    result = temporal_smoothness_metrics(observed, predicted)

    assert all(np.isfinite(value) for value in result.values())


def test_effective_rank_matches_known_representations() -> None:
    collapsed = np.column_stack([np.arange(20, dtype=np.float64), np.zeros((20, 3))])
    full_rank = np.vstack([np.eye(4, dtype=np.float64), -np.eye(4, dtype=np.float64)])

    collapsed_rank, collapsed_fraction, _ = effective_rank(collapsed)
    full_rank_value, full_rank_fraction, _ = effective_rank(full_rank)

    assert collapsed_rank == pytest.approx(1.0)
    assert collapsed_fraction == pytest.approx(0.25)
    assert full_rank_value == pytest.approx(4.0)
    assert full_rank_fraction == pytest.approx(1.0)


def test_collapse_requires_multiple_indicators() -> None:
    thresholds = {
        "posterior_collapse_effective_rank_fraction": 0.20,
        "low_latent_variance_fraction": 0.10,
    }

    assert detect_posterior_collapse(0.03, 0.05, 1.0e-5, thresholds) is True
    assert detect_posterior_collapse(1.0, 1.0, 1.0e-5, thresholds) is False


def test_integrity_defect_blocks_all_next_experiments() -> None:
    result = recommend_next_action(False, "implementation_or_protocol_defect", 1.0, True, _config())

    assert result["recommended_next_action"] == "block_due_to_integrity_issue"
    assert result["full_lfads_evaluation_allowed"] is False


def test_clear_recoverable_failure_recommends_targeted_repair() -> None:
    result = recommend_next_action(True, "posterior_or_latent_collapse", 0.06, True, _config())

    assert result["recommended_next_action"] == "targeted_lfads_repair_pilot"
    assert result["targeted_repair_available"] is True
    assert result["full_lfads_evaluation_allowed"] is False


def test_stable_unexplained_deficit_recommends_neural_ode() -> None:
    result = recommend_next_action(
        True, "model_class_or_objective_limitation", 0.01, False, _config()
    )

    assert result["recommended_next_action"] == "retire_lfads_and_start_neural_ode_pilot"
    assert result["full_lfads_evaluation_allowed"] is False
