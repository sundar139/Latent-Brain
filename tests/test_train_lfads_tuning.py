from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from latentbrain.train import lfads_tuning
from latentbrain.train.lfads_tuning import build_lfads_run_config, run_lfads_tuning


def _base_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "dataset": {
            "name": "unit",
            "processed_path": str(tmp_path / "dataset.npz"),
            "expected_hash": "abc",
            "bin_size_ms": 5,
        },
        "splits": {
            "seed": 2027,
            "train_fraction": 0.7,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "heldout_neuron_fraction": 0.25,
        },
        "data": {
            "input_neuron_group": "heldin",
            "target_neuron_group": "heldout",
            "max_time_bins": 256,
            "batch_size": 4,
            "num_workers": 0,
            "drop_last": False,
        },
        "runtime": {"device": "cuda", "fail_if_cuda_unavailable": True},
        "search": {
            "max_runs": 2,
            "selection_metric": "validation_bits_per_spike",
            "selection_mode": "max",
            "run_order": "deterministic",
        },
        "grid": {
            "encoder_hidden_dim": [64, 96],
            "generator_hidden_dim": [64],
            "latent_dim": [16],
            "factor_dim": [32],
            "dropout": [0.0],
            "learning_rate": [0.001],
            "weight_decay": [0.0],
            "heldout_loss_weight": [1.0],
            "kl_warmup_epochs": [5],
        },
        "model": {
            "name": "lfads_gru",
            "input_dim": None,
            "output_dim": "all",
            "min_rate_hz": 1e-4,
            "max_rate_hz": 500.0,
        },
        "training": {
            "seed": 2027,
            "epochs": 10,
            "gradient_clip_norm": 5.0,
            "heldin_loss_weight": 1.0,
            "loss_normalization": "per_observed_spike_bin",
            "log_every_batches": 10,
            "checkpoint_metric": "validation_total_loss",
            "checkpoint_mode": "min",
        },
        "evaluation": {
            "primary_split": "validation",
            "primary_metric": "bits_per_spike",
            "direct_model_primary": True,
            "also_evaluate_factor_decoder": True,
            "behavior_decoder_enabled": True,
            "baseline_references": {
                "window_matched_mean_rate_validation_bits_per_spike": 0.7,
                "window_matched_factor_latent_validation_bits_per_spike": 0.03,
                "previous_lfads_masked_direct_validation_bits_per_spike": -0.04,
            },
        },
        "reporting": {"output_dir": str(tmp_path / "out")},
    }


def test_run_config_applies_params_and_does_not_mutate_base(tmp_path: Path) -> None:
    base = _base_config(tmp_path)
    original = deepcopy(base)
    run_dir = tmp_path / "out" / "runs" / "run_000"

    config = build_lfads_run_config(
        base,
        {
            "encoder_hidden_dim": 96,
            "generator_hidden_dim": 64,
            "latent_dim": 16,
            "factor_dim": 32,
            "dropout": 0.1,
            "learning_rate": 0.002,
            "weight_decay": 1e-5,
            "heldout_loss_weight": 2.0,
            "kl_warmup_epochs": 5,
        },
        run_dir,
    )

    assert base == original
    assert config["model"]["encoder_hidden_dim"] == 96
    assert config["model"]["dropout"] == 0.1
    assert config["training"]["device"] == "cuda"
    assert config["training"]["learning_rate"] == 0.002
    assert config["training"]["heldout_loss_weight"] == 2.0
    assert config["reporting"]["output_dir"] == str(run_dir)
    assert config["data"]["max_time_bins"] == 256


def test_run_output_dirs_are_unique(tmp_path: Path) -> None:
    base = _base_config(tmp_path)
    dirs = [
        build_lfads_run_config(base, {"encoder_hidden_dim": 64}, tmp_path / f"run_{i}")[
            "reporting"
        ]["output_dir"]
        for i in range(2)
    ]

    assert len(set(dirs)) == 2


def test_toy_tuning_run_can_be_monkeypatched_without_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _base_config(tmp_path)
    monkeypatch.setattr(lfads_tuning, "_validate_cuda", lambda config: "Unit GPU")
    monkeypatch.setattr(
        lfads_tuning,
        "_load_windowed_dataset",
        lambda config: (None, "hash", {"cropped_time_bins": 256, "window_seconds": 1.28}),
    )

    def fake_run(
        run_config: dict[str, Any], run_index: int, run_id: str, dataset: object
    ) -> dict[str, Any]:
        return {
            "validation_bits_per_spike": 0.1 + run_index,
            "validation_poisson_nll": 5.0 - run_index,
            "validation_behavior_mean_r2": 0.0,
            "validation_total_loss": 1.0,
            "validation_heldout_prediction_loss": 2.0,
        }

    monkeypatch.setattr(lfads_tuning, "_train_and_evaluate_run", fake_run)

    results, summary = run_lfads_tuning(config)

    assert len(results) == 2
    assert summary["best_run_id"] == results.iloc[1]["run_id"]
    assert (tmp_path / "out" / "best_config.yaml").exists()


def test_expected_failed_run_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _base_config(tmp_path)
    monkeypatch.setattr(lfads_tuning, "_validate_cuda", lambda config: "Unit GPU")
    monkeypatch.setattr(
        lfads_tuning,
        "_load_windowed_dataset",
        lambda config: (None, "hash", {"cropped_time_bins": 256, "window_seconds": 1.28}),
    )

    def fake_run(
        run_config: dict[str, Any], run_index: int, run_id: str, dataset: object
    ) -> dict[str, Any]:
        if run_index == 0:
            raise lfads_tuning.RecoverableRunError("unit recoverable")
        return {
            "validation_bits_per_spike": 0.2,
            "validation_poisson_nll": 5.0,
            "validation_behavior_mean_r2": 0.1,
            "validation_total_loss": 1.0,
            "validation_heldout_prediction_loss": 2.0,
        }

    monkeypatch.setattr(lfads_tuning, "_train_and_evaluate_run", fake_run)

    results, summary = run_lfads_tuning(config)

    assert results.iloc[0]["status"] == "failed"
    assert "unit recoverable" in results.iloc[0]["notes"]
    assert summary["successful_runs"] == 1
