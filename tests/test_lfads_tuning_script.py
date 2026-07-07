from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pytest
import yaml


def _script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "tune_lfads_gru", Path("scripts/tune_lfads_gru.py")
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(output_dir: Path) -> dict[str, Any]:
    return {
        "dataset": {
            "name": "unit",
            "processed_path": "missing.npz",
            "expected_hash": "abc",
            "bin_size_ms": 5,
        },
        "splits": {
            "seed": 1,
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
            "max_runs": 1,
            "selection_metric": "validation_bits_per_spike",
            "selection_mode": "max",
            "run_order": "deterministic",
        },
        "grid": {
            "encoder_hidden_dim": [64],
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
            "seed": 1,
            "epochs": 1,
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
        "reporting": {"output_dir": str(output_dir)},
    }


def test_missing_config_fails_clearly(capsys: pytest.CaptureFixture[str]) -> None:
    module = _script_module()

    assert module.main(["--config", "missing_tuning_config.yaml"]) == 2
    assert "Config file is missing" in capsys.readouterr().out


def test_cuda_unavailable_path_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _script_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config(tmp_path / "out")), encoding="utf-8")
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)

    assert module.main(["--config", str(config_path)]) == 2
    assert "CUDA was requested" in capsys.readouterr().out


def test_script_like_run_with_monkeypatched_tuning_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script_module()
    output_dir = tmp_path / "out"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config(output_dir)), encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_cuda_diagnostic",
        lambda: {
            "torch_version": "unit",
            "cuda_available": True,
            "torch_cuda": "unit",
            "gpu_name": "Unit GPU",
        },
    )

    def fake_tuning(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
        output = Path(config["reporting"]["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        results = pd.DataFrame(
            {
                "run_id": ["run_000"],
                "run_index": [0],
                "status": ["completed"],
                "validation_bits_per_spike": [0.2],
                "validation_poisson_nll": [5.0],
                "validation_behavior_mean_r2": [0.1],
                "validation_total_loss": [1.0],
                "validation_heldout_prediction_loss": [2.0],
                "beats_window_matched_mean_rate": [False],
                "beats_window_matched_factor_latent": [True],
                "beats_previous_lfads_masked_direct": [True],
                "output_dir": [str(output / "runs" / "run_000")],
                "notes": [""],
            }
        )
        summary = {
            "dataset_name": "unit",
            "dataset_hash": "abc",
            "window_time_bins": 256,
            "window_seconds": 1.28,
            "cuda_device": "Unit GPU",
            "runs_attempted": 1,
            "successful_runs": 1,
            "best_run_id": "run_000",
            "best_run_params": {"latent_dim": 16},
            "best_validation_bits_per_spike": 0.2,
            "best_validation_poisson_nll": 5.0,
            "best_validation_behavior_mean_r2": 0.1,
            "beats_window_matched_mean_rate": False,
            "beats_window_matched_factor_latent": True,
            "beats_previous_lfads_masked_direct": True,
            "baseline_references": config["evaluation"]["baseline_references"],
        }
        return results, summary

    monkeypatch.setattr(module, "run_lfads_tuning", fake_tuning)

    assert module.main(["--config", str(config_path)]) == 0
    assert (
        json.loads((output_dir / "tuning_summary.json").read_text(encoding="utf-8"))["best_run_id"]
        == "run_000"
    )
    assert (output_dir / "tuning_results.csv").exists()
    assert (output_dir / "validation_leaderboard.csv").exists()
    report = (output_dir / "tuning_report.md").read_text(encoding="utf-8")
    assert "local validation tuning only" in report
