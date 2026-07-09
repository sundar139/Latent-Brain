from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pytest
import yaml


def _script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "tune_lfads_controller", Path("scripts/tune_lfads_controller.py")
    )
    assert spec is not None and spec.loader is not None
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
            "original_bin_size_ms": 5,
        },
        "splits": {
            "seed": 2027,
            "train_fraction": 0.7,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "heldout_neuron_fraction": 0.25,
        },
        "window": {"duration_seconds": 1.28, "crop_policy": "from_start"},
        "binning": {"target_bin_size_ms": 20},
        "runtime": {"device": "cuda", "fail_if_cuda_unavailable": True},
        "scoring": {
            "reference_model": "train_heldout_mean_rate",
            "include_poisson_constant": True,
            "min_rate_hz": 1e-4,
            "max_rate_hz": 500.0,
            "primary_split": "validation",
            "primary_metric": "unified_bits_per_spike",
        },
        "search": {
            "max_runs": 1,
            "run_order": "deterministic",
            "selection_metric": "validation_unified_bits_per_spike",
            "selection_mode": "max",
        },
        "grid": {
            "encoder_hidden_dim": [64],
            "controller_hidden_dim": [64],
            "generator_hidden_dim": [96],
            "latent_dim": [16],
            "factor_dim": [32],
            "inferred_input_dim": [4],
            "input_dropout_rate": [0.0],
            "heldout_loss_weight": [4.0],
            "kl_warmup_epochs": [5],
            "kl_scale": [0.1],
            "inferred_input_kl_scale": [0.01],
            "epochs": [1],
        },
        "model": {
            "name": "lfads_controller",
            "input_neuron_group": "heldin",
            "output_dim": "all",
            "batch_size": 4,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "gradient_clip_norm": 5.0,
            "heldin_loss_weight": 1.0,
            "loss_normalization": "per_observed_spike_bin",
            "model_dropout": 0.0,
            "min_rate_hz": 1e-4,
            "max_rate_hz": 500.0,
            "checkpoint_metric": "validation_total_loss",
            "checkpoint_mode": "min",
        },
        "references": {
            "train_mean_validation_bits_per_spike": 0.0,
            "factor_latent_unified_validation_bits_per_spike": 0.03,
            "previous_best_lfads_family_validation_bits_per_spike": 0.01,
            "oracle_validation_bits_per_spike": 3.0,
        },
        "evaluation": {
            "evaluate_splits": ["train", "validation", "test"],
            "direct_model_primary": True,
            "also_evaluate_factor_decoder": True,
            "behavior_decoder_enabled": True,
        },
        "reporting": {"output_dir": str(output_dir)},
    }


def test_missing_processed_data_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _script_module()
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config(tmp_path / "out")), encoding="utf-8")

    assert module.main(["--config", str(config_path)]) == 2
    assert "Processed dataset is missing" in capsys.readouterr().out


def test_cuda_unavailable_path_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _script_module()
    processed = tmp_path / "dataset.npz"
    processed.write_bytes(b"x")
    config_path = tmp_path / "config.yaml"
    config = _config(tmp_path / "out")
    config["dataset"]["processed_path"] = str(processed)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)

    assert module.main(["--config", str(config_path)]) == 2
    assert "CUDA was requested" in capsys.readouterr().out


def test_script_like_run_writes_expected_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script_module()
    output_dir = tmp_path / "out"
    processed = tmp_path / "dataset.npz"
    processed.write_bytes(b"x")
    config_path = tmp_path / "config.yaml"
    config = _config(output_dir)
    config["dataset"]["processed_path"] = str(processed)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_cuda_diagnostic",
        lambda: {"cuda_available": True, "device_name": "Unit GPU"},
    )

    def fake_tuning(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
        results = pd.DataFrame(
            {
                "run_id": ["run_000"],
                "run_index": [0],
                "status": ["completed"],
                "validation_unified_bits_per_spike": [0.02],
                "validation_poisson_nll": [5.0],
                "validation_behavior_mean_r2": [0.1],
                "validation_factor_decoder_unified_bits_per_spike": [0.01],
                "input_dropout_rate": [0.0],
                "heldout_loss_weight": [4.0],
                "kl_scale": [0.1],
                "inferred_input_kl_scale": [0.01],
                "latent_dim": [16],
                "inferred_input_dim": [4],
                "beats_factor_latent_unified": [False],
                "beats_previous_best_lfads_family": [True],
                "output_dir": [str(output_dir / "runs" / "run_000")],
                "notes": [""],
            }
        )
        summary = {
            "dataset_name": "unit",
            "dataset_hash": "abc",
            "cuda_device": "Unit GPU",
            "bin_size_ms": 20,
            "window_seconds": 1.28,
            "reference_model": "train_heldout_mean_rate",
            "train_mean_validation_bits_per_spike": 0.0,
            "runs_attempted": 1,
            "successful_runs": 1,
            "best_run_id": "run_000",
            "best_run_params": {"latent_dim": 16},
            "best_validation_unified_bits_per_spike": 0.02,
            "best_validation_poisson_nll": 5.0,
            "best_factor_decoder_unified_bits_per_spike": 0.01,
            "best_inferred_input_kl_loss": 0.4,
            "factor_latent_unified_reference": 0.03,
            "previous_best_lfads_family_reference": 0.01,
            "beats_factor_latent_unified": False,
            "beats_previous_best_lfads_family": True,
            "old_incompatible_mean_rate_values_used_as_targets": False,
            "output_dir": str(output_dir),
        }
        return results, summary

    monkeypatch.setattr(module, "run_lfads_controller_tuning", fake_tuning)

    assert module.main(["--config", str(config_path)]) == 0
    assert (output_dir / "controller_tuning_summary.json").exists()
    report = (output_dir / "controller_tuning_report.md").read_text(encoding="utf-8")
    assert "Train-mean-as-model equals 0.0" in report
    assert "Old incompatible mean-rate values are not used as tuning targets" in report


def test_script_verifies_train_mean_as_model_zero() -> None:
    module = _script_module()

    assert module._validate_reference_zero(0.0) == 0.0
    with pytest.raises(RuntimeError, match="0.0"):
        module._validate_reference_zero(1.0)
