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
        "run_lfads_rate_calibration", Path("scripts/run_lfads_rate_calibration.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(output: Path, checkpoint: Path) -> dict[str, Any]:
    return {
        "dataset": {
            "name": "unit",
            "processed_path": "data.npz",
            "expected_hash": "abc",
            "original_bin_size_ms": 5,
        },
        "splits": {
            "seed": 1,
            "train_fraction": 0.7,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "heldout_neuron_fraction": 0.25,
        },
        "window": {"duration_seconds": 1.28, "crop_policy": "from_start"},
        "runtime": {"device": "cuda", "fail_if_cuda_unavailable": True},
        "binning": {"target_bin_size_ms": 20},
        "references": {
            "same_bin_mean_rate_validation_bits_per_spike": 0.7,
            "same_bin_factor_latent_validation_bits_per_spike": 0.03,
            "previous_20ms_lfads_validation_bits_per_spike": 0.01,
        },
        "existing_lfads": {"checkpoint_path": str(checkpoint), "prediction_source": "direct_model"},
        "posthoc_calibration": {
            "enabled": True,
            "methods": ["multiplicative_per_neuron", "log_bias_per_neuron", "mean_rate_blend"],
            "blend_alpha_grid": [0.0, 0.5, 1.0],
            "min_rate_hz": 1.0e-4,
            "max_rate_hz": 500.0,
            "fit_train_trials_only": True,
        },
        "initialized_lfads": {
            "enabled": True,
            "output_dir_name": "initialized_20ms",
            "initialize_readout_bias_from_train_rates": True,
            "encoder_hidden_dim": 8,
            "generator_hidden_dim": 8,
            "latent_dim": 2,
            "factor_dim": 2,
            "dropout": 0.0,
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-5,
            "epochs": 1,
            "batch_size": 1,
            "gradient_clip_norm": 1.0,
            "heldin_loss_weight": 1.0,
            "heldout_loss_weight": 1.0,
            "kl_warmup_epochs": 1,
            "loss_normalization": "mean",
            "min_rate_hz": 1.0e-4,
            "max_rate_hz": 500.0,
        },
        "evaluation": {
            "primary_split": "validation",
            "primary_metric": "bits_per_spike",
            "evaluate_splits": ["train", "validation", "test"],
        },
        "reporting": {"output_dir": str(output)},
    }


def test_missing_checkpoint_fails_clearly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _script_module()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_config(tmp_path / "out", tmp_path / "missing.pt")))
    monkey_cuda = {"available": True, "gpu": "Unit GPU"}
    module._cuda_diagnostic = lambda: monkey_cuda

    assert module.main(["--config", str(path)]) == 2
    assert "LFADS-style checkpoint is missing" in capsys.readouterr().out


def test_cuda_unavailable_path_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _script_module()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"x")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_config(tmp_path / "out", checkpoint)))
    monkeypatch.setattr(module, "_cuda_diagnostic", lambda: {"available": False, "gpu": "NONE"})

    assert module.main(["--config", str(path)]) == 2
    assert "CUDA was requested" in capsys.readouterr().out


def test_script_like_run_with_monkeypatched_predictions_and_training_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script_module()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"x")
    output = tmp_path / "out"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_config(output, checkpoint)))
    monkeypatch.setattr(module, "_cuda_diagnostic", lambda: {"available": True, "gpu": "Unit GPU"})
    summary = {
        "dataset_name": "unit",
        "dataset_hash": "abc",
        "bin_size_ms": 20,
        "window_seconds": 1.28,
        "cuda_device": "Unit GPU",
        "existing_checkpoint_path": "checkpoint.pt",
        "raw_lfads_validation_bits_per_spike": 0.01,
        "multiplicative_calibrated_validation_bits_per_spike": 0.02,
        "log_bias_calibrated_validation_bits_per_spike": 0.02,
        "best_blend_alpha": 0.0,
        "best_blend_validation_bits_per_spike": 0.0,
        "initialized_lfads_validation_bits_per_spike": 0.03,
        "initialized_calibrated_validation_bits_per_spike": 0.04,
        "same_bin_mean_rate_reference": 0.7,
        "same_bin_factor_latent_reference": 0.03,
        "calibration_improves_lfads": True,
        "initialization_improves_lfads": True,
        "beats_same_bin_factor_latent": True,
        "beats_same_bin_mean_rate": False,
        "best_lfads_family_method": "initialized_calibrated",
    }
    calibration = pd.DataFrame(
        {
            "method_name": ["raw_lfads", "multiplicative_per_neuron"],
            "split": ["validation", "validation"],
            "bits_per_spike": [0.01, 0.02],
            "mean_predicted_rate_hz": [1.0, 2.0],
        }
    )
    blend = pd.DataFrame(
        {"alpha": [0.0, 1.0], "split": ["validation", "validation"], "bits_per_spike": [0.0, 0.01]}
    )
    monkeypatch.setattr(
        module,
        "run_lfads_rate_calibration",
        lambda _: (
            summary,
            {
                "calibration_metrics": calibration,
                "blend_metrics": blend,
                "initialized_lfads_metrics": pd.DataFrame(),
            },
        ),
    )

    assert module.main(["--config", str(path)]) == 0
    assert (
        json.loads((output / "rate_calibration_summary.json").read_text())[
            "best_lfads_family_method"
        ]
        == "initialized_calibrated"
    )
    assert (
        "local rate-calibration diagnostic work" in (output / "calibration_report.md").read_text()
    )
