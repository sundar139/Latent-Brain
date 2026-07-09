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
        "audit_lfads_gru", Path("scripts/audit_lfads_gru.py")
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(processed: Path, checkpoint: Path, output: Path) -> dict[str, Any]:
    return {
        "dataset": {
            "name": "unit",
            "processed_path": str(processed),
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
        "window": {"max_time_bins": 256, "crop_policy": "from_start"},
        "runtime": {"device": "cuda", "fail_if_cuda_unavailable": True},
        "references": {
            "window_matched_mean_rate_validation_bits_per_spike": 0.7,
            "window_matched_factor_latent_validation_bits_per_spike": 0.03,
            "best_tuned_lfads_validation_bits_per_spike": 0.005,
            "best_tuned_lfads_run_id": "run_000",
        },
        "checkpoints": {
            "tuned_lfads_best": str(checkpoint),
            "masked_cosmoothing_best": str(checkpoint),
        },
        "audit": {
            "evaluate_splits": ["train", "validation", "test"],
            "primary_split": "validation",
            "rate_bins": 3,
            "tiny_subset_trials": 2,
            "tiny_subset_epochs": 2,
            "tiny_subset_max_time_bins": 8,
            "tiny_subset_learning_rate": 1.0e-3,
            "tiny_subset_expected_loss_drop_fraction": 0.1,
            "save_figures": False,
        },
        "reporting": {"output_dir": str(output)},
    }


def test_missing_processed_data_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _script_module()
    monkeypatch.setattr(module, "_cuda_diagnostic", lambda: {"available": False, "gpu": "NONE"})
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"x")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(_config(tmp_path / "missing.npz", checkpoint, tmp_path / "out")),
        encoding="utf-8",
    )

    assert module.main(["--config", str(config_path)]) == 2
    assert "Processed dataset is missing" in capsys.readouterr().out


def test_missing_checkpoint_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _script_module()
    monkeypatch.setattr(module, "_cuda_diagnostic", lambda: {"available": False, "gpu": "NONE"})
    processed = tmp_path / "dataset.npz"
    processed.write_bytes(b"x")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(_config(processed, tmp_path / "missing.pt", tmp_path / "out")),
        encoding="utf-8",
    )

    assert module.main(["--config", str(config_path)]) == 2
    captured = capsys.readouterr().out
    assert "LFADS checkpoint is missing" in captured
    assert "python scripts/tune_lfads_gru.py" in captured


def test_script_like_run_with_monkeypatched_audit_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script_module()
    processed = tmp_path / "dataset.npz"
    processed.write_bytes(b"x")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"x")
    output = tmp_path / "out"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config(processed, checkpoint, output)), encoding="utf-8")
    monkeypatch.setattr(module, "_cuda_diagnostic", lambda: {"available": True, "gpu": "Unit GPU"})

    def fake_audit(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
        summary = {
            "dataset_name": "unit",
            "dataset_hash": "abc",
            "window_time_bins": 256,
            "cuda_device": "Unit GPU",
            "checkpoint_audited": str(checkpoint),
            "validation_bits_per_spike": 0.1,
            "mean_rate_reference_bits_per_spike": 0.7,
            "mean_predicted_rate_hz": 2.0,
            "observed_rate_hz": 3.0,
            "prediction_reference_correlation": 0.5,
            "active_factor_count": 1,
            "tiny_overfit_initial_loss": 10.0,
            "tiny_overfit_final_loss": 5.0,
            "tiny_overfit_loss_drop_fraction": 0.5,
            "likely_issue_flags": ["underfitting"],
        }
        tables = {
            "split_diagnostics": pd.DataFrame({"split": ["validation"]}),
            "neuron_diagnostics": pd.DataFrame({"split": ["validation"]}),
            "rate_calibration": pd.DataFrame({"rate_bin": [0]}),
            "loss_scale_diagnostics": pd.DataFrame({"split": ["validation"]}),
            "tiny_subset_overfit": pd.DataFrame({"epoch": [0]}),
            "factor_usage": pd.DataFrame({"split": ["validation"], "active": [True]}),
        }
        return summary, tables

    monkeypatch.setattr(module, "run_lfads_audit", fake_audit)

    assert module.main(["--config", str(config_path)]) == 0
    assert (
        json.loads((output / "audit_summary.json").read_text())["validation_bits_per_spike"] == 0.1
    )
    assert "local diagnostic audit" in (output / "audit_report.md").read_text(encoding="utf-8")


def test_cuda_unavailable_fails_after_inputs_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _script_module()
    processed = tmp_path / "dataset.npz"
    processed.write_bytes(b"x")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"x")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(_config(processed, checkpoint, tmp_path / "out")),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "_cuda_diagnostic", lambda: {"available": False, "gpu": "NONE"})

    assert module.main(["--config", str(config_path)]) == 2
    assert "CUDA was requested" in capsys.readouterr().out
