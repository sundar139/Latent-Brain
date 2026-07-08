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
        "run_temporal_rebinning_diagnostic", Path("scripts/run_temporal_rebinning_diagnostic.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(processed: Path, output: Path) -> dict[str, Any]:
    return {
        "dataset": {
            "name": "unit",
            "processed_path": str(processed),
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
        "binning": {"target_bin_size_ms": [5, 10], "train_lfads_for_bin_size_ms": [10]},
        "baseline_settings": {
            "mean_rate": {"enabled": True},
            "factor_latent": {
                "enabled": True,
                "latent_dim": 2,
                "smoothing_sigma_ms": 20.0,
                "heldout_decoder_alpha": 1.0,
                "standardize_features": True,
            },
        },
        "lfads_settings": {
            "encoder_hidden_dim": 8,
            "generator_hidden_dim": 8,
            "latent_dim": 2,
            "factor_dim": 2,
            "dropout": 0.0,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "epochs": 1,
            "batch_size": 1,
            "gradient_clip_norm": 1.0,
            "heldin_loss_weight": 1.0,
            "heldout_loss_weight": 1.0,
            "kl_warmup_epochs": 1,
            "loss_normalization": "per_observed_spike_bin",
            "min_rate_hz": 1e-4,
            "max_rate_hz": 500.0,
        },
        "evaluation": {
            "primary_split": "validation",
            "primary_metric": "bits_per_spike",
            "evaluate_splits": ["train", "validation", "test"],
            "references": {
                "window_matched_5ms_mean_rate_validation_bits_per_spike": 0.7,
                "best_tuned_5ms_lfads_validation_bits_per_spike": 0.1,
            },
        },
        "reporting": {"output_dir": str(output)},
    }


def test_missing_processed_data_fails_clearly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _script_module()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_config(tmp_path / "missing.npz", tmp_path / "out")))
    assert module.main(["--config", str(path)]) == 2
    assert "Processed dataset is missing" in capsys.readouterr().out


def test_cuda_unavailable_path_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _script_module()
    processed = tmp_path / "data.npz"
    processed.write_bytes(b"x")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_config(processed, tmp_path / "out")))
    monkeypatch.setattr(module, "_cuda_diagnostic", lambda: {"available": False, "gpu": "NONE"})
    assert module.main(["--config", str(path)]) == 2
    assert "CUDA was requested" in capsys.readouterr().out


def test_script_like_run_with_monkeypatched_diagnostic_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script_module()
    processed = tmp_path / "data.npz"
    processed.write_bytes(b"x")
    output = tmp_path / "out"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_config(processed, output)))
    monkeypatch.setattr(module, "_cuda_diagnostic", lambda: {"available": True, "gpu": "Unit GPU"})
    summary = {
        "dataset_name": "unit",
        "dataset_hash": "abc",
        "original_bin_size_ms": 5,
        "target_bin_sizes_ms": [5, 10],
        "window_seconds": 1.28,
        "best_lfads_bin_size_ms": 10,
        "lfads_beat_same_bin_mean_rate": False,
        "coarser_lfads_improved_over_5ms": True,
    }
    tables = {
        "sparsity": pd.DataFrame({"bin_size_ms": [5], "split": ["validation"]}),
        "baseline_metrics": pd.DataFrame({"bin_size_ms": [5], "method_name": ["mean_rate"]}),
        "lfads_metrics": pd.DataFrame({"bin_size_ms": [10], "bits_per_spike": [0.2]}),
    }
    monkeypatch.setattr(module, "run_temporal_rebinning_diagnostic", lambda _: (summary, tables))
    assert module.main(["--config", str(path)]) == 0
    assert (
        json.loads((output / "rebinning_summary.json").read_text())["best_lfads_bin_size_ms"] == 10
    )
    assert (
        "local temporal-binning diagnostic" in (output / "temporal_rebinning_report.md").read_text()
    )
