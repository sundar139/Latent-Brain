from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pytest


def _script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_lfads_coordinated_dropout", Path("scripts/run_lfads_coordinated_dropout.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(tmp_path: Path) -> dict[str, Any]:
    return {
        "dataset": {
            "name": "mc_maze_small",
            "processed_path": str(tmp_path / "missing.npz"),
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
        "runtime": {"device": "cuda", "fail_if_cuda_unavailable": True},
        "binning": {"target_bin_size_ms": 20},
        "dropout": {
            "enabled": True,
            "mode": "neuron",
            "rates": [0.10],
            "apply_to": ["train"],
            "resample_each_batch": True,
            "keep_at_least_one_neuron": True,
            "seed": 2027,
        },
        "lfads_settings": {
            "encoder_hidden_dim": 8,
            "generator_hidden_dim": 8,
            "latent_dim": 4,
            "factor_dim": 4,
            "dropout": 0.0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "epochs": 1,
            "batch_size": 2,
            "gradient_clip_norm": 5.0,
            "heldin_loss_weight": 1.0,
            "heldout_loss_weight": 1.0,
            "kl_warmup_epochs": 1,
            "loss_normalization": "mean",
            "min_rate_hz": 1e-4,
            "max_rate_hz": 500.0,
        },
        "evaluation": {
            "primary_split": "validation",
            "primary_metric": "bits_per_spike",
            "evaluate_splits": ["train", "validation", "test"],
            "direct_model_primary": True,
            "also_evaluate_factor_decoder": True,
            "behavior_decoder_enabled": True,
        },
        "references": {
            "same_bin_mean_rate_validation_bits_per_spike": 0.7,
            "same_bin_factor_latent_validation_bits_per_spike": 0.03,
            "previous_20ms_lfads_validation_bits_per_spike": 0.01,
            "multiplicative_calibrated_validation_bits_per_spike": 0.007,
            "initialized_lfads_validation_bits_per_spike": -0.005,
        },
        "reporting": {"output_dir": str(tmp_path / "results")},
    }


def test_missing_processed_data_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script_module()
    monkeypatch.setattr(module, "_cuda_diagnostic", lambda: {"available": True, "gpu": "GPU"})

    with pytest.raises(FileNotFoundError, match="Processed dataset is missing"):
        module.run_lfads_coordinated_dropout(_config(tmp_path))


def test_cuda_unavailable_path_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script_module()
    monkeypatch.setattr(module, "_cuda_diagnostic", lambda: {"available": False, "gpu": "NONE"})

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        module.run_lfads_coordinated_dropout(_config(tmp_path))


def test_script_like_run_writes_expected_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script_module()
    config = _config(tmp_path)
    config_path = tmp_path / "config.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    summary = {
        "dataset_name": "mc_maze_small",
        "dataset_hash": "abc",
        "bin_size_ms": 20,
        "window_seconds": 1.28,
        "cuda_device": "Unit GPU",
        "dropout_rates_tested": [0.1],
        "best_dropout_rate": 0.1,
        "best_validation_bits_per_spike": 0.02,
        "best_validation_poisson_nll": 1.2,
        "best_validation_factor_decoder_bits_per_spike": 0.01,
        "same_bin_mean_rate_reference": 0.7,
        "same_bin_factor_latent_reference": 0.03,
        "previous_20ms_lfads_reference": 0.01,
        "coordinated_dropout_improves_lfads": True,
        "beats_same_bin_factor_latent": False,
        "beats_same_bin_mean_rate": False,
    }
    tables = {
        "training_metrics": pd.DataFrame(
            {
                "run_id": ["dropout_0p10"],
                "epoch": [1],
                "train_total_loss": [1.0],
                "validation_total_loss": [1.1],
            }
        ),
        "evaluation_metrics": pd.DataFrame(
            {
                "run_id": ["dropout_0p10"],
                "dropout_rate": [0.1],
                "validation_bits_per_spike": [0.02],
                "validation_poisson_nll": [1.2],
            }
        ),
        "dropout_diagnostics": pd.DataFrame(
            {"run_id": ["dropout_0p10"], "realized_input_dropout_fraction": [0.1]}
        ),
    }
    monkeypatch.setattr(module, "run_lfads_coordinated_dropout", lambda _config: (summary, tables))

    assert module.main(["--config", str(config_path)]) == 0
    report = Path(config["reporting"]["output_dir"]) / "coordinated_dropout_report.md"
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "not an official NLB leaderboard result" in text
    assert "LFADS-style only, not full LFADS" in text
