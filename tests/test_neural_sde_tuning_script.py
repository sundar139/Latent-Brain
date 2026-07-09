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
        "tune_neural_sde", Path("scripts/tune_neural_sde.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(tmp_path: Path) -> dict[str, Any]:
    return {
        "dataset": {
            "name": "unit",
            "processed_path": str(tmp_path / "data.npz"),
            "expected_hash": "abc",
            "original_bin_size_ms": 5,
        },
        "splits": {
            "seed": 2027,
            "train_fraction": 0.5,
            "validation_fraction": 0.25,
            "test_fraction": 0.25,
            "heldout_neuron_fraction": 0.5,
        },
        "window": {"duration_seconds": 0.04, "crop_policy": "from_start"},
        "binning": {"target_bin_size_ms": 20},
        "runtime": {"device": "cuda", "fail_if_cuda_unavailable": True},
        "scoring": {
            "reference_model": "train_heldout_mean_rate",
            "include_poisson_constant": True,
            "min_rate_hz": 1.0e-4,
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
            "encoder_hidden_dim": [4],
            "drift_hidden_dim": [4],
            "diffusion_hidden_dim": [4],
            "latent_dim": [2],
            "factor_dim": [2],
            "input_dropout_rate": [0.0],
            "heldout_loss_weight": [1.0],
            "kl_warmup_epochs": [1],
            "kl_scale": [0.1],
            "diffusion_scale": [0.0],
            "epochs": [1],
        },
        "model": {
            "name": "neural_sde",
            "input_neuron_group": "heldin",
            "output_dim": "all",
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "heldin_loss_weight": 1.0,
            "loss_normalization": "mean",
            "model_dropout": 0.0,
            "min_rate_hz": 1.0e-4,
            "max_rate_hz": 500.0,
            "dt_seconds": 0.02,
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
        "reporting": {"output_dir": str(tmp_path / "out")},
    }


def test_missing_processed_data_fails_before_cuda(tmp_path: Path) -> None:
    module = _script_module()
    config = _config(tmp_path)
    config["dataset"]["processed_path"] = str(tmp_path / "missing.npz")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert module.main(["--config", str(path)]) == 2


def test_cuda_unavailable_fails_after_input_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script_module()
    config = _config(tmp_path)
    Path(config["dataset"]["processed_path"]).write_bytes(b"exists")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)

    assert module.main(["--config", str(path)]) == 2


def test_script_like_run_writes_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _script_module()
    config = _config(tmp_path)
    Path(config["dataset"]["processed_path"]).write_bytes(b"exists")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    results = pd.DataFrame(
        {
            "run_id": ["run_000"],
            "status": ["completed"],
            "validation_unified_bits_per_spike": [0.02],
            "validation_poisson_nll": [2.0],
            "validation_factor_decoder_unified_bits_per_spike": [0.01],
            "input_dropout_rate": [0.0],
            "heldout_loss_weight": [1.0],
            "kl_scale": [0.1],
            "diffusion_scale": [0.0],
            "latent_dim": [2],
            "factor_dim": [2],
            "beats_factor_latent_unified": [False],
            "beats_previous_best_lfads_family": [True],
            "notes": [""],
            "run_index": [0],
            "validation_behavior_mean_r2": [0.0],
            "drift_norm": [0.3],
            "diffusion_mean": [0.0],
        }
    )
    summary = {
        "dataset_name": "unit",
        "dataset_hash": "abc",
        "cuda_device": "Unit GPU",
        "bin_size_ms": 20,
        "window_seconds": 0.04,
        "reference_model": "train_heldout_mean_rate",
        "train_mean_validation_bits_per_spike": 0.0,
        "runs_attempted": 1,
        "successful_runs": 1,
        "best_run_id": "run_000",
        "best_run_params": {"latent_dim": 2},
        "best_validation_unified_bits_per_spike": 0.02,
        "best_validation_poisson_nll": 2.0,
        "best_factor_decoder_unified_bits_per_spike": 0.01,
        "best_drift_norm": 0.3,
        "best_diffusion_mean": 0.0,
        "factor_latent_unified_reference": 0.03,
        "previous_best_lfads_family_reference": 0.01,
        "beats_factor_latent_unified": False,
        "beats_previous_best_lfads_family": True,
        "old_incompatible_mean_rate_values_used_as_targets": False,
        "output_dir": str(config["reporting"]["output_dir"]),
    }
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "get_device_name", lambda _idx: "Unit GPU")
    monkeypatch.setattr(module, "run_neural_sde_tuning", lambda _config: (results, summary))

    assert module.main(["--config", str(path)]) == 0

    output = Path(config["reporting"]["output_dir"])
    assert (
        json.loads((output / "neural_sde_tuning_summary.json").read_text())["best_run_id"]
        == "run_000"
    )
    report = (output / "neural_sde_tuning_report.md").read_text(encoding="utf-8")
    assert "Old incompatible mean-rate values are not used as tuning targets" in report
    assert "not an official NLB leaderboard result" in report
