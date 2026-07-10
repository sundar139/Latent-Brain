from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pytest
import yaml

# Mirrors the keys of the script's own _cuda_diagnostic() so tests never touch real CUDA.
_FAKE_CUDA: dict[str, Any] = {
    "torch": "unit-test",
    "cuda_available": True,
    "torch_cuda": "unit-test",
    "device_count": 1,
    "device_name": "Unit Test GPU",
}


def _script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "refine_neural_ode", Path("scripts/refine_neural_ode.py")
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
            "drift_regularization": [0.0],
            "learning_rate": [0.001],
            "scheduler": ["cosine"],
            "diffusion_scale": [0.0],
            "epochs": [1],
        },
        "model": {
            "name": "neural_ode_refinement",
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
            "diffusion_scale": 0.0,
            "checkpoint_metric": "validation_total_loss",
            "checkpoint_mode": "min",
            "save_unified_checkpoints": True,
            "evaluate_checkpoints_by_unified_metric": True,
        },
        "references": {
            "train_mean_validation_bits_per_spike": 0.0,
            "factor_latent_unified_validation_bits_per_spike": 0.03,
            "previous_neural_ode_validation_bits_per_spike": 0.018,
            "previous_switching_ode_validation_bits_per_spike": 0.006,
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
            "run_index": [0],
            "status": ["completed"],
            "validation_unified_bits_per_spike": [0.02],
            "validation_poisson_nll": [1.0],
            "validation_behavior_mean_r2": [0.0],
            "validation_factor_decoder_unified_bits_per_spike": [0.01],
            "drift_regularization_loss": [0.0],
            "input_dropout_rate": [0.0],
            "heldout_loss_weight": [1.0],
            "kl_warmup_epochs": [1],
            "kl_scale": [0.1],
            "drift_regularization": [0.0],
            "scheduler": ["cosine"],
            "latent_dim": [2],
            "factor_dim": [2],
            "best_checkpoint_source": ["latest"],
            "beats_factor_latent_unified": [False],
            "beats_previous_neural_ode": [True],
            "notes": [""],
            "output_dir": [str(tmp_path / "out" / "runs" / "run_000")],
        }
    )
    summary = {
        "train_mean_validation_bits_per_spike": 0.0,
        "dataset_name": "unit",
        "reference_model": "train_heldout_mean_rate",
        "best_run_id": "run_000",
        "best_validation_unified_bits_per_spike": 0.02,
        "best_validation_poisson_nll": 1.0,
        "best_factor_decoder_unified_bits_per_spike": 0.01,
        "best_drift_norm": 0.2,
        "best_drift_regularization_loss": 0.0,
        "best_learning_rate": 0.0001,
        "best_checkpoint_source": "latest",
        "factor_latent_unified_reference": 0.03,
        "previous_neural_ode_reference": 0.018,
        "previous_switching_ode_reference": 0.006,
        "beats_factor_latent_unified": False,
        "beats_previous_neural_ode": True,
        "old_incompatible_mean_rate_values_used_as_targets": False,
    }
    out = Path(config["reporting"]["output_dir"])
    out.mkdir(parents=True)
    pd.DataFrame({"checkpoint_source": ["latest"], "selected_by_unified": [True]}).to_csv(
        out / "checkpoint_selection.csv", index=False
    )
    monkeypatch.setattr(module, "_cuda_diagnostic", lambda: _FAKE_CUDA)
    monkeypatch.setattr(module, "run_neural_ode_refinement", lambda _config: (results, summary))

    assert module.main(["--config", str(path)]) == 0
    assert (out / "neural_ode_refinement_summary.json").exists()
    assert (out / "neural_ode_refinement_report.md").exists()
    report = (out / "neural_ode_refinement_report.md").read_text(encoding="utf-8")
    assert "Old incompatible mean-rate values are not used as tuning targets" in report


def test_reference_zero_validation_rejects_nonzero() -> None:
    module = _script_module()

    with pytest.raises(RuntimeError, match="must be 0.0"):
        module._validate_reference_zero(0.1)
