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
        "run_seed_robustness", Path("scripts/run_seed_robustness.py")
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
            "base_seed": 2027,
            "train_fraction": 0.7,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "heldout_neuron_fraction": 0.25,
            "split_seed_mode": "fixed",
            "split_seed": 2027,
            "initialization_seed_mode": "varied",
        },
        "window": {"duration_seconds": 1.28, "crop_policy": "from_start"},
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
        "seeds": [2027, 2028, 2029],
        "methods": [
            {
                "name": "factor_latent",
                "type": "factor_latent",
                "valid_model": True,
                "fallback_config": {"latent_dim": 8},
                "notes": "baseline",
            },
            {
                "name": "neural_ode_refinement",
                "type": "neural_ode",
                "valid_model": True,
                "fallback_config": {"epochs": 1, "diffusion_scale": 0.0},
                "notes": "dynamics",
            },
        ],
        "references": {
            "train_mean_validation_bits_per_spike": 0.0,
            "factor_latent_single_seed_reference": 0.0316,
            "neural_ode_refinement_single_seed_reference": 0.0283,
            "neural_ode_objective_single_seed_reference": 0.0115,
            "oracle_validation_bits_per_spike": 3.54,
        },
        "statistics": {
            "confidence_interval": 0.95,
            "bootstrap_repeats": 100,
            "bootstrap_seed": 1337,
            "paired_by_seed": True,
        },
        "evaluation": {
            "evaluate_splits": ["train", "validation", "test"],
            "direct_model_primary": True,
            "also_evaluate_factor_decoder": True,
            "behavior_decoder_enabled": True,
        },
        "reporting": {"output_dir": str(tmp_path / "out")},
    }


def _results() -> pd.DataFrame:
    rows = []
    for seed, factor_bits, neural_bits in (
        (2027, 0.031, 0.010),
        (2028, 0.032, 0.012),
        (2029, 0.030, 0.011),
    ):
        for name, method_type, bits in (
            ("factor_latent", "factor_latent", factor_bits),
            ("neural_ode_refinement", "neural_ode", neural_bits),
        ):
            rows.append(
                {
                    "method_name": name,
                    "method_type": method_type,
                    "seed": seed,
                    "split_seed": 2027,
                    "initialization_seed": seed,
                    "config_hash": "abc123",
                    "valid_model": True,
                    "status": "completed",
                    "validation_unified_bits_per_spike": bits,
                    "validation_poisson_nll": 2000.0,
                    "validation_behavior_mean_r2": 0.0,
                    "validation_factor_decoder_unified_bits_per_spike": bits,
                    "train_unified_bits_per_spike": bits + 0.01,
                    "test_unified_bits_per_spike": bits - 0.001,
                    "beats_train_mean_reference": True,
                    "beats_factor_latent_single_seed_reference": False,
                    "beats_neural_ode_refinement_single_seed_reference": False,
                    "output_dir": "out",
                    "notes": "",
                }
            )
    return pd.DataFrame(rows)


def _summary() -> dict[str, Any]:
    return {
        "dataset_name": "unit",
        "dataset_hash": "abc",
        "bin_size_ms": 20,
        "window_seconds": 1.28,
        "cuda_device": "Unit GPU",
        "reference_model": "train_heldout_mean_rate",
        "train_mean_validation_bits_per_spike": 0.0,
        "split_seed_mode": "fixed",
        "split_seed": 2027,
        "initialization_seed_mode": "varied",
        "seed_list_shared_across_methods": True,
        "methods_evaluated": ["factor_latent", "neural_ode_refinement"],
        "seeds_evaluated": [2027, 2028, 2029],
        "total_jobs": 6,
        "successful_jobs": 6,
        "method_config_hashes": {"factor_latent": "abc", "neural_ode_refinement": "def"},
        "confidence_interval": 0.95,
        "bootstrap_repeats": 100,
        "bootstrap_seed": 1337,
        "best_mean_method": "factor_latent",
        "best_mean_validation_unified_bits_per_spike": 0.031,
        "best_lower_ci_method": "factor_latent",
        "best_lower_ci_validation_unified_bits_per_spike": 0.0303,
        "factor_latent_mean_validation_unified_bits_per_spike": 0.031,
        "best_neural_method": "neural_ode_refinement",
        "best_neural_method_mean_validation_unified_bits_per_spike": 0.011,
        "paired_mean_difference_best_neural_minus_factor_latent": -0.02,
        "any_neural_beats_factor_latent_mean": False,
        "any_neural_beats_factor_latent_lower_ci": False,
        "carried_forward_method": "factor_latent",
        "carried_forward_reason": "No neural method beats factor-latent across seeds.",
        "output_dir": "out",
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


def test_too_few_seeds_fail_config_validation(tmp_path: Path) -> None:
    module = _script_module()
    config = _config(tmp_path)
    Path(config["dataset"]["processed_path"]).write_bytes(b"exists")
    config["seeds"] = [2027, 2028]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert module.main(["--config", str(path)]) == 2


def test_nonzero_diffusion_scale_fails_config_validation(tmp_path: Path) -> None:
    module = _script_module()
    config = _config(tmp_path)
    Path(config["dataset"]["processed_path"]).write_bytes(b"exists")
    config["methods"][1]["fallback_config"]["diffusion_scale"] = 0.2
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert module.main(["--config", str(path)]) == 2


def test_reference_zero_validation_rejects_nonzero() -> None:
    module = _script_module()

    with pytest.raises(RuntimeError, match="must be 0.0"):
        module._validate_reference_zero(0.1)


def test_script_like_run_writes_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _script_module()
    config = _config(tmp_path)
    Path(config["dataset"]["processed_path"]).write_bytes(b"exists")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(module, "_cuda_diagnostic", lambda: _FAKE_CUDA)
    monkeypatch.setattr(module, "run_seed_robustness", lambda _config: (_results(), _summary()))

    assert module.main(["--config", str(path)]) == 0

    out = Path(config["reporting"]["output_dir"])
    assert (out / "seed_robustness_summary.json").exists()
    assert (out / "seed_robustness_results.csv").exists()
    assert (out / "seed_robustness_leaderboard.csv").exists()
    assert (out / "method_summary.csv").exists()
    assert (out / "seed_effects.csv").exists()
    assert (out / "carried_forward_config.yaml").exists()
    assert (out / "figures" / "validation_bits_by_method_seed.png").exists()
    assert (out / "figures" / "method_mean_ci.png").exists()
    assert (out / "figures" / "seed_effects.png").exists()
    assert (out / "figures" / "neural_ode_vs_factor_latent_by_seed.png").exists()

    report = (out / "seed_robustness_report.md").read_text(encoding="utf-8")
    assert "Single-seed model leaderboards are not sufficient for claims." in report
    assert "not an official NLB leaderboard result" in report
    assert "Old incompatible mean-rate values are not used as tuning targets" in report
    assert "Train-mean-as-model equals 0.0 bits/spike." in report


def test_script_rejects_nonzero_train_mean_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script_module()
    config = _config(tmp_path)
    Path(config["dataset"]["processed_path"]).write_bytes(b"exists")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    broken = _summary() | {"train_mean_validation_bits_per_spike": 0.5}
    monkeypatch.setattr(module, "_cuda_diagnostic", lambda: _FAKE_CUDA)
    monkeypatch.setattr(module, "run_seed_robustness", lambda _config: (_results(), broken))

    assert module.main(["--config", str(path)]) == 2
