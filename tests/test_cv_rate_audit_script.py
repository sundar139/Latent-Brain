from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
import yaml

from latentbrain.data.io import save_neural_dataset
from latentbrain.data.schemas import NeuralDataset
from latentbrain.eval.cv_rate_audit import (
    METHOD_SUMMARY_COLUMNS,
    RATE_CONTROL_COLUMNS,
    REPEATED_SPLIT_COLUMNS,
)
from latentbrain.eval.rate_controls import SPLIT_MEAN_RATE_INVALID, TRAIN_MEAN_RATE


def _script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_cv_rate_audit", Path("scripts/run_cv_rate_audit.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_dataset(path: Path, trials: int = 40, time_bins: int = 32, neurons: int = 16) -> None:
    generator = np.random.default_rng(5)
    spikes = generator.poisson(0.4, size=(trials, time_bins, neurons)).astype(np.int64)
    dataset = NeuralDataset(
        spikes=spikes,
        rates=None,
        latents=None,
        trial_ids=np.arange(trials),
        time_ms=np.arange(time_bins) * 5.0,
        bin_size_ms=5,
        metadata={"name": "unit"},
        behavior=None,
        behavior_names=None,
    )
    save_neural_dataset(dataset, path)


def _config(tmp_path: Path, processed: Path) -> dict[str, Any]:
    return {
        "dataset": {"name": "unit", "processed_path": str(processed), "original_bin_size_ms": 5},
        "splits": {
            "train_fraction": 0.7,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "heldout_neuron_fraction": 0.25,
            "split_seeds": list(range(2027, 2037)),
            "factor_analysis_random_states": [0, 2027, 2028],
        },
        "window": {"duration_seconds": 0.08, "crop_policy": "from_start"},
        "binning": {"target_bin_size_ms": 20},
        "scoring": {
            "reference_model": "train_heldout_mean_rate",
            "include_poisson_constant": True,
            "min_rate_hz": 1.0e-4,
            "max_rate_hz": 500.0,
            "primary_metric": "unified_bits_per_spike",
        },
        "factor_latent": {
            "latent_dim": 4,
            "smoothing_sigma_ms": 200.0,
            "heldout_decoder_alpha": 10000.0,
            "standardize_features": True,
            "fit_intercept": True,
        },
        "rate_controls": {
            "include_train_mean_rate": True,
            "include_split_mean_rate_invalid": True,
            "include_train_per_neuron_mean_rate": True,
            "include_train_population_scaled_mean_rate": True,
            "include_train_split_rate_calibrated_factor_latent": True,
            "include_oracle_split_scaled_factor_latent_invalid": True,
        },
        "statistics": {
            "confidence_interval": 0.95,
            "bootstrap_repeats": 50,
            "bootstrap_seed": 1337,
        },
        "inputs": {
            "split_audit_summary_path": str(tmp_path / "split_audit.json"),
            "seed_robustness_summary_path": str(tmp_path / "robustness.json"),
        },
        "references": {
            "accepted_split_seed": 2027,
            "accepted_factor_latent_validation_mean": 0.029,
            "accepted_factor_latent_test_mean": -0.0083,
            "repeated_split_factor_latent_test_mean": 0.0082,
            "split_mean_rate_validation_reference": 0.0879,
            "split_mean_rate_test_reference": 0.0924,
            "train_mean_validation_bits_per_spike": 0.0,
        },
        "reporting": {"output_dir": str(tmp_path / "out")},
    }


def test_missing_processed_data_fails_clearly(tmp_path: Path) -> None:
    module = _script_module()
    config = _config(tmp_path, tmp_path / "missing.npz")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert module.main(["--config", str(path)]) == 2


def test_too_few_split_seeds_fail_validation(tmp_path: Path) -> None:
    module = _script_module()
    processed = tmp_path / "data.npz"
    _write_dataset(processed)
    config = _config(tmp_path, processed)
    config["splits"]["split_seeds"] = [2027, 2028]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert module.main(["--config", str(path)]) == 2


def test_too_few_random_states_fail_validation(tmp_path: Path) -> None:
    module = _script_module()
    processed = tmp_path / "data.npz"
    _write_dataset(processed)
    config = _config(tmp_path, processed)
    config["splits"]["factor_analysis_random_states"] = [0, 1]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert module.main(["--config", str(path)]) == 2


def test_unrecognized_rate_control_fails_validation(tmp_path: Path) -> None:
    module = _script_module()
    processed = tmp_path / "data.npz"
    _write_dataset(processed)
    config = _config(tmp_path, processed)
    config["rate_controls"]["include_magic_control"] = True
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert module.main(["--config", str(path)]) == 2


def test_script_run_writes_expected_outputs(tmp_path: Path) -> None:
    module = _script_module()
    processed = tmp_path / "data.npz"
    _write_dataset(processed)
    config = _config(tmp_path, processed)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert module.main(["--config", str(path)]) == 0

    out = Path(config["reporting"]["output_dir"])
    for name in (
        "cv_rate_audit_summary.json",
        "repeated_split_scores.csv",
        "factor_analysis_random_state_sensitivity.csv",
        "rate_control_scores.csv",
        "rate_offset_decomposition.csv",
        "method_summary.csv",
        "reporting_recommendations.json",
        "cv_rate_audit_report.md",
    ):
        assert (out / name).exists(), name
    for name in (
        "repeated_split_score_distribution.png",
        "factor_analysis_random_state_sensitivity.png",
        "rate_control_comparison.png",
        "rate_offset_decomposition.png",
        "validation_test_by_split.png",
    ):
        assert (out / "figures" / name).exists(), name

    repeated = pd.read_csv(out / "repeated_split_scores.csv")
    assert list(repeated.columns) == REPEATED_SPLIT_COLUMNS
    assert len(repeated) == 10 * 3

    controls = pd.read_csv(out / "rate_control_scores.csv")
    assert list(controls.columns) == RATE_CONTROL_COLUMNS
    # The canonical reference scored against itself is exactly zero on every split.
    train_mean = controls[controls["method_name"] == TRAIN_MEAN_RATE]
    assert np.allclose(train_mean["unified_bits_per_spike"], 0.0, atol=1e-12)
    invalid = controls[controls["method_name"] == SPLIT_MEAN_RATE_INVALID]
    assert not bool(invalid["valid_model"].any())

    summary = pd.read_csv(out / "method_summary.csv")
    assert list(summary.columns) == METHOD_SUMMARY_COLUMNS


def test_script_handles_missing_split_audit_inputs(tmp_path: Path) -> None:
    module = _script_module()
    processed = tmp_path / "data.npz"
    _write_dataset(processed)
    config = _config(tmp_path, processed)
    config["inputs"]["split_audit_summary_path"] = str(tmp_path / "absent.json")
    config["inputs"]["seed_robustness_summary_path"] = str(tmp_path / "absent2.json")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert module.main(["--config", str(path)]) == 0
    assert (Path(config["reporting"]["output_dir"]) / "cv_rate_audit_summary.json").exists()


def test_best_valid_method_is_never_an_invalid_control(tmp_path: Path) -> None:
    module = _script_module()
    processed = tmp_path / "data.npz"
    _write_dataset(processed)
    config = _config(tmp_path, processed)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert module.main(["--config", str(path)]) == 0

    out = Path(config["reporting"]["output_dir"])
    summary = json.loads((out / "cv_rate_audit_summary.json").read_text(encoding="utf-8"))
    assert summary["best_valid_rate_control_method"] not in summary["invalid_control_methods"]
    assert summary["invalid_controls_excluded_from_best_valid_model"] is True
    assert summary["single_split_results_reportable"] is False

    recommendations = json.loads(
        (out / "reporting_recommendations.json").read_text(encoding="utf-8")
    )
    assert recommendations["recommended_reporting_mode"] == "repeated_split"
    assert recommendations["neural_models_carried_forward"] is False


def test_report_states_invalid_controls_and_single_split_limits(tmp_path: Path) -> None:
    module = _script_module()
    processed = tmp_path / "data.npz"
    _write_dataset(processed)
    config = _config(tmp_path, processed)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert module.main(["--config", str(path)]) == 0

    report = (Path(config["reporting"]["output_dir"]) / "cv_rate_audit_report.md").read_text(
        encoding="utf-8"
    )
    assert (
        "Invalid controls use evaluation split targets and cannot be reported as model performance."
        in report
    )
    assert "Single-split numbers are not reportable as final performance." in report
    assert "not an official NLB leaderboard result" in report
    assert "Old incompatible mean-rate values are not used as tuning targets" in report
    assert "Invalid diagnostic controls" in report


def test_reference_validation_rejects_missing_keys(tmp_path: Path) -> None:
    module = _script_module()
    processed = tmp_path / "data.npz"
    _write_dataset(processed)
    config = _config(tmp_path, processed)
    del config["references"]["accepted_split_seed"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert module.main(["--config", str(path)]) == 2


def test_reference_zero_is_preserved_for_train_mean(tmp_path: Path) -> None:
    module = _script_module()
    processed = tmp_path / "data.npz"
    _write_dataset(processed)
    config = _config(tmp_path, processed)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert module.main(["--config", str(path)]) == 0

    summary = pd.read_csv(Path(config["reporting"]["output_dir"]) / "method_summary.csv")
    train_mean = summary[summary["method_name"] == TRAIN_MEAN_RATE].iloc[0]
    assert train_mean["mean_validation_unified_bits_per_spike"] == pytest.approx(0.0, abs=1e-12)
    assert train_mean["mean_test_unified_bits_per_spike"] == pytest.approx(0.0, abs=1e-12)
