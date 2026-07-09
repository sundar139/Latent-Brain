from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml

from latentbrain.data.io import save_neural_dataset
from latentbrain.data.schemas import NeuralDataset
from latentbrain.eval.split_audit import REPEATED_SPLIT_COLUMNS


def _script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_split_audit", Path("scripts/run_split_audit.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_dataset(path: Path, trials: int = 40, time_bins: int = 64, neurons: int = 16) -> None:
    generator = np.random.default_rng(11)
    spikes = generator.poisson(0.4, size=(trials, time_bins, neurons)).astype(np.int64)
    behavior = np.cumsum(generator.normal(0.0, 0.1, size=(trials, time_bins, 4)), axis=1)
    dataset = NeuralDataset(
        spikes=spikes,
        rates=None,
        latents=None,
        trial_ids=np.arange(trials),
        time_ms=np.arange(time_bins) * 5.0,
        bin_size_ms=5,
        metadata={"name": "unit"},
        behavior=behavior,
        behavior_names=["hand_pos_x", "hand_pos_y", "cursor_pos_x", "cursor_pos_y"],
    )
    save_neural_dataset(dataset, path)


def _config(tmp_path: Path, processed: Path) -> dict[str, Any]:
    return {
        "dataset": {
            "name": "unit",
            "processed_path": str(processed),
            "original_bin_size_ms": 5,
        },
        "splits": {
            "seed": 2027,
            "train_fraction": 0.7,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "heldout_neuron_fraction": 0.25,
        },
        "window": {"duration_seconds": 0.16, "crop_policy": "from_start"},
        "binning": {"target_bin_size_ms": 20},
        "scoring": {
            "reference_model": "train_heldout_mean_rate",
            "include_poisson_constant": True,
            "min_rate_hz": 1.0e-4,
            "max_rate_hz": 500.0,
            "primary_metric": "unified_bits_per_spike",
        },
        "audit": {
            "bootstrap_repeats": 50,
            "bootstrap_seed": 1337,
            "confidence_interval": 0.95,
            "repeated_split_seeds": [2027, 2028, 2029, 2030, 2031],
            "repeated_split_methods": ["factor_latent", "train_mean_rate", "split_mean_rate"],
            "behavior_variables": ["hand_pos_x", "hand_pos_y"],
            "derive_behavior_summaries": True,
            "derive_endpoint_direction": True,
            "trial_level_bootstrap": True,
        },
        "inputs": {
            "seed_robustness_results_path": str(tmp_path / "robustness.csv"),
            "seed_robustness_summary_path": str(tmp_path / "robustness.json"),
        },
        "references": {"train_mean_validation_bits_per_spike": 0.0},
        "reporting": {"output_dir": str(tmp_path / "out")},
    }


def _write_robustness(path: Path) -> None:
    rows = []
    for seed, validation, test in (
        (2027, 0.031, -0.008),
        (2028, 0.027, -0.009),
        (2029, 0.029, -0.007),
    ):
        rows.append(
            {
                "method_name": "factor_latent",
                "seed": seed,
                "status": "completed",
                "validation_unified_bits_per_spike": validation,
                "test_unified_bits_per_spike": test,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_missing_processed_data_fails_clearly(tmp_path: Path) -> None:
    module = _script_module()
    config = _config(tmp_path, tmp_path / "missing.npz")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert module.main(["--config", str(path)]) == 2


def test_too_few_repeated_split_seeds_fail_validation(tmp_path: Path) -> None:
    module = _script_module()
    processed = tmp_path / "data.npz"
    _write_dataset(processed)
    config = _config(tmp_path, processed)
    config["audit"]["repeated_split_seeds"] = [2027, 2028]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert module.main(["--config", str(path)]) == 2


def test_script_run_writes_expected_outputs(tmp_path: Path) -> None:
    module = _script_module()
    processed = tmp_path / "data.npz"
    _write_dataset(processed)
    config = _config(tmp_path, processed)
    _write_robustness(Path(config["inputs"]["seed_robustness_results_path"]))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert module.main(["--config", str(path)]) == 0

    out = Path(config["reporting"]["output_dir"])
    for name in (
        "split_audit_summary.json",
        "split_statistics.csv",
        "trial_statistics.csv",
        "neuron_split_statistics.csv",
        "behavior_split_statistics.csv",
        "validation_test_gap.csv",
        "repeated_split_factor_latent.csv",
        "split_audit_report.md",
    ):
        assert (out / name).exists(), name
    for name in (
        "validation_test_gap.png",
        "split_trial_rate_distributions.png",
        "split_behavior_distributions.png",
        "repeated_split_factor_latent.png",
        "validation_test_gap_bootstrap.png",
    ):
        assert (out / "figures" / name).exists(), name


def test_repeated_split_output_has_expected_columns(tmp_path: Path) -> None:
    module = _script_module()
    processed = tmp_path / "data.npz"
    _write_dataset(processed)
    config = _config(tmp_path, processed)
    _write_robustness(Path(config["inputs"]["seed_robustness_results_path"]))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert module.main(["--config", str(path)]) == 0

    repeated = pd.read_csv(
        Path(config["reporting"]["output_dir"]) / "repeated_split_factor_latent.csv"
    )
    assert list(repeated.columns) == REPEATED_SPLIT_COLUMNS
    assert set(repeated["method_name"]) == {"factor_latent", "train_mean_rate", "split_mean_rate"}
    assert len(repeated) == 15
    # The canonical reference scored against itself must be exactly zero bits/spike.
    train_mean = repeated[repeated["method_name"] == "train_mean_rate"]
    assert np.allclose(train_mean["validation_unified_bits_per_spike"], 0.0, atol=1e-12)
    assert np.allclose(train_mean["test_unified_bits_per_spike"], 0.0, atol=1e-12)


def test_script_handles_missing_seed_robustness_results(tmp_path: Path) -> None:
    module = _script_module()
    processed = tmp_path / "data.npz"
    _write_dataset(processed)
    config = _config(tmp_path, processed)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert module.main(["--config", str(path)]) == 0

    out = Path(config["reporting"]["output_dir"])
    summary = yaml.safe_load((out / "split_audit_summary.json").read_text(encoding="utf-8"))
    assert summary["model_gap_diagnostics_available"] is False
    assert summary["generalization_risk"] == "unresolved_missing_data"
    report = (out / "split_audit_report.md").read_text(encoding="utf-8")
    assert "Model gap diagnostics are unavailable" in report


def test_report_states_no_claim_until_instability_is_resolved(tmp_path: Path) -> None:
    module = _script_module()
    processed = tmp_path / "data.npz"
    _write_dataset(processed)
    config = _config(tmp_path, processed)
    _write_robustness(Path(config["inputs"]["seed_robustness_results_path"]))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert module.main(["--config", str(path)]) == 0

    report = (Path(config["reporting"]["output_dir"]) / "split_audit_report.md").read_text(
        encoding="utf-8"
    )
    assert (
        "No model performance claim should be made until validation/test instability is resolved."
        in report
    )
    assert "not an official NLB leaderboard result" in report
    assert "Old incompatible mean-rate values are not used as tuning targets" in report
    assert "Generalization risk: high" in report
