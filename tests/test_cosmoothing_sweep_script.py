from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml

from latentbrain.data.io import save_neural_dataset
from latentbrain.data.schemas import NeuralDataset


def _script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_cosmoothing_sweep", Path("scripts/run_cosmoothing_sweep.py")
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SweepConfig = _script_module().CosmoothingSweepConfig
main = _script_module().main


def _dataset() -> NeuralDataset:
    rng = np.random.default_rng(4)
    heldin = rng.poisson(0.2, size=(12, 8, 3)).astype(np.int64)
    heldout_a = heldin[:, :, 0] + heldin[:, :, 1]
    heldout_b = heldin[:, :, 2] + 1
    spikes = np.concatenate([heldin, np.stack([heldout_a, heldout_b], axis=2)], axis=2)
    return NeuralDataset(
        spikes=spikes,
        rates=None,
        latents=None,
        trial_ids=np.arange(spikes.shape[0], dtype=np.int64),
        time_ms=np.arange(spikes.shape[1], dtype=np.float64) * 5,
        bin_size_ms=5,
        metadata={},
    )


def _config_dict(processed_path: str, output_dir: str) -> dict[str, object]:
    return {
        "dataset": {
            "name": "unit",
            "processed_path": processed_path,
            "expected_hash": "abc",
            "bin_size_ms": 5,
        },
        "splits": {
            "seed": 1,
            "train_fraction": 0.5,
            "validation_fraction": 0.25,
            "test_fraction": 0.25,
            "heldout_neuron_fraction": 0.4,
        },
        "features": {
            "input_neuron_group": "heldin",
            "target_neuron_group": "heldout",
            "convert_to_hz": True,
        },
        "sweep": {
            "smoothing_sigma_ms": [5.0, 10.0],
            "ridge_alpha": [0.1, 1.0],
            "standardize_features": [True, False],
            "fit_intercept": [True],
        },
        "targets": {"fit_target_type": "rate_hz", "min_rate_hz": 1.0e-4, "max_rate_hz": 5000.0},
        "reference": {"name": "train_mean_rate", "fit_train_trials_only": True},
        "evaluation": {
            "primary_split": "validation",
            "primary_metric": "bits_per_spike",
            "evaluate_splits": ["train", "validation", "test"],
        },
        "reporting": {"output_dir": output_dir},
    }


def test_sweep_config_validation_accepts_temp_path(tmp_path: Path) -> None:
    config = SweepConfig.model_validate(_config_dict(str(tmp_path / "data.npz"), str(tmp_path)))

    assert config.sweep.smoothing_sigma_ms == [5.0, 10.0]
    assert config.evaluation.primary_split == "validation"


def test_missing_processed_data_fails_gracefully(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(_config_dict(str(tmp_path / "missing.npz"), str(tmp_path / "out"))),
        encoding="utf-8",
    )

    assert main(["--config", str(config_path)]) == 2


def test_toy_dataset_sweep_writes_expected_files_and_rows(tmp_path: Path) -> None:
    dataset = _dataset()
    processed_path = tmp_path / "dataset.npz"
    save_neural_dataset(dataset, processed_path)
    output_dir = tmp_path / "out"
    config = _config_dict(str(processed_path), str(output_dir))
    config["dataset"]["expected_hash"] = dataset.metadata["dataset_hash"]  # type: ignore[index]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert main(["--config", str(config_path)]) == 0

    expected = {
        "sweep_results.csv",
        "best_config.json",
        "best_split_metrics.csv",
        "best_neuron_metrics.csv",
        "sweep_report.md",
    }
    assert expected == {path.name for path in output_dir.iterdir() if path.is_file()}
    sweep_results = pd.read_csv(output_dir / "sweep_results.csv")
    assert len(sweep_results) == 8 * 3
    assert set(sweep_results["split"]) == {"train", "validation", "test"}


def test_all_negative_bits_warning_does_not_fail(tmp_path: Path, capsys: object) -> None:
    dataset = _dataset()
    processed_path = tmp_path / "dataset.npz"
    save_neural_dataset(dataset, processed_path)
    output_dir = tmp_path / "out"
    config = _config_dict(str(processed_path), str(output_dir))
    config["dataset"]["expected_hash"] = dataset.metadata["dataset_hash"]  # type: ignore[index]
    config["targets"] = {"fit_target_type": "rate_hz", "min_rate_hz": 1000.0, "max_rate_hz": 1001.0}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert main(["--config", str(config_path)]) == 0

    captured = capsys.readouterr()
    assert "warning" in captured.out.lower()
