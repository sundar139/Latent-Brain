from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import yaml

from latentbrain.data.io import save_neural_dataset
from latentbrain.data.schemas import NeuralDataset
from latentbrain.eval.reporting import write_cosmoothing_outputs


def _script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_cosmoothing_baseline", Path("scripts/run_cosmoothing_baseline.py")
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CosmoothingConfig = _script_module().CosmoothingConfig
main = _script_module().main


def _config_dict(processed_path: str) -> dict[str, object]:
    return {
        "dataset": {
            "name": "unit",
            "processed_path": processed_path,
            "expected_hash": "abc",
            "bin_size_ms": 5,
        },
        "splits": {
            "seed": 1,
            "train_fraction": 0.6,
            "validation_fraction": 0.2,
            "test_fraction": 0.2,
            "heldout_neuron_fraction": 0.5,
        },
        "features": {
            "input_neuron_group": "heldin",
            "target_neuron_group": "heldout",
            "smoothing": {"method": "gaussian", "sigma_ms": 5.0, "truncate": 1.0},
            "convert_to_hz": True,
            "standardize_features": True,
        },
        "targets": {
            "target_transform": "counts",
            "fit_target_type": "rate_hz",
            "min_rate_hz": 1.0e-4,
            "max_rate_hz": 5000.0,
        },
        "decoder": {
            "name": "ridge",
            "alpha": 1.0,
            "fit_intercept": True,
            "train_trials_only": True,
        },
        "reference": {"name": "train_mean_rate", "fit_train_trials_only": True},
        "evaluation": {
            "evaluate_splits": ["train", "validation", "test"],
            "primary_split": "validation",
            "metrics": [
                "poisson_nll",
                "poisson_log_likelihood",
                "bits_per_spike",
                "mse_rate_hz",
                "mae_rate_hz",
            ],
        },
        "reporting": {"output_dir": "results/unit/cosmoothing_ridge"},
    }


def _dataset() -> NeuralDataset:
    spikes = np.arange(10 * 6 * 4, dtype=np.int64).reshape(10, 6, 4) % 5
    return NeuralDataset(
        spikes=spikes,
        rates=None,
        latents=None,
        trial_ids=np.arange(10, dtype=np.int64),
        time_ms=np.arange(6, dtype=np.float64) * 5,
        bin_size_ms=5,
        metadata={},
    )


def test_cosmoothing_config_validation_accepts_temp_path(tmp_path: Path) -> None:
    config = CosmoothingConfig.model_validate(_config_dict(str(tmp_path / "data.npz")))

    assert config.features.input_neuron_group == "heldin"
    assert config.features.target_neuron_group == "heldout"


def test_missing_processed_data_fails_gracefully(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(_config_dict(str(tmp_path / "missing.npz"))),
        encoding="utf-8",
    )

    assert main(["--config", str(config_path)]) == 2


def test_report_writer_creates_expected_files(tmp_path: Path) -> None:
    summary = {
        "dataset_name": "unit",
        "dataset_hash": "abc",
        "input_neuron_group": "heldin",
        "target_neuron_group": "heldout",
        "smoothing": {"method": "gaussian", "sigma_ms": 5.0},
        "decoder_name": "ridge",
        "decoder_alpha": 1.0,
        "fit_policy": "train trials only",
        "standardization_policy": "train-only held-in features",
        "reference_policy": "train-only held-out mean rates",
        "primary_split": "validation",
        "primary_bits_per_spike": 0.1,
        "primary_poisson_nll": 5.0,
        "intercept": [0.0],
    }
    split_metrics = pd.DataFrame({"split": ["validation"], "bits_per_spike": [0.1]})
    neuron_metrics = pd.DataFrame({"split": ["validation"], "target_neuron_index": [2]})
    coefficients = pd.DataFrame(
        {"input_neuron_index": [0], "target_neuron_index": [2], "coefficient": [1.0]}
    )

    paths = write_cosmoothing_outputs(
        tmp_path,
        summary,
        split_metrics,
        neuron_metrics,
        coefficients,
    )

    assert {path.name for path in paths.values()} == {
        "metrics_summary.json",
        "split_metrics.csv",
        "neuron_metrics.csv",
        "decoder_coefficients.csv",
        "cosmoothing_report.md",
    }
    assert "No neural network model was trained" in paths["report"].read_text(encoding="utf-8")


def test_script_runs_on_toy_data_and_writes_all_splits(tmp_path: Path) -> None:
    dataset = _dataset()
    processed_path = tmp_path / "dataset.npz"
    save_neural_dataset(dataset, processed_path)
    config = _config_dict(str(processed_path))
    config["dataset"]["expected_hash"] = dataset.metadata["dataset_hash"]  # type: ignore[index]
    config["reporting"] = {"output_dir": str(tmp_path / "out")}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert main(["--config", str(config_path)]) == 0
    split_metrics = pd.read_csv(tmp_path / "out" / "split_metrics.csv")
    assert split_metrics["split"].tolist() == ["train", "validation", "test"]
