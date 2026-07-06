from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from latentbrain.data.io import save_neural_dataset
from latentbrain.data.schemas import NeuralDataset
from latentbrain.eval.reporting import write_behavior_decoder_outputs


def _script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_behavior_decoder", Path("scripts/run_behavior_decoder.py")
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BehaviorDecoderConfig = _script_module().BehaviorDecoderConfig
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
            "heldout_neuron_fraction": 0.25,
        },
        "features": {
            "neuron_group": "heldin",
            "smoothing": {"method": "gaussian", "sigma_ms": 50.0, "truncate": 4.0},
            "convert_to_hz": True,
            "standardize_features": True,
        },
        "targets": {
            "source_behavior_prefixes": ["hand_pos"],
            "derive_velocity": True,
            "velocity_method": "central_difference",
            "standardize_targets": True,
        },
        "decoder": {
            "name": "ridge",
            "alpha": 100.0,
            "fit_intercept": True,
            "train_trials_only": True,
        },
        "evaluation": {
            "evaluate_splits": ["train", "validation", "test"],
            "primary_split": "validation",
            "primary_target_prefix": "hand_pos_velocity",
            "metrics": ["r2", "mse", "mae"],
        },
        "reporting": {"output_dir": "results/unit/behavior_decoder"},
    }


def test_behavior_decoder_config_validation_accepts_temp_path(tmp_path: Path) -> None:
    config = BehaviorDecoderConfig.model_validate(_config_dict(str(tmp_path / "data.npz")))

    assert config.dataset.bin_size_ms == 5
    assert config.evaluation.primary_split == "validation"


def test_missing_processed_data_fails_gracefully(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    import yaml

    config_path.write_text(
        yaml.safe_dump(_config_dict(str(tmp_path / "missing.npz"))),
        encoding="utf-8",
    )

    assert main(["--config", str(config_path)]) == 2


def test_report_writer_creates_expected_files(tmp_path: Path) -> None:
    summary = {
        "dataset_name": "unit",
        "dataset_hash": "abc",
        "feature_neuron_group": "heldin",
        "smoothing": {"method": "gaussian", "sigma_ms": 50.0},
        "target_names": ["hand_pos_velocity_x"],
        "decoder_name": "ridge",
        "decoder_alpha": 1.0,
        "fit_policy": "train trials only",
        "standardization_policy": "train-only statistics",
        "primary_split": "validation",
        "primary_mean_r2": 0.5,
        "intercept": [0.0],
    }
    split_metrics = pd.DataFrame({"split": ["validation"], "mean_r2": [0.5]})
    target_metrics = pd.DataFrame(
        {"split": ["validation"], "target_name": ["hand_pos_velocity_x"], "r2": [0.5]}
    )
    coefficients = pd.DataFrame(
        {"feature_index": [0], "target_name": ["hand_pos_velocity_x"], "coefficient": [1.0]}
    )

    paths = write_behavior_decoder_outputs(
        tmp_path, summary, split_metrics, target_metrics, coefficients
    )

    assert {path.name for path in paths.values()} == {
        "metrics_summary.json",
        "split_metrics.csv",
        "target_metrics.csv",
        "decoder_coefficients.csv",
        "behavior_decoder_report.md",
    }
    assert "No neural network model was trained" in paths["report"].read_text(encoding="utf-8")


def test_script_runs_on_synthetic_behavior_dataset(tmp_path: Path) -> None:
    dataset = NeuralDataset(
        spikes=np.arange(10 * 6 * 4, dtype=np.int64).reshape(10, 6, 4) % 3,
        rates=None,
        latents=None,
        trial_ids=np.arange(10, dtype=np.int64),
        time_ms=np.arange(6, dtype=np.float64) * 5,
        bin_size_ms=5,
        metadata={},
        behavior=np.stack(
            [
                np.tile(np.arange(6, dtype=np.float64), (10, 1)),
                np.tile(np.arange(6, dtype=np.float64) * 2.0, (10, 1)),
            ],
            axis=2,
        ),
        behavior_names=["hand_pos_x", "hand_pos_y"],
    )
    processed_path = tmp_path / "dataset.npz"
    save_neural_dataset(dataset, processed_path)
    config = _config_dict(str(processed_path))
    config["dataset"]["expected_hash"] = dataset.metadata["dataset_hash"]  # type: ignore[index]
    config["reporting"] = {"output_dir": str(tmp_path / "out")}
    config_path = tmp_path / "config.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert main(["--config", str(config_path)]) == 0
    assert (tmp_path / "out" / "behavior_decoder_report.md").exists()
