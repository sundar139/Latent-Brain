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
        "run_factor_latent_baseline", Path("scripts/run_factor_latent_baseline.py")
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FactorLatentConfig = _script_module().FactorLatentConfig
main = _script_module().main


def _dataset() -> NeuralDataset:
    rng = np.random.default_rng(11)
    heldin = rng.poisson(0.25, size=(8, 6, 4)).astype(np.int64)
    heldout = np.stack([heldin[:, :, 0] + 1, heldin[:, :, 2] + 1], axis=2).astype(np.int64)
    spikes = np.concatenate([heldin, heldout], axis=2)
    t = np.arange(6, dtype=np.float64)[None, :, None]
    trial = np.arange(8, dtype=np.float64)[:, None, None]
    t_all = np.broadcast_to(t, (8, 6, 1))
    behavior = np.concatenate([t + trial, t - trial, 2.0 * t_all, -t_all], axis=2)
    return NeuralDataset(
        spikes=spikes,
        rates=None,
        latents=None,
        trial_ids=np.arange(8, dtype=np.int64),
        time_ms=np.arange(6, dtype=np.float64) * 5,
        bin_size_ms=5,
        metadata={},
        behavior=behavior,
        behavior_names=["hand_pos_x", "hand_pos_y", "cursor_pos_x", "cursor_pos_y"],
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
            "heldout_neuron_fraction": 0.33,
        },
        "features": {
            "input_neuron_group": "heldin",
            "target_neuron_group": "heldout",
            "smoothing": {"method": "gaussian", "sigma_ms": 5.0, "truncate": 1.0},
            "convert_to_hz": True,
            "standardize_features": True,
        },
        "latent_model": {
            "name": "factor_analysis",
            "latent_dim": 2,
            "random_state": 1,
            "max_iter": 300,
            "tol": 1.0e-4,
            "train_trials_only": True,
        },
        "heldout_decoder": {
            "name": "ridge",
            "alpha": 1.0,
            "fit_intercept": True,
            "min_rate_hz": 1.0e-4,
            "max_rate_hz": 5000.0,
            "train_trials_only": True,
        },
        "behavior_decoder": {
            "enabled": True,
            "alpha": 1.0,
            "fit_intercept": True,
            "target_prefixes": ["hand_pos", "cursor_pos"],
            "derive_velocity": True,
            "velocity_method": "central_difference",
            "standardize_targets": True,
            "train_trials_only": True,
        },
        "reference": {"name": "train_mean_rate", "fit_train_trials_only": True},
        "evaluation": {
            "evaluate_splits": ["train", "validation", "test"],
            "primary_split": "validation",
            "primary_metric": "bits_per_spike",
            "metrics": [
                "poisson_nll",
                "poisson_log_likelihood",
                "bits_per_spike",
                "mse_rate_hz",
                "mae_rate_hz",
                "behavior_r2",
            ],
        },
        "reporting": {"output_dir": output_dir},
    }


def test_config_validation_accepts_temp_path(tmp_path: Path) -> None:
    config = FactorLatentConfig.model_validate(
        _config_dict(str(tmp_path / "dataset.npz"), str(tmp_path / "out"))
    )

    assert config.latent_model.latent_dim == 2
    assert config.evaluation.primary_split == "validation"


def test_missing_processed_data_fails_gracefully(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(_config_dict(str(tmp_path / "missing.npz"), str(tmp_path / "out"))),
        encoding="utf-8",
    )

    assert main(["--config", str(config_path)]) == 2


def test_script_execution_on_toy_data_writes_outputs(tmp_path: Path) -> None:
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
        "metrics_summary.json",
        "split_metrics.csv",
        "neuron_metrics.csv",
        "behavior_metrics.csv",
        "latent_summary.csv",
        "factor_loadings.csv",
        "heldout_decoder_coefficients.csv",
        "behavior_decoder_coefficients.csv",
        "factor_latent_report.md",
    }
    assert expected == {path.name for path in output_dir.iterdir() if path.is_file()}
    split_metrics = pd.read_csv(output_dir / "split_metrics.csv")
    assert set(split_metrics["split"]) == {"train", "validation", "test"}
