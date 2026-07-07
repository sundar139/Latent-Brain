from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import yaml

from latentbrain.data.io import save_neural_dataset
from latentbrain.data.schemas import NeuralDataset
from latentbrain.models.lfads_gru import LFADSGRU, LFADSGRUConfig
from latentbrain.torch.checkpoints import save_checkpoint


def _script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "evaluate_lfads_gru", Path("scripts/evaluate_lfads_gru.py")
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dataset() -> NeuralDataset:
    rng = np.random.default_rng(13)
    heldin = rng.poisson(0.2, size=(8, 8, 4)).astype(np.int64)
    heldout = np.stack([heldin[:, :, 0] + 1, heldin[:, :, 1] + 1], axis=2).astype(np.int64)
    t = np.arange(8, dtype=np.float64)[None, :, None]
    trial = np.arange(8, dtype=np.float64)[:, None, None]
    t_all = np.broadcast_to(t, (8, 8, 1))
    return NeuralDataset(
        spikes=np.concatenate([heldin, heldout], axis=2),
        rates=None,
        latents=None,
        trial_ids=np.arange(8, dtype=np.int64),
        time_ms=np.arange(8, dtype=np.float64) * 10,
        bin_size_ms=10,
        metadata={},
        behavior=np.concatenate([t_all + trial, t_all - trial, 2 * t_all, -t_all], axis=2),
        behavior_names=["hand_pos_x", "hand_pos_y", "cursor_pos_x", "cursor_pos_y"],
    )


def _checkpoint(path: Path) -> Path:
    model = LFADSGRU(LFADSGRUConfig(4, 4, 6, 6, 3, 5, 0.0, 1.0e-4, 500.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    return save_checkpoint(path, model, optimizer, 1, {"validation_loss": 1.0}, {"unit": True})


def _config(
    processed_path: str, checkpoint_path: str, output_dir: str, expected_hash: str = "abc"
) -> dict[str, object]:
    return {
        "dataset": {
            "name": "unit",
            "processed_path": processed_path,
            "expected_hash": expected_hash,
            "bin_size_ms": 10,
        },
        "splits": {
            "seed": 2,
            "train_fraction": 0.5,
            "validation_fraction": 0.25,
            "test_fraction": 0.25,
            "heldout_neuron_fraction": 0.33,
        },
        "data": {
            "input_neuron_group": "heldin",
            "target_neuron_group": "heldout",
            "max_time_bins": 6,
            "batch_size": 2,
            "num_workers": 0,
            "drop_last": False,
        },
        "model": {
            "name": "lfads_gru",
            "checkpoint_path": checkpoint_path,
            "encoder_hidden_dim": 6,
            "generator_hidden_dim": 6,
            "latent_dim": 3,
            "factor_dim": 5,
            "dropout": 0.0,
            "min_rate_hz": 1.0e-4,
            "max_rate_hz": 500.0,
        },
        "heldout_decoder": {
            "name": "ridge",
            "alpha": 1.0,
            "fit_intercept": True,
            "min_rate_hz": 1.0e-4,
            "max_rate_hz": 500.0,
            "standardize_factors": True,
            "train_trials_only": True,
        },
        "behavior_decoder": {
            "enabled": True,
            "alpha": 1.0,
            "fit_intercept": True,
            "standardize_factors": True,
            "standardize_targets": True,
            "target_prefixes": ["hand_pos", "cursor_pos"],
            "derive_velocity": True,
            "velocity_method": "central_difference",
            "train_trials_only": True,
        },
        "reference": {"name": "train_mean_rate", "fit_train_trials_only": True},
        "evaluation": {
            "evaluate_splits": ["train", "validation", "test"],
            "primary_split": "validation",
            "baseline_references": {
                "mean_rate_validation_bits_per_spike": 0.5,
                "factor_latent_best_validation_bits_per_spike": 0.1,
                "factor_latent_best_behavior_mean_r2": 0.03,
            },
        },
        "reporting": {"output_dir": output_dir},
    }


def test_missing_processed_data_fails_gracefully(tmp_path: Path) -> None:
    module = _script_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            _config(
                str(tmp_path / "missing.npz"), str(tmp_path / "missing.pt"), str(tmp_path / "out")
            )
        ),
        encoding="utf-8",
    )

    assert module.main(["--config", str(config_path)]) == 2


def test_missing_checkpoint_fails_gracefully(tmp_path: Path) -> None:
    module = _script_module()
    dataset = _dataset()
    processed = tmp_path / "dataset.npz"
    save_neural_dataset(dataset, processed)
    config = _config(
        str(processed),
        str(tmp_path / "missing.pt"),
        str(tmp_path / "out"),
        dataset.metadata["dataset_hash"],
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert module.main(["--config", str(config_path)]) == 2


def test_script_like_execution_writes_outputs_and_report(tmp_path: Path) -> None:
    module = _script_module()
    dataset = _dataset()
    processed = tmp_path / "dataset.npz"
    save_neural_dataset(dataset, processed)
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt")
    output_dir = tmp_path / "out"
    config = _config(
        str(processed), str(checkpoint), str(output_dir), dataset.metadata["dataset_hash"]
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert module.main(["--config", str(config_path)]) == 0

    expected = {
        "metrics_summary.json",
        "split_metrics.csv",
        "neuron_metrics.csv",
        "behavior_metrics.csv",
        "factor_summary.csv",
        "heldout_decoder_coefficients.csv",
        "behavior_decoder_coefficients.csv",
        "lfads_gru_eval_report.md",
    }
    assert expected == {path.name for path in output_dir.iterdir() if path.is_file()}
    assert not (output_dir / "checkpoints").exists()
    report = (output_dir / "lfads_gru_eval_report.md").read_text(encoding="utf-8")
    assert "No new neural network model was trained by this evaluation script" in report
    assert "not an official NLB leaderboard result" in report
