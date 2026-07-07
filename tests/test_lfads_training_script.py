from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import yaml

from latentbrain.data.io import save_neural_dataset
from latentbrain.data.schemas import NeuralDataset


def _script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "train_lfads_gru", Path("scripts/train_lfads_gru.py")
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dataset() -> NeuralDataset:
    rng = np.random.default_rng(5)
    return NeuralDataset(
        spikes=rng.poisson(0.2, size=(8, 10, 6)).astype(np.int64),
        rates=None,
        latents=None,
        trial_ids=np.arange(8, dtype=np.int64),
        time_ms=np.arange(10, dtype=np.float64) * 10.0,
        bin_size_ms=10,
        metadata={},
    )


def _config(
    processed_path: str, output_dir: str, expected_hash: str | None = "abc"
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
            "max_time_bins": 8,
            "batch_size": 2,
            "num_workers": 0,
            "drop_last": False,
        },
        "model": {
            "name": "lfads_gru",
            "encoder_hidden_dim": 8,
            "generator_hidden_dim": 8,
            "latent_dim": 3,
            "factor_dim": 4,
            "dropout": 0.0,
            "min_rate_hz": 1.0e-4,
            "max_rate_hz": 500.0,
        },
        "training": {
            "seed": 2,
            "device": "cpu",
            "epochs": 1,
            "learning_rate": 1.0e-3,
            "weight_decay": 0.0,
            "gradient_clip_norm": 5.0,
            "kl_warmup_epochs": 1,
            "log_every_batches": 10,
            "checkpoint_metric": "validation_loss",
            "checkpoint_mode": "min",
        },
        "evaluation": {
            "evaluate_splits": ["train", "validation"],
            "primary_split": "validation",
            "metrics": ["poisson_nll", "bits_per_spike"],
        },
        "reporting": {"output_dir": output_dir},
    }


def test_missing_processed_data_fails_gracefully(tmp_path: Path) -> None:
    module = _script_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(_config(str(tmp_path / "missing.npz"), str(tmp_path / "out"))),
        encoding="utf-8",
    )

    assert module.main(["--config", str(config_path)]) == 2


def test_script_like_run_on_tiny_data_writes_outputs_and_report(tmp_path: Path) -> None:
    module = _script_module()
    dataset = _dataset()
    processed_path = tmp_path / "dataset.npz"
    save_neural_dataset(dataset, processed_path)
    output_dir = tmp_path / "out"
    config = _config(str(processed_path), str(output_dir), dataset.metadata["dataset_hash"])
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert module.main(["--config", str(config_path)]) == 0

    expected = {
        "config_snapshot.yaml",
        "metrics_history.csv",
        "final_metrics.json",
        "lfads_gru_report.md",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir() if path.is_file()})
    assert (output_dir / "checkpoints" / "latest.pt").exists()
    assert (output_dir / "checkpoints" / "best_validation.pt").exists()
    final_metrics = json.loads((output_dir / "final_metrics.json").read_text(encoding="utf-8"))
    assert np.isfinite(final_metrics["validation_loss"])
    report = (output_dir / "lfads_gru_report.md").read_text(encoding="utf-8")
    assert "not a full LFADS implementation" in report
    assert "No official NLB leaderboard result is reported" in report
