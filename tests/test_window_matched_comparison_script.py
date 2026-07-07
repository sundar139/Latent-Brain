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
        "run_window_matched_comparison", Path("scripts/run_window_matched_comparison.py")
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dataset() -> NeuralDataset:
    rng = np.random.default_rng(7)
    behavior_t = np.arange(6, dtype=np.float64)[None, :, None]
    behavior_trial = np.arange(8, dtype=np.float64)[:, None, None]
    return NeuralDataset(
        spikes=rng.poisson(0.2, size=(8, 6, 6)).astype(np.int64),
        rates=None,
        latents=None,
        trial_ids=np.arange(8, dtype=np.int64),
        time_ms=np.arange(6, dtype=np.float64) * 5.0,
        bin_size_ms=5,
        metadata={},
        behavior=np.concatenate([behavior_t + behavior_trial, behavior_t - behavior_trial], axis=2),
        behavior_names=["hand_pos_x", "hand_pos_y"],
    )


def _config(processed_path: str, output_dir: str, expected_hash: str = "abc") -> dict[str, object]:
    return {
        "dataset": {
            "name": "unit",
            "processed_path": processed_path,
            "expected_hash": expected_hash,
            "bin_size_ms": 5,
        },
        "splits": {
            "seed": 2,
            "train_fraction": 0.5,
            "validation_fraction": 0.25,
            "test_fraction": 0.25,
            "heldout_neuron_fraction": 0.33,
        },
        "window": {"max_time_bins": 4, "crop_policy": "from_start"},
        "references": {
            "full_window_mean_rate_bits_per_spike": 0.5,
            "full_window_factor_latent_best_bits_per_spike": 0.1,
            "lfads_heldin_checkpoint_path": "missing.pt",
            "lfads_cosmoothing_checkpoint_path": "missing2.pt",
        },
        "methods": {
            "include": [
                "mean_rate_windowed",
                "ridge_cosmoothing_windowed",
                "factor_latent_windowed",
                "lfads_gru_factor_decoder",
            ]
        },
        "evaluation": {
            "primary_split": "validation",
            "primary_metric": "bits_per_spike",
            "behavior_metric": "mean_r2",
            "evaluate_splits": ["train", "validation", "test"],
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


def test_toy_methods_write_outputs_and_missing_checkpoint_note(tmp_path: Path) -> None:
    module = _script_module()
    dataset = _dataset()
    processed = tmp_path / "dataset.npz"
    save_neural_dataset(dataset, processed)
    output_dir = tmp_path / "out"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(_config(str(processed), str(output_dir), dataset.metadata["dataset_hash"])),
        encoding="utf-8",
    )

    assert module.main(["--config", str(config_path)]) == 0

    expected = {
        "comparison_summary.json",
        "comparison_metrics.csv",
        "validation_leaderboard.csv",
        "behavior_comparison.csv",
        "comparison_report.md",
    }
    assert expected == {path.name for path in output_dir.iterdir() if path.is_file()}
    summary = json.loads((output_dir / "comparison_summary.json").read_text(encoding="utf-8"))
    assert summary["official_benchmark_claim"] is False
    metrics_text = (output_dir / "comparison_metrics.csv").read_text(encoding="utf-8")
    assert "lfads_gru_factor_decoder" in metrics_text
    assert "checkpoint missing" in metrics_text
    report = (output_dir / "comparison_report.md").read_text(encoding="utf-8")
    assert "not an official NLB leaderboard result" in report
    assert "No new neural network model was trained" in report
