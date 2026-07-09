from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
import yaml

from latentbrain.data.schemas import NeuralDataset


def _script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_unified_scoreboard", Path("scripts/run_unified_scoreboard.py")
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
            "processed_path": str(tmp_path / "missing.npz"),
            "expected_hash": "abc",
            "original_bin_size_ms": 5,
        },
        "splits": {
            "seed": 2027,
            "train_fraction": 0.5,
            "validation_fraction": 0.25,
            "test_fraction": 0.25,
            "heldout_neuron_fraction": 0.5,
        },
        "window": {"duration_seconds": 0.04, "crop_policy": "from_start"},
        "binning": {"target_bin_size_ms": 20},
        "scoring": {
            "reference_model": "train_heldout_mean_rate",
            "include_poisson_constant": True,
            "min_rate_hz": 1e-4,
            "max_rate_hz": 500.0,
            "primary_split": "validation",
            "primary_metric": "bits_per_spike",
        },
        "inputs": {
            "metric_audit_dir": str(tmp_path / "metric_audit"),
            "temporal_rebinning_dir": str(tmp_path / "temporal_rebinning"),
            "coordinated_dropout_dir": str(tmp_path / "dropout"),
            "rate_calibration_dir": str(tmp_path / "calibration"),
        },
        "known_unified_values": {
            "train_mean_as_model_validation_bits_per_spike": 0.0,
            "split_mean_validation_bits_per_spike": 0.08477702283785095,
            "factor_latent_unified_validation_bits_per_spike": 0.0316438194429199,
            "lfads_unified_validation_bits_per_spike": 0.0094416136085523,
            "coordinated_dropout_unified_validation_bits_per_spike": 0.0094749929355431,
            "best_oracle_validation_bits_per_spike": 3.5417067067892387,
        },
        "historical_incompatible_values": {
            "old_window_matched_mean_rate_validation_bits_per_spike": 0.7019005614036485,
            "old_full_window_mean_rate_validation_bits_per_spike": 0.5465273967210786,
        },
        "reporting": {"output_dir": str(tmp_path / "unified_scoreboard")},
    }


def _toy_dataset() -> NeuralDataset:
    spikes = np.array(
        [
            [[0, 1, 2, 0], [1, 0, 1, 0]],
            [[0, 1, 1, 1], [1, 0, 0, 1]],
            [[1, 0, 2, 0], [0, 1, 1, 0]],
            [[0, 2, 0, 1], [1, 1, 0, 1]],
        ],
        dtype=np.int64,
    )
    return NeuralDataset(
        spikes=spikes,
        rates=None,
        latents=None,
        behavior=None,
        trial_ids=np.arange(4),
        time_ms=np.array([0.0, 5.0]),
        bin_size_ms=5,
        metadata={},
    )


def test_missing_processed_data_fails_clearly(tmp_path: Path) -> None:
    module = _script_module()

    with pytest.raises(FileNotFoundError, match="Processed dataset is missing"):
        module.run_unified_scoreboard(_config(tmp_path))


def test_toy_script_run_writes_expected_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script_module()
    config = _config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(module, "_prepare_dataset", lambda _config: (_toy_dataset(), "abc", 2))

    assert module.main(["--config", str(config_path)]) == 0

    output_dir = Path(config["reporting"]["output_dir"])
    assert (output_dir / "unified_scoreboard_summary.json").exists()
    report = (output_dir / "unified_scoreboard_report.md").read_text(encoding="utf-8")
    scores = pd.read_csv(output_dir / "unified_split_scores.csv")
    train_mean = scores[
        (scores["method_name"] == "train_heldout_mean_rate") & (scores["split"] == "validation")
    ].iloc[0]
    assert abs(float(train_mean["bits_per_spike"])) < 1e-12
    assert "Old mean-rate values are historical-only" in report
    assert "not an official NLB leaderboard result" in report
