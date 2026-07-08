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
        "run_metric_audit", Path("scripts/run_metric_audit.py")
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
        "audit": {
            "evaluate_splits": ["train", "validation", "test"],
            "primary_split": "validation",
            "smoothing_sigma_ms": [40.0],
            "shuffle_repeats": 1,
            "shuffle_seed": 2027,
            "min_rate_hz": 1e-4,
            "max_rate_hz": 500.0,
        },
        "references": {
            "use_train_heldout_mean_rate_reference": True,
            "include_global_mean_reference": True,
            "include_split_mean_reference": True,
        },
        "model_outputs": {
            "include_existing_outputs": False,
            "coordinated_dropout_eval_dir": str(tmp_path / "missing_dropout"),
            "temporal_20ms_eval_dir": str(tmp_path / "missing_lfads"),
            "factor_latent_reference_bits_per_spike": 0.03,
            "mean_rate_reference_bits_per_spike": 0.7,
        },
        "reporting": {"output_dir": str(tmp_path / "metric_audit")},
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
        module.run_metric_audit(_config(tmp_path))


def test_script_like_toy_run_writes_expected_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script_module()
    config = _config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(module, "_prepare_dataset", lambda _config: (_toy_dataset(), "abc", 2))

    assert module.main(["--config", str(config_path)]) == 0

    output_dir = Path(config["reporting"]["output_dir"])
    assert (output_dir / "metric_audit_summary.json").exists()
    report = (output_dir / "metric_audit_report.md").read_text(encoding="utf-8")
    scores = pd.read_csv(output_dir / "unified_scores.csv")
    train_mean = scores[
        (scores["method_name"] == "train_heldout_mean_rate") & (scores["split"] == "validation")
    ].iloc[0]
    assert abs(float(train_mean["bits_per_spike"])) < 1e-12
    assert "Oracle controls are not valid models" in report
    assert "local metric-audit work" in report
