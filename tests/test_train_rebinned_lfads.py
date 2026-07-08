from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from latentbrain.train.rebinned_lfads import (
    build_rebinned_lfads_eval_config,
    build_rebinned_lfads_train_config,
)


def _config() -> dict[str, Any]:
    return {
        "dataset": {"name": "unit", "processed_path": "data.npz", "expected_hash": "abc"},
        "splits": {"seed": 7, "heldout_neuron_fraction": 0.25},
        "runtime": {"device": "cuda"},
        "evaluation": {"evaluate_splits": ["train", "validation", "test"]},
        "lfads_settings": {
            "encoder_hidden_dim": 64,
            "generator_hidden_dim": 96,
            "latent_dim": 16,
            "factor_dim": 32,
            "dropout": 0.0,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "epochs": 10,
            "batch_size": 4,
            "gradient_clip_norm": 5.0,
            "heldin_loss_weight": 1.0,
            "heldout_loss_weight": 2.0,
            "kl_warmup_epochs": 5,
            "loss_normalization": "per_observed_spike_bin",
            "min_rate_hz": 1e-4,
            "max_rate_hz": 500.0,
        },
    }


def test_train_config_builder_applies_bin_size_window_and_output_dir(tmp_path: Path) -> None:
    base = _config()
    before = deepcopy(base)
    cfg = build_rebinned_lfads_train_config(base, 10, 128, tmp_path / "bin_10ms")
    assert cfg["dataset"]["bin_size_ms"] == 10
    assert cfg["data"]["max_time_bins"] == 128
    assert cfg["model"]["output_dim"] == "all"
    assert cfg["training"]["device"] == "cuda"
    assert cfg["reporting"]["output_dir"].endswith("bin_10ms")
    assert base == before


def test_eval_config_builder_points_to_checkpoint(tmp_path: Path) -> None:
    cfg = build_rebinned_lfads_eval_config(
        _config(), 20, 64, tmp_path / "best.pt", tmp_path / "bin_20ms"
    )
    assert cfg["model"]["checkpoint_path"].endswith("best.pt")
    assert cfg["data"]["max_time_bins"] == 64
    assert cfg["evaluation_mode"]["use_direct_model_rates_for_heldout"] is True
