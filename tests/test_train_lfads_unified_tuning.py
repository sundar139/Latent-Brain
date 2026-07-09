from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from latentbrain.eval.lfads_unified_tuning import rank_lfads_unified_results
from latentbrain.train.lfads_unified_tuning import (
    build_unified_lfads_train_config,
    expand_unified_lfads_grid,
    make_unified_lfads_run_id,
)


def _config(tmp_path: Path) -> dict[str, Any]:
    return {
        "dataset": {"name": "unit", "bin_size_ms": 20},
        "splits": {
            "seed": 2027,
            "train_fraction": 0.7,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "heldout_neuron_fraction": 0.25,
        },
        "window": {"duration_seconds": 1.28, "crop_policy": "from_start"},
        "binning": {"target_bin_size_ms": 20},
        "runtime": {"device": "cuda", "fail_if_cuda_unavailable": True},
        "scoring": {"primary_split": "validation"},
        "grid": {},
        "lfads_settings": {
            "batch_size": 4,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "gradient_clip_norm": 5.0,
            "heldin_loss_weight": 1.0,
            "loss_normalization": "per_observed_spike_bin",
            "model_dropout": 0.0,
            "min_rate_hz": 1e-4,
            "max_rate_hz": 500.0,
            "checkpoint_metric": "validation_total_loss",
            "checkpoint_mode": "min",
        },
        "references": {
            "train_mean_validation_bits_per_spike": 0.0,
            "factor_latent_unified_validation_bits_per_spike": 0.03,
            "previous_best_lfads_family_validation_bits_per_spike": 0.01,
        },
        "evaluation": {
            "evaluate_splits": ["train", "validation", "test"],
            "direct_model_primary": True,
            "also_evaluate_factor_decoder": True,
            "behavior_decoder_enabled": True,
        },
        "reporting": {"output_dir": str(tmp_path / "out")},
        "_window_bins": 64,
    }


def _params(
    dropout: float = 0.25, loss_weight: float = 4.0, kl_scale: float = 0.1
) -> dict[str, Any]:
    return {
        "encoder_hidden_dim": 64,
        "generator_hidden_dim": 96,
        "latent_dim": 16,
        "factor_dim": 32,
        "input_dropout_rate": dropout,
        "heldout_loss_weight": loss_weight,
        "kl_warmup_epochs": 5,
        "kl_scale": kl_scale,
        "epochs": 20,
    }


def test_grid_expansion_is_deterministic() -> None:
    grid = {"a": [1, 2], "b": [3, 4]}

    assert expand_unified_lfads_grid(grid) == [
        {"a": 1, "b": 3},
        {"a": 1, "b": 4},
        {"a": 2, "b": 3},
        {"a": 2, "b": 4},
    ]


def test_run_ids_are_stable() -> None:
    assert make_unified_lfads_run_id(0, _params()) == (
        "run_000_enc64_gen96_lat16_fac32_idr0p25_hw4p0_kl0p1"
    )


def test_train_config_does_not_mutate_base(tmp_path: Path) -> None:
    base = _config(tmp_path)
    original = deepcopy(base)

    build_unified_lfads_train_config(base, _params(), tmp_path / "run")

    assert base == original


def test_train_config_applies_dropout_loss_weight_and_kl_scale(tmp_path: Path) -> None:
    base_config = _config(tmp_path)
    config = build_unified_lfads_train_config(base_config, _params(), tmp_path / "run")

    assert config["training"]["input_dropout"]["rate"] == 0.25
    assert config["training"]["heldout_loss_weight"] == 4.0
    assert config["training"]["kl_scale"] == 0.1
    assert config["data"]["max_time_bins"] == base_config["_window_bins"]


def test_train_config_omits_disabled_input_dropout(tmp_path: Path) -> None:
    config = build_unified_lfads_train_config(
        _config(tmp_path), _params(dropout=0.0), tmp_path / "run"
    )

    assert "input_dropout" not in config["training"]


def test_best_selection_uses_unified_bits_not_validation_loss() -> None:
    results = pd.DataFrame(
        {
            "run_id": ["low_loss", "high_bits"],
            "run_index": [0, 1],
            "status": ["completed", "completed"],
            "validation_unified_bits_per_spike": [0.01, 0.02],
            "validation_poisson_nll": [1.0, 2.0],
            "validation_behavior_mean_r2": [0.0, 0.0],
            "kl_scale": [0.1, 1.0],
            "validation_loss": [0.1, 10.0],
            "validation_factor_decoder_unified_bits_per_spike": [0.0, 0.0],
            "input_dropout_rate": [0.0, 0.25],
            "heldout_loss_weight": [2.0, 2.0],
            "latent_dim": [16, 16],
            "factor_dim": [32, 32],
            "beats_factor_latent_unified": [False, False],
            "beats_previous_best_lfads_family": [False, True],
            "notes": ["", ""],
        }
    )

    assert rank_lfads_unified_results(results).iloc[0]["run_id"] == "high_bits"
