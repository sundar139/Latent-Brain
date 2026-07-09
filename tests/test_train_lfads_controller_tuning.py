from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pytest

from latentbrain.eval.lfads_controller_tuning import rank_controller_results
from latentbrain.train.lfads_controller_tuning import (
    _resolve_processed_path,
    build_controller_train_config,
    expand_controller_grid,
    make_controller_run_id,
)


def _config(tmp_path: Path) -> dict[str, Any]:
    return {
        "dataset": {"name": "unit", "processed_path": "missing.npz", "original_bin_size_ms": 5},
        "splits": {"seed": 2027, "heldout_neuron_fraction": 0.25},
        "window": {"duration_seconds": 1.28, "crop_policy": "from_start"},
        "binning": {"target_bin_size_ms": 20},
        "runtime": {"device": "cuda", "fail_if_cuda_unavailable": True},
        "scoring": {"primary_split": "validation"},
        "grid": {},
        "model": {
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
            "oracle_validation_bits_per_spike": 3.0,
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


def _params() -> dict[str, Any]:
    return {
        "encoder_hidden_dim": 64,
        "controller_hidden_dim": 64,
        "generator_hidden_dim": 96,
        "latent_dim": 16,
        "factor_dim": 32,
        "inferred_input_dim": 4,
        "input_dropout_rate": 0.25,
        "heldout_loss_weight": 4.0,
        "kl_warmup_epochs": 5,
        "kl_scale": 0.1,
        "inferred_input_kl_scale": 0.01,
        "epochs": 20,
    }


def test_grid_expansion_is_deterministic() -> None:
    assert expand_controller_grid({"a": [1, 2], "b": [3, 4]}) == [
        {"a": 1, "b": 3},
        {"a": 1, "b": 4},
        {"a": 2, "b": 3},
        {"a": 2, "b": 4},
    ]


def test_run_ids_are_stable() -> None:
    assert make_controller_run_id(0, _params()) == (
        "run_000_enc64_ctrl64_gen96_lat16_fac32_u4_idr0p25_hw4p0_kl0p1_ukl0p01"
    )


def test_base_config_is_not_mutated(tmp_path: Path) -> None:
    base = _config(tmp_path)
    original = deepcopy(base)

    build_controller_train_config(base, _params(), tmp_path / "run")

    assert base == original


def test_train_config_applies_inferred_input_settings(tmp_path: Path) -> None:
    config = build_controller_train_config(_config(tmp_path), _params(), tmp_path / "run")

    assert config["model"]["inferred_input_dim"] == 4
    assert config["training"]["inferred_input_kl_scale"] == 0.01


def test_best_selection_uses_unified_bits_not_validation_loss() -> None:
    results = pd.DataFrame(
        {
            "run_id": ["low_loss", "high_bits"],
            "run_index": [0, 1],
            "status": ["completed", "completed"],
            "validation_unified_bits_per_spike": [0.01, 0.02],
            "validation_poisson_nll": [1.0, 2.0],
            "validation_behavior_mean_r2": [0.0, 0.0],
            "inferred_input_kl_scale": [0.1, 0.1],
            "validation_loss": [0.1, 10.0],
            "validation_factor_decoder_unified_bits_per_spike": [0.0, 0.0],
            "input_dropout_rate": [0.0, 0.25],
            "heldout_loss_weight": [2.0, 2.0],
            "kl_scale": [0.1, 0.1],
            "latent_dim": [16, 16],
            "inferred_input_dim": [4, 4],
            "beats_factor_latent_unified": [False, False],
            "beats_previous_best_lfads_family": [False, True],
            "notes": ["", ""],
        }
    )

    assert rank_controller_results(results).iloc[0]["run_id"] == "high_bits"


def test_missing_processed_data_fails_before_cuda_validation(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Processed dataset is missing"):
        _resolve_processed_path(_config(tmp_path))
