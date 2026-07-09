from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest

from latentbrain.eval.neural_ode_tuning import rank_neural_ode_results
from latentbrain.train.neural_ode_tuning import (
    build_neural_ode_train_config,
    expand_neural_ode_grid,
    make_neural_ode_run_id,
    run_neural_ode_tuning,
)


def _base_config(tmp_path: Path) -> dict[str, object]:
    return {
        "dataset": {"processed_path": str(tmp_path / "missing.npz"), "original_bin_size_ms": 5},
        "splits": {"seed": 2027},
        "window": {"duration_seconds": 1.28},
        "binning": {"target_bin_size_ms": 20},
        "runtime": {"device": "cuda", "fail_if_cuda_unavailable": True},
        "scoring": {"reference_model": "train_heldout_mean_rate"},
        "grid": {},
        "search": {"max_runs": 1},
        "model": {
            "name": "neural_ode",
            "batch_size": 4,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "heldin_loss_weight": 1.0,
            "loss_normalization": "mean",
            "model_dropout": 0.0,
            "min_rate_hz": 1.0e-4,
            "max_rate_hz": 500.0,
            "dt_seconds": 0.02,
            "checkpoint_metric": "validation_total_loss",
            "checkpoint_mode": "min",
            "save_unified_checkpoints": True,
            "evaluate_checkpoints_by_unified_metric": True,
        },
        "evaluation": {
            "evaluate_splits": ["train", "validation", "test"],
            "direct_model_primary": True,
            "also_evaluate_factor_decoder": True,
            "behavior_decoder_enabled": True,
        },
        "references": {
            "train_mean_validation_bits_per_spike": 0.0,
            "factor_latent_unified_validation_bits_per_spike": 0.03,
            "previous_neural_sde_validation_bits_per_spike": 0.02,
            "previous_best_lfads_family_validation_bits_per_spike": 0.01,
            "oracle_validation_bits_per_spike": 3.0,
        },
        "reporting": {"output_dir": str(tmp_path / "out")},
    }


def _params() -> dict[str, object]:
    return {
        "encoder_hidden_dim": 64,
        "drift_hidden_dim": 96,
        "diffusion_hidden_dim": 32,
        "latent_dim": 32,
        "factor_dim": 48,
        "input_dropout_rate": 0.25,
        "heldout_loss_weight": 6.0,
        "kl_warmup_epochs": 5,
        "kl_scale": 0.05,
        "diffusion_scale": 0.0,
        "epochs": 2,
    }


def test_grid_expansion_and_run_ids_are_deterministic() -> None:
    grid = {"latent_dim": [32, 48], "kl_scale": [0.05, 0.1], "diffusion_scale": [0.0]}

    assert expand_neural_ode_grid(grid) == [
        {"latent_dim": 32, "kl_scale": 0.05, "diffusion_scale": 0.0},
        {"latent_dim": 32, "kl_scale": 0.1, "diffusion_scale": 0.0},
        {"latent_dim": 48, "kl_scale": 0.05, "diffusion_scale": 0.0},
        {"latent_dim": 48, "kl_scale": 0.1, "diffusion_scale": 0.0},
    ]
    assert make_neural_ode_run_id(2, _params()).startswith("run_002_enc64_drift96")


def test_nonzero_diffusion_grid_is_rejected() -> None:
    with pytest.raises(ValueError, match="diffusion_scale == 0.0"):
        expand_neural_ode_grid({"diffusion_scale": [0.03]})


def test_build_train_config_does_not_mutate_and_forces_zero_diffusion(tmp_path: Path) -> None:
    base = _base_config(tmp_path)
    before = copy.deepcopy(base)

    config = build_neural_ode_train_config(
        base, _params() | {"diffusion_scale": 0.2}, tmp_path / "run"
    )

    assert base == before
    assert config["model"]["name"] == "neural_ode"
    assert config["model"]["drift_hidden_dim"] == 96
    assert config["model"]["diffusion_scale"] == 0.0
    assert config["training"]["heldout_loss_weight"] == 6.0


def test_selection_uses_unified_bits_not_validation_loss() -> None:
    ranked = rank_neural_ode_results(
        pd.DataFrame(
            [
                {
                    "run_id": "loss",
                    "status": "completed",
                    "validation_unified_bits_per_spike": 0.01,
                    "validation_poisson_nll": 1.0,
                    "validation_behavior_mean_r2": 0.0,
                    "run_index": 0,
                    "validation_factor_decoder_unified_bits_per_spike": 0.0,
                    "input_dropout_rate": 0.25,
                    "heldout_loss_weight": 4.0,
                    "kl_scale": 0.1,
                    "latent_dim": 32,
                    "factor_dim": 32,
                    "best_checkpoint_source": "best_validation",
                    "beats_factor_latent_unified": False,
                    "beats_previous_neural_sde": False,
                    "notes": "",
                },
                {
                    "run_id": "bits",
                    "status": "completed",
                    "validation_unified_bits_per_spike": 0.02,
                    "validation_poisson_nll": 3.0,
                    "validation_behavior_mean_r2": 0.0,
                    "run_index": 1,
                    "validation_factor_decoder_unified_bits_per_spike": 0.0,
                    "input_dropout_rate": 0.25,
                    "heldout_loss_weight": 4.0,
                    "kl_scale": 0.1,
                    "latent_dim": 32,
                    "factor_dim": 32,
                    "best_checkpoint_source": "latest",
                    "beats_factor_latent_unified": False,
                    "beats_previous_neural_sde": True,
                    "notes": "",
                },
            ]
        )
    )

    assert ranked.iloc[0]["run_id"] == "bits"


def test_checkpoint_selection_prefers_higher_unified_bits() -> None:
    ranked = pd.DataFrame(
        [
            {
                "checkpoint_source": "best_validation",
                "validation_total_loss": 1.0,
                "validation_unified_bits_per_spike": 0.01,
            },
            {
                "checkpoint_source": "latest",
                "validation_total_loss": 2.0,
                "validation_unified_bits_per_spike": 0.03,
            },
        ]
    ).sort_values("validation_unified_bits_per_spike", ascending=False)

    assert ranked.iloc[0]["checkpoint_source"] == "latest"


def test_missing_processed_data_fails_before_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_cuda(_config: dict[str, object]) -> str:  # pragma: no cover
        raise AssertionError("cuda checked too early")

    monkeypatch.setattr("latentbrain.train.neural_ode_tuning._validate_cuda", fail_cuda)

    with pytest.raises(FileNotFoundError, match="Processed dataset is missing"):
        run_neural_ode_tuning(_base_config(tmp_path))
