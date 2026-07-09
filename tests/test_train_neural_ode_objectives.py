from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest
import torch

from latentbrain.eval.neural_ode_objectives import rank_neural_ode_objective_results
from latentbrain.train.neural_ode_objectives import (
    _train_and_evaluate_run,
    build_neural_ode_objective_train_config,
    build_objective_variants,
    make_neural_ode_objective_run_id,
    run_neural_ode_objective_variants,
    select_best_unified_checkpoint_index,
    weighted_poisson_loss,
)


def _base_model() -> dict[str, object]:
    return {
        "name": "neural_ode_objectives",
        "batch_size": 4,
        "encoder_hidden_dim": 64,
        "drift_hidden_dim": 64,
        "diffusion_hidden_dim": 32,
        "latent_dim": 32,
        "factor_dim": 32,
        "input_dropout_rate": 0.10,
        "heldin_loss_weight": 1.0,
        "heldout_loss_weight": 6.0,
        "kl_warmup_epochs": 10,
        "kl_scale": 0.01,
        "drift_regularization": 1.0e-4,
        "learning_rate": 7.5e-4,
        "scheduler": "cosine",
        "weight_decay": 1.0e-5,
        "gradient_clip_norm": 5.0,
        "loss_normalization": "per_observed_spike_bin",
        "model_dropout": 0.0,
        "min_rate_hz": 1.0e-4,
        "max_rate_hz": 500.0,
        "dt_seconds": 0.02,
        "diffusion_scale": 0.0,
        "epochs": 50,
        "checkpoint_metric": "validation_total_loss",
        "checkpoint_mode": "min",
        "save_unified_checkpoints": True,
        "evaluate_checkpoints_by_unified_metric": True,
    }


def _variants() -> list[dict[str, object]]:
    return [
        {
            "name": "refined_baseline",
            "heldin_loss_weight": 1.0,
            "heldout_loss_weight": 6.0,
            "zero_count_weight": 1.0,
            "positive_count_weight": 1.0,
            "rate_calibration_loss_weight": 0.0,
            "notes": "baseline",
        },
        {
            "name": "zero_downweighted",
            "heldin_loss_weight": 0.5,
            "heldout_loss_weight": 8.0,
            "zero_count_weight": 0.5,
            "positive_count_weight": 1.5,
            "rate_calibration_loss_weight": 0.05,
            "notes": "sparse",
        },
    ]


def _base_config(tmp_path: Path) -> dict[str, object]:
    return {
        "dataset": {"processed_path": str(tmp_path / "missing.npz"), "original_bin_size_ms": 5},
        "splits": {"seed": 2027},
        "window": {"duration_seconds": 1.28},
        "binning": {"target_bin_size_ms": 20},
        "runtime": {"device": "cuda", "fail_if_cuda_unavailable": True},
        "scoring": {"reference_model": "train_heldout_mean_rate"},
        "search": {"max_runs": 2},
        "base_model": _base_model(),
        "objective_variants": _variants(),
        "evaluation": {
            "evaluate_splits": ["train", "validation", "test"],
            "behavior_decoder_enabled": True,
        },
        "references": {
            "train_mean_validation_bits_per_spike": 0.0,
            "factor_latent_unified_validation_bits_per_spike": 0.03,
            "previous_neural_ode_refinement_validation_bits_per_spike": 0.028,
            "switching_ode_validation_bits_per_spike": 0.006,
            "oracle_validation_bits_per_spike": 3.0,
        },
        "reporting": {"output_dir": str(tmp_path / "out")},
    }


def test_objective_variants_are_deterministic_and_names_unique() -> None:
    first = build_objective_variants(_base_model(), _variants(), 8)
    second = build_objective_variants(_base_model(), _variants(), 8)

    assert first == second
    assert [variant["name"] for variant in first] == ["refined_baseline", "zero_downweighted"]
    assert len({variant["name"] for variant in first}) == len(first)
    assert all(variant["diffusion_scale"] == 0.0 for variant in first)


def test_duplicate_variant_names_are_rejected() -> None:
    duplicated = [*_variants(), dict(_variants()[0])]

    with pytest.raises(ValueError, match="unique"):
        build_objective_variants(_base_model(), duplicated, 8)


def test_max_runs_limits_variant_count() -> None:
    assert len(build_objective_variants(_base_model(), _variants(), 1)) == 1


def test_run_ids_are_stable() -> None:
    variants = build_objective_variants(_base_model(), _variants(), 8)

    assert make_neural_ode_objective_run_id(0, variants[0]) == "run_000_refined_baseline"
    assert make_neural_ode_objective_run_id(3, variants[1]) == "run_003_zero_downweighted"


def test_build_train_config_does_not_mutate_and_applies_objective_weights(tmp_path: Path) -> None:
    base = _base_config(tmp_path)
    before = copy.deepcopy(base)
    variant = build_objective_variants(_base_model(), _variants(), 8)[1]

    config = build_neural_ode_objective_train_config(base, variant, tmp_path / "run")

    assert base == before
    assert config["model"]["name"] == "neural_ode"
    assert config["model"]["diffusion_scale"] == 0.0
    assert config["training"]["heldout_loss_weight"] == 8.0
    assert config["training"]["heldin_loss_weight"] == 0.5
    assert config["training"]["zero_count_weight"] == 0.5
    assert config["training"]["positive_count_weight"] == 1.5
    assert config["training"]["rate_calibration_loss_weight"] == 0.05
    assert config["training"]["drift_regularization_scale"] == 1.0e-4
    assert config["training"]["objective_name"] == "zero_downweighted"


def test_weighted_poisson_loss_upweights_positive_bins() -> None:
    counts = torch.tensor([[[0.0, 2.0]]])
    rates = torch.full_like(counts, 5.0)

    baseline = weighted_poisson_loss(counts, rates, 20, 1.0, 1.0, "sum")
    positive_heavy = weighted_poisson_loss(counts, rates, 20, 1.0, 3.0, "sum")
    zero_heavy = weighted_poisson_loss(counts, rates, 20, 3.0, 1.0, "sum")

    zero_term = weighted_poisson_loss(counts[:, :, :1], rates[:, :, :1], 20, 1.0, 1.0, "sum")
    positive_term = weighted_poisson_loss(counts[:, :, 1:], rates[:, :, 1:], 20, 1.0, 1.0, "sum")

    assert positive_heavy > baseline
    assert zero_heavy > baseline
    assert torch.isclose(positive_heavy, zero_term + 3.0 * positive_term, atol=1e-5)
    assert torch.isclose(baseline, zero_term + positive_term, atol=1e-5)


def test_weighted_poisson_loss_rejects_non_positive_weights() -> None:
    counts = torch.zeros((1, 1, 2))
    rates = torch.ones_like(counts)

    with pytest.raises(ValueError, match="count weights must be positive"):
        weighted_poisson_loss(counts, rates, 20, 0.0, 1.0, "sum")


def test_selection_uses_unified_bits_not_validation_loss() -> None:
    base = {
        "status": "completed",
        "objective_name": "variant",
        "validation_behavior_mean_r2": 0.0,
        "validation_factor_decoder_unified_bits_per_spike": 0.0,
        "heldout_loss_weight": 8.0,
        "zero_count_weight": 1.0,
        "positive_count_weight": 1.0,
        "rate_calibration_loss_weight": 0.0,
        "kl_scale": 0.01,
        "drift_regularization": 1.0e-4,
        "input_dropout_rate": 0.1,
        "best_checkpoint_source": "latest",
        "beats_factor_latent_unified": False,
        "beats_previous_neural_ode_refinement": False,
        "notes": "",
    }
    ranked = rank_neural_ode_objective_results(
        pd.DataFrame(
            [
                base
                | {
                    "run_id": "low_loss",
                    "run_index": 0,
                    "validation_unified_bits_per_spike": 0.01,
                    "validation_poisson_nll": 1.0,
                    "validation_total_loss": 0.5,
                },
                base
                | {
                    "run_id": "high_bits",
                    "run_index": 1,
                    "validation_unified_bits_per_spike": 0.02,
                    "validation_poisson_nll": 3.0,
                    "validation_total_loss": 9.0,
                },
            ]
        )
    )

    assert ranked.iloc[0]["run_id"] == "high_bits"


def test_checkpoint_selection_prefers_higher_unified_bits() -> None:
    rows = [
        {
            "validation_total_loss": 1.0,
            "validation_unified_bits_per_spike": 0.01,
            "validation_poisson_nll": 1.0,
        },
        {
            "validation_total_loss": 2.0,
            "validation_unified_bits_per_spike": 0.03,
            "validation_poisson_nll": 2.0,
        },
    ]

    assert select_best_unified_checkpoint_index(rows) == 1


def test_every_variant_trains_under_the_same_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seeding per run index would confound the objective with initialization."""
    seeds: list[int] = []
    monkeypatch.setattr(
        "latentbrain.train.neural_ode_objectives.seed_everything",
        lambda seed: seeds.append(int(seed)),
    )
    monkeypatch.setattr(
        "latentbrain.train.neural_ode_objectives.create_torch_datasets",
        lambda *_args, **_kwargs: pytest.fail("training must not start in this test"),
    )
    base = _base_config(tmp_path) | {"_window_bins": 64}
    variants = build_objective_variants(_base_model(), _variants(), 8)
    for variant in variants:
        run_config = build_neural_ode_objective_train_config(base, variant, tmp_path / "run")
        with pytest.raises(pytest.fail.Exception):
            _train_and_evaluate_run(run_config, "run", object(), object(), object())

    assert seeds == [2027] * len(variants)


def test_missing_processed_data_fails_before_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_cuda(_config: dict[str, object]) -> str:  # pragma: no cover
        raise AssertionError("cuda checked too early")

    monkeypatch.setattr("latentbrain.train.neural_ode_objectives._validate_cuda", fail_cuda)

    with pytest.raises(FileNotFoundError, match="Processed dataset is missing"):
        run_neural_ode_objective_variants(_base_config(tmp_path))
