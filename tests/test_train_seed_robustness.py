from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from latentbrain.train.seed_robustness import (
    build_seed_plan,
    config_hash,
    load_or_build_method_config,
    make_seed_run_id,
    run_seed_robustness,
)


def _methods(tmp_path: Path) -> list[dict[str, Any]]:
    return [
        {
            "name": "factor_latent",
            "type": "factor_latent",
            "valid_model": True,
            "fallback_config": {
                "latent_dim": 8,
                "smoothing_sigma_ms": 200.0,
                "heldout_decoder_alpha": 10000.0,
                "standardize_features": True,
                "max_iter": 1000,
                "tol": 1.0e-4,
                "min_rate_hz": 1.0e-4,
                "max_rate_hz": 500.0,
            },
            "notes": "baseline",
        },
        {
            "name": "neural_ode_refinement",
            "type": "neural_ode",
            "valid_model": True,
            "config_source": str(tmp_path / "missing_best_config.yaml"),
            "fallback_config": {
                "encoder_hidden_dim": 64,
                "drift_hidden_dim": 64,
                "latent_dim": 32,
                "factor_dim": 32,
                "input_dropout_rate": 0.10,
                "heldout_loss_weight": 6.0,
                "heldin_loss_weight": 1.0,
                "kl_warmup_epochs": 10,
                "kl_scale": 0.01,
                "drift_regularization": 1.0e-4,
                "learning_rate": 7.5e-4,
                "scheduler": "cosine",
                "weight_decay": 1.0e-5,
                "gradient_clip_norm": 5.0,
                "batch_size": 4,
                "epochs": 50,
                "diffusion_scale": 0.0,
            },
            "notes": "dynamics",
        },
    ]


def _config(tmp_path: Path) -> dict[str, Any]:
    return {
        "dataset": {
            "processed_path": str(tmp_path / "missing.npz"),
            "original_bin_size_ms": 5,
            "name": "unit",
        },
        "splits": {
            "base_seed": 2027,
            "train_fraction": 0.7,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "heldout_neuron_fraction": 0.25,
            "split_seed_mode": "fixed",
            "split_seed": 2027,
            "initialization_seed_mode": "varied",
        },
        "window": {"duration_seconds": 1.28, "crop_policy": "from_start"},
        "binning": {"target_bin_size_ms": 20},
        "runtime": {"device": "cuda", "fail_if_cuda_unavailable": True},
        "scoring": {"reference_model": "train_heldout_mean_rate"},
        "seeds": [2027, 2028, 2029, 2030, 2031],
        "methods": _methods(tmp_path),
        "references": {
            "train_mean_validation_bits_per_spike": 0.0,
            "factor_latent_single_seed_reference": 0.0316,
            "neural_ode_refinement_single_seed_reference": 0.0283,
            "neural_ode_objective_single_seed_reference": 0.0115,
            "oracle_validation_bits_per_spike": 3.54,
        },
        "statistics": {
            "confidence_interval": 0.95,
            "bootstrap_repeats": 100,
            "bootstrap_seed": 1337,
            "paired_by_seed": True,
        },
        "evaluation": {
            "evaluate_splits": ["train", "validation", "test"],
            "behavior_decoder_enabled": True,
        },
        "reporting": {"output_dir": str(tmp_path / "out")},
    }


def test_seed_plan_is_deterministic(tmp_path: Path) -> None:
    config = _config(tmp_path)

    first = build_seed_plan(config)
    second = build_seed_plan(config)

    assert first.equals(second)
    assert len(first) == 10


def test_every_method_receives_the_same_seed_list(tmp_path: Path) -> None:
    plan = build_seed_plan(_config(tmp_path))

    seed_lists = plan.groupby("method_name")["seed"].apply(list).to_dict()
    assert set(seed_lists) == {"factor_latent", "neural_ode_refinement"}
    for seeds in seed_lists.values():
        assert seeds == [2027, 2028, 2029, 2030, 2031]


def test_split_seed_is_fixed_and_initialization_seed_varies(tmp_path: Path) -> None:
    plan = build_seed_plan(_config(tmp_path))

    assert set(plan["split_seed"]) == {2027}
    assert sorted(set(plan["initialization_seed"])) == [2027, 2028, 2029, 2030, 2031]
    assert plan["initialization_seed"].tolist() == plan["seed"].tolist()


def test_method_order_does_not_change_seeds(tmp_path: Path) -> None:
    """Regression: `seed + run_index` would make seeds depend on method position."""
    config = _config(tmp_path)
    reversed_config = copy.deepcopy(config)
    reversed_config["methods"] = list(reversed(config["methods"]))

    forward = build_seed_plan(config)
    backward = build_seed_plan(reversed_config)

    for method in ("factor_latent", "neural_ode_refinement"):
        forward_seeds = forward[forward["method_name"] == method]["initialization_seed"].tolist()
        backward_seeds = backward[backward["method_name"] == method]["initialization_seed"].tolist()
        assert forward_seeds == backward_seeds == [2027, 2028, 2029, 2030, 2031]


def test_no_seed_is_derived_from_a_run_index(tmp_path: Path) -> None:
    plan = build_seed_plan(_config(tmp_path))
    configured = [2027, 2028, 2029, 2030, 2031]

    # A `seed + run_index` scheme would produce values outside the configured list.
    assert set(plan["initialization_seed"]) == set(configured)
    assert set(plan["split_seed"]) == {2027}


def test_varied_split_seed_mode_tracks_the_seed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["splits"]["split_seed_mode"] = "varied"

    plan = build_seed_plan(config)

    assert plan["split_seed"].tolist() == plan["seed"].tolist()


def test_seed_plan_rejects_too_few_seeds(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["seeds"] = [2027, 2028]

    with pytest.raises(ValueError, match="at least three seeds"):
        build_seed_plan(config)


def test_seed_plan_rejects_duplicate_method_names(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["methods"] = [config["methods"][0], dict(config["methods"][0])]

    with pytest.raises(ValueError, match="method names must be unique"):
        build_seed_plan(config)


def test_seed_plan_rejects_unknown_method_type(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["methods"][1]["type"] = "transformer"

    with pytest.raises(ValueError, match="unknown method type"):
        build_seed_plan(config)


def test_run_ids_are_stable() -> None:
    assert make_seed_run_id("factor_latent", 2029) == "factor_latent/seed_2029"


def test_method_configs_are_not_mutated(tmp_path: Path) -> None:
    config = _config(tmp_path)
    method = config["methods"][1]
    before = copy.deepcopy(method)

    resolved = load_or_build_method_config(config, method)
    resolved["heldout_loss_weight"] = 999.0

    assert method == before


def test_config_source_fallback_is_used_when_file_is_missing(tmp_path: Path) -> None:
    config = _config(tmp_path)

    resolved = load_or_build_method_config(config, config["methods"][1])

    assert resolved["source"] == "fallback_config"
    assert resolved["heldout_loss_weight"] == 6.0
    assert resolved["diffusion_scale"] == 0.0


def test_config_source_is_preferred_when_present(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = tmp_path / "best_config.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                # `model` intentionally carries the stale pre-variant weights that the
                # generating workflow leaves behind; `training` is authoritative.
                "model": {"encoder_hidden_dim": 96, "heldout_loss_weight": 6.0},
                "training": {
                    "heldout_loss_weight": 10.0,
                    "heldin_loss_weight": 0.5,
                    "kl_scale": 0.01,
                    "drift_regularization_scale": 1.0e-5,
                    "input_dropout": {"rate": 0.0},
                },
            }
        ),
        encoding="utf-8",
    )
    config["methods"][1]["config_source"] = str(source)

    resolved = load_or_build_method_config(config, config["methods"][1])

    assert resolved["source"] == str(source)
    assert resolved["encoder_hidden_dim"] == 96
    assert resolved["heldout_loss_weight"] == 10.0
    assert resolved["heldin_loss_weight"] == 0.5
    assert resolved["drift_regularization"] == 1.0e-5
    assert resolved["input_dropout_rate"] == 0.0


def test_neural_method_rejects_nonzero_diffusion_scale(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["methods"][1]["fallback_config"]["diffusion_scale"] = 0.2

    with pytest.raises(ValueError, match="diffusion_scale == 0.0"):
        load_or_build_method_config(config, config["methods"][1])


def test_config_hash_is_stable_and_order_independent() -> None:
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_missing_processed_data_fails_before_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_cuda(_config: dict[str, Any]) -> str:  # pragma: no cover
        raise AssertionError("cuda checked too early")

    monkeypatch.setattr("latentbrain.train.seed_robustness._validate_cuda", fail_cuda)

    with pytest.raises(FileNotFoundError, match="Processed dataset is missing"):
        run_seed_robustness(_config(tmp_path))


def test_neural_method_requires_cuda_after_inputs_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    Path(config["dataset"]["processed_path"]).write_bytes(b"exists")
    monkeypatch.setattr(
        "latentbrain.train.seed_robustness._load_dataset",
        lambda _config: pytest.fail("dataset loaded before CUDA validation"),
    )
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    with pytest.raises(RuntimeError, match="torch.cuda.is_available"):
        run_seed_robustness(config)
