from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from latentbrain.data.schemas import NeuralDataset, NeuronMask, TrialSplit
from latentbrain.eval.latent_sweep import run_factor_latent_sweep
from latentbrain.eval.sweeps import rank_sweep_results


def _split() -> TrialSplit:
    return TrialSplit(
        train=np.array([0, 1, 2, 3]),
        validation=np.array([4, 5]),
        test=np.array([6, 7]),
    )


def _mask() -> NeuronMask:
    return NeuronMask(
        heldin=np.array([True, True, True, True, False, False]),
        heldout=np.array([False, False, False, False, True, True]),
    )


def _dataset(with_behavior: bool = True) -> NeuralDataset:
    rng = np.random.default_rng(17)
    heldin = rng.poisson(0.3, size=(8, 6, 4)).astype(np.int64)
    heldout = np.stack([heldin[:, :, 0] + 1, heldin[:, :, 2] + 1], axis=2).astype(np.int64)
    spikes = np.concatenate([heldin, heldout], axis=2)
    behavior = None
    behavior_names = None
    if with_behavior:
        t = np.arange(6, dtype=np.float64)[None, :, None]
        trial = np.arange(8, dtype=np.float64)[:, None, None]
        t_all = np.broadcast_to(t, (8, 6, 1))
        behavior = np.concatenate([t + trial, t - trial, 2.0 * t_all, -t_all], axis=2)
        behavior_names = ["hand_pos_x", "hand_pos_y", "cursor_pos_x", "cursor_pos_y"]
    return NeuralDataset(
        spikes=spikes,
        rates=None,
        latents=None,
        trial_ids=np.arange(8, dtype=np.int64),
        time_ms=np.arange(6, dtype=np.float64) * 5,
        bin_size_ms=5,
        metadata={},
        behavior=behavior,
        behavior_names=behavior_names,
    )


def _config(behavior_enabled: bool = True) -> dict[str, object]:
    return {
        "features": {
            "input_neuron_group": "heldin",
            "target_neuron_group": "heldout",
            "convert_to_hz": True,
        },
        "sweep": {
            "latent_dim": [1, 2],
            "smoothing_sigma_ms": [5.0, 10.0],
            "heldout_decoder_alpha": [1.0, 10.0],
            "standardize_features": [True, False],
        },
        "latent_model": {
            "name": "factor_analysis",
            "random_state": 3,
            "max_iter": 300,
            "tol": 1.0,
            "train_trials_only": True,
        },
        "heldout_decoder": {
            "name": "ridge",
            "fit_intercept": True,
            "min_rate_hz": 1.0e-4,
            "max_rate_hz": 5000.0,
            "train_trials_only": True,
        },
        "behavior_decoder": {
            "enabled": behavior_enabled,
            "alpha": 1.0,
            "fit_intercept": True,
            "target_prefixes": ["hand_pos", "cursor_pos"],
            "derive_velocity": True,
            "velocity_method": "central_difference",
            "standardize_targets": True,
            "train_trials_only": True,
        },
        "reference": {"name": "train_mean_rate", "fit_train_trials_only": True},
        "evaluation": {
            "evaluate_splits": ["train", "validation", "test"],
            "primary_split": "validation",
            "primary_metric": "bits_per_spike",
            "secondary_metric": "behavior_mean_r2",
        },
    }


def _rank_row(
    run_id: str,
    *,
    behavior_mean_r2: float,
    latent_dim: int,
    alpha: float,
    sigma: float,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "split": "validation",
        "bits_per_spike": 1.0,
        "poisson_nll": 5.0,
        "behavior_mean_r2": behavior_mean_r2,
        "latent_dim": latent_dim,
        "heldout_decoder_alpha": alpha,
        "smoothing_sigma_ms": sigma,
    }


def test_sweep_produces_one_row_per_config_per_split() -> None:
    sweep_results, best_config, best_split, best_neuron, best_behavior, best_latent = (
        run_factor_latent_sweep(_dataset(), _split(), _mask(), _config())
    )

    assert len(sweep_results) == 2 * 2 * 2 * 2 * 3
    assert set(sweep_results["split"]) == {"train", "validation", "test"}
    assert best_config["run_id"] in set(sweep_results["run_id"])
    assert set(best_split["split"]) == {"train", "validation", "test"}
    assert not best_neuron.empty
    assert not best_behavior.empty
    assert not best_latent.empty


def test_best_config_selected_by_validation_bits_per_spike() -> None:
    sweep_results, best_config, *_ = run_factor_latent_sweep(
        _dataset(), _split(), _mask(), _config()
    )
    validation = sweep_results[sweep_results["split"] == "validation"]
    expected = validation.sort_values(
        ["bits_per_spike", "poisson_nll", "behavior_mean_r2", "latent_dim"],
        ascending=[False, True, False, True],
        kind="mergesort",
    ).iloc[0]

    assert best_config["run_id"] == expected["run_id"]


def test_factor_latent_tie_breakers_are_deterministic() -> None:
    results = pd.DataFrame(
        [
            _rank_row("a", behavior_mean_r2=0.1, latent_dim=4, alpha=10.0, sigma=50.0),
            _rank_row("b", behavior_mean_r2=0.2, latent_dim=8, alpha=1.0, sigma=25.0),
            _rank_row("c", behavior_mean_r2=0.2, latent_dim=4, alpha=10.0, sigma=25.0),
            _rank_row("d", behavior_mean_r2=0.2, latent_dim=4, alpha=1.0, sigma=50.0),
            _rank_row("e", behavior_mean_r2=0.2, latent_dim=4, alpha=1.0, sigma=25.0),
        ]
    )

    ranked = rank_sweep_results(results, "validation", "bits_per_spike")

    assert ranked.iloc[0]["run_id"] == "e"


def test_invalid_latent_dimension_is_rejected() -> None:
    config = _config()
    config["sweep"]["latent_dim"] = [4]  # type: ignore[index]

    with pytest.raises(ValueError, match="no valid factor latent sweep results"):
        run_factor_latent_sweep(_dataset(), _split(), _mask(), config)


def test_train_only_fitting_is_enforced() -> None:
    changed = _dataset()
    changed.spikes[4:, :, :4] += 50
    changed.spikes[4:, :, 4:] += 50

    _, best_a, *_ = run_factor_latent_sweep(_dataset(), _split(), _mask(), _config())
    _, best_b, *_ = run_factor_latent_sweep(changed, _split(), _mask(), _config())

    assert best_a["train_only_fit"] is True
    assert best_b["train_only_fit"] is True


def test_sweep_works_with_behavior_decoder_disabled() -> None:
    sweep_results, _, _, _, best_behavior, _ = run_factor_latent_sweep(
        _dataset(with_behavior=False), _split(), _mask(), _config(behavior_enabled=False)
    )

    assert sweep_results["behavior_mean_r2"].isna().all()
    assert best_behavior.empty


def test_required_output_columns_exist() -> None:
    sweep_results, _, *_ = run_factor_latent_sweep(_dataset(), _split(), _mask(), _config())

    assert {
        "run_id",
        "split",
        "latent_dim",
        "smoothing_sigma_ms",
        "heldout_decoder_alpha",
        "standardize_features",
        "n_train_trials",
        "n_eval_trials",
        "n_input_neurons",
        "n_target_neurons",
        "spike_count",
        "poisson_nll",
        "poisson_log_likelihood",
        "reference_log_likelihood",
        "bits_per_spike",
        "mse_rate_hz",
        "mae_rate_hz",
        "behavior_mean_r2",
        "behavior_mean_mse",
        "behavior_mean_mae",
        "mean_predicted_rate_hz",
        "mean_reference_rate_hz",
    }.issubset(sweep_results.columns)
