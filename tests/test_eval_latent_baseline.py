from __future__ import annotations

import numpy as np

from latentbrain.data.schemas import NeuralDataset, NeuronMask, TrialSplit
from latentbrain.eval.latent_baseline import run_factor_latent_baseline


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
    rng = np.random.default_rng(5)
    heldin = rng.poisson(0.3, size=(8, 6, 4)).astype(np.int64)
    heldout = np.stack(
        [heldin[:, :, 0] + heldin[:, :, 1] + 1, heldin[:, :, 2] + 1],
        axis=2,
    ).astype(np.int64)
    spikes = np.concatenate([heldin, heldout], axis=2)
    behavior = None
    behavior_names = None
    if with_behavior:
        t = np.arange(6, dtype=np.float64)[None, :, None]
        trial = np.arange(8, dtype=np.float64)[:, None, None]
        behavior = np.concatenate(
            [
                t + 0.1 * trial,
                2.0 * t - 0.1 * trial,
                -t + 0.2 * trial,
                0.5 * t + 0.3 * trial,
            ],
            axis=2,
        )
        behavior_names = ["hand_pos_x", "hand_pos_y", "cursor_pos_x", "cursor_pos_y"]
    return NeuralDataset(
        spikes=spikes,
        rates=None,
        latents=None,
        trial_ids=np.arange(spikes.shape[0], dtype=np.int64),
        time_ms=np.arange(spikes.shape[1], dtype=np.float64) * 5,
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
            "smoothing": {"method": "gaussian", "sigma_ms": 5.0, "truncate": 1.0},
            "convert_to_hz": True,
            "standardize_features": True,
        },
        "latent_model": {
            "name": "factor_analysis",
            "latent_dim": 2,
            "random_state": 3,
            "max_iter": 300,
            "tol": 1.0e-4,
            "train_trials_only": True,
        },
        "heldout_decoder": {
            "name": "ridge",
            "alpha": 1.0,
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
        },
    }


def test_pipeline_outputs_expected_frames_and_shapes() -> None:
    split = _split()

    split_metrics, neuron_metrics, behavior_metrics, latent_summary, metadata = (
        run_factor_latent_baseline(_dataset(), split, _mask(), _config())
    )

    assert set(split_metrics["split"]) == {"train", "validation", "test"}
    assert set(behavior_metrics["split"]) == {"train", "validation", "test"}
    assert set(latent_summary["split"]) == {"train", "validation", "test"}
    assert set(neuron_metrics["target_neuron_rank"]) == {0, 1}
    assert metadata["latent_dim"] == 2
    assert metadata["latent_shapes"]["validation"] == [2, 6, 2]
    assert set(metadata["input_neuron_indices"]).isdisjoint(set(metadata["target_neuron_indices"]))


def test_predicted_rates_are_positive_and_finite() -> None:
    split = _split()

    split_metrics, _, _, _, _ = run_factor_latent_baseline(_dataset(), split, _mask(), _config())

    assert np.all(np.isfinite(split_metrics["mean_predicted_rate_hz"]))
    assert (split_metrics["mean_predicted_rate_hz"] > 0.0).all()


def test_validation_and_test_samples_do_not_affect_factor_fit() -> None:
    split = _split()
    changed = _dataset()
    changed.spikes[4:, :, :4] += 20
    changed.spikes[4:, :, 4:] += 50

    _, _, _, _, meta_a = run_factor_latent_baseline(_dataset(), split, _mask(), _config())
    _, _, _, _, meta_b = run_factor_latent_baseline(changed, split, _mask(), _config())

    np.testing.assert_allclose(meta_a["factor_components"], meta_b["factor_components"])
    np.testing.assert_allclose(meta_a["feature_stats"]["mean"], meta_b["feature_stats"]["mean"])


def test_pipeline_works_without_behavior_when_decoder_disabled() -> None:
    split = _split()

    _, _, behavior_metrics, _, metadata = run_factor_latent_baseline(
        _dataset(with_behavior=False), split, _mask(), _config(behavior_enabled=False)
    )

    assert behavior_metrics.empty
    assert metadata["behavior_decoder_enabled"] is False


def test_required_columns_exist() -> None:
    split = _split()

    split_metrics, neuron_metrics, behavior_metrics, latent_summary, _ = run_factor_latent_baseline(
        _dataset(), split, _mask(), _config()
    )

    assert {
        "split",
        "n_trials",
        "n_time_bins",
        "n_input_neurons",
        "n_target_neurons",
        "latent_dim",
        "spike_count",
        "poisson_nll",
        "poisson_log_likelihood",
        "reference_log_likelihood",
        "bits_per_spike",
        "mse_rate_hz",
        "mae_rate_hz",
        "mean_predicted_rate_hz",
        "mean_reference_rate_hz",
    }.issubset(split_metrics.columns)
    assert {"split", "target_neuron_index", "bits_per_spike"}.issubset(neuron_metrics.columns)
    assert {"split", "target_name", "r2", "mse", "mae", "target_variance"}.issubset(
        behavior_metrics.columns
    )
    assert {"split", "latent_dim_index", "mean", "std", "variance"}.issubset(latent_summary.columns)
