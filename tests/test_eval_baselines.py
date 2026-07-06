from __future__ import annotations

from typing import Any

import numpy as np

from latentbrain.data.schemas import NeuralDataset, NeuronMask, TrialSplit
from latentbrain.eval.baselines import (
    evaluate_mean_rate_baseline,
    fit_mean_rate_baseline,
    predict_mean_rate,
)


def _dataset() -> NeuralDataset:
    spikes = np.array(
        [
            [[1, 0, 2], [3, 0, 2]],
            [[2, 1, 0], [0, 1, 0]],
            [[100, 100, 100], [100, 100, 100]],
            [[200, 200, 200], [200, 200, 200]],
        ],
        dtype=np.int64,
    )
    return NeuralDataset(
        spikes=spikes,
        rates=None,
        latents=None,
        trial_ids=np.array([10, 11, 12, 13], dtype=np.int64),
        time_ms=np.array([0.0, 50.0]),
        bin_size_ms=50,
        metadata={},
    )


def _split() -> TrialSplit:
    return TrialSplit(
        train=np.array([10, 11], dtype=np.int64),
        validation=np.array([12], dtype=np.int64),
        test=np.array([13], dtype=np.int64),
    )


def _mask() -> NeuronMask:
    return NeuronMask(
        heldin=np.array([True, True, False]),
        heldout=np.array([False, False, True]),
    )


def _config() -> dict[str, Any]:
    return {
        "baseline": {"name": "mean_rate", "min_rate_hz": 1.0e-4, "max_rate_hz": 500.0},
        "evaluation": {
            "evaluate_splits": ["train", "validation", "test"],
            "evaluate_neuron_groups": ["heldin", "heldout", "all"],
        },
    }


def test_mean_rate_baseline_fits_using_train_trials_only() -> None:
    dataset = _dataset()
    train_spikes = dataset.spikes[:2]

    rates = fit_mean_rate_baseline(
        train_spikes,
        bin_size_ms=50,
        min_rate_hz=1.0e-4,
        max_rate_hz=500.0,
    )

    np.testing.assert_allclose(rates, np.array([30.0, 10.0, 20.0]))


def test_predict_mean_rate_returns_expected_tensor_shape() -> None:
    predictions = predict_mean_rate(np.array([1.0, 2.0, 3.0]), n_trials=4, n_time_bins=5)

    assert predictions.shape == (4, 5, 3)
    np.testing.assert_allclose(predictions[0, 0], np.array([1.0, 2.0, 3.0]))


def test_validation_and_test_spikes_do_not_affect_fitted_rates() -> None:
    dataset = _dataset()
    changed = _dataset()
    changed.spikes[2:] = 999

    metrics, _, metadata = evaluate_mean_rate_baseline(dataset, _split(), _mask(), _config())
    changed_metrics, _, changed_metadata = evaluate_mean_rate_baseline(
        changed,
        _split(),
        _mask(),
        _config(),
    )

    assert metadata["fitted_rates_hz"] == changed_metadata["fitted_rates_hz"]
    train_metrics = metrics[metrics["split"] == "train"].reset_index(drop=True)
    changed_train_metrics = changed_metrics[changed_metrics["split"] == "train"].reset_index(
        drop=True
    )
    np.testing.assert_allclose(
        train_metrics["poisson_log_likelihood"],
        changed_train_metrics["poisson_log_likelihood"],
    )


def test_evaluation_outputs_required_split_and_neuron_group_tables() -> None:
    split_metrics, neuron_metrics, metadata = evaluate_mean_rate_baseline(
        _dataset(),
        _split(),
        _mask(),
        _config(),
    )

    assert set(split_metrics["split"]) == {"train", "validation", "test"}
    assert set(split_metrics["neuron_group"]) == {"heldin", "heldout", "all"}
    assert set(split_metrics.columns) >= {
        "split",
        "neuron_group",
        "n_trials",
        "n_neurons",
        "n_time_bins",
        "spike_count",
        "poisson_nll",
        "poisson_log_likelihood",
        "reference_log_likelihood",
        "bits_per_spike",
        "mean_predicted_rate_hz",
    }
    assert set(neuron_metrics.columns) >= {
        "neuron_index",
        "neuron_group",
        "train_mean_rate_hz",
        "total_spikes_all_trials",
        "validation_spikes",
        "test_spikes",
    }
    assert metadata["train_only_fit"] is True
    assert metadata["fit_trials"] == [10, 11]
