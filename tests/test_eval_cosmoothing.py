from __future__ import annotations

import numpy as np
import pytest

from latentbrain.data.schemas import NeuralDataset, NeuronMask, TrialSplit
from latentbrain.eval.cosmoothing import (
    evaluate_cosmoothing_predictions,
    fit_cosmoothing_ridge,
    flatten_trial_time,
    predict_cosmoothing_rates,
    run_cosmoothing_baseline,
    run_cosmoothing_sweep,
    select_neuron_group,
)


def _mask() -> NeuronMask:
    return NeuronMask(
        heldin=np.array([True, True, False, False]),
        heldout=np.array([False, False, True, True]),
    )


def _easy_dataset() -> NeuralDataset:
    base = np.array(
        [
            [[0, 1], [1, 0], [2, 1], [3, 0]],
            [[1, 1], [2, 0], [3, 1], [4, 0]],
            [[2, 1], [3, 0], [4, 1], [5, 0]],
            [[3, 1], [4, 0], [5, 1], [6, 0]],
            [[4, 1], [5, 0], [6, 1], [7, 0]],
        ],
        dtype=np.int64,
    )
    heldout = np.stack([base[:, :, 0] + 1, base[:, :, 0] + base[:, :, 1] + 1], axis=2)
    spikes = np.concatenate([base, heldout], axis=2)
    return NeuralDataset(
        spikes=spikes,
        rates=None,
        latents=None,
        trial_ids=np.arange(spikes.shape[0], dtype=np.int64),
        time_ms=np.arange(spikes.shape[1], dtype=np.float64) * 5,
        bin_size_ms=5,
        metadata={},
    )


def _config() -> dict[str, object]:
    return {
        "features": {
            "smoothing": {"method": "gaussian", "sigma_ms": 5.0, "truncate": 1.0},
            "convert_to_hz": True,
            "standardize_features": True,
        },
        "targets": {"min_rate_hz": 1.0e-4, "max_rate_hz": 5000.0},
        "decoder": {"alpha": 0.0, "fit_intercept": True, "train_trials_only": True},
        "reference": {"fit_train_trials_only": True},
        "evaluation": {"evaluate_splits": ["train", "validation", "test"]},
    }


def _sweep_config() -> dict[str, object]:
    return {
        "features": {
            "input_neuron_group": "heldin",
            "target_neuron_group": "heldout",
            "convert_to_hz": True,
        },
        "sweep": {
            "smoothing_sigma_ms": [5.0, 10.0],
            "ridge_alpha": [0.0],
            "standardize_features": [True],
            "fit_intercept": [True, False],
        },
        "targets": {"fit_target_type": "rate_hz", "min_rate_hz": 1.0e-4, "max_rate_hz": 5000.0},
        "reference": {"fit_train_trials_only": True},
        "evaluation": {
            "primary_split": "validation",
            "primary_metric": "bits_per_spike",
            "evaluate_splits": ["train", "validation", "test"],
        },
    }


def test_neuron_group_selection_shapes_and_nonoverlap() -> None:
    spikes = np.zeros((2, 3, 4), dtype=np.int64)

    heldin, heldin_indices = select_neuron_group(spikes, _mask(), "heldin")
    heldout, heldout_indices = select_neuron_group(spikes, _mask(), "heldout")

    assert heldin.shape == (2, 3, 2)
    assert heldout.shape == (2, 3, 2)
    assert set(heldin_indices).isdisjoint(set(heldout_indices))


def test_flattening_preserves_sample_count() -> None:
    values = np.zeros((2, 3, 4), dtype=np.float64)

    flat = flatten_trial_time(values)

    assert flat.shape == (6, 4)


def test_cosmoothing_ridge_prediction_shape_positive_and_clipped() -> None:
    x = np.arange(12, dtype=np.float64).reshape(6, 2)
    counts = np.stack([x[:, 0] + 1.0, x[:, 1] + 1.0], axis=1)

    model = fit_cosmoothing_ridge(x, counts, 5, alpha=1.0, min_rate_hz=0.1, max_rate_hz=20.0)
    pred = predict_cosmoothing_rates(x, model, min_rate_hz=0.1, max_rate_hz=20.0)

    assert pred.shape == counts.shape
    assert np.all(np.isfinite(pred))
    assert pred.min() >= 0.1
    assert pred.max() <= 20.0


def test_validation_targets_do_not_affect_fitted_model() -> None:
    train_x = np.array([[0.0], [1.0], [2.0]])
    train_counts = train_x + 1.0
    model_a = fit_cosmoothing_ridge(
        train_x, train_counts, 5, alpha=0.0, min_rate_hz=0.1, max_rate_hz=1000
    )
    model_b = fit_cosmoothing_ridge(
        train_x, train_counts, 5, alpha=0.0, min_rate_hz=0.1, max_rate_hz=1000
    )

    np.testing.assert_allclose(model_a["coefficients"], model_b["coefficients"])
    np.testing.assert_allclose(model_a["feature_stats"]["mean"], np.array([1.0]))


def test_reference_rates_and_bits_per_spike_use_train_targets_only() -> None:
    dataset = _easy_dataset()
    split = TrialSplit(train=np.array([0, 1, 2]), validation=np.array([3]), test=np.array([4]))

    split_metrics, _, metadata = run_cosmoothing_baseline(dataset, split, _mask(), _config())

    train_target = dataset.spikes[:3, :, 2:]
    train_seconds = train_target.shape[0] * train_target.shape[1] * 0.005
    expected_reference = train_target.sum(axis=(0, 1)) / train_seconds
    np.testing.assert_allclose(metadata["reference_rates_hz"], expected_reference)
    assert split_metrics.loc[split_metrics["split"] == "validation", "bits_per_spike"].iloc[0] > 0


def test_sweep_uses_train_only_fit_and_evaluates_all_splits() -> None:
    dataset = _easy_dataset()
    split = TrialSplit(train=np.array([0, 1, 2]), validation=np.array([3]), test=np.array([4]))

    sweep_results, best_config, best_split_metrics, _ = run_cosmoothing_sweep(
        dataset, split, _mask(), _sweep_config()
    )

    assert set(sweep_results["split"]) == {"train", "validation", "test"}
    assert set(best_split_metrics["split"]) == {"train", "validation", "test"}
    assert set(sweep_results["n_train_trials"]) == {3}
    assert best_config["run_id"] in set(sweep_results["run_id"])


def test_sweep_validation_and_test_targets_do_not_affect_fit() -> None:
    dataset = _easy_dataset()
    split = TrialSplit(train=np.array([0, 1, 2]), validation=np.array([3]), test=np.array([4]))
    changed = _easy_dataset()
    changed.spikes[3:, :, 2:] += 100

    results_a, _, _, _ = run_cosmoothing_sweep(dataset, split, _mask(), _sweep_config())
    results_b, _, _, _ = run_cosmoothing_sweep(changed, split, _mask(), _sweep_config())

    train_a = results_a[results_a["split"] == "train"].reset_index(drop=True)
    train_b = results_b[results_b["split"] == "train"].reset_index(drop=True)
    np.testing.assert_allclose(train_a["mean_predicted_rate_hz"], train_b["mean_predicted_rate_hz"])


def test_sweep_results_contain_required_columns() -> None:
    dataset = _easy_dataset()
    split = TrialSplit(train=np.array([0, 1, 2]), validation=np.array([3]), test=np.array([4]))

    sweep_results, _, _, _ = run_cosmoothing_sweep(dataset, split, _mask(), _sweep_config())

    assert {
        "run_id",
        "split",
        "smoothing_sigma_ms",
        "ridge_alpha",
        "standardize_features",
        "fit_intercept",
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
        "mean_predicted_rate_hz",
        "mean_reference_rate_hz",
    }.issubset(set(sweep_results.columns))


def test_evaluate_predictions_reports_poisson_and_rate_metrics() -> None:
    counts = np.array([[[1, 2], [3, 4]]], dtype=np.int64)
    predicted = counts / 0.005
    reference = np.full_like(predicted, 100.0, dtype=np.float64)

    metrics = evaluate_cosmoothing_predictions(counts, predicted, reference, 5)

    assert metrics["poisson_nll"] > 0
    assert metrics["bits_per_spike"] > 0
    assert metrics["mse_rate_hz"] == 0.0


def test_invalid_shapes_raise_clear_error() -> None:
    with pytest.raises(ValueError, match="rank 3"):
        select_neuron_group(np.zeros((2, 3)), _mask(), "heldin")
