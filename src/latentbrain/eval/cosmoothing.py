from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from latentbrain.data.schemas import NeuralDataset, NeuronMask, TrialSplit
from latentbrain.eval.decoding import (
    fit_ridge_decoder,
    predict_ridge_decoder,
    standardize_train_apply,
)
from latentbrain.eval.metrics import (
    poisson_log_likelihood,
    safe_clip_rates,
    summarize_poisson_metrics,
)
from latentbrain.eval.smoothing import smooth_spike_counts, spike_counts_to_rates_hz
from latentbrain.eval.sweeps import expand_grid, rank_sweep_results, select_best_config


def _validate_3d(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3:
        msg = f"{name} must have rank 3; got shape {array.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must be finite"
        raise ValueError(msg)
    return array


def _group_mask(mask: NeuronMask, group: str) -> np.ndarray:
    if group == "heldin":
        return np.asarray(mask.heldin, dtype=bool)
    if group == "heldout":
        return np.asarray(mask.heldout, dtype=bool)
    if group == "all":
        return np.asarray(mask.heldin | mask.heldout, dtype=bool)
    msg = f"unknown neuron group: {group}"
    raise ValueError(msg)


def _trial_mask(dataset: NeuralDataset, trial_ids: np.ndarray) -> np.ndarray:
    return np.isin(dataset.trial_ids, trial_ids)


def _split_ids(split: TrialSplit, name: str) -> np.ndarray:
    if name == "train":
        return split.train
    if name == "validation":
        return split.validation
    if name == "test":
        return split.test
    msg = f"unknown split: {name}"
    raise ValueError(msg)


def select_neuron_group(
    spikes: np.ndarray,
    mask: NeuronMask,
    group: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Select one neuron group and return selected spikes plus original indices."""
    spikes_array = _validate_3d("spikes", spikes)
    group_values = _group_mask(mask, group)
    if group_values.shape[0] != spikes_array.shape[2]:
        msg = "neuron mask length must match spikes neuron dimension"
        raise ValueError(msg)
    indices = np.flatnonzero(group_values)
    if indices.size == 0:
        msg = f"neuron group {group} is empty"
        raise ValueError(msg)
    return spikes_array[:, :, indices], indices


def flatten_trial_time(values: np.ndarray) -> np.ndarray:
    """Flatten trial/time axes into samples."""
    array = _validate_3d("values", values)
    return array.reshape(array.shape[0] * array.shape[1], array.shape[2])


def _rates_from_counts(counts: np.ndarray, bin_size_ms: int) -> np.ndarray:
    if bin_size_ms <= 0:
        msg = "bin_size_ms must be positive"
        raise ValueError(msg)
    return np.asarray(counts, dtype=np.float64) / (bin_size_ms / 1000.0)


def fit_cosmoothing_ridge(
    train_input_rates_hz: np.ndarray,
    train_target_counts: np.ndarray,
    bin_size_ms: int,
    alpha: float,
    min_rate_hz: float,
    max_rate_hz: float,
    standardize_features: bool = True,
    fit_intercept: bool = True,
) -> dict[str, Any]:
    """Fit train-only ridge from held-in rates to held-out target rates."""
    x = np.asarray(train_input_rates_hz, dtype=np.float64)
    y_counts = np.asarray(train_target_counts, dtype=np.float64)
    if x.ndim != 2 or y_counts.ndim != 2:
        msg = "train_input_rates_hz and train_target_counts must have rank 2"
        raise ValueError(msg)
    if x.shape[0] != y_counts.shape[0]:
        msg = "input rates and target counts must have the same sample count"
        raise ValueError(msg)
    y = safe_clip_rates(_rates_from_counts(y_counts, bin_size_ms), min_rate_hz, max_rate_hz)
    if standardize_features:
        x_fit, feature_stats = standardize_train_apply(x, x)
    else:
        x_fit = x
        feature_stats = {}
    decoder = fit_ridge_decoder(x_fit, y, alpha=alpha, fit_intercept=fit_intercept)
    return {**decoder, "feature_stats": feature_stats, "bin_size_ms": np.array(bin_size_ms)}


def fit_reduced_rank_cosmoothing(
    train_input_rates_hz: np.ndarray,
    train_target_counts: np.ndarray,
    bin_size_ms: int,
    alpha: float,
    rank: int,
    min_rate_hz: float,
    max_rate_hz: float,
    standardize_features: bool = True,
    fit_intercept: bool = True,
) -> dict[str, Any]:
    """Reduced-rank ridge from held-in rates to held-out rates, fit on training samples only.

    This is linear reduced-rank regression: ridge coefficients projected onto the top `rank`
    right-singular directions of the fitted training predictions. It carries no temporal or
    dynamical assumptions, so it is not GPFA and not a latent dynamical model.
    """
    model = fit_cosmoothing_ridge(
        train_input_rates_hz,
        train_target_counts,
        bin_size_ms,
        alpha,
        min_rate_hz,
        max_rate_hz,
        standardize_features=standardize_features,
        fit_intercept=fit_intercept,
    )
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    max_rank = int(min(coefficients.shape))
    if rank <= 0:
        msg = "rank must be positive"
        raise ValueError(msg)
    if rank > max_rank:
        msg = f"rank {rank} exceeds the maximum rank {max_rank} of the coefficient matrix"
        raise ValueError(msg)

    x = np.asarray(train_input_rates_hz, dtype=np.float64)
    stats = model.get("feature_stats", {})
    if stats:
        x = (x - stats["mean"]) / stats["std"]
    # Truncate in the output basis of the fitted training predictions, which is the standard
    # reduced-rank-regression projection and keeps the intercept unconstrained.
    centered = x @ coefficients
    _, _, right = np.linalg.svd(centered - centered.mean(axis=0), full_matrices=False)
    projection = right[:rank].T @ right[:rank]
    return {**model, "coefficients": coefficients @ projection, "rank": np.array(rank)}


def predict_cosmoothing_rates(
    input_rates_hz: np.ndarray,
    model: dict[str, Any],
    min_rate_hz: float,
    max_rate_hz: float,
) -> np.ndarray:
    """Predict clipped held-out rates from held-in rates."""
    array = np.asarray(input_rates_hz, dtype=np.float64)
    original_shape = array.shape
    if array.ndim == 3:
        flat = flatten_trial_time(array)
    elif array.ndim == 2:
        flat = array
    else:
        msg = f"input_rates_hz must have rank 2 or 3; got shape {array.shape}"
        raise ValueError(msg)
    stats = model.get("feature_stats", {})
    if stats:
        flat = (flat - stats["mean"]) / stats["std"]
    predicted = safe_clip_rates(predict_ridge_decoder(flat, model), min_rate_hz, max_rate_hz)
    if len(original_shape) == 3:
        return predicted.reshape(original_shape[0], original_shape[1], predicted.shape[1])
    return predicted


def evaluate_cosmoothing_predictions(
    target_counts: np.ndarray,
    predicted_rates_hz: np.ndarray,
    reference_rates_hz: np.ndarray,
    bin_size_ms: int,
) -> dict[str, float]:
    """Evaluate Poisson and rate-error metrics for held-out predictions."""
    counts = _validate_3d("target_counts", target_counts)
    predicted = _validate_3d("predicted_rates_hz", predicted_rates_hz)
    reference = _validate_3d("reference_rates_hz", reference_rates_hz)
    target_rates = _rates_from_counts(counts, bin_size_ms)
    metrics = summarize_poisson_metrics(counts, predicted, reference, bin_size_ms)
    metrics["mse_rate_hz"] = float(np.mean((predicted - target_rates) ** 2))
    metrics["mae_rate_hz"] = float(np.mean(np.abs(predicted - target_rates)))
    metrics["mean_reference_rate_hz"] = float(np.mean(reference))
    return metrics


def _reference_rates(
    train_target_counts: np.ndarray,
    bin_size_ms: int,
    min_rate: float,
    max_rate: float,
) -> np.ndarray:
    seconds = train_target_counts.shape[0] * train_target_counts.shape[1] * (bin_size_ms / 1000.0)
    return safe_clip_rates(train_target_counts.sum(axis=(0, 1)) / seconds, min_rate, max_rate)


def _broadcast_reference(reference: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return np.broadcast_to(reference, counts.shape).copy()


def _neuron_rows(
    split_name: str,
    target_counts: np.ndarray,
    predicted_rates: np.ndarray,
    reference_rates: np.ndarray,
    target_indices: np.ndarray,
    bin_size_ms: int,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    target_rates = _rates_from_counts(target_counts, bin_size_ms)
    for rank, neuron_index in enumerate(target_indices):
        counts = target_counts[:, :, rank : rank + 1]
        predicted = predicted_rates[:, :, rank : rank + 1]
        reference = reference_rates[:, :, rank : rank + 1]
        target_rate = target_rates[:, :, rank : rank + 1]
        spike_count = float(np.sum(counts))
        if spike_count > 0.0:
            metrics = evaluate_cosmoothing_predictions(counts, predicted, reference, bin_size_ms)
        else:
            model_ll = poisson_log_likelihood(counts, predicted, bin_size_ms)
            reference_ll = poisson_log_likelihood(counts, reference, bin_size_ms)
            metrics = {
                "spike_count": spike_count,
                "poisson_nll": -model_ll,
                "poisson_log_likelihood": model_ll,
                "reference_log_likelihood": reference_ll,
                "bits_per_spike": float("nan"),
                "mse_rate_hz": float(np.mean((predicted - target_rate) ** 2)),
                "mae_rate_hz": float(np.mean(np.abs(predicted - target_rate))),
                "mean_reference_rate_hz": float(np.mean(reference)),
            }
        rows.append(
            {
                "split": split_name,
                "target_neuron_index": int(neuron_index),
                "target_neuron_rank": int(rank),
                **metrics,
                "train_reference_rate_hz": float(reference[0, 0, 0]),
            }
        )
    return rows


def run_cosmoothing_baseline(
    dataset: NeuralDataset,
    split: TrialSplit,
    neuron_mask: NeuronMask,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run train-only held-in to held-out ridge co-smoothing baseline."""
    features_config = config["features"]
    targets_config = config["targets"]
    decoder_config = config["decoder"]
    evaluation_config = config["evaluation"]
    min_rate = float(targets_config["min_rate_hz"])
    max_rate = float(targets_config["max_rate_hz"])

    input_spikes, input_indices = select_neuron_group(
        dataset.spikes, neuron_mask, features_config.get("input_neuron_group", "heldin")
    )
    target_counts, target_indices = select_neuron_group(
        dataset.spikes, neuron_mask, features_config.get("target_neuron_group", "heldout")
    )
    smoothed = smooth_spike_counts(
        input_spikes,
        dataset.bin_size_ms,
        method=features_config["smoothing"]["method"],
        sigma_ms=float(features_config["smoothing"]["sigma_ms"]),
        truncate=float(features_config["smoothing"]["truncate"]),
    )
    input_rates = (
        spike_counts_to_rates_hz(smoothed, dataset.bin_size_ms)
        if bool(features_config.get("convert_to_hz", True))
        else smoothed
    )

    train_mask = _trial_mask(dataset, split.train)
    train_input = flatten_trial_time(input_rates[train_mask])
    train_target = flatten_trial_time(target_counts[train_mask])
    model = fit_cosmoothing_ridge(
        train_input,
        train_target,
        dataset.bin_size_ms,
        alpha=float(decoder_config["alpha"]),
        min_rate_hz=min_rate,
        max_rate_hz=max_rate,
        standardize_features=bool(features_config.get("standardize_features", True)),
        fit_intercept=bool(decoder_config.get("fit_intercept", True)),
    )
    reference = _reference_rates(target_counts[train_mask], dataset.bin_size_ms, min_rate, max_rate)

    split_rows: list[dict[str, float | int | str]] = []
    neuron_rows: list[dict[str, float | int | str]] = []
    for split_name in evaluation_config["evaluate_splits"]:
        mask = _trial_mask(dataset, _split_ids(split, split_name))
        counts = target_counts[mask]
        predicted = predict_cosmoothing_rates(input_rates[mask], model, min_rate, max_rate)
        reference_rates = _broadcast_reference(reference, counts)
        metrics = evaluate_cosmoothing_predictions(
            counts,
            predicted,
            reference_rates,
            dataset.bin_size_ms,
        )
        split_rows.append(
            {
                "split": split_name,
                "n_trials": int(counts.shape[0]),
                "n_time_bins": int(counts.shape[1]),
                "n_input_neurons": int(input_indices.size),
                "n_target_neurons": int(target_indices.size),
                **metrics,
            }
        )
        neuron_rows.extend(
            _neuron_rows(
                split_name,
                counts,
                predicted,
                reference_rates,
                target_indices,
                dataset.bin_size_ms,
            )
        )

    metadata = {
        "input_neuron_indices": input_indices.tolist(),
        "target_neuron_indices": target_indices.tolist(),
        "coefficients": model["coefficients"],
        "intercept": model["intercept"],
        "feature_stats": model["feature_stats"],
        "reference_rates_hz": reference,
        "train_only_fit": True,
    }
    split_columns = [
        "split",
        "n_trials",
        "n_time_bins",
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
    ]
    neuron_columns = [
        "split",
        "target_neuron_index",
        "target_neuron_rank",
        "spike_count",
        "poisson_nll",
        "poisson_log_likelihood",
        "reference_log_likelihood",
        "bits_per_spike",
        "mse_rate_hz",
        "mae_rate_hz",
        "train_reference_rate_hz",
        "mean_predicted_rate_hz",
    ]
    return (
        pd.DataFrame(split_rows, columns=split_columns),
        pd.DataFrame(neuron_rows, columns=neuron_columns),
        metadata,
    )


SWEEP_RESULT_COLUMNS = [
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
]


def _sweep_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    sweep = config["sweep"]
    return expand_grid(
        {
            "smoothing_sigma_ms": list(sweep["smoothing_sigma_ms"]),
            "ridge_alpha": list(sweep["ridge_alpha"]),
            "standardize_features": list(sweep["standardize_features"]),
            "fit_intercept": list(sweep["fit_intercept"]),
        }
    )


def _input_rates_for_sigma(
    input_spikes: np.ndarray,
    bin_size_ms: int,
    sigma_ms: float,
    convert_to_hz: bool,
) -> np.ndarray:
    smoothed = smooth_spike_counts(
        input_spikes,
        bin_size_ms,
        method="gaussian",
        sigma_ms=sigma_ms,
        truncate=4.0,
    )
    return spike_counts_to_rates_hz(smoothed, bin_size_ms) if convert_to_hz else smoothed


def run_cosmoothing_sweep(
    dataset: NeuralDataset,
    split: TrialSplit,
    neuron_mask: NeuronMask,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Run a train-only ridge co-smoothing diagnostic sweep."""
    if config["targets"].get("fit_target_type", "rate_hz") != "rate_hz":
        msg = "only rate_hz co-smoothing targets are supported"
        raise ValueError(msg)

    features_config = config["features"]
    targets_config = config["targets"]
    evaluation_config = config["evaluation"]
    min_rate = float(targets_config["min_rate_hz"])
    max_rate = float(targets_config["max_rate_hz"])
    evaluate_splits = list(evaluation_config["evaluate_splits"])

    input_spikes, input_indices = select_neuron_group(
        dataset.spikes, neuron_mask, features_config.get("input_neuron_group", "heldin")
    )
    target_counts, target_indices = select_neuron_group(
        dataset.spikes, neuron_mask, features_config.get("target_neuron_group", "heldout")
    )
    train_mask = _trial_mask(dataset, split.train)
    reference = _reference_rates(target_counts[train_mask], dataset.bin_size_ms, min_rate, max_rate)

    rows: list[dict[str, Any]] = []
    neuron_rows: list[dict[str, Any]] = []
    rates_by_sigma: dict[float, np.ndarray] = {}
    for run_index, params in enumerate(_sweep_grid(config)):
        run_id = f"run_{run_index:03d}"
        sigma_ms = float(params["smoothing_sigma_ms"])
        if sigma_ms not in rates_by_sigma:
            rates_by_sigma[sigma_ms] = _input_rates_for_sigma(
                input_spikes,
                dataset.bin_size_ms,
                sigma_ms,
                convert_to_hz=bool(features_config.get("convert_to_hz", True)),
            )
        input_rates = rates_by_sigma[sigma_ms]
        model = fit_cosmoothing_ridge(
            flatten_trial_time(input_rates[train_mask]),
            flatten_trial_time(target_counts[train_mask]),
            dataset.bin_size_ms,
            alpha=float(params["ridge_alpha"]),
            min_rate_hz=min_rate,
            max_rate_hz=max_rate,
            standardize_features=bool(params["standardize_features"]),
            fit_intercept=bool(params["fit_intercept"]),
        )
        for split_name in evaluate_splits:
            eval_mask = _trial_mask(dataset, _split_ids(split, split_name))
            counts = target_counts[eval_mask]
            predicted = predict_cosmoothing_rates(input_rates[eval_mask], model, min_rate, max_rate)
            reference_rates = _broadcast_reference(reference, counts)
            metrics = evaluate_cosmoothing_predictions(
                counts,
                predicted,
                reference_rates,
                dataset.bin_size_ms,
            )
            rows.append(
                {
                    "run_id": run_id,
                    "split": split_name,
                    "smoothing_sigma_ms": sigma_ms,
                    "ridge_alpha": float(params["ridge_alpha"]),
                    "standardize_features": bool(params["standardize_features"]),
                    "fit_intercept": bool(params["fit_intercept"]),
                    "n_train_trials": int(np.count_nonzero(train_mask)),
                    "n_eval_trials": int(np.count_nonzero(eval_mask)),
                    "n_input_neurons": int(input_indices.size),
                    "n_target_neurons": int(target_indices.size),
                    **metrics,
                }
            )
            for neuron_row in _neuron_rows(
                split_name,
                counts,
                predicted,
                reference_rates,
                target_indices,
                dataset.bin_size_ms,
            ):
                neuron_rows.append(
                    {
                        "run_id": run_id,
                        "smoothing_sigma_ms": sigma_ms,
                        "ridge_alpha": float(params["ridge_alpha"]),
                        "standardize_features": bool(params["standardize_features"]),
                        "fit_intercept": bool(params["fit_intercept"]),
                        **neuron_row,
                    }
                )

    sweep_results = pd.DataFrame(rows, columns=SWEEP_RESULT_COLUMNS)
    if sweep_results.empty:
        msg = "co-smoothing sweep produced no results"
        raise ValueError(msg)
    ranked = rank_sweep_results(
        sweep_results,
        primary_split=str(evaluation_config["primary_split"]),
        primary_metric=str(evaluation_config["primary_metric"]),
    )
    best_config = select_best_config(ranked)
    best_config.update(
        {
            "input_neuron_group": features_config.get("input_neuron_group", "heldin"),
            "target_neuron_group": features_config.get("target_neuron_group", "heldout"),
            "fit_target_type": config["targets"].get("fit_target_type", "rate_hz"),
            "min_rate_hz": min_rate,
            "max_rate_hz": max_rate,
            "reference": config.get("reference", {}).get("name", "train_mean_rate"),
            "train_only_fit": True,
        }
    )
    best_split_metrics = sweep_results[
        sweep_results["run_id"] == best_config["run_id"]
    ].reset_index(drop=True)
    best_neuron_metrics = pd.DataFrame(neuron_rows)
    best_neuron_metrics = best_neuron_metrics[
        best_neuron_metrics["run_id"] == best_config["run_id"]
    ].reset_index(drop=True)
    return sweep_results, best_config, best_split_metrics, best_neuron_metrics
