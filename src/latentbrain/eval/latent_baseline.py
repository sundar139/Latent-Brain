from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from latentbrain.data.schemas import NeuralDataset, NeuronMask, TrialSplit
from latentbrain.eval.behavior import derive_velocity_targets, select_behavior_targets
from latentbrain.eval.cosmoothing import (
    _broadcast_reference,
    _neuron_rows,
    _rates_from_counts,
    _reference_rates,
    _split_ids,
    _trial_mask,
    evaluate_cosmoothing_predictions,
    flatten_trial_time,
    select_neuron_group,
)
from latentbrain.eval.decoding import (
    fit_ridge_decoder,
    predict_ridge_decoder,
    regression_metrics,
    standardize_train_apply,
)
from latentbrain.eval.metrics import safe_clip_rates
from latentbrain.eval.smoothing import smooth_spike_counts, spike_counts_to_rates_hz
from latentbrain.models.factor_latent import FactorLatentModel

SPLIT_COLUMNS = [
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
]

NEURON_COLUMNS = [
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

BEHAVIOR_COLUMNS = ["split", "target_name", "r2", "mse", "mae", "target_variance"]
LATENT_COLUMNS = ["split", "latent_dim_index", "mean", "std", "min", "max", "variance"]


def _flatten_2d(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    subset = values[mask]
    return np.asarray(subset.reshape(subset.shape[0] * subset.shape[1], subset.shape[2]))


def _transform_latents(
    model: FactorLatentModel,
    features_3d: np.ndarray,
    feature_stats: dict[str, np.ndarray],
) -> np.ndarray:
    flat = flatten_trial_time(features_3d)
    if feature_stats:
        flat = (flat - feature_stats["mean"]) / feature_stats["std"]
    latents = model.transform(flat)
    return latents.reshape(features_3d.shape[0], features_3d.shape[1], latents.shape[1])


def _latent_summary(split_name: str, latents: np.ndarray) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    flat = flatten_trial_time(latents)
    for index in range(flat.shape[1]):
        values = flat[:, index]
        rows.append(
            {
                "split": split_name,
                "latent_dim_index": index,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "variance": float(np.var(values)),
            }
        )
    return rows


def _fit_behavior_decoder(
    dataset: NeuralDataset,
    train_mask: np.ndarray,
    train_latents: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[str], dict[str, Any], dict[str, np.ndarray]]:
    if dataset.behavior is None or dataset.behavior_names is None:
        msg = "behavior decoder is enabled but dataset has no behavior"
        raise ValueError(msg)
    behavior_config = config["behavior_decoder"]
    positions, position_names = select_behavior_targets(
        dataset.behavior, dataset.behavior_names, list(behavior_config["target_prefixes"])
    )
    if bool(behavior_config.get("derive_velocity", True)):
        targets_3d, target_names = derive_velocity_targets(
            positions,
            position_names,
            dataset.bin_size_ms,
            method=str(behavior_config.get("velocity_method", "central_difference")),
        )
    else:
        targets_3d, target_names = positions, position_names
    train_targets = _flatten_2d(targets_3d, train_mask)
    if bool(behavior_config.get("standardize_targets", True)):
        train_targets_fit, target_stats = standardize_train_apply(train_targets, train_targets)
    else:
        train_targets_fit = train_targets
        target_stats = {}
    decoder = fit_ridge_decoder(
        flatten_trial_time(train_latents),
        train_targets_fit,
        alpha=float(behavior_config["alpha"]),
        fit_intercept=bool(behavior_config.get("fit_intercept", True)),
    )
    return targets_3d, target_names, decoder, target_stats


def run_factor_latent_baseline(
    dataset: NeuralDataset,
    split: TrialSplit,
    neuron_mask: NeuronMask,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    features_config = config["features"]
    latent_config = config["latent_model"]
    decoder_config = config["heldout_decoder"]
    evaluation_config = config["evaluation"]
    latent_dim = int(latent_config["latent_dim"])

    input_spikes, input_indices = select_neuron_group(
        dataset.spikes, neuron_mask, features_config.get("input_neuron_group", "heldin")
    )
    target_counts, target_indices = select_neuron_group(
        dataset.spikes, neuron_mask, features_config.get("target_neuron_group", "heldout")
    )
    if latent_dim >= input_indices.size:
        msg = "latent_dim must be less than the number of held-in neurons"
        raise ValueError(msg)

    smoothed = smooth_spike_counts(
        input_spikes,
        dataset.bin_size_ms,
        method=features_config["smoothing"]["method"],
        sigma_ms=float(features_config["smoothing"]["sigma_ms"]),
        truncate=float(features_config["smoothing"].get("truncate", 4.0)),
    )
    input_rates = (
        spike_counts_to_rates_hz(smoothed, dataset.bin_size_ms)
        if bool(features_config.get("convert_to_hz", True))
        else smoothed
    )
    train_mask = _trial_mask(dataset, split.train)
    train_features_raw = flatten_trial_time(input_rates[train_mask])
    if bool(features_config.get("standardize_features", True)):
        train_features, feature_stats = standardize_train_apply(
            train_features_raw, train_features_raw
        )
    else:
        train_features = train_features_raw
        feature_stats = {}

    factor_model = FactorLatentModel(
        latent_dim=latent_dim,
        random_state=int(latent_config["random_state"]),
        max_iter=int(latent_config["max_iter"]),
        tol=float(latent_config["tol"]),
    ).fit(train_features)

    latents_by_split: dict[str, np.ndarray] = {}
    for split_name in evaluation_config["evaluate_splits"]:
        mask = _trial_mask(dataset, _split_ids(split, split_name))
        latents_by_split[split_name] = _transform_latents(
            factor_model, input_rates[mask], feature_stats
        )

    min_rate = float(decoder_config["min_rate_hz"])
    max_rate = float(decoder_config["max_rate_hz"])
    train_latents = latents_by_split["train"]
    train_target_rates = safe_clip_rates(
        _rates_from_counts(flatten_trial_time(target_counts[train_mask]), dataset.bin_size_ms),
        min_rate,
        max_rate,
    )
    heldout_decoder = fit_ridge_decoder(
        flatten_trial_time(train_latents),
        train_target_rates,
        alpha=float(decoder_config["alpha"]),
        fit_intercept=bool(decoder_config.get("fit_intercept", True)),
    )
    reference = _reference_rates(target_counts[train_mask], dataset.bin_size_ms, min_rate, max_rate)

    split_rows: list[dict[str, Any]] = []
    neuron_rows: list[dict[str, Any]] = []
    latent_rows: list[dict[str, float | int | str]] = []
    for split_name in evaluation_config["evaluate_splits"]:
        mask = _trial_mask(dataset, _split_ids(split, split_name))
        counts = target_counts[mask]
        predicted = safe_clip_rates(
            predict_ridge_decoder(
                flatten_trial_time(latents_by_split[split_name]), heldout_decoder
            ),
            min_rate,
            max_rate,
        ).reshape(counts.shape)
        reference_rates = _broadcast_reference(reference, counts)
        metrics = evaluate_cosmoothing_predictions(
            counts, predicted, reference_rates, dataset.bin_size_ms
        )
        split_rows.append(
            {
                "split": split_name,
                "n_trials": int(counts.shape[0]),
                "n_time_bins": int(counts.shape[1]),
                "n_input_neurons": int(input_indices.size),
                "n_target_neurons": int(target_indices.size),
                "latent_dim": latent_dim,
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
        latent_rows.extend(_latent_summary(split_name, latents_by_split[split_name]))

    behavior_config = config.get("behavior_decoder", {})
    behavior_decoder: dict[str, Any] = {}
    behavior_target_names: list[str] = []
    target_stats: dict[str, np.ndarray] = {}
    if bool(behavior_config.get("enabled", False)):
        targets_3d, behavior_target_names, behavior_decoder, target_stats = _fit_behavior_decoder(
            dataset, train_mask, train_latents, config
        )
        frames = []
        for split_name in evaluation_config["evaluate_splits"]:
            mask = _trial_mask(dataset, _split_ids(split, split_name))
            pred_fit = predict_ridge_decoder(
                flatten_trial_time(latents_by_split[split_name]), behavior_decoder
            )
            pred = (
                pred_fit * target_stats["std"] + target_stats["mean"] if target_stats else pred_fit
            )
            metrics = regression_metrics(
                flatten_trial_time(targets_3d[mask]),
                pred,
                behavior_target_names,
            )
            metrics.insert(0, "split", split_name)
            frames.append(metrics)
        behavior_metrics = pd.concat(frames, ignore_index=True)[BEHAVIOR_COLUMNS]
    else:
        behavior_metrics = pd.DataFrame(columns=BEHAVIOR_COLUMNS)

    metadata = {
        "model_name": "factor_analysis",
        "latent_dim": latent_dim,
        "input_neuron_indices": input_indices.tolist(),
        "target_neuron_indices": target_indices.tolist(),
        "feature_stats": feature_stats,
        "factor_components": factor_model.components_,
        "factor_noise_variance": factor_model.noise_variance_,
        "heldout_decoder_coefficients": heldout_decoder["coefficients"],
        "heldout_decoder_intercept": heldout_decoder["intercept"],
        "reference_rates_hz": reference,
        "behavior_decoder_enabled": bool(behavior_config.get("enabled", False)),
        "behavior_target_names": behavior_target_names,
        "behavior_decoder_coefficients": behavior_decoder.get(
            "coefficients", np.empty((latent_dim, 0))
        ),
        "behavior_decoder_intercept": behavior_decoder.get("intercept", np.empty(0)),
        "behavior_target_stats": target_stats,
        "latent_shapes": {
            name: [int(value) for value in latents.shape]
            for name, latents in latents_by_split.items()
        },
        "train_only_fit": True,
    }
    return (
        pd.DataFrame(split_rows, columns=SPLIT_COLUMNS),
        pd.DataFrame(neuron_rows, columns=NEURON_COLUMNS),
        behavior_metrics,
        pd.DataFrame(latent_rows, columns=LATENT_COLUMNS),
        metadata,
    )
