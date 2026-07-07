from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from latentbrain.data.schemas import NeuralDataset, NeuronMask, TrialSplit
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
from latentbrain.eval.decoding import fit_ridge_decoder, predict_ridge_decoder, regression_metrics
from latentbrain.eval.latent_baseline import (
    _fit_behavior_decoder,
    _latent_summary,
    _transform_latents,
)
from latentbrain.eval.metrics import safe_clip_rates
from latentbrain.eval.smoothing import smooth_spike_counts, spike_counts_to_rates_hz
from latentbrain.eval.sweeps import expand_grid, rank_sweep_results
from latentbrain.models.factor_latent import FactorLatentModel

SWEEP_RESULT_COLUMNS = [
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
]


def _sweep_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    sweep = config["sweep"]
    return expand_grid(
        {
            "latent_dim": list(sweep["latent_dim"]),
            "smoothing_sigma_ms": list(sweep["smoothing_sigma_ms"]),
            "heldout_decoder_alpha": list(sweep["heldout_decoder_alpha"]),
            "standardize_features": list(sweep["standardize_features"]),
        }
    )


def _baseline_config(config: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    return {
        "features": {
            "input_neuron_group": config["features"].get("input_neuron_group", "heldin"),
            "target_neuron_group": config["features"].get("target_neuron_group", "heldout"),
            "smoothing": {
                "method": "gaussian",
                "sigma_ms": float(params["smoothing_sigma_ms"]),
                "truncate": 4.0,
            },
            "convert_to_hz": bool(config["features"].get("convert_to_hz", True)),
            "standardize_features": bool(params["standardize_features"]),
        },
        "latent_model": {
            **config["latent_model"],
            "latent_dim": int(params["latent_dim"]),
        },
        "heldout_decoder": {
            **config["heldout_decoder"],
            "alpha": float(params["heldout_decoder_alpha"]),
        },
        "behavior_decoder": dict(config["behavior_decoder"]),
        "reference": dict(config["reference"]),
        "evaluation": dict(config["evaluation"]),
    }


def _mean_behavior_by_split(behavior_metrics: pd.DataFrame) -> dict[str, dict[str, float]]:
    if behavior_metrics.empty:
        return {}
    grouped = behavior_metrics.groupby("split", sort=False)[["r2", "mse", "mae"]].mean()
    return {
        str(split): {
            "behavior_mean_r2": float(row["r2"]),
            "behavior_mean_mse": float(row["mse"]),
            "behavior_mean_mae": float(row["mae"]),
        }
        for split, row in grouped.iterrows()
    }


def _best_config_from_row(row: pd.Series) -> dict[str, Any]:
    return {
        "run_id": str(row["run_id"]),
        "latent_dim": int(row["latent_dim"]),
        "smoothing_sigma_ms": float(row["smoothing_sigma_ms"]),
        "heldout_decoder_alpha": float(row["heldout_decoder_alpha"]),
        "standardize_features": bool(row["standardize_features"]),
    }


def _rates_for_sigma(
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


def _standardize_train_only(
    train_features: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    mean = np.mean(train_features, axis=0, keepdims=True)
    std = np.std(train_features, axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (train_features - mean) / std, {"mean": mean, "std": std}


def _fit_latent_cache_entry(
    dataset: NeuralDataset,
    split: TrialSplit,
    config: dict[str, Any],
    input_rates: np.ndarray,
    params: dict[str, Any],
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    train_mask = _trial_mask(dataset, split.train)
    train_features_raw = flatten_trial_time(input_rates[train_mask])
    if bool(params["standardize_features"]):
        train_features, feature_stats = _standardize_train_only(train_features_raw)
    else:
        train_features = train_features_raw
        feature_stats = {}

    latent_config = config["latent_model"]
    factor_model = FactorLatentModel(
        latent_dim=int(params["latent_dim"]),
        random_state=int(latent_config["random_state"]),
        max_iter=int(latent_config["max_iter"]),
        tol=float(latent_config["tol"]),
    ).fit(train_features)

    latents_by_split = {
        split_name: _transform_latents(
            factor_model,
            input_rates[_trial_mask(dataset, _split_ids(split, split_name))],
            feature_stats,
        )
        for split_name in config["evaluation"]["evaluate_splits"]
    }
    latent_summary = pd.DataFrame(
        [
            row
            for split_name, latents in latents_by_split.items()
            for row in _latent_summary(split_name, latents)
        ]
    )
    behavior_metrics = _behavior_metrics(dataset, split, config, latents_by_split, train_mask)
    return latents_by_split, behavior_metrics, latent_summary


def _behavior_metrics(
    dataset: NeuralDataset,
    split: TrialSplit,
    config: dict[str, Any],
    latents_by_split: dict[str, np.ndarray],
    train_mask: np.ndarray,
) -> pd.DataFrame:
    if not bool(config.get("behavior_decoder", {}).get("enabled", False)):
        return pd.DataFrame(columns=["split", "target_name", "r2", "mse", "mae", "target_variance"])
    targets_3d, target_names, decoder, target_stats = _fit_behavior_decoder(
        dataset,
        train_mask,
        latents_by_split["train"],
        config,
    )
    frames = []
    for split_name in config["evaluation"]["evaluate_splits"]:
        mask = _trial_mask(dataset, _split_ids(split, split_name))
        pred_fit = predict_ridge_decoder(flatten_trial_time(latents_by_split[split_name]), decoder)
        pred = pred_fit * target_stats["std"] + target_stats["mean"] if target_stats else pred_fit
        metrics = regression_metrics(flatten_trial_time(targets_3d[mask]), pred, target_names)
        metrics.insert(0, "split", split_name)
        frames.append(metrics)
    return pd.concat(frames, ignore_index=True)


def run_factor_latent_sweep(
    dataset: NeuralDataset,
    split: TrialSplit,
    neuron_mask: NeuronMask,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run a local train-only factor latent diagnostic sweep."""
    features_config = config["features"]
    decoder_config = config["heldout_decoder"]
    input_spikes, input_indices = select_neuron_group(
        dataset.spikes, neuron_mask, features_config.get("input_neuron_group", "heldin")
    )
    target_counts, target_indices = select_neuron_group(
        dataset.spikes, neuron_mask, features_config.get("target_neuron_group", "heldout")
    )
    train_mask = _trial_mask(dataset, split.train)
    min_rate = float(decoder_config["min_rate_hz"])
    max_rate = float(decoder_config["max_rate_hz"])
    reference = _reference_rates(target_counts[train_mask], dataset.bin_size_ms, min_rate, max_rate)
    train_target_rates = safe_clip_rates(
        _rates_from_counts(flatten_trial_time(target_counts[train_mask]), dataset.bin_size_ms),
        min_rate,
        max_rate,
    )

    rows: list[dict[str, Any]] = []
    split_by_run: dict[str, pd.DataFrame] = {}
    neuron_by_run: dict[str, pd.DataFrame] = {}
    behavior_by_run: dict[str, pd.DataFrame] = {}
    latent_by_run: dict[str, pd.DataFrame] = {}
    rates_by_sigma: dict[float, np.ndarray] = {}
    latent_cache: dict[
        tuple[int, float, bool], tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]
    ] = {}
    skipped_invalid = 0

    for run_index, params in enumerate(_sweep_grid(config)):
        latent_dim = int(params["latent_dim"])
        if latent_dim >= input_indices.size:
            skipped_invalid += 1
            continue
        run_id = f"run_{run_index:03d}"
        sigma_ms = float(params["smoothing_sigma_ms"])
        if sigma_ms not in rates_by_sigma:
            rates_by_sigma[sigma_ms] = _rates_for_sigma(
                input_spikes,
                dataset.bin_size_ms,
                sigma_ms,
                bool(features_config.get("convert_to_hz", True)),
            )
        cache_key = (latent_dim, sigma_ms, bool(params["standardize_features"]))
        if cache_key not in latent_cache:
            latent_cache[cache_key] = _fit_latent_cache_entry(
                dataset,
                split,
                _baseline_config(config, params),
                rates_by_sigma[sigma_ms],
                params,
            )
        latents_by_split, behavior_metrics, latent_summary = latent_cache[cache_key]
        behavior_summary = _mean_behavior_by_split(behavior_metrics)

        heldout_decoder = fit_ridge_decoder(
            flatten_trial_time(latents_by_split["train"]),
            train_target_rates,
            alpha=float(params["heldout_decoder_alpha"]),
            fit_intercept=bool(decoder_config.get("fit_intercept", True)),
        )
        split_rows: list[dict[str, Any]] = []
        neuron_rows: list[dict[str, Any]] = []
        for split_name in config["evaluation"]["evaluate_splits"]:
            eval_mask = _trial_mask(dataset, _split_ids(split, split_name))
            counts = target_counts[eval_mask]
            predicted = safe_clip_rates(
                predict_ridge_decoder(
                    flatten_trial_time(latents_by_split[split_name]), heldout_decoder
                ),
                min_rate,
                max_rate,
            ).reshape(counts.shape)
            reference_rates = _broadcast_reference(reference, counts)
            metrics = evaluate_cosmoothing_predictions(
                counts,
                predicted,
                reference_rates,
                dataset.bin_size_ms,
            )
            behavior = behavior_summary.get(
                split_name,
                {
                    "behavior_mean_r2": np.nan,
                    "behavior_mean_mse": np.nan,
                    "behavior_mean_mae": np.nan,
                },
            )
            row = {
                "run_id": run_id,
                "split": split_name,
                "latent_dim": latent_dim,
                "smoothing_sigma_ms": sigma_ms,
                "heldout_decoder_alpha": float(params["heldout_decoder_alpha"]),
                "standardize_features": bool(params["standardize_features"]),
                "n_train_trials": int(len(split.train)),
                "n_eval_trials": int(np.count_nonzero(eval_mask)),
                "n_input_neurons": int(input_indices.size),
                "n_target_neurons": int(target_indices.size),
                **metrics,
                **behavior,
            }
            rows.append(row)
            split_rows.append(row)
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
        split_by_run[run_id] = pd.DataFrame(split_rows, columns=SWEEP_RESULT_COLUMNS)
        neuron_by_run[run_id] = pd.DataFrame(neuron_rows)
        behavior_by_run[run_id] = behavior_metrics
        latent_by_run[run_id] = latent_summary

    sweep_results = pd.DataFrame(rows, columns=SWEEP_RESULT_COLUMNS)
    if sweep_results.empty:
        msg = f"no valid factor latent sweep results; skipped_invalid={skipped_invalid}"
        raise ValueError(msg)

    ranked = rank_sweep_results(
        sweep_results,
        primary_split=str(config["evaluation"]["primary_split"]),
        primary_metric=str(config["evaluation"]["primary_metric"]),
    )
    best_config = _best_config_from_row(ranked.iloc[0])
    best_config.update(
        {
            "input_neuron_group": features_config.get("input_neuron_group", "heldin"),
            "target_neuron_group": features_config.get("target_neuron_group", "heldout"),
            "behavior_decoder_enabled": bool(
                config.get("behavior_decoder", {}).get("enabled", False)
            ),
            "reference": config.get("reference", {}).get("name", "train_mean_rate"),
            "train_only_fit": True,
            "skipped_invalid_configs": skipped_invalid,
        }
    )
    run_id = best_config["run_id"]
    return (
        sweep_results,
        best_config,
        split_by_run[run_id].reset_index(drop=True),
        neuron_by_run[run_id].reset_index(drop=True),
        behavior_by_run[run_id].reset_index(drop=True),
        latent_by_run[run_id].reset_index(drop=True),
    )
