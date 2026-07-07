from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
from torch.utils.data import DataLoader

from latentbrain.data.schemas import NeuralDataset, NeuronMask, TrialSplit
from latentbrain.eval.behavior import derive_velocity_targets, select_behavior_targets
from latentbrain.eval.cosmoothing import (
    _broadcast_reference,
    _rates_from_counts,
    _reference_rates,
    evaluate_cosmoothing_predictions,
)
from latentbrain.eval.decoding import (
    apply_standardization,
    fit_ridge_decoder,
    predict_ridge_decoder,
    regression_metrics,
    standardize_train_apply,
)
from latentbrain.eval.metrics import poisson_log_likelihood, safe_clip_rates
from latentbrain.eval.neural_predictions import (
    flatten_batch_time,
    reshape_flat_predictions,
    summarize_factor_activity,
)
from latentbrain.models.lfads_gru import LFADSGRU, LFADSGRUConfig
from latentbrain.torch.checkpoints import load_checkpoint
from latentbrain.torch.datasets import create_dataloaders, create_torch_datasets

SPLIT_COLUMNS = [
    "split",
    "n_trials",
    "n_time_bins",
    "factor_dim",
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


def load_lfads_gru_from_checkpoint(
    checkpoint_path: Path,
    input_dim: int,
    output_dim: int,
    config: dict[str, Any],
    device: torch.device,
) -> LFADSGRU:
    """Load a checkpointed LFADS-style GRU model for evaluation."""
    model_config = dict(config["model"])
    model = LFADSGRU(
        LFADSGRUConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            encoder_hidden_dim=int(model_config["encoder_hidden_dim"]),
            generator_hidden_dim=int(model_config["generator_hidden_dim"]),
            latent_dim=int(model_config["latent_dim"]),
            factor_dim=int(model_config["factor_dim"]),
            dropout=float(model_config.get("dropout", 0.0)),
            min_rate_hz=float(model_config["min_rate_hz"]),
            max_rate_hz=float(model_config["max_rate_hz"]),
        )
    ).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    return model


def extract_lfads_factors(
    model: LFADSGRU,
    dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
    device: torch.device,
) -> dict[str, dict[str, np.ndarray]]:
    """Run held-in spikes through the model and collect factors in dataloader order."""
    model.eval()
    extracted: dict[str, dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for split_name, loader in dataloaders.items():
            chunks: dict[str, list[np.ndarray]] = {
                "factors": [],
                "heldin_rates_hz": [],
                "heldin_spikes": [],
                "heldout_spikes": [],
                "trial_ids": [],
            }
            for batch in loader:
                heldin = batch["heldin_spikes"].to(device)
                output = model(heldin)
                chunks["factors"].append(output["factors"].detach().cpu().numpy())
                chunks["heldin_rates_hz"].append(output["rates_hz"].detach().cpu().numpy())
                chunks["heldin_spikes"].append(batch["heldin_spikes"].detach().cpu().numpy())
                chunks["heldout_spikes"].append(batch["heldout_spikes"].detach().cpu().numpy())
                chunks["trial_ids"].append(batch["trial_id"].detach().cpu().numpy())
            extracted[split_name] = {
                key: np.concatenate(value, axis=0) for key, value in chunks.items()
            }
    return extracted


def _apply_optional_standardization(
    train: np.ndarray,
    values: np.ndarray,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if not enabled:
        return train, values, {}
    train_fit, stats = standardize_train_apply(train, train)
    return train_fit, apply_standardization(values, stats), stats


def _behavior_targets_for_split(
    dataset: NeuralDataset,
    trial_ids: np.ndarray,
    time_bins: int,
    behavior_config: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    if dataset.behavior is None or dataset.behavior_names is None:
        msg = "behavior decoder is enabled but dataset has no behavior"
        raise ValueError(msg)
    positions, position_names = select_behavior_targets(
        dataset.behavior[:, :time_bins, :],
        dataset.behavior_names,
        list(behavior_config["target_prefixes"]),
    )
    targets, target_names = (
        derive_velocity_targets(
            positions,
            position_names,
            dataset.bin_size_ms,
            method=str(behavior_config.get("velocity_method", "central_difference")),
        )
        if bool(behavior_config.get("derive_velocity", True))
        else (positions, position_names)
    )
    index_by_trial = {int(trial_id): index for index, trial_id in enumerate(dataset.trial_ids)}
    indices = [index_by_trial[int(trial_id)] for trial_id in trial_ids]
    return targets[indices], target_names


def _checkpoint_epoch(path: Path) -> int | None:
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    epoch = checkpoint.get("epoch")
    return None if epoch is None else int(epoch)


def _safe_neuron_rows(
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
        model_ll = poisson_log_likelihood(counts, predicted, bin_size_ms)
        reference_ll = poisson_log_likelihood(counts, reference, bin_size_ms)
        spike_count = float(np.sum(counts))
        bits = (
            float((model_ll - reference_ll) / (np.log(2.0) * spike_count))
            if spike_count > 0.0
            else float("nan")
        )
        rows.append(
            {
                "split": split_name,
                "target_neuron_index": int(neuron_index),
                "target_neuron_rank": int(rank),
                "spike_count": spike_count,
                "poisson_nll": -model_ll,
                "poisson_log_likelihood": model_ll,
                "reference_log_likelihood": reference_ll,
                "bits_per_spike": bits,
                "mse_rate_hz": float(np.mean((predicted - target_rate) ** 2)),
                "mae_rate_hz": float(np.mean(np.abs(predicted - target_rate))),
                "train_reference_rate_hz": float(reference[0, 0, 0]),
                "mean_predicted_rate_hz": float(np.mean(predicted)),
            }
        )
    return rows


def run_lfads_gru_evaluation(
    dataset: NeuralDataset,
    split: TrialSplit,
    neuron_mask: NeuronMask,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate held-out prediction from checkpointed LFADS-style factors."""
    data_config = dict(config["data"])
    decoder_config = dict(config["heldout_decoder"])
    behavior_config = dict(config.get("behavior_decoder", {}))
    eval_config = dict(config["evaluation"])
    target_indices = np.flatnonzero(neuron_mask.heldout)
    input_dim = int(np.count_nonzero(neuron_mask.heldin))
    checkpoint_path = Path(str(config["model"]["checkpoint_path"]))
    model = load_lfads_gru_from_checkpoint(checkpoint_path, input_dim, input_dim, config, device)
    dataloaders = create_dataloaders(
        create_torch_datasets(dataset, split, neuron_mask, data_config.get("max_time_bins")),
        batch_size=int(data_config["batch_size"]),
        num_workers=int(data_config.get("num_workers", 0)),
        drop_last=bool(data_config.get("drop_last", False)),
        seed=int(config.get("splits", {}).get("seed", 0)),
    )
    extracted = extract_lfads_factors(model, dataloaders, device)

    min_rate = float(decoder_config["min_rate_hz"])
    max_rate = float(decoder_config["max_rate_hz"])
    train_factors_raw = flatten_batch_time(extracted["train"]["factors"])
    train_counts = flatten_batch_time(extracted["train"]["heldout_spikes"])
    train_targets = safe_clip_rates(
        _rates_from_counts(train_counts, dataset.bin_size_ms), min_rate, max_rate
    )
    train_factors, _, factor_stats = _apply_optional_standardization(
        train_factors_raw,
        train_factors_raw,
        bool(decoder_config.get("standardize_factors", True)),
    )
    heldout_decoder = fit_ridge_decoder(
        train_factors,
        train_targets,
        alpha=float(decoder_config["alpha"]),
        fit_intercept=bool(decoder_config.get("fit_intercept", True)),
    )
    reference = _reference_rates(
        extracted["train"]["heldout_spikes"], dataset.bin_size_ms, min_rate, max_rate
    )

    split_rows: list[dict[str, Any]] = []
    neuron_rows: list[dict[str, Any]] = []
    factor_tables = []
    for split_name in eval_config["evaluate_splits"]:
        factors = extracted[str(split_name)]["factors"]
        counts = extracted[str(split_name)]["heldout_spikes"]
        flat_factors = flatten_batch_time(factors)
        if factor_stats:
            flat_factors = apply_standardization(flat_factors, factor_stats)
        predicted_flat = safe_clip_rates(
            predict_ridge_decoder(flat_factors, heldout_decoder), min_rate, max_rate
        )
        predicted = reshape_flat_predictions(
            predicted_flat, counts.shape[0], counts.shape[1], counts.shape[2]
        )
        reference_rates = _broadcast_reference(reference, counts)
        metrics = evaluate_cosmoothing_predictions(
            counts, predicted, reference_rates, dataset.bin_size_ms
        )
        split_rows.append(
            {
                "split": split_name,
                "n_trials": int(counts.shape[0]),
                "n_time_bins": int(counts.shape[1]),
                "factor_dim": int(factors.shape[2]),
                "n_target_neurons": int(counts.shape[2]),
                **metrics,
            }
        )
        neuron_rows.extend(
            _safe_neuron_rows(
                split_name, counts, predicted, reference_rates, target_indices, dataset.bin_size_ms
            )
        )
        factor_tables.append(summarize_factor_activity(factors, str(split_name)))

    if bool(behavior_config.get("enabled", False)):
        behavior_feature_stats: dict[str, np.ndarray] = {}
        train_behavior_factors = train_factors_raw
        if bool(behavior_config.get("standardize_factors", True)):
            train_behavior_factors, behavior_feature_stats = standardize_train_apply(
                train_factors_raw, train_factors_raw
            )
        train_behavior_targets, behavior_target_names = _behavior_targets_for_split(
            dataset,
            extracted["train"]["trial_ids"],
            extracted["train"]["factors"].shape[1],
            behavior_config,
        )
        train_behavior_flat = flatten_batch_time(train_behavior_targets)
        if bool(behavior_config.get("standardize_targets", True)):
            train_behavior_fit, behavior_target_stats = standardize_train_apply(
                train_behavior_flat, train_behavior_flat
            )
        else:
            train_behavior_fit = train_behavior_flat
            behavior_target_stats = {}
        behavior_decoder = fit_ridge_decoder(
            train_behavior_factors,
            train_behavior_fit,
            alpha=float(behavior_config["alpha"]),
            fit_intercept=bool(behavior_config.get("fit_intercept", True)),
        )
        behavior_frames = []
        for split_name in eval_config["evaluate_splits"]:
            split_key = str(split_name)
            features = flatten_batch_time(extracted[split_key]["factors"])
            if behavior_feature_stats:
                features = apply_standardization(features, behavior_feature_stats)
            pred_fit = predict_ridge_decoder(features, behavior_decoder)
            pred = (
                pred_fit * behavior_target_stats["std"] + behavior_target_stats["mean"]
                if behavior_target_stats
                else pred_fit
            )
            targets, _ = _behavior_targets_for_split(
                dataset,
                extracted[split_key]["trial_ids"],
                extracted[split_key]["factors"].shape[1],
                behavior_config,
            )
            metrics = regression_metrics(flatten_batch_time(targets), pred, behavior_target_names)
            metrics.insert(0, "split", split_key)
            behavior_frames.append(metrics)
        behavior_metrics = pd.concat(behavior_frames, ignore_index=True)[BEHAVIOR_COLUMNS]
    else:
        behavior_decoder = {
            "coefficients": np.empty((model.config.factor_dim, 0)),
            "intercept": np.empty(0),
        }
        behavior_target_names = []
        behavior_metrics = pd.DataFrame(columns=BEHAVIOR_COLUMNS)

    metadata = {
        "model_name": "lfads_gru",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": _checkpoint_epoch(checkpoint_path),
        "factor_dim": model.config.factor_dim,
        "latent_dim": model.config.latent_dim,
        "input_neuron_indices": np.flatnonzero(neuron_mask.heldin).tolist(),
        "target_neuron_indices": target_indices.tolist(),
        "heldout_decoder_coefficients": heldout_decoder["coefficients"],
        "heldout_decoder_intercept": heldout_decoder["intercept"],
        "reference_rates_hz": reference,
        "behavior_decoder_enabled": bool(behavior_config.get("enabled", False)),
        "behavior_target_names": behavior_target_names,
        "behavior_decoder_coefficients": behavior_decoder["coefficients"],
        "behavior_decoder_intercept": behavior_decoder["intercept"],
        "factor_stats": factor_stats,
        "train_only_fit": True,
    }
    return (
        pd.DataFrame(split_rows, columns=SPLIT_COLUMNS),
        pd.DataFrame(neuron_rows, columns=NEURON_COLUMNS),
        behavior_metrics,
        pd.concat(factor_tables, ignore_index=True),
        metadata,
    )
