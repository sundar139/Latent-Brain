from __future__ import annotations

import copy
import csv
import itertools
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
import yaml
from torch.utils.data import DataLoader

from latentbrain.eval.cosmoothing import _broadcast_reference, _rates_from_counts, _reference_rates
from latentbrain.eval.decoding import (
    apply_standardization,
    fit_ridge_decoder,
    predict_ridge_decoder,
    regression_metrics,
    standardize_train_apply,
)
from latentbrain.eval.lfads_eval import (
    BEHAVIOR_COLUMNS,
    NEURON_COLUMNS,
    SPLIT_COLUMNS,
    _append_prediction_metrics,
    _apply_optional_standardization,
    _behavior_targets_for_split,
    _checkpoint_epoch,
)
from latentbrain.eval.metrics import safe_clip_rates
from latentbrain.eval.neural_predictions import (
    flatten_batch_time,
    reshape_flat_predictions,
    summarize_factor_activity,
)
from latentbrain.eval.switching_ode_tuning import (
    build_switching_ode_result_row,
    compute_regime_diagnostics,
    rank_switching_ode_results,
    summarize_regime_diagnostics,
    summarize_switching_ode_tuning,
)
from latentbrain.models.switching_ode import SwitchingODE, SwitchingODEConfig
from latentbrain.paths import get_repo_root, resolve_configured_path
from latentbrain.randomness import seed_everything
from latentbrain.torch.checkpoints import load_checkpoint, save_checkpoint
from latentbrain.torch.datasets import create_dataloaders, create_torch_datasets
from latentbrain.torch.device import resolve_device
from latentbrain.torch.losses import gaussian_kl_standard_normal, lfads_cosmoothing_loss
from latentbrain.torch.masking import (
    apply_input_neuron_dropout,
    sample_neuron_dropout_mask,
    summarize_dropout_mask,
)
from latentbrain.torch.rate_initialization import compute_train_mean_rates_hz
from latentbrain.torch.schedules import linear_warmup
from latentbrain.train.neural_sde_tuning import (
    _behavior_mean_r2,
    _ensure_finite,
    _json_default,
    _load_dataset,
    _mean,
    _relative,
    _resolve_processed_path,
    _split_and_mask,
    _trial_mask,
    _validate_cuda,
    _verify_reference_zero,
)

CheckpointPayload = tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
]


def expand_switching_ode_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    values = [list(grid[key]) for key in keys]
    runs = [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]
    for run in runs:
        if int(run["n_regimes"]) < 2:
            msg = "n_regimes must be at least 2"
            raise ValueError(msg)
        if float(run["regime_temperature"]) <= 0.0:
            msg = "regime_temperature must be positive"
            raise ValueError(msg)
        if float(run.get("entropy_regularization", 0.0)) < 0.0:
            msg = "entropy_regularization must be non-negative"
            raise ValueError(msg)
    return runs


def make_switching_ode_run_id(index: int, params: dict[str, Any]) -> str:
    def value(key: str) -> str:
        return str(params[key]).replace(".", "p")

    return (
        f"run_{index:03d}_enc{value('encoder_hidden_dim')}"
        f"_drift{value('drift_hidden_dim')}_lat{value('latent_dim')}"
        f"_fac{value('factor_dim')}_reg{value('n_regimes')}"
        f"_temp{value('regime_temperature')}_drop{value('input_dropout_rate')}"
        f"_hw{value('heldout_loss_weight')}_kl{value('kl_scale')}"
        f"_ent{value('entropy_regularization')}"
    )


def build_switching_ode_train_config(
    base_config: dict[str, Any], run_params: dict[str, Any], run_output_dir: Path
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config.setdefault("data", {})
    config["data"].update(
        {
            "batch_size": int(config["model"]["batch_size"]),
            "max_time_bins": int(config.get("_window_bins", 0)) or None,
        }
    )
    config["model"].update(
        {
            "name": "switching_ode",
            "encoder_hidden_dim": int(run_params["encoder_hidden_dim"]),
            "drift_hidden_dim": int(run_params["drift_hidden_dim"]),
            "latent_dim": int(run_params["latent_dim"]),
            "factor_dim": int(run_params["factor_dim"]),
            "n_regimes": int(run_params["n_regimes"]),
            "regime_hidden_dim": int(run_params["regime_hidden_dim"]),
            "regime_temperature": float(run_params["regime_temperature"]),
            "dropout": float(config["model"].get("model_dropout", 0.0)),
            "diffusion_scale": 0.0,
        }
    )
    config["training"] = {
        "device": str(config["runtime"]["device"]),
        "seed": int(config["splits"]["seed"]),
        "epochs": int(run_params["epochs"]),
        "learning_rate": float(config["model"]["learning_rate"]),
        "weight_decay": float(config["model"]["weight_decay"]),
        "gradient_clip_norm": float(config["model"]["gradient_clip_norm"]),
        "heldin_loss_weight": float(config["model"].get("heldin_loss_weight", 1.0)),
        "heldout_loss_weight": float(run_params["heldout_loss_weight"]),
        "loss_normalization": str(config["model"].get("loss_normalization", "mean")),
        "kl_warmup_epochs": int(run_params["kl_warmup_epochs"]),
        "kl_scale": float(run_params["kl_scale"]),
        "entropy_regularization": float(run_params["entropy_regularization"]),
        "checkpoint_metric": str(config["model"].get("checkpoint_metric", "validation_total_loss")),
        "checkpoint_mode": str(config["model"].get("checkpoint_mode", "min")),
        "save_unified_checkpoints": bool(config["model"].get("save_unified_checkpoints", True)),
        "evaluate_checkpoints_by_unified_metric": bool(
            config["model"].get("evaluate_checkpoints_by_unified_metric", True)
        ),
        "input_dropout": {
            "enabled": float(run_params["input_dropout_rate"]) > 0.0,
            "rate": float(run_params["input_dropout_rate"]),
            "apply_to": ["train"],
            "keep_at_least_one_neuron": True,
            "seed": int(config["splits"]["seed"]),
        },
    }
    config["reporting"] = dict(config["reporting"])
    config["reporting"]["output_dir"] = str(run_output_dir)
    return config


def _first_indices(batch: dict[str, torch.Tensor], key: str, device: torch.device) -> torch.Tensor:
    indices = batch[key]
    if indices.ndim == 2:
        indices = indices[0]
    return indices.to(device=device, dtype=torch.long)


def _input_dropout_settings(training_config: dict[str, Any]) -> dict[str, Any]:
    settings = dict(training_config.get("input_dropout", {}))
    if not bool(settings.get("enabled", False)):
        return {"enabled": False, "rate": 0.0, "keep_at_least_one_neuron": True}
    rate = float(settings.get("rate", 0.0))
    if rate < 0.0 or rate >= 1.0:
        msg = "input_dropout.rate must be in [0, 1)"
        raise ValueError(msg)
    apply_to = {str(value) for value in settings.get("apply_to", ["train"])}
    return {
        "enabled": "train" in apply_to and rate > 0.0,
        "rate": rate,
        "keep_at_least_one_neuron": bool(settings.get("keep_at_least_one_neuron", True)),
        "seed": int(settings.get("seed", training_config.get("seed", 0))),
    }


def _make_dropout_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    generator.manual_seed(seed)
    return generator


def _regime_entropy(regime_probs: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(regime_probs, min=1.0e-12)
    return -(clipped * clipped.log()).sum(dim=-1).mean()


def _active_regime_count(regime_probs: torch.Tensor) -> torch.Tensor:
    occupancy = regime_probs.mean(dim=(0, 1))
    return (occupancy > 0.05).sum().to(dtype=regime_probs.dtype)


def _loss_for_batch(
    model: SwitchingODE,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    bin_size_ms: int,
    z0_beta: float,
    training_config: dict[str, Any],
    input_heldin_spikes: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
    heldin = batch["heldin_spikes"].to(device)
    output = model(heldin if input_heldin_spikes is None else input_heldin_spikes)
    base = lfads_cosmoothing_loss(
        heldin_counts=heldin,
        heldout_counts=batch["heldout_spikes"].to(device),
        all_rates_hz=output["rates_hz"],
        heldin_indices=_first_indices(batch, "heldin_indices", device),
        heldout_indices=_first_indices(batch, "heldout_indices", device),
        posterior_mean=output["z0_mean"],
        posterior_logvar=output["z0_logvar"],
        bin_size_ms=bin_size_ms,
        kl_beta=z0_beta,
        heldin_loss_weight=float(training_config.get("heldin_loss_weight", 1.0)),
        heldout_loss_weight=float(training_config.get("heldout_loss_weight", 1.0)),
        normalization=str(training_config.get("loss_normalization", "mean")),
    )
    entropy = _regime_entropy(output["regime_probs"])
    base["loss"] = (
        base["loss"]
        + heldin.new_tensor(float(training_config.get("entropy_regularization", 0.0))) * entropy
    )
    base["z0_kl_loss"] = gaussian_kl_standard_normal(output["z0_mean"], output["z0_logvar"])
    base["drift_norm"] = output["mixed_drift"].norm(dim=-1).mean()
    base["regime_entropy"] = entropy
    base["active_regime_count"] = _active_regime_count(output["regime_probs"])
    return base, output["rates_hz"], output


def _is_better(value: float, best: float | None, mode: str) -> bool:
    if best is None:
        return True
    if mode == "min":
        return value < best
    if mode == "max":
        return value > best
    msg = "checkpoint_mode must be min or max"
    raise ValueError(msg)


def _write_history(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def _split_metrics(
    model: SwitchingODE,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    bin_size_ms: int,
    z0_beta: float,
    training_config: dict[str, Any],
) -> dict[str, float]:
    model.eval()
    losses: dict[str, list[float]] = {
        "total_loss": [],
        "heldin_reconstruction_loss": [],
        "heldout_prediction_loss": [],
        "z0_kl_loss": [],
        "drift_norm": [],
        "regime_entropy": [],
        "active_regime_count": [],
        "mean_rate_hz": [],
    }
    with torch.no_grad():
        for batch in loader:
            loss, rates, _ = _loss_for_batch(
                model, batch, device, bin_size_ms, z0_beta, training_config
            )
            _ensure_finite("evaluation loss", loss["loss"])
            losses["total_loss"].append(float(loss["loss"].detach().cpu()))
            losses["heldin_reconstruction_loss"].append(
                float(loss["heldin_reconstruction_loss"].detach().cpu())
            )
            losses["heldout_prediction_loss"].append(
                float(loss["heldout_prediction_loss"].detach().cpu())
            )
            losses["z0_kl_loss"].append(float(loss["z0_kl_loss"].detach().cpu()))
            losses["drift_norm"].append(float(loss["drift_norm"].detach().cpu()))
            losses["regime_entropy"].append(float(loss["regime_entropy"].detach().cpu()))
            losses["active_regime_count"].append(float(loss["active_regime_count"].detach().cpu()))
            losses["mean_rate_hz"].append(float(rates.mean().detach().cpu()))
    return {key: _mean(value) for key, value in losses.items()}


def _train_switching_ode(
    model: SwitchingODE,
    dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> list[dict[str, float]]:
    training_config = dict(config["training"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    model.to(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    input_dropout = _input_dropout_settings(training_config)
    dropout_generator = _make_dropout_generator(device, int(input_dropout.get("seed", 0)))
    history: list[dict[str, float]] = []
    best_metric: float | None = None
    for epoch in range(int(training_config["epochs"])):
        model.train()
        kl_beta = linear_warmup(epoch, int(training_config["kl_warmup_epochs"])) * float(
            training_config.get("kl_scale", 1.0)
        )
        train_losses: list[float] = []
        dropout_fractions: list[float] = []
        for batch in dataloaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            model_input = None
            if bool(input_dropout["enabled"]):
                heldin = batch["heldin_spikes"].to(device)
                mask = sample_neuron_dropout_mask(
                    heldin.shape[0],
                    heldin.shape[2],
                    float(input_dropout["rate"]),
                    heldin.device,
                    generator=dropout_generator,
                    keep_at_least_one=bool(input_dropout["keep_at_least_one_neuron"]),
                )
                model_input = apply_input_neuron_dropout(heldin, mask)
                dropout_fractions.append(float(summarize_dropout_mask(mask)["dropout_fraction"]))
            loss, _, _ = _loss_for_batch(
                model,
                batch,
                device,
                int(config["dataset"]["bin_size_ms"]),
                kl_beta,
                training_config,
                model_input,
            )
            _ensure_finite("training loss", loss["loss"])
            torch.autograd.backward(loss["loss"])
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training_config["gradient_clip_norm"])
            )
            optimizer.step()
            train_losses.append(float(loss["loss"].detach().cpu()))
        row: dict[str, float] = {
            "epoch": float(epoch),
            "kl_beta": float(kl_beta),
            "training_batch_loss": _mean(train_losses),
            "configured_input_dropout_rate": float(input_dropout["rate"]),
            "realized_input_dropout_fraction": _mean(dropout_fractions)
            if dropout_fractions
            else 0.0,
        }
        for split_name in config["evaluation"]["evaluate_splits"]:
            metrics = _split_metrics(
                model,
                dataloaders[str(split_name)],
                device,
                int(config["dataset"]["bin_size_ms"]),
                kl_beta,
                training_config,
            )
            row[f"{split_name}_total_loss"] = metrics["total_loss"]
            row[f"{split_name}_heldin_reconstruction_loss"] = metrics["heldin_reconstruction_loss"]
            row[f"{split_name}_heldout_prediction_loss"] = metrics["heldout_prediction_loss"]
            row[f"{split_name}_z0_kl_loss"] = metrics["z0_kl_loss"]
            row[f"{split_name}_drift_norm"] = metrics["drift_norm"]
            row[f"{split_name}_regime_entropy"] = metrics["regime_entropy"]
            row[f"{split_name}_active_regime_count"] = metrics["active_regime_count"]
            row[f"{split_name}_mean_rate_hz"] = metrics["mean_rate_hz"]
        row["train_total_loss"] = row.get("train_total_loss", row["training_batch_loss"])
        row["z0_kl_loss"] = row.get("validation_z0_kl_loss", float("nan"))
        row["drift_norm"] = row.get("validation_drift_norm", float("nan"))
        row["regime_entropy"] = row.get("validation_regime_entropy", float("nan"))
        row["active_regime_count"] = row.get("validation_active_regime_count", float("nan"))
        row["mean_predicted_rate"] = row.get("validation_mean_rate_hz", float("nan"))
        history.append(row)
        save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, epoch, row, config)
        metric = float(row[str(training_config.get("checkpoint_metric", "validation_total_loss"))])
        if _is_better(metric, best_metric, str(training_config.get("checkpoint_mode", "min"))):
            best_metric = metric
            save_checkpoint(
                checkpoint_dir / "best_validation.pt", model, optimizer, epoch, row, config
            )
        _write_history(output_dir / "metrics_history.csv", history)
    final = history[-1] | {"best_metric": best_metric}
    (output_dir / "final_metrics.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return history


def _make_model(input_dim: int, output_dim: int, config: dict[str, Any]) -> SwitchingODE:
    model_config = dict(config["model"])
    return SwitchingODE(
        SwitchingODEConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            encoder_hidden_dim=int(model_config["encoder_hidden_dim"]),
            drift_hidden_dim=int(model_config["drift_hidden_dim"]),
            latent_dim=int(model_config["latent_dim"]),
            factor_dim=int(model_config["factor_dim"]),
            n_regimes=int(model_config["n_regimes"]),
            regime_hidden_dim=int(model_config["regime_hidden_dim"]),
            regime_temperature=float(model_config["regime_temperature"]),
            dropout=float(model_config.get("dropout", 0.0)),
            min_rate_hz=float(model_config["min_rate_hz"]),
            max_rate_hz=float(model_config["max_rate_hz"]),
            dt_seconds=float(model_config["dt_seconds"]),
            diffusion_scale=float(model_config["diffusion_scale"]),
        )
    )


def _load_switching_from_checkpoint(
    checkpoint_path: Path,
    input_dim: int,
    output_dim: int,
    config: dict[str, Any],
    device: torch.device,
) -> SwitchingODE:
    model = _make_model(input_dim, output_dim, config).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    return model


def _extract_switching(
    model: SwitchingODE,
    dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
    device: torch.device,
) -> dict[str, dict[str, np.ndarray]]:
    extracted: dict[str, dict[str, np.ndarray]] = {}
    model.eval()
    with torch.no_grad():
        for split_name, loader in dataloaders.items():
            chunks: dict[str, list[np.ndarray]] = {
                "factors": [],
                "model_rates_hz": [],
                "latents": [],
                "mixed_drift": [],
                "regime_probs": [],
                "regime_logits": [],
                "regime_drifts": [],
                "heldout_spikes": [],
                "trial_ids": [],
            }
            for batch in loader:
                output = model(batch["heldin_spikes"].to(device), deterministic=True)
                mapping = {"rates_hz": "model_rates_hz", "drift": "mixed_drift"}
                for key in (
                    "factors",
                    "rates_hz",
                    "latents",
                    "drift",
                    "regime_probs",
                    "regime_logits",
                    "regime_drifts",
                ):
                    chunks[mapping.get(key, key)].append(output[key].detach().cpu().numpy())
                chunks["heldout_spikes"].append(batch["heldout_spikes"].detach().cpu().numpy())
                chunks["trial_ids"].append(batch["trial_id"].detach().cpu().numpy())
            extracted[split_name] = {
                key: np.concatenate(value, axis=0) for key, value in chunks.items()
            }
    return extracted


def _run_switching_evaluation(
    dataset: Any,
    split: Any,
    mask: Any,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
]:
    input_dim = int(np.count_nonzero(mask.heldin))
    output_dim = int(dataset.spikes.shape[2])
    target_indices = np.flatnonzero(mask.heldout)
    model = _load_switching_from_checkpoint(
        Path(str(config["model"]["checkpoint_path"])), input_dim, output_dim, config, device
    )
    dataloaders = create_dataloaders(
        create_torch_datasets(dataset, split, mask, int(config["data"]["max_time_bins"])),
        batch_size=int(config["data"]["batch_size"]),
        num_workers=0,
        drop_last=False,
        seed=int(config["splits"]["seed"]),
    )
    extracted = _extract_switching(model, dataloaders, device)
    min_rate = float(config["model"]["min_rate_hz"])
    max_rate = float(config["model"]["max_rate_hz"])
    reference = _reference_rates(
        extracted["train"]["heldout_spikes"], dataset.bin_size_ms, min_rate, max_rate
    )
    split_rows: list[dict[str, Any]] = []
    neuron_rows: list[dict[str, Any]] = []
    factor_tables: list[pd.DataFrame] = []
    latent_rows: list[dict[str, Any]] = []
    regime_tables: list[pd.DataFrame] = []
    train_factors_raw = flatten_batch_time(extracted["train"]["factors"])
    train_counts = flatten_batch_time(extracted["train"]["heldout_spikes"])
    train_targets = safe_clip_rates(
        _rates_from_counts(train_counts, dataset.bin_size_ms), min_rate, max_rate
    )
    train_factors, _, factor_stats = _apply_optional_standardization(
        train_factors_raw, train_factors_raw, True
    )
    heldout_decoder = fit_ridge_decoder(
        train_factors, train_targets, alpha=1000.0, fit_intercept=True
    )
    for split_name in config["evaluation"]["evaluate_splits"]:
        split_key = str(split_name)
        factors = extracted[split_key]["factors"]
        counts = extracted[split_key]["heldout_spikes"]
        reference_rates = _broadcast_reference(reference, counts)
        direct_rates = safe_clip_rates(
            extracted[split_key]["model_rates_hz"][:, :, target_indices], min_rate, max_rate
        )
        _append_prediction_metrics(
            split_rows,
            neuron_rows,
            split_key,
            "direct_model",
            factors,
            counts,
            direct_rates,
            reference_rates,
            target_indices,
            dataset.bin_size_ms,
        )
        flat_factors = apply_standardization(flatten_batch_time(factors), factor_stats)
        predicted = reshape_flat_predictions(
            safe_clip_rates(
                predict_ridge_decoder(flat_factors, heldout_decoder), min_rate, max_rate
            ),
            counts.shape[0],
            counts.shape[1],
            counts.shape[2],
        )
        _append_prediction_metrics(
            split_rows,
            neuron_rows,
            split_key,
            "factor_decoder",
            factors,
            counts,
            predicted,
            reference_rates,
            target_indices,
            dataset.bin_size_ms,
        )
        factor_tables.append(summarize_factor_activity(factors, split_key))
        regime_tables.append(
            compute_regime_diagnostics(extracted[split_key]["regime_probs"], split_key)
        )
        latent_rows.append(
            {
                "split": split_key,
                "drift_norm": float(
                    np.linalg.norm(extracted[split_key]["mixed_drift"], axis=-1).mean()
                ),
                "latent_std": float(extracted[split_key]["latents"].std()),
                **summarize_regime_diagnostics(regime_tables[-1], split_key),
            }
        )
    behavior_metrics = pd.DataFrame(columns=BEHAVIOR_COLUMNS)
    behavior_decoder = {
        "coefficients": np.empty((model.config.factor_dim, 0)),
        "intercept": np.empty(0),
    }
    behavior_target_names: list[str] = []
    if (
        bool(config["evaluation"].get("behavior_decoder_enabled", True))
        and dataset.behavior is not None
    ):
        train_behavior_factors, behavior_feature_stats = standardize_train_apply(
            train_factors_raw, train_factors_raw
        )
        behavior_config = {
            "target_prefixes": ["hand_pos", "cursor_pos"],
            "derive_velocity": True,
            "velocity_method": "central_difference",
        }
        train_behavior_targets, behavior_target_names = _behavior_targets_for_split(
            dataset,
            extracted["train"]["trial_ids"],
            extracted["train"]["factors"].shape[1],
            behavior_config,
        )
        train_behavior_fit, behavior_target_stats = standardize_train_apply(
            flatten_batch_time(train_behavior_targets), flatten_batch_time(train_behavior_targets)
        )
        behavior_decoder = fit_ridge_decoder(
            train_behavior_factors, train_behavior_fit, alpha=100.0, fit_intercept=True
        )
        behavior_frames = []
        for split_name in config["evaluation"]["evaluate_splits"]:
            split_key = str(split_name)
            features = apply_standardization(
                flatten_batch_time(extracted[split_key]["factors"]), behavior_feature_stats
            )
            pred_fit = predict_ridge_decoder(features, behavior_decoder)
            pred = pred_fit * behavior_target_stats["std"] + behavior_target_stats["mean"]
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
    metadata = {
        "model_name": "switching_ode",
        "checkpoint_path": str(config["model"]["checkpoint_path"]),
        "checkpoint_epoch": _checkpoint_epoch(Path(str(config["model"]["checkpoint_path"]))),
        "factor_dim": model.config.factor_dim,
        "latent_dim": model.config.latent_dim,
        "n_regimes": model.config.n_regimes,
        "target_neuron_indices": target_indices.tolist(),
        "heldout_decoder_coefficients": heldout_decoder["coefficients"],
        "behavior_decoder_coefficients": behavior_decoder["coefficients"],
        "behavior_target_names": behavior_target_names,
        "train_only_fit": True,
    }
    return (
        pd.DataFrame(split_rows, columns=SPLIT_COLUMNS),
        pd.DataFrame(neuron_rows, columns=NEURON_COLUMNS),
        behavior_metrics,
        pd.concat(factor_tables, ignore_index=True),
        metadata,
        pd.DataFrame(latent_rows),
        pd.concat(regime_tables, ignore_index=True),
    )


def _checkpoint_metric_row(
    checkpoint_path: Path,
    source: str,
    run_config: dict[str, Any],
    dataset: Any,
    split: Any,
    mask: Any,
    device: torch.device,
) -> CheckpointPayload:
    eval_config = copy.deepcopy(run_config)
    eval_config["model"]["checkpoint_path"] = str(checkpoint_path)
    split_metrics, neuron_metrics, behavior_metrics, factor_summary, metadata, latent, regime = (
        _run_switching_evaluation(dataset, split, mask, eval_config, device)
    )
    validation = split_metrics[
        (split_metrics["split"] == "validation")
        & (split_metrics["prediction_source"] == "direct_model")
    ].iloc[0]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metrics = dict(checkpoint.get("metrics", {}))
    row = {
        "checkpoint_source": source,
        "epoch": int(checkpoint.get("epoch", -1)),
        "validation_total_loss": float(metrics.get("validation_total_loss", float("nan"))),
        "validation_unified_bits_per_spike": float(validation["bits_per_spike"]),
        "validation_poisson_nll": float(validation["poisson_nll"]),
        "checkpoint_path": str(checkpoint_path),
        "selected_by_loss": source == "best_validation",
        "selected_by_unified": False,
    }
    return (
        row,
        split_metrics,
        neuron_metrics,
        behavior_metrics,
        factor_summary,
        metadata,
        latent,
        regime,
    )


def _write_final_evaluation(
    output_dir: Path,
    run_config: dict[str, Any],
    split_metrics: pd.DataFrame,
    neuron_metrics: pd.DataFrame,
    behavior_metrics: pd.DataFrame,
    factor_summary: pd.DataFrame,
    metadata: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    from latentbrain.eval.reporting import write_lfads_gru_evaluation_outputs

    validation = split_metrics[
        (split_metrics["split"] == "validation")
        & (split_metrics["prediction_source"] == "direct_model")
    ].iloc[0]
    factor_rows = split_metrics[
        (split_metrics["split"] == "validation")
        & (split_metrics["prediction_source"] == "factor_decoder")
    ]
    factor_bits = None if factor_rows.empty else float(factor_rows.iloc[0]["bits_per_spike"])
    write_lfads_gru_evaluation_outputs(
        output_dir / "evaluation",
        {
            "dataset_name": run_config["dataset"]["name"],
            "dataset_hash": run_config["dataset"].get("expected_hash"),
            "checkpoint_path": _relative(checkpoint_path, get_repo_root()),
            "checkpoint_epoch": metadata.get("checkpoint_epoch"),
            "model_name": "switching_ode",
            "factor_dim": int(metadata["factor_dim"]),
            "latent_dim": int(metadata["latent_dim"]),
            "primary_split": "validation",
            "primary_bits_per_spike": float(validation["bits_per_spike"]),
            "primary_poisson_nll": float(validation["poisson_nll"]),
            "primary_behavior_mean_r2": _behavior_mean_r2(
                output_dir / "evaluation" / "behavior_metrics.csv"
            ),
            "primary_prediction_source": "direct_model",
            "direct_model_available": True,
            "factor_decoder_evaluated": True,
            "direct_model_validation_bits_per_spike": float(validation["bits_per_spike"]),
            "factor_decoder_validation_bits_per_spike": factor_bits,
            "heldout_decoder_alpha": 1000.0,
            "behavior_decoder_enabled": bool(
                run_config["evaluation"].get("behavior_decoder_enabled", True)
            ),
            "behavior_decoder_alpha": 100.0,
            "fit_policy": "train trials only",
            "official_benchmark_claim": False,
            "full_lfads_claim": False,
        },
        split_metrics,
        neuron_metrics,
        behavior_metrics,
        factor_summary,
        metadata,
    )


def _unified_scores(run_dir: Path) -> pd.DataFrame:
    scores = pd.read_csv(run_dir / "evaluation" / "split_metrics.csv")
    scores.insert(0, "reference_name", "train_heldout_mean_rate")
    scores.insert(0, "valid_model", True)
    scores.to_csv(run_dir / "unified_scores.csv", index=False)
    return scores


def _selected_checkpoint_metrics(run_dir: Path, device: str) -> dict[str, Any]:
    checkpoint_scores = pd.read_csv(run_dir / "checkpoint_scores.csv")
    selected = checkpoint_scores[checkpoint_scores["selected_by_unified"].astype(bool)].iloc[0]
    checkpoint = torch.load(
        run_dir / "checkpoints" / "best_unified.pt", map_location="cpu", weights_only=False
    )
    metrics = dict(checkpoint.get("metrics", {}))
    return {
        "status": "completed",
        "device": device,
        "validation_behavior_mean_r2": _behavior_mean_r2(
            run_dir / "evaluation" / "behavior_metrics.csv"
        ),
        "train_total_loss": metrics.get("train_total_loss", metrics.get("training_batch_loss")),
        "validation_total_loss": metrics.get("validation_total_loss"),
        "validation_heldout_prediction_loss": metrics.get("validation_heldout_prediction_loss"),
        "z0_kl_loss": metrics.get("z0_kl_loss"),
        "drift_norm": metrics.get("drift_norm"),
        "best_checkpoint_source": selected.get("checkpoint_source", "best_unified"),
    }


def _train_and_evaluate_run(
    run_config: dict[str, Any], run_index: int, dataset: Any, split: Any, mask: Any
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_everything(int(run_config["training"]["seed"]) + run_index)
    dataloaders = create_dataloaders(
        create_torch_datasets(dataset, split, mask, int(run_config["data"]["max_time_bins"])),
        batch_size=int(run_config["data"]["batch_size"]),
        num_workers=0,
        drop_last=False,
        seed=int(run_config["training"]["seed"]) + run_index,
    )
    input_dim = int(mask.heldin.sum())
    output_dim = int(dataset.spikes.shape[2])
    model = _make_model(input_dim, output_dim, run_config)
    train_spikes = dataset.spikes[_trial_mask(dataset, split.train)]
    mean_rates = compute_train_mean_rates_hz(
        train_spikes,
        dataset.bin_size_ms,
        float(run_config["model"]["min_rate_hz"]),
        float(run_config["model"]["max_rate_hz"]),
    )
    model.initialize_output_bias_from_rates(torch.as_tensor(mean_rates, dtype=torch.float32))
    device = resolve_device(str(run_config["training"]["device"]))
    output_dir = Path(str(run_config["reporting"]["output_dir"]))
    snapshot = copy.deepcopy(run_config)
    snapshot["dataset"]["bin_size_ms"] = dataset.bin_size_ms
    snapshot["model"]["input_dim"] = input_dim
    snapshot["model"]["resolved_output_dim"] = output_dim
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8"
    )
    _train_switching_ode(model, dataloaders, snapshot, output_dir, device)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_rows: list[dict[str, Any]] = []
    checkpoint_payloads = []
    for source, checkpoint_path in (
        ("best_validation", checkpoint_dir / "best_validation.pt"),
        ("latest", checkpoint_dir / "latest.pt"),
    ):
        payload = _checkpoint_metric_row(
            checkpoint_path, source, snapshot, dataset, split, mask, device
        )
        checkpoint_rows.append(payload[0])
        checkpoint_payloads.append(payload)
    selected_index = max(
        range(len(checkpoint_rows)),
        key=lambda index: (
            float(checkpoint_rows[index]["validation_unified_bits_per_spike"]),
            -float(checkpoint_rows[index]["validation_poisson_nll"]),
        ),
    )
    checkpoint_rows[selected_index]["selected_by_unified"] = True
    selected = checkpoint_payloads[selected_index]
    selected_row = selected[0]
    best_unified = checkpoint_dir / "best_unified.pt"
    shutil.copy2(Path(str(selected_row["checkpoint_path"])), best_unified)
    checkpoint_scores = pd.DataFrame(checkpoint_rows)
    checkpoint_scores.to_csv(output_dir / "checkpoint_scores.csv", index=False)
    _write_final_evaluation(
        output_dir,
        snapshot,
        selected[1],
        selected[2],
        selected[3],
        selected[4],
        selected[5],
        best_unified,
    )
    selected[6].to_csv(output_dir / "latent_diagnostics.csv", index=False)
    selected[7].to_csv(output_dir / "regime_diagnostics.csv", index=False)
    return checkpoint_scores, selected[7]


def run_switching_ode_tuning(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    _resolve_processed_path(config)
    gpu_name = _validate_cuda(config)
    if float(config["model"].get("diffusion_scale", 0.0)) != 0.0:
        msg = "switching neural-ODE-style tuning requires diffusion_scale == 0.0"
        raise ValueError(msg)
    dataset, dataset_hash, window_bins = _load_dataset(config)
    split, mask = _split_and_mask(dataset, config)
    reference_zero = _verify_reference_zero(dataset, split, mask, config)
    repo_root = get_repo_root()
    output_dir = resolve_configured_path(str(config["reporting"]["output_dir"]), repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    refs = dict(config["references"])
    refs["train_mean_validation_bits_per_spike"] = reference_zero
    grid = expand_switching_ode_grid(dict(config["grid"]))[: int(config["search"]["max_runs"])]
    rows: list[dict[str, Any]] = []
    all_checkpoint_scores: list[pd.DataFrame] = []
    all_regime_diagnostics: list[pd.DataFrame] = []
    base = copy.deepcopy(config)
    base["_window_bins"] = window_bins
    for run_index, params in enumerate(grid):
        run_id = make_switching_ode_run_id(run_index, params)
        run_dir = output_dir / "runs" / run_id
        run_config = build_switching_ode_train_config(base, params, run_dir)
        checkpoint_scores, regime_diagnostics = _train_and_evaluate_run(
            run_config, run_index, dataset, split, mask
        )
        checkpoint_scores.insert(0, "run_id", run_id)
        regime_diagnostics.insert(0, "run_id", run_id)
        all_checkpoint_scores.append(checkpoint_scores)
        all_regime_diagnostics.append(regime_diagnostics)
        scores = _unified_scores(run_dir)
        rows.append(
            build_switching_ode_result_row(
                run_id,
                run_index,
                params,
                scores,
                _selected_checkpoint_metrics(run_dir, str(config["runtime"]["device"])),
                checkpoint_scores,
                summarize_regime_diagnostics(regime_diagnostics),
                refs,
                run_dir,
            )
        )
    results = pd.DataFrame(rows)
    leaderboard = rank_switching_ode_results(results)
    summary = summarize_switching_ode_tuning(results, refs)
    summary.update(
        {
            "dataset_name": config["dataset"]["name"],
            "dataset_hash": dataset_hash,
            "cuda_device": gpu_name,
            "bin_size_ms": int(config["binning"]["target_bin_size_ms"]),
            "window_seconds": float(config["window"]["duration_seconds"]),
            "window_bins": window_bins,
            "output_dir": str(output_dir),
        }
    )
    if all_checkpoint_scores:
        pd.concat(all_checkpoint_scores, ignore_index=True).to_csv(
            output_dir / "checkpoint_selection.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(output_dir / "checkpoint_selection.csv", index=False)
    if all_regime_diagnostics:
        pd.concat(all_regime_diagnostics, ignore_index=True).to_csv(
            output_dir / "regime_diagnostics.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(output_dir / "regime_diagnostics.csv", index=False)
    best_config: dict[str, Any] = {}
    if not leaderboard.empty:
        best = leaderboard.iloc[0]
        best_result = results.loc[results["run_id"] == best["run_id"]].iloc[0]
        best_params = {
            key: best_result[key]
            for key in config["grid"]
            if key in best_result and pd.notna(best_result[key])
        }
        best_config = build_switching_ode_train_config(
            base, best_params, Path(str(best_result["output_dir"]))
        )
    (output_dir / "best_config.yaml").write_text(
        yaml.safe_dump(json.loads(json.dumps(best_config, default=_json_default)), sort_keys=False),
        encoding="utf-8",
    )
    return results, summary
