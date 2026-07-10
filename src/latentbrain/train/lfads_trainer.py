from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from latentbrain.models.lfads_gru import LFADSGRU
from latentbrain.torch.checkpoints import save_checkpoint
from latentbrain.torch.losses import lfads_cosmoothing_loss, lfads_elbo_loss
from latentbrain.torch.masking import (
    apply_input_neuron_dropout,
    sample_neuron_dropout_mask,
    summarize_dropout_mask,
)
from latentbrain.torch.rate_initialization import compute_train_mean_rates_hz
from latentbrain.torch.schedules import linear_warmup

TrainingMode = Literal["heldin_reconstruction", "cosmoothing"]


@dataclass(slots=True)
class TrainingState:
    epoch: int
    best_metric: float | None
    history: list[dict[str, float]]
    best_epoch: int | None
    early_stopping_triggered: bool


def _ensure_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all().item():
        msg = f"{name} is not finite"
        raise RuntimeError(msg)


def _write_history(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _training_mode(config: dict[str, Any], model: LFADSGRU) -> TrainingMode:
    training_config = dict(config.get("training", {}))
    model_config = dict(config.get("model", {}))
    heldout_weight = float(training_config.get("heldout_loss_weight", 0.0))
    output_policy = model_config.get("output_dim")
    if heldout_weight > 0.0 and (
        output_policy == "all" or model.config.output_dim > model.config.input_dim
    ):
        return "cosmoothing"
    return "heldin_reconstruction"


def _first_indices(batch: dict[str, torch.Tensor], key: str, device: torch.device) -> torch.Tensor:
    indices = batch[key]
    if indices.ndim == 2:
        indices = indices[0]
    return indices.to(device=device, dtype=torch.long)


def _loss_for_batch(
    model: LFADSGRU,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    bin_size_ms: int,
    kl_beta: float,
    training_config: dict[str, Any],
    mode: TrainingMode,
    input_heldin_spikes: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    heldin = batch["heldin_spikes"].to(device)
    output = model(heldin if input_heldin_spikes is None else input_heldin_spikes)
    if mode == "cosmoothing":
        loss = lfads_cosmoothing_loss(
            heldin_counts=heldin,
            heldout_counts=batch["heldout_spikes"].to(device),
            all_rates_hz=output["rates_hz"],
            heldin_indices=_first_indices(batch, "heldin_indices", device),
            heldout_indices=_first_indices(batch, "heldout_indices", device),
            posterior_mean=output["z0_mean"],
            posterior_logvar=output["z0_logvar"],
            bin_size_ms=bin_size_ms,
            kl_beta=kl_beta,
            heldin_loss_weight=float(training_config.get("heldin_loss_weight", 1.0)),
            heldout_loss_weight=float(training_config.get("heldout_loss_weight", 1.0)),
            normalization=str(training_config.get("loss_normalization", "mean")),
        )
    else:
        elbo = lfads_elbo_loss(
            heldin,
            output["rates_hz"],
            output["z0_mean"],
            output["z0_logvar"],
            bin_size_ms,
            kl_beta,
        )
        zero = elbo["loss"].new_tensor(0.0)
        loss = {
            "loss": elbo["loss"],
            "heldin_reconstruction_loss": elbo["reconstruction_loss"],
            "heldout_prediction_loss": zero,
            "kl_loss": elbo["kl_loss"],
            "kl_beta": elbo["kl_beta"],
        }
    return loss, output["rates_hz"]


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


def _masked_training_input(
    heldin: torch.Tensor,
    settings: dict[str, Any],
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = sample_neuron_dropout_mask(
        heldin.shape[0],
        heldin.shape[2],
        float(settings["rate"]),
        heldin.device,
        generator=generator,
        keep_at_least_one=bool(settings["keep_at_least_one_neuron"]),
    )
    return apply_input_neuron_dropout(heldin, mask), summarize_dropout_mask(mask)


def _split_metrics(
    model: LFADSGRU,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    bin_size_ms: int,
    kl_beta: float,
    training_config: dict[str, Any],
    mode: TrainingMode,
) -> dict[str, float]:
    model.eval()
    losses: dict[str, list[float]] = {
        "total_loss": [],
        "heldin_reconstruction_loss": [],
        "heldout_prediction_loss": [],
        "kl_loss": [],
        "mean_rate_hz": [],
        "min_rate_hz": [],
        "max_rate_hz": [],
    }
    with torch.no_grad():
        for batch in loader:
            loss, rates = _loss_for_batch(
                model, batch, device, bin_size_ms, kl_beta, training_config, mode
            )
            _ensure_finite("evaluation loss", loss["loss"])
            losses["total_loss"].append(float(loss["loss"].detach().cpu()))
            losses["heldin_reconstruction_loss"].append(
                float(loss["heldin_reconstruction_loss"].detach().cpu())
            )
            losses["heldout_prediction_loss"].append(
                float(loss["heldout_prediction_loss"].detach().cpu())
            )
            losses["kl_loss"].append(float(loss["kl_loss"].detach().cpu()))
            losses["mean_rate_hz"].append(float(rates.mean().detach().cpu()))
            losses["min_rate_hz"].append(float(rates.min().detach().cpu()))
            losses["max_rate_hz"].append(float(rates.max().detach().cpu()))
    return {key: _mean(value) for key, value in losses.items()}


def _is_better(value: float, best: float | None, mode: str) -> bool:
    if best is None:
        return True
    if mode == "min":
        return value < best
    if mode == "max":
        return value > best
    msg = "checkpoint_mode must be min or max"
    raise ValueError(msg)


def _initialize_output_bias_from_train_loader(
    model: LFADSGRU,
    train_loader: DataLoader[dict[str, torch.Tensor]],
    bin_size_ms: int,
) -> None:
    chunks: list[np.ndarray] = []
    for batch in train_loader:
        key = (
            "all_spikes"
            if model.config.output_dim == batch["all_spikes"].shape[-1]
            else "heldin_spikes"
        )
        chunks.append(batch[key].detach().cpu().numpy())
    train_spikes = np.concatenate(chunks, axis=0)
    rates = compute_train_mean_rates_hz(
        train_spikes,
        bin_size_ms,
        model.config.min_rate_hz,
        model.config.max_rate_hz,
    )
    model.initialize_output_bias_from_rates(torch.as_tensor(rates, dtype=torch.float32))


def train_lfads_gru(
    model: LFADSGRU,
    dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    checkpoint_scorer: Callable[[LFADSGRU], float] | None = None,
) -> TrainingState:
    """Train the LFADS-style GRU model and write local ignored outputs."""
    train_loader = dataloaders["train"]
    training_config = dict(config["training"])
    bin_size_ms = int(config["dataset"]["bin_size_ms"])
    evaluate_splits = [
        str(value) for value in config.get("evaluation", {}).get("evaluate_splits", [])
    ]
    if not evaluate_splits:
        evaluate_splits = ["train", "validation"]
    epochs = int(training_config["epochs"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    clip_norm = float(training_config["gradient_clip_norm"])
    warmup_epochs = int(training_config["kl_warmup_epochs"])
    mode = _training_mode(config, model)
    checkpoint_metric = str(training_config.get("checkpoint_metric", "validation_total_loss"))
    if checkpoint_metric == "validation_loss":
        checkpoint_metric = "validation_total_loss"
    checkpoint_mode = str(training_config.get("checkpoint_mode", "min"))
    if checkpoint_metric == "inner_validation_unified_bits_per_spike" and checkpoint_scorer is None:
        msg = "checkpoint_scorer is required for inner-validation unified checkpoint selection"
        raise ValueError(msg)
    scheduler_name = str(training_config.get("scheduler", "none"))
    if scheduler_name not in {"none", "cosine"}:
        msg = "training.scheduler must be none or cosine"
        raise ValueError(msg)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        if scheduler_name == "cosine"
        else None
    )
    amp_enabled = bool(training_config.get("mixed_precision", False)) and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=amp_enabled)
    minimum_epochs = int(training_config.get("minimum_epochs", epochs))
    patience = int(training_config.get("early_stopping_patience", epochs))
    if minimum_epochs <= 0 or patience <= 0:
        msg = "minimum_epochs and early_stopping_patience must be positive"
        raise ValueError(msg)
    save_latest = bool(training_config.get("save_latest", True))
    save_best = bool(training_config.get("save_best", True))
    input_dropout = _input_dropout_settings(training_config)
    dropout_generator = _make_dropout_generator(device, int(input_dropout.get("seed", 0)))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    history: list[dict[str, float]] = []
    best_metric: float | None = None
    best_epoch: int | None = None
    epochs_without_improvement = 0
    early_stopping_triggered = False

    if bool(training_config.get("initialize_readout_bias_from_train_rates", False)):
        _initialize_output_bias_from_train_loader(model, train_loader, bin_size_ms)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        kl_beta = linear_warmup(epoch, warmup_epochs) * float(training_config.get("kl_scale", 1.0))
        train_losses: list[float] = []
        gradient_norms: list[float] = []
        realized_dropout: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            model_input = None
            if bool(input_dropout["enabled"]):
                heldin = batch["heldin_spikes"].to(device)
                model_input, stats = _masked_training_input(
                    heldin, input_dropout, dropout_generator
                )
                realized_dropout.append(float(stats["dropout_fraction"]))
            with autocast(device_type=device.type, enabled=amp_enabled):
                loss, _ = _loss_for_batch(
                    model,
                    batch,
                    device,
                    bin_size_ms,
                    kl_beta,
                    training_config,
                    mode,
                    input_heldin_spikes=model_input,
                )
            _ensure_finite("training loss", loss["loss"])
            scaler.scale(loss["loss"]).backward()  # type: ignore[no-untyped-call]
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            _ensure_finite("gradient norm", torch.as_tensor(grad_norm, device=device))
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss["loss"].detach().cpu()))
            gradient_norms.append(float(torch.as_tensor(grad_norm).detach().cpu()))

        row: dict[str, float] = {
            "epoch": float(epoch),
            "kl_beta": float(kl_beta),
            "gradient_norm": _mean(gradient_norms),
            "training_batch_loss": _mean(train_losses),
            "training_mode_cosmoothing": 1.0 if mode == "cosmoothing" else 0.0,
            "configured_input_dropout_rate": float(input_dropout["rate"]),
            "realized_input_dropout_fraction": _mean(realized_dropout) if realized_dropout else 0.0,
        }
        for split_name in evaluate_splits:
            metrics = _split_metrics(
                model,
                dataloaders[split_name],
                device,
                bin_size_ms,
                kl_beta,
                training_config,
                mode,
            )
            row[f"{split_name}_total_loss"] = metrics["total_loss"]
            row[f"{split_name}_heldin_reconstruction_loss"] = metrics["heldin_reconstruction_loss"]
            row[f"{split_name}_heldout_prediction_loss"] = metrics["heldout_prediction_loss"]
            row[f"{split_name}_kl_loss"] = metrics["kl_loss"]
            row[f"{split_name}_mean_rate_hz"] = metrics["mean_rate_hz"]
            row[f"{split_name}_min_rate_hz"] = metrics["min_rate_hz"]
            row[f"{split_name}_max_rate_hz"] = metrics["max_rate_hz"]
        if "validation_total_loss" in row:
            row["validation_loss"] = row["validation_total_loss"]
            row["validation_reconstruction_loss"] = row["validation_heldin_reconstruction_loss"]
            row["loss"] = row["validation_total_loss"]
            row["reconstruction_loss"] = row["validation_heldin_reconstruction_loss"]
            row["kl_loss"] = row["validation_kl_loss"]
            row["mean_predicted_rate_hz"] = row["validation_mean_rate_hz"]
            row["mean_predicted_rate"] = row["validation_mean_rate_hz"]
            row["min_predicted_rate_hz"] = row["validation_min_rate_hz"]
            row["max_predicted_rate_hz"] = row["validation_max_rate_hz"]
        if checkpoint_scorer is not None:
            metric = float(checkpoint_scorer(model))
            if not np.isfinite(metric):
                msg = "checkpoint selection metric is not finite"
                raise RuntimeError(msg)
            row["inner_validation_unified_bits_per_spike"] = metric
        row["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        history.append(row)
        if save_latest:
            save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, epoch, row, config)
        metric_value = row[checkpoint_metric]
        if _is_better(metric_value, best_metric, checkpoint_mode):
            best_metric = metric_value
            best_epoch = epoch
            epochs_without_improvement = 0
            if save_best:
                save_checkpoint(
                    checkpoint_dir / "best_validation.pt", model, optimizer, epoch, row, config
                )
        else:
            epochs_without_improvement += 1
        if scheduler is not None:
            scheduler.step()
        _write_history(output_dir / "metrics_history.csv", history)
        if epoch + 1 >= minimum_epochs and epochs_without_improvement >= patience:
            early_stopping_triggered = True
            break

    final_metrics = history[-1] | {
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "early_stopping_triggered": early_stopping_triggered,
        "mixed_precision_enabled": amp_enabled,
    }
    (output_dir / "final_metrics.json").write_text(
        json.dumps(final_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return TrainingState(
        epoch=len(history) - 1,
        best_metric=best_metric,
        history=history,
        best_epoch=best_epoch,
        early_stopping_triggered=early_stopping_triggered,
    )
