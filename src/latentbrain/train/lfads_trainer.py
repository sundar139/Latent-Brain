from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from latentbrain.models.lfads_gru import LFADSGRU
from latentbrain.torch.checkpoints import save_checkpoint
from latentbrain.torch.losses import lfads_elbo_loss
from latentbrain.torch.schedules import linear_warmup


@dataclass(slots=True)
class TrainingState:
    epoch: int
    best_metric: float | None
    history: list[dict[str, float]]


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


def _split_metrics(
    model: LFADSGRU,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    bin_size_ms: int,
    kl_beta: float,
) -> dict[str, float]:
    model.eval()
    losses: dict[str, list[float]] = {
        "loss": [],
        "reconstruction_loss": [],
        "kl_loss": [],
        "mean_rate_hz": [],
        "min_rate_hz": [],
        "max_rate_hz": [],
    }
    with torch.no_grad():
        for batch in loader:
            heldin = batch["heldin_spikes"].to(device)
            output = model(heldin)
            loss = lfads_elbo_loss(
                heldin,
                output["rates_hz"],
                output["z0_mean"],
                output["z0_logvar"],
                bin_size_ms,
                kl_beta,
            )
            _ensure_finite("evaluation loss", loss["loss"])
            losses["loss"].append(float(loss["loss"].detach().cpu()))
            losses["reconstruction_loss"].append(float(loss["reconstruction_loss"].detach().cpu()))
            losses["kl_loss"].append(float(loss["kl_loss"].detach().cpu()))
            losses["mean_rate_hz"].append(float(output["rates_hz"].mean().detach().cpu()))
            losses["min_rate_hz"].append(float(output["rates_hz"].min().detach().cpu()))
            losses["max_rate_hz"].append(float(output["rates_hz"].max().detach().cpu()))
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


def train_lfads_gru(
    model: LFADSGRU,
    dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> TrainingState:
    """Train the minimal LFADS-style GRU model and write local ignored outputs."""
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
    checkpoint_metric = str(training_config.get("checkpoint_metric", "validation_loss"))
    checkpoint_mode = str(training_config.get("checkpoint_mode", "min"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    history: list[dict[str, float]] = []
    best_metric: float | None = None

    model.to(device)
    for epoch in range(epochs):
        model.train()
        kl_beta = linear_warmup(epoch, warmup_epochs)
        train_losses: list[float] = []
        gradient_norms: list[float] = []
        for batch in train_loader:
            heldin = batch["heldin_spikes"].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(heldin)
            loss = lfads_elbo_loss(
                heldin,
                output["rates_hz"],
                output["z0_mean"],
                output["z0_logvar"],
                bin_size_ms,
                kl_beta,
            )
            _ensure_finite("training loss", loss["loss"])
            torch.autograd.backward(loss["loss"])
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            _ensure_finite("gradient norm", torch.as_tensor(grad_norm, device=device))
            optimizer.step()
            train_losses.append(float(loss["loss"].detach().cpu()))
            gradient_norms.append(float(torch.as_tensor(grad_norm).detach().cpu()))

        row: dict[str, float] = {
            "epoch": float(epoch),
            "kl_beta": float(kl_beta),
            "gradient_norm": _mean(gradient_norms),
            "training_batch_loss": _mean(train_losses),
        }
        for split_name in evaluate_splits:
            metrics = _split_metrics(model, dataloaders[split_name], device, bin_size_ms, kl_beta)
            row[f"{split_name}_loss"] = metrics["loss"]
            row[f"{split_name}_reconstruction_loss"] = metrics["reconstruction_loss"]
            row[f"{split_name}_kl_loss"] = metrics["kl_loss"]
            row[f"{split_name}_mean_rate_hz"] = metrics["mean_rate_hz"]
            row[f"{split_name}_min_rate_hz"] = metrics["min_rate_hz"]
            row[f"{split_name}_max_rate_hz"] = metrics["max_rate_hz"]
        if "validation_loss" in row:
            row["loss"] = row["validation_loss"]
            row["reconstruction_loss"] = row["validation_reconstruction_loss"]
            row["kl_loss"] = row["validation_kl_loss"]
            row["mean_predicted_rate_hz"] = row["validation_mean_rate_hz"]
            row["min_predicted_rate_hz"] = row["validation_min_rate_hz"]
            row["max_predicted_rate_hz"] = row["validation_max_rate_hz"]
        history.append(row)
        save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, epoch, row, config)
        metric_value = row[checkpoint_metric]
        if _is_better(metric_value, best_metric, checkpoint_mode):
            best_metric = metric_value
            save_checkpoint(
                checkpoint_dir / "best_validation.pt", model, optimizer, epoch, row, config
            )
        _write_history(output_dir / "metrics_history.csv", history)

    final_metrics = history[-1] | {"best_metric": best_metric}
    (output_dir / "final_metrics.json").write_text(
        json.dumps(final_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return TrainingState(epoch=epochs - 1, best_metric=best_metric, history=history)
