from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import torch
import yaml
from torch.utils.data import DataLoader

from latentbrain.eval.neural_ode_objectives import (
    build_neural_ode_objective_diagnostics,
    build_neural_ode_objective_result_row,
    rank_neural_ode_objective_results,
    summarize_neural_ode_objectives,
)
from latentbrain.models.neural_sde import NeuralSDE, NeuralSDEConfig
from latentbrain.paths import get_repo_root, resolve_configured_path
from latentbrain.randomness import seed_everything
from latentbrain.torch.checkpoints import save_checkpoint
from latentbrain.torch.datasets import create_dataloaders, create_torch_datasets
from latentbrain.torch.device import resolve_device
from latentbrain.torch.losses import gaussian_kl_standard_normal, poisson_nll_torch
from latentbrain.torch.masking import (
    apply_input_neuron_dropout,
    sample_neuron_dropout_mask,
    summarize_dropout_mask,
)
from latentbrain.torch.rate_initialization import compute_train_mean_rates_hz
from latentbrain.torch.schedules import linear_warmup
from latentbrain.train.neural_ode_refinement import (
    _is_better,
    _make_dropout_generator,
    _scheduler_factor,
    _set_optimizer_lr,
    _write_final_evaluation,
    _write_history,
    select_best_unified_checkpoint_index,
)
from latentbrain.train.neural_sde_tuning import (
    _behavior_mean_r2,
    _ensure_finite,
    _first_indices,
    _input_dropout_settings,
    _json_default,
    _load_dataset,
    _mean,
    _resolve_processed_path,
    _run_neural_sde_evaluation,
    _split_and_mask,
    _trial_mask,
    _validate_cuda,
    _verify_reference_zero,
)

_HISTORY_KEYS = (
    "total_loss",
    "heldin_reconstruction_loss",
    "heldout_prediction_loss",
    "z0_kl_loss",
    "drift_regularization_loss",
    "rate_calibration_loss",
    "drift_norm",
    "diffusion_mean",
    "mean_rate_hz",
)


def build_objective_variants(
    base_model_config: dict[str, Any],
    variants: list[dict[str, Any]],
    max_runs: int,
) -> list[dict[str, Any]]:
    """Merge each objective variant onto the base model config, deterministically ordered."""
    if max_runs <= 0:
        msg = "max_runs must be positive"
        raise ValueError(msg)
    names = [str(variant["name"]) for variant in variants]
    if len(set(names)) != len(names):
        msg = "objective variant names must be unique"
        raise ValueError(msg)
    resolved: list[dict[str, Any]] = []
    for variant in variants[:max_runs]:
        merged = copy.deepcopy(base_model_config) | copy.deepcopy(variant)
        merged["diffusion_scale"] = 0.0
        merged.setdefault("zero_count_weight", 1.0)
        merged.setdefault("positive_count_weight", 1.0)
        merged.setdefault("rate_calibration_loss_weight", 0.0)
        if float(merged["zero_count_weight"]) <= 0.0:
            msg = "zero_count_weight must be positive"
            raise ValueError(msg)
        if float(merged["positive_count_weight"]) <= 0.0:
            msg = "positive_count_weight must be positive"
            raise ValueError(msg)
        if float(merged["rate_calibration_loss_weight"]) < 0.0:
            msg = "rate_calibration_loss_weight must be non-negative"
            raise ValueError(msg)
        if float(merged["drift_regularization"]) < 0.0:
            msg = "drift_regularization must be non-negative"
            raise ValueError(msg)
        if str(merged["scheduler"]) not in {"none", "cosine"}:
            msg = "scheduler must be none or cosine"
            raise ValueError(msg)
        resolved.append(merged)
    return resolved


def make_neural_ode_objective_run_id(index: int, variant: dict[str, Any]) -> str:
    return f"run_{index:03d}_{variant['name']}"


def build_neural_ode_objective_train_config(
    config: dict[str, Any], variant: dict[str, Any], run_output_dir: Path
) -> dict[str, Any]:
    """Build a run config from the shared base model config plus one objective variant."""
    run_config = copy.deepcopy(config)
    base_model = dict(run_config.pop("base_model"))
    params = copy.deepcopy(base_model) | copy.deepcopy(variant)
    params["diffusion_scale"] = 0.0
    run_config["data"] = {
        "batch_size": int(params["batch_size"]),
        "max_time_bins": int(run_config.get("_window_bins", 0)) or None,
    }
    run_config["model"] = dict(base_model) | {
        "name": "neural_ode",
        "encoder_hidden_dim": int(params["encoder_hidden_dim"]),
        "drift_hidden_dim": int(params["drift_hidden_dim"]),
        "diffusion_hidden_dim": int(params["diffusion_hidden_dim"]),
        "latent_dim": int(params["latent_dim"]),
        "factor_dim": int(params["factor_dim"]),
        "dropout": float(params.get("model_dropout", 0.0)),
        "diffusion_scale": 0.0,
    }
    run_config["training"] = {
        "device": str(run_config["runtime"]["device"]),
        "seed": int(run_config["splits"]["seed"]),
        "epochs": int(params["epochs"]),
        "learning_rate": float(params["learning_rate"]),
        "weight_decay": float(params["weight_decay"]),
        "gradient_clip_norm": float(params["gradient_clip_norm"]),
        "objective_name": str(params["name"]),
        "heldin_loss_weight": float(params["heldin_loss_weight"]),
        "heldout_loss_weight": float(params["heldout_loss_weight"]),
        "zero_count_weight": float(params["zero_count_weight"]),
        "positive_count_weight": float(params["positive_count_weight"]),
        "rate_calibration_loss_weight": float(params["rate_calibration_loss_weight"]),
        "loss_normalization": str(params["loss_normalization"]),
        "kl_warmup_epochs": int(params["kl_warmup_epochs"]),
        "kl_scale": float(params["kl_scale"]),
        "drift_regularization_scale": float(params["drift_regularization"]),
        "scheduler": str(params["scheduler"]),
        "checkpoint_metric": str(params.get("checkpoint_metric", "validation_total_loss")),
        "checkpoint_mode": str(params.get("checkpoint_mode", "min")),
        "save_unified_checkpoints": bool(params.get("save_unified_checkpoints", True)),
        "evaluate_checkpoints_by_unified_metric": bool(
            params.get("evaluate_checkpoints_by_unified_metric", True)
        ),
        "input_dropout": {
            "enabled": float(params["input_dropout_rate"]) > 0.0,
            "rate": float(params["input_dropout_rate"]),
            "apply_to": ["train"],
            "keep_at_least_one_neuron": True,
            "seed": int(run_config["splits"]["seed"]),
        },
    }
    run_config["reporting"] = dict(run_config["reporting"])
    run_config["reporting"]["output_dir"] = str(run_output_dir)
    return run_config


def weighted_poisson_loss(
    counts: torch.Tensor,
    rates_hz: torch.Tensor,
    bin_size_ms: int,
    zero_count_weight: float,
    positive_count_weight: float,
    normalization: str,
) -> torch.Tensor:
    """Poisson NLL where zero-count and positive-count bins carry separate weights."""
    if counts.shape != rates_hz.shape:
        msg = f"counts and rates_hz must match; got {counts.shape} and {rates_hz.shape}"
        raise ValueError(msg)
    if zero_count_weight <= 0.0 or positive_count_weight <= 0.0:
        msg = "count weights must be positive"
        raise ValueError(msg)
    if zero_count_weight == positive_count_weight:
        nll = poisson_nll_torch(counts, rates_hz, bin_size_ms, include_constant=True)
        return _normalize_loss(zero_count_weight * nll, counts, normalization)
    expected = torch.clamp(rates_hz, min=torch.finfo(rates_hz.dtype).tiny) * (bin_size_ms / 1000.0)
    elementwise = -(counts * torch.log(expected) - expected - torch.lgamma(counts + 1.0))
    weights = torch.where(
        counts > 0.0,
        counts.new_tensor(float(positive_count_weight)),
        counts.new_tensor(float(zero_count_weight)),
    )
    return _normalize_loss(torch.sum(weights * elementwise), counts, normalization)


def _normalize_loss(total: torch.Tensor, counts: torch.Tensor, normalization: str) -> torch.Tensor:
    if normalization == "sum":
        return total
    if normalization in {"mean", "per_observed_spike_bin"}:
        return total / max(int(counts.numel()), 1)
    if normalization == "batch_mean":
        return total / max(int(counts.shape[0]), 1)
    msg = "normalization must be sum, mean, batch_mean, or per_observed_spike_bin"
    raise ValueError(msg)


def _objective_loss_for_batch(
    model: NeuralSDE,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    bin_size_ms: int,
    kl_beta: float,
    training_config: dict[str, Any],
    observed_mean_rates_hz: torch.Tensor,
    input_heldin_spikes: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    heldin = batch["heldin_spikes"].to(device)
    heldout = batch["heldout_spikes"].to(device)
    output = model(heldin if input_heldin_spikes is None else input_heldin_spikes)
    rates = output["rates_hz"]
    heldin_indices = _first_indices(batch, "heldin_indices", device)
    heldout_indices = _first_indices(batch, "heldout_indices", device)
    zero_weight = float(training_config["zero_count_weight"])
    positive_weight = float(training_config["positive_count_weight"])
    normalization = str(training_config.get("loss_normalization", "mean"))
    heldin_loss = weighted_poisson_loss(
        heldin,
        rates.index_select(dim=2, index=heldin_indices),
        bin_size_ms,
        zero_weight,
        positive_weight,
        normalization,
    )
    heldout_loss = weighted_poisson_loss(
        heldout,
        rates.index_select(dim=2, index=heldout_indices),
        bin_size_ms,
        zero_weight,
        positive_weight,
        normalization,
    )
    kl_loss = gaussian_kl_standard_normal(output["z0_mean"], output["z0_logvar"])
    smoothness = output["drift"].pow(2).mean()
    drift_scale = float(training_config.get("drift_regularization_scale", 0.0))
    calibration_scale = float(training_config.get("rate_calibration_loss_weight", 0.0))
    calibration = (rates.mean(dim=(0, 1)) - observed_mean_rates_hz).pow(2).mean()
    total = (
        float(training_config["heldin_loss_weight"]) * heldin_loss
        + float(training_config["heldout_loss_weight"]) * heldout_loss
        + rates.new_tensor(float(kl_beta)) * kl_loss
        + rates.new_tensor(drift_scale) * smoothness
        + rates.new_tensor(calibration_scale) * calibration
    )
    losses = {
        "loss": total,
        "heldin_reconstruction_loss": heldin_loss,
        "heldout_prediction_loss": heldout_loss,
        "z0_kl_loss": kl_loss,
        "drift_regularization_loss": drift_scale * smoothness,
        "rate_calibration_loss": calibration_scale * calibration,
        "drift_norm": output["drift"].norm(dim=-1).mean(),
        "diffusion_mean": output["diffusion"].mean(),
    }
    return losses, rates


def _split_metrics(
    model: NeuralSDE,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    bin_size_ms: int,
    kl_beta: float,
    training_config: dict[str, Any],
    observed_mean_rates_hz: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    collected: dict[str, list[float]] = {key: [] for key in _HISTORY_KEYS}
    with torch.no_grad():
        for batch in loader:
            losses, rates = _objective_loss_for_batch(
                model, batch, device, bin_size_ms, kl_beta, training_config, observed_mean_rates_hz
            )
            _ensure_finite("evaluation loss", losses["loss"])
            collected["total_loss"].append(float(losses["loss"].detach().cpu()))
            for key in _HISTORY_KEYS[1:-1]:
                collected[key].append(float(losses[key].detach().cpu()))
            collected["mean_rate_hz"].append(float(rates.mean().detach().cpu()))
    return {key: _mean(value) for key, value in collected.items()}


def _train_objective_variant(
    model: NeuralSDE,
    dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    observed_mean_rates_hz: torch.Tensor,
) -> list[dict[str, float]]:
    training_config = dict(config["training"])
    base_lr = float(training_config["learning_rate"])
    epochs = int(training_config["epochs"])
    scheduler_name = str(training_config.get("scheduler", "none"))
    bin_size_ms = int(config["dataset"]["bin_size_ms"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr, weight_decay=float(training_config["weight_decay"])
    )
    model.to(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    input_dropout = _input_dropout_settings(training_config)
    dropout_generator = _make_dropout_generator(device, int(input_dropout.get("seed", 0)))
    mean_observed_rate = float(observed_mean_rates_hz.mean().detach().cpu())
    history: list[dict[str, float]] = []
    best_metric: float | None = None
    for epoch in range(epochs):
        learning_rate = base_lr * _scheduler_factor(scheduler_name, epoch, epochs)
        _set_optimizer_lr(optimizer, learning_rate)
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
                dropout_mask = sample_neuron_dropout_mask(
                    heldin.shape[0],
                    heldin.shape[2],
                    float(input_dropout["rate"]),
                    heldin.device,
                    generator=dropout_generator,
                    keep_at_least_one=bool(input_dropout["keep_at_least_one_neuron"]),
                )
                model_input = apply_input_neuron_dropout(heldin, dropout_mask)
                dropout_fractions.append(
                    float(summarize_dropout_mask(dropout_mask)["dropout_fraction"])
                )
            losses, _ = _objective_loss_for_batch(
                model,
                batch,
                device,
                bin_size_ms,
                kl_beta,
                training_config,
                observed_mean_rates_hz,
                model_input,
            )
            _ensure_finite("training loss", losses["loss"])
            torch.autograd.backward(losses["loss"])
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training_config["gradient_clip_norm"])
            )
            optimizer.step()
            train_losses.append(float(losses["loss"].detach().cpu()))
        row: dict[str, float] = {
            "epoch": float(epoch),
            "kl_beta": float(kl_beta),
            "learning_rate": float(learning_rate),
            "training_batch_loss": _mean(train_losses),
            "configured_input_dropout_rate": float(input_dropout["rate"]),
            "realized_input_dropout_fraction": _mean(dropout_fractions)
            if dropout_fractions
            else 0.0,
            "zero_count_weight": float(training_config["zero_count_weight"]),
            "positive_count_weight": float(training_config["positive_count_weight"]),
            "mean_observed_rate": mean_observed_rate,
        }
        for split_name in config["evaluation"]["evaluate_splits"]:
            metrics = _split_metrics(
                model,
                dataloaders[str(split_name)],
                device,
                bin_size_ms,
                kl_beta,
                training_config,
                observed_mean_rates_hz,
            )
            for key, value in metrics.items():
                row[f"{split_name}_{key}"] = value
        row["train_total_loss"] = row.get("train_total_loss", row["training_batch_loss"])
        for key in (
            "z0_kl_loss",
            "drift_regularization_loss",
            "rate_calibration_loss",
            "drift_norm",
            "diffusion_mean",
        ):
            row[key] = row.get(f"validation_{key}", float("nan"))
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


def _checkpoint_metric_row(
    checkpoint_path: Path,
    source: str,
    run_config: dict[str, Any],
    dataset: Any,
    split: Any,
    mask: Any,
    device: torch.device,
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    eval_config = copy.deepcopy(run_config)
    eval_config["model"]["checkpoint_path"] = str(checkpoint_path)
    payload = _run_neural_sde_evaluation(dataset, split, mask, eval_config, device)
    split_metrics = payload[0]
    validation = split_metrics[
        (split_metrics["split"] == "validation")
        & (split_metrics["prediction_source"] == "direct_model")
    ].iloc[0]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metrics = dict(checkpoint.get("metrics", {}))
    row = {
        "checkpoint_source": source,
        "epoch_or_source": source,
        "epoch": int(checkpoint.get("epoch", -1)),
        "validation_total_loss": float(metrics.get("validation_total_loss", float("nan"))),
        "validation_unified_bits_per_spike": float(validation["bits_per_spike"]),
        "validation_poisson_nll": float(validation["poisson_nll"]),
        "checkpoint_path": str(checkpoint_path),
        "selected_by_loss": source == "best_validation",
        "selected_by_unified": False,
    }
    return row, payload


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
        "drift_regularization_loss": metrics.get("drift_regularization_loss"),
        "rate_calibration_loss": metrics.get("rate_calibration_loss"),
        "drift_norm": metrics.get("drift_norm"),
        "diffusion_mean": metrics.get("diffusion_mean"),
        "learning_rate": metrics.get("learning_rate"),
        "mean_predicted_rate": metrics.get("mean_predicted_rate"),
        "mean_observed_rate": metrics.get("mean_observed_rate"),
        "best_checkpoint_source": selected.get("checkpoint_source", "best_unified"),
    }


def _unified_scores(run_dir: Path) -> pd.DataFrame:
    scores = pd.read_csv(run_dir / "evaluation" / "split_metrics.csv")
    scores.insert(0, "reference_name", "train_heldout_mean_rate")
    scores.insert(0, "valid_model", True)
    scores.to_csv(run_dir / "unified_scores.csv", index=False)
    return scores


def _train_and_evaluate_run(
    run_config: dict[str, Any], run_id: str, dataset: Any, split: Any, mask: Any
) -> pd.DataFrame:
    # Every variant shares one seed so that score differences are attributable to the
    # objective alone. Seeding per run index (as the grid workflows do) confounds the
    # objective with initialization: a seed change alone moves validation unified
    # bits/spike by more than any objective effect measured here.
    seed = int(run_config["training"]["seed"])
    seed_everything(seed)
    dataloaders = create_dataloaders(
        create_torch_datasets(dataset, split, mask, int(run_config["data"]["max_time_bins"])),
        batch_size=int(run_config["data"]["batch_size"]),
        num_workers=0,
        drop_last=False,
        seed=seed,
    )
    input_dim = int(mask.heldin.sum())
    output_dim = int(dataset.spikes.shape[2])
    model = NeuralSDE(
        NeuralSDEConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            encoder_hidden_dim=int(run_config["model"]["encoder_hidden_dim"]),
            drift_hidden_dim=int(run_config["model"]["drift_hidden_dim"]),
            diffusion_hidden_dim=int(run_config["model"]["diffusion_hidden_dim"]),
            latent_dim=int(run_config["model"]["latent_dim"]),
            factor_dim=int(run_config["model"]["factor_dim"]),
            dropout=float(run_config["model"].get("dropout", 0.0)),
            min_rate_hz=float(run_config["model"]["min_rate_hz"]),
            max_rate_hz=float(run_config["model"]["max_rate_hz"]),
            dt_seconds=float(run_config["model"]["dt_seconds"]),
            diffusion_scale=0.0,
        )
    )
    train_spikes = dataset.spikes[_trial_mask(dataset, split.train)]
    mean_rates = compute_train_mean_rates_hz(
        train_spikes,
        dataset.bin_size_ms,
        float(run_config["model"]["min_rate_hz"]),
        float(run_config["model"]["max_rate_hz"]),
    )
    mean_rates_tensor = torch.as_tensor(mean_rates, dtype=torch.float32)
    model.initialize_output_bias_from_rates(mean_rates_tensor)
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
    _train_objective_variant(
        model, dataloaders, snapshot, output_dir, device, mean_rates_tensor.to(device)
    )
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_rows: list[dict[str, Any]] = []
    payloads: list[tuple[Any, ...]] = []
    for source, checkpoint_path in (
        ("best_validation", checkpoint_dir / "best_validation.pt"),
        ("latest", checkpoint_dir / "latest.pt"),
    ):
        row, payload = _checkpoint_metric_row(
            checkpoint_path, source, snapshot, dataset, split, mask, device
        )
        checkpoint_rows.append(row)
        payloads.append(payload)
    selected_index = select_best_unified_checkpoint_index(checkpoint_rows)
    checkpoint_rows[selected_index]["selected_by_unified"] = True
    best_unified = checkpoint_dir / "best_unified.pt"
    shutil.copy2(Path(str(checkpoint_rows[selected_index]["checkpoint_path"])), best_unified)
    checkpoint_scores = pd.DataFrame(checkpoint_rows)
    checkpoint_scores.to_csv(output_dir / "checkpoint_scores.csv", index=False)
    selected = payloads[selected_index]
    _write_final_evaluation(
        output_dir,
        snapshot,
        selected[0],
        selected[1],
        selected[2],
        selected[3],
        selected[4],
        best_unified,
        "neural_ode_objectives",
    )
    latent_diagnostics = selected[5].copy()
    latent_diagnostics.insert(0, "run_id", run_id)
    latent_diagnostics.to_csv(output_dir / "latent_diagnostics.csv", index=False)
    return checkpoint_scores


def run_neural_ode_objective_variants(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _resolve_processed_path(config)
    gpu_name = _validate_cuda(config)
    if float(config["base_model"].get("diffusion_scale", 0.0)) != 0.0:
        msg = "deterministic neural-ODE objective variants require diffusion_scale == 0.0"
        raise ValueError(msg)
    dataset, dataset_hash, window_bins = _load_dataset(config)
    split, mask = _split_and_mask(dataset, config)
    reference_zero = _verify_reference_zero(dataset, split, mask, config)
    repo_root = get_repo_root()
    output_dir = resolve_configured_path(str(config["reporting"]["output_dir"]), repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    refs = dict(config["references"])
    refs["train_mean_validation_bits_per_spike"] = reference_zero
    variants = build_objective_variants(
        dict(config["base_model"]),
        [dict(variant) for variant in config["objective_variants"]],
        int(config["search"]["max_runs"]),
    )
    base = copy.deepcopy(config)
    base["_window_bins"] = window_bins
    rows: list[dict[str, Any]] = []
    all_checkpoint_scores: list[pd.DataFrame] = []
    all_diagnostics: list[pd.DataFrame] = []
    for run_index, variant in enumerate(variants):
        run_id = make_neural_ode_objective_run_id(run_index, variant)
        run_dir = output_dir / "runs" / run_id
        run_config = build_neural_ode_objective_train_config(base, variant, run_dir)
        checkpoint_scores = _train_and_evaluate_run(run_config, run_id, dataset, split, mask)
        checkpoint_scores.insert(0, "run_id", run_id)
        all_checkpoint_scores.append(checkpoint_scores)
        checkpoint_metrics = _selected_checkpoint_metrics(run_dir, str(config["runtime"]["device"]))
        row = build_neural_ode_objective_result_row(
            run_id,
            run_index,
            variant,
            _unified_scores(run_dir),
            checkpoint_metrics,
            checkpoint_scores,
            refs,
            run_dir,
        )
        run_diagnostics = build_neural_ode_objective_diagnostics(
            pd.DataFrame(
                [
                    row
                    | {
                        "mean_predicted_rate": checkpoint_metrics["mean_predicted_rate"],
                        "mean_observed_rate": checkpoint_metrics["mean_observed_rate"],
                    }
                ]
            )
        )
        run_diagnostics.to_csv(run_dir / "objective_diagnostics.csv", index=False)
        all_diagnostics.append(run_diagnostics)
        rows.append(row)
    results = pd.DataFrame(rows)
    diagnostics = (
        pd.concat(all_diagnostics, ignore_index=True)
        if all_diagnostics
        else build_neural_ode_objective_diagnostics(results)
    )
    leaderboard = rank_neural_ode_objective_results(results)
    summary = summarize_neural_ode_objectives(results, refs)
    summary.update(
        {
            "dataset_name": config["dataset"]["name"],
            "dataset_hash": dataset_hash,
            "cuda_device": gpu_name,
            "bin_size_ms": int(config["binning"]["target_bin_size_ms"]),
            "window_seconds": float(config["window"]["duration_seconds"]),
            "window_bins": window_bins,
            "shared_seed": int(config["splits"]["seed"]),
            "seed_held_constant_across_variants": True,
            "output_dir": str(output_dir),
        }
    )
    if all_checkpoint_scores:
        pd.concat(all_checkpoint_scores, ignore_index=True).to_csv(
            output_dir / "checkpoint_selection.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(output_dir / "checkpoint_selection.csv", index=False)
    diagnostics.to_csv(output_dir / "objective_diagnostics.csv", index=False)
    best_config: dict[str, Any] = {}
    if not leaderboard.empty:
        best_name = str(leaderboard.iloc[0]["objective_name"])
        best_variant = next(item for item in variants if str(item["name"]) == best_name)
        best_result = results.loc[results["objective_name"] == best_name].iloc[0]
        best_config = build_neural_ode_objective_train_config(
            base, best_variant, Path(str(best_result["output_dir"]))
        )
    (output_dir / "best_config.yaml").write_text(
        yaml.safe_dump(json.loads(json.dumps(best_config, default=_json_default)), sort_keys=False),
        encoding="utf-8",
    )
    return results, summary


__all__ = [
    "build_neural_ode_objective_train_config",
    "build_objective_variants",
    "make_neural_ode_objective_run_id",
    "run_neural_ode_objective_variants",
    "select_best_unified_checkpoint_index",
    "weighted_poisson_loss",
]
