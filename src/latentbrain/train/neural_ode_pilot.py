"""Controlled deterministic neural-ODE feasibility pilot on MC_Maze Large repeat 0.

The deterministic neural-ODE is the existing NeuralSDE Euler latent generator with
diffusion disabled (diffusion_scale == 0.0). Model dimensions and objective are frozen
from the accepted MC_Maze Small refinement best run; only input/output dimensions are
adapted to the Large repeat-0 neuron mask. Orchestration, leakage discipline, inner
checkpoint selection, and resume mirror the accepted LFADS pilot.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
from torch.utils.data import DataLoader

from latentbrain.data.schemas import NeuralDataset, NeuronMask, TrialSplit
from latentbrain.eval.lfads_diagnostics import effective_rank, time_bin_diagnostics
from latentbrain.eval.scoring import (
    ScoringConfig,
    score_heldout_prediction,
    train_heldout_mean_rate_reference,
)
from latentbrain.eval.seed_robustness import bootstrap_mean_ci
from latentbrain.models.neural_sde import NeuralSDE, NeuralSDEConfig
from latentbrain.paths import get_repo_root, resolve_configured_path
from latentbrain.randomness import seed_everything
from latentbrain.torch.checkpoints import load_checkpoint, save_checkpoint
from latentbrain.torch.datasets import NeuralTrialDataset, create_dataloaders, create_torch_datasets
from latentbrain.torch.masking import (
    apply_input_neuron_dropout,
    sample_neuron_dropout_mask,
    summarize_dropout_mask,
)
from latentbrain.torch.rate_initialization import compute_train_mean_rates_hz
from latentbrain.torch.schedules import linear_warmup
from latentbrain.train.lfads_pilot import (
    EXPECTED_FOLDS,
    EXPECTED_REPEAT,
    EXPECTED_SEEDS,
    EXPECTED_SHAPE,
    PilotRun,
    _behavior_mean_r2,
    _counts,
    _finite,
    _load_protocol_inputs,
    _mask_from_fold,
    _scoring,
    build_inner_split,
    checkpoint_sha256,
    validate_checkpoint_record,
    validate_input_target_separation,
)
from latentbrain.train.neural_sde_tuning import (
    _ensure_finite,
    _input_dropout_settings,
    _loss_for_batch,
    _make_dropout_generator,
    _mean,
    _split_metrics,
    _trial_mask,
)

INPUT_NEURON_COUNT = 122
HELDOUT_NEURON_COUNT = 40

RUN_COLUMNS = [
    "repeat_index",
    "fold_index",
    "split_seed",
    "neuron_mask_seed",
    "initialization_seed",
    "status",
    "best_epoch",
    "checkpoint_source",
    "inner_validation_unified_bits_per_spike",
    "outer_unified_bits_per_spike",
    "outer_poisson_nll",
    "baseline_outer_unified_bits_per_spike",
    "paired_difference_vs_baseline",
    "lfads_outer_mean_reference",
    "training_seconds",
    "integration_seconds",
    "peak_cuda_memory_mb",
    "final_train_loss",
    "final_inner_validation_loss",
    "maximum_state_norm",
    "mean_state_norm",
    "maximum_drift_norm",
    "mean_drift_norm",
    "solver_failure_count",
    "nonfinite_state_count",
    "notes",
]
CHECKPOINT_COLUMNS = [
    "repeat_index",
    "fold_index",
    "initialization_seed",
    "epoch",
    "checkpoint_type",
    "selection_split",
    "selection_metric",
    "selection_metric_value",
    "checkpoint_path",
    "checkpoint_sha256",
    "model_config_digest",
    "solver_config_digest",
]
SOLVER_COLUMNS = [
    "repeat_index",
    "fold_index",
    "initialization_seed",
    "solver",
    "integration_step_seconds",
    "integration_steps",
    "solver_failure_count",
    "nonfinite_state_count",
    "maximum_state_norm",
    "mean_state_norm",
    "maximum_drift_norm",
    "mean_drift_norm",
    "terminal_state_norm",
    "gradient_norm",
    "integration_seconds",
]
SELECTION_METRIC = "inner_validation_unified_bits_per_spike"


@dataclass(slots=True)
class NeuralODETrainingState:
    best_epoch: int = -1
    best_metric: float = float("nan")
    history: list[dict[str, float]] = field(default_factory=list)
    early_stopping_triggered: bool = False
    final_gradient_norm: float = float("nan")


def _resolve(path: str) -> Path:
    return resolve_configured_path(path, get_repo_root())


def validate_neural_ode_pilot_config(config: dict[str, Any]) -> None:
    """Reject any protocol drift before touching data. Mirrors the LFADS pilot gate."""
    outer = config["outer_protocol"]
    initialization = config["initialization"]
    if int(outer["repeat_index"]) != EXPECTED_REPEAT:
        msg = "outer_protocol.repeat_index must be 0 for the feasibility pilot"
        raise ValueError(msg)
    if [int(value) for value in outer["fold_indices"]] != EXPECTED_FOLDS:
        msg = "outer_protocol.fold_indices must be exactly [0, 1, 2, 3, 4]"
        raise ValueError(msg)
    if [int(value) for value in initialization["seeds"]] != EXPECTED_SEEDS:
        msg = "initialization.seeds must be exactly [2027, 2028, 2029, 2030, 2031]"
        raise ValueError(msg)
    if str(initialization["seed_policy"]) != "exact_declared_seed":
        msg = "initialization.seed_policy must be exact_declared_seed; seed arithmetic is forbidden"
        raise ValueError(msg)
    if str(config["trial_source"]["type"]) != "trial_aware_raw" or bool(
        config["trial_source"]["allow_global_crop_to_min"]
    ):
        msg = "trial-aware raw input is required and global crop is forbidden"
        raise ValueError(msg)
    if not bool(config["window"]["extract_before_rebin"]):
        msg = "event-centered extraction must occur before rebinning"
        raise ValueError(msg)
    if not bool(outer["reuse_exact_assignments"]) or not bool(outer["reuse_exact_neuron_mask"]):
        msg = "exact outer assignments and neuron mask must be reused"
        raise ValueError(msg)
    inner = config["inner_checkpoint_selection"]
    if not bool(inner["enabled"]) or not bool(inner["stratified"]):
        msg = "stratified inner checkpoint selection must be enabled"
        raise ValueError(msg)
    if bool(inner["use_outer_evaluation_for_selection"]):
        msg = "outer evaluation cannot be used for checkpoint selection"
        raise ValueError(msg)
    if str(config["training"]["device"]) != "cuda":
        msg = "training.device must be cuda"
        raise ValueError(msg)
    if str(config["training"]["optimizer"]) != "adamw":
        msg = "training.optimizer must be adamw"
        raise ValueError(msg)
    if str(config["training"]["scheduler"]) != "cosine":
        msg = "training.scheduler must be cosine"
        raise ValueError(msg)
    if (
        str(config["training"]["checkpoint_metric"]) != SELECTION_METRIC
        or str(config["training"]["checkpoint_mode"]) != "max"
    ):
        msg = "checkpoints must maximize inner_validation_unified_bits_per_spike"
        raise ValueError(msg)
    if bool(config["model"]["diffusion_enabled"]) or bool(config["dynamics"]["adjoint"]):
        msg = "the pilot is a deterministic neural ODE: diffusion and adjoint must be disabled"
        raise ValueError(msg)
    if str(config["baseline"]["method"]) != "factor_latent_train_selected":
        msg = "baseline.method must be factor_latent_train_selected"
        raise ValueError(msg)


def build_pilot_run_schedule(config: dict[str, Any]) -> list[PilotRun]:
    validate_neural_ode_pilot_config(config)
    repeat = int(config["outer_protocol"]["repeat_index"])
    return [
        PilotRun(repeat, int(fold), int(seed))
        for fold in config["outer_protocol"]["fold_indices"]
        for seed in config["initialization"]["seeds"]
    ]


def _model_config_digest(config: dict[str, Any], input_dim: int, output_dim: int) -> str:
    model = config["model"]
    payload = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "latent_dim": int(model["latent_dim"]),
        "factor_dim": int(model["factor_dim"]),
        "encoder_hidden_dim": int(model["encoder_hidden_dim"]),
        "drift_hidden_dim": int(model["drift_hidden_dim"]),
        "diffusion_hidden_dim": int(model["diffusion_hidden_dim"]),
        "dropout": float(model["dropout_rate"]),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _solver_config_digest(config: dict[str, Any]) -> str:
    dynamics = config["dynamics"]
    payload = {
        "solver": str(dynamics["solver"]),
        "integration_step_seconds": float(dynamics["integration_step_seconds"]),
        "integration_horizon_seconds": float(dynamics["integration_horizon_seconds"]),
        "adjoint": bool(dynamics["adjoint"]),
        "diffusion_enabled": bool(config["model"]["diffusion_enabled"]),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def build_pilot_model(
    config: dict[str, Any], input_dim: int, output_dim: int, initialization_seed: int
) -> NeuralSDE:
    """Seed before construction so initialization equals the declared seed exactly."""
    seed_everything(
        initialization_seed,
        deterministic=bool(config["initialization"]["deterministic_algorithms"]),
    )
    model = config["model"]
    return NeuralSDE(
        NeuralSDEConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            encoder_hidden_dim=int(model["encoder_hidden_dim"]),
            drift_hidden_dim=int(model["drift_hidden_dim"]),
            diffusion_hidden_dim=int(model["diffusion_hidden_dim"]),
            latent_dim=int(model["latent_dim"]),
            factor_dim=int(model["factor_dim"]),
            dropout=float(model["dropout_rate"]),
            min_rate_hz=float(model["min_rate_hz"]),
            max_rate_hz=float(model["max_rate_hz"]),
            dt_seconds=float(config["dynamics"]["integration_step_seconds"]),
            diffusion_scale=0.0,
        )
    )


def _predict_dynamics(
    model: NeuralSDE, loader: DataLoader[dict[str, torch.Tensor]], device: torch.device
) -> dict[str, np.ndarray]:
    model.eval()
    chunks: dict[str, list[np.ndarray]] = {
        "rates": [],
        "factors": [],
        "latents": [],
        "drift": [],
        "diffusion": [],
        "trial_ids": [],
    }
    with torch.no_grad():
        for batch in loader:
            output = model(batch["heldin_spikes"].to(device), deterministic=True)
            chunks["rates"].append(output["rates_hz"].detach().cpu().numpy())
            chunks["factors"].append(output["factors"].detach().cpu().numpy())
            chunks["latents"].append(output["latents"].detach().cpu().numpy())
            chunks["drift"].append(output["drift"].detach().cpu().numpy())
            chunks["diffusion"].append(output["diffusion"].detach().cpu().numpy())
            chunks["trial_ids"].append(batch["trial_id"].detach().cpu().numpy())
    return {key: np.concatenate(value) for key, value in chunks.items()}


def _loader(
    dataset: NeuralDataset, trial_ids: np.ndarray, mask: NeuronMask, batch_size: int, name: str
) -> DataLoader[dict[str, torch.Tensor]]:
    return DataLoader(
        NeuralTrialDataset(dataset, trial_ids, mask, dataset.spikes.shape[1], name),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def _inner_scorer(
    dataset: NeuralDataset,
    train_trials: np.ndarray,
    validation_trials: np.ndarray,
    heldout: np.ndarray,
    validation_loader: DataLoader[dict[str, torch.Tensor]],
    scoring: ScoringConfig,
    device: torch.device,
) -> Any:
    train_counts = _counts(dataset, train_trials, heldout)
    validation_counts = _counts(dataset, validation_trials, heldout)
    reference = train_heldout_mean_rate_reference(train_counts, validation_counts.shape, scoring)

    def score(model: NeuralSDE) -> float:
        predicted = _predict_dynamics(model, validation_loader, device)["rates"][:, :, heldout]
        row = score_heldout_prediction(
            validation_counts,
            predicted,
            reference,
            scoring,
            "neural_ode",
            "inner_validation",
            "direct_model",
            True,
        )
        return float(row["bits_per_spike"])

    return score


def _concrete_training(config: dict[str, Any], bin_size_ms: int) -> dict[str, Any]:
    training = config["training"]
    return {
        "dataset": {"bin_size_ms": bin_size_ms},
        "evaluation": {"evaluate_splits": ["train", "validation"]},
        "training": {
            "learning_rate": float(training["learning_rate"]),
            "weight_decay": float(training["weight_decay"]),
            "gradient_clip_norm": float(training["gradient_clip_norm"]),
            "epochs": int(training["epochs"]),
            "heldin_loss_weight": float(training["heldin_loss_weight"]),
            "heldout_loss_weight": float(training["heldout_loss_weight"]),
            "loss_normalization": str(training["loss_normalization"]),
            "kl_scale": float(training["kl_scale"]),
            "kl_warmup_epochs": int(training["kl_warmup_epochs"]),
            "drift_regularization_scale": float(training["drift_regularization"]),
            "input_dropout": {
                "enabled": float(training["input_dropout_rate"]) > 0.0,
                "rate": float(training["input_dropout_rate"]),
                "apply_to": ["train"],
                "keep_at_least_one_neuron": True,
                "seed": int(config["outer_protocol"]["repeat_index"]),
            },
        },
    }


def _train_neural_ode(
    model: NeuralSDE,
    dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
    concrete: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    checkpoint_scorer: Any,
    early_stopping_patience: int,
    minimum_epochs: int,
    snapshot: dict[str, Any],
) -> NeuralODETrainingState:
    training = concrete["training"]
    base_lr = float(training["learning_rate"])
    epochs = int(training["epochs"])
    bin_size_ms = int(concrete["dataset"]["bin_size_ms"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr, weight_decay=float(training["weight_decay"])
    )
    model.to(device)
    checkpoint_dir = output_dir / "checkpoints"
    input_dropout = _input_dropout_settings(training)
    dropout_generator = _make_dropout_generator(device, int(input_dropout.get("seed", 0)))
    state = NeuralODETrainingState()
    epochs_without_improvement = 0
    for epoch in range(epochs):
        learning_rate = base_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / max(epochs, 1)))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        kl_beta = linear_warmup(epoch, int(training["kl_warmup_epochs"])) * float(
            training["kl_scale"]
        )
        train_losses: list[float] = []
        grad_norms: list[float] = []
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
                summarize_dropout_mask(dropout_mask)
            loss, _, _ = _loss_for_batch(
                model, batch, device, bin_size_ms, kl_beta, training, model_input
            )
            _ensure_finite("training loss", loss["loss"])
            torch.autograd.backward(loss["loss"])
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            grad_norms.append(float(grad_norm))
            optimizer.step()
            train_losses.append(float(loss["loss"].detach().cpu()))
        validation_metrics = _split_metrics(
            model, dataloaders["validation"], device, bin_size_ms, kl_beta, training
        )
        inner_metric = float(checkpoint_scorer(model))
        row: dict[str, float] = {
            "epoch": float(epoch),
            "kl_beta": float(kl_beta),
            "learning_rate": float(learning_rate),
            "train_total_loss": _mean(train_losses),
            "validation_total_loss": validation_metrics["total_loss"],
            "inner_validation_unified_bits_per_spike": inner_metric,
            "drift_norm": validation_metrics["drift_norm"],
            "gradient_norm": _mean(grad_norms),
        }
        state.history.append(row)
        state.final_gradient_norm = _mean(grad_norms)
        save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, epoch, row, snapshot)
        if math.isnan(state.best_metric) or inner_metric > state.best_metric:
            state.best_metric = inner_metric
            state.best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_dir / "best_validation.pt", model, optimizer, epoch, row, snapshot
            )
        else:
            epochs_without_improvement += 1
        if epoch + 1 >= minimum_epochs and epochs_without_improvement >= early_stopping_patience:
            state.early_stopping_triggered = True
            break
    pd.DataFrame(state.history).to_csv(output_dir / "metrics_history.csv", index=False)
    return state


def _solver_diagnostics(prediction: dict[str, np.ndarray]) -> dict[str, float]:
    latents = np.asarray(prediction["latents"], dtype=np.float64)
    drift = np.asarray(prediction["drift"], dtype=np.float64)
    state_norms = np.linalg.norm(latents, axis=-1)
    drift_norms = np.linalg.norm(drift, axis=-1)
    terminal = np.linalg.norm(latents[:, -1, :], axis=-1)
    nonfinite = int(np.count_nonzero(~np.isfinite(latents)))
    return {
        "integration_steps": int(latents.shape[1]),
        "nonfinite_state_count": nonfinite,
        "maximum_state_norm": float(state_norms.max()),
        "mean_state_norm": float(state_norms.mean()),
        "maximum_drift_norm": float(drift_norms.max()),
        "mean_drift_norm": float(drift_norms.mean()),
        "terminal_state_norm": float(terminal.mean()),
    }


def _latent_diagnostic_rows(
    prediction: dict[str, np.ndarray], fold_index: int, seed: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for representation, tensor in (
        ("factor", prediction["factors"]),
        ("latent", prediction["latents"]),
    ):
        matrix = np.asarray(tensor, dtype=np.float64)
        flat = matrix.reshape(-1, matrix.shape[-1])
        rank, rank_fraction, eigenvalues = effective_rank(flat)
        variances = np.var(flat, axis=0)
        maximum = float(np.max(variances)) if variances.size else 0.0
        near_zero = variances <= max(maximum * 1.0e-6, np.finfo(np.float64).eps)
        first_diff = float(np.mean(np.square(np.diff(matrix, axis=1))))
        second_diff = float(np.mean(np.square(np.diff(matrix, n=2, axis=1))))
        for dimension, variance in enumerate(variances):
            rows.append(
                {
                    "fold_index": fold_index,
                    "initialization_seed": seed,
                    "representation": representation,
                    "dimension": int(dimension),
                    "variance": float(variance),
                    "covariance_eigenvalue": float(eigenvalues[dimension]),
                    "effective_rank": rank,
                    "effective_rank_fraction": rank_fraction,
                    "near_zero_variance_dimensions": int(np.sum(near_zero)),
                    "near_zero_variance_fraction": float(np.mean(near_zero)),
                    "temporal_first_difference_variance": first_diff,
                    "temporal_second_difference_variance": second_diff,
                }
            )
    return pd.DataFrame(rows)


def _near_peak_scores(
    dataset: NeuralDataset,
    fold: Any,
    prediction: dict[str, np.ndarray],
    reference: np.ndarray,
    scoring: ScoringConfig,
) -> dict[str, float]:
    counts = _counts(dataset, fold.eval_trials, fold.heldout)
    predicted = prediction["rates"][:, :, fold.heldout]
    speed = np.zeros(counts.shape[1], dtype=np.float64)
    frame = time_bin_diagnostics(
        counts, predicted, reference, speed, dataset.bin_size_ms, int(fold.fold_index), 0
    )
    before = frame[frame["relative_time_seconds"] < -0.10]["unified_bits_per_spike"]
    peak = frame[frame["relative_time_seconds"].abs() <= 0.10]["unified_bits_per_spike"]
    after = frame[frame["relative_time_seconds"] > 0.10]["unified_bits_per_spike"]
    return {
        "before_peak": float(before.mean()),
        "near_peak": float(peak.mean()),
        "after_peak": float(after.mean()),
    }


def _train_one(
    run: PilotRun,
    fold: Any,
    dataset: NeuralDataset,
    assignments: pd.DataFrame,
    baseline_score: float,
    lfads_reference_mean: float,
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], pd.DataFrame, dict[str, Any], dict[str, float]
]:
    inner_seed = int(config["inner_checkpoint_selection"]["split_seed_base"]) + run.fold_index
    inner_train, inner_validation = build_inner_split(
        fold.train_trials,
        assignments,
        float(config["inner_checkpoint_selection"]["validation_fraction"]),
        inner_seed,
    )
    if (
        np.intersect1d(inner_train, fold.eval_trials).size
        or np.intersect1d(inner_validation, fold.eval_trials).size
    ):
        msg = "outer-evaluation trials entered the inner checkpoint split"
        raise RuntimeError(msg)
    validate_input_target_separation(
        fold.heldin, fold.heldout, int(fold.heldin.size), EXPECTED_SHAPE[2]
    )
    if (
        int(fold.heldin.size) != INPUT_NEURON_COUNT
        or int(fold.heldout.size) != HELDOUT_NEURON_COUNT
    ):
        msg = "repeat-0 mask must expose exactly 122 held-in and 40 held-out neurons"
        raise RuntimeError(msg)
    mask = _mask_from_fold(fold.heldin, fold.heldout, EXPECTED_SHAPE[2])
    split = TrialSplit(train=inner_train, validation=inner_validation, test=fold.eval_trials)
    batch_size = int(config["training"]["batch_size"])
    loaders = create_dataloaders(
        create_torch_datasets(dataset, split, mask, EXPECTED_SHAPE[1]),
        batch_size=batch_size,
        num_workers=int(config["training"]["num_workers"]),
        drop_last=False,
        seed=run.initialization_seed,
    )
    run_dir = output_dir / "runs" / run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    input_dim = int(fold.heldin.size)
    output_dim = EXPECTED_SHAPE[2]
    snapshot = {
        "dataset": {"name": config["dataset"]["name"], "bin_size_ms": dataset.bin_size_ms},
        "model": {
            "name": "deterministic_neural_ode",
            "input_dim": input_dim,
            "resolved_output_dim": output_dim,
            "diffusion_scale": 0.0,
        },
        "pilot": {
            "repeat_index": run.repeat_index,
            "fold_index": run.fold_index,
            "initialization_seed": run.initialization_seed,
            "inner_split_seed": inner_seed,
            "selection_split": "inner_validation",
            "outer_evaluation_used_for_selection": False,
            "normalization_fit_trials": inner_train.tolist(),
        },
    }
    model = build_pilot_model(config, input_dim, output_dim, run.initialization_seed)
    train_spikes = dataset.spikes[_trial_mask(dataset, inner_train)]
    mean_rates = compute_train_mean_rates_hz(
        train_spikes,
        dataset.bin_size_ms,
        float(config["model"]["min_rate_hz"]),
        float(config["model"]["max_rate_hz"]),
    )
    model.initialize_output_bias_from_rates(torch.as_tensor(mean_rates, dtype=torch.float32))
    scoring = _scoring(config)
    scorer = _inner_scorer(
        dataset, inner_train, inner_validation, fold.heldout, loaders["validation"], scoring, device
    )
    concrete = _concrete_training(config, dataset.bin_size_ms)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    state = _train_neural_ode(
        model,
        loaders,
        concrete,
        run_dir,
        device,
        scorer,
        int(config["training"]["early_stopping_patience"]),
        int(config["training"]["minimum_epochs"]),
        snapshot,
    )
    training_seconds = time.perf_counter() - started
    peak_memory = (
        float(torch.cuda.max_memory_allocated(device) / (1024.0**2))
        if device.type == "cuda"
        else 0.0
    )
    checkpoint = run_dir / "checkpoints" / "best_validation.pt"
    load_checkpoint(checkpoint, model, map_location=device)
    selected = {
        "repeat_index": run.repeat_index,
        "fold_index": run.fold_index,
        "initialization_seed": run.initialization_seed,
        "epoch": state.best_epoch,
        "checkpoint_type": "best",
        "selection_split": "inner_validation",
        "selection_metric": SELECTION_METRIC,
        "selection_metric_value": state.best_metric,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "model_config_digest": _model_config_digest(config, input_dim, output_dim),
        "solver_config_digest": _solver_config_digest(config),
    }
    validate_checkpoint_record(selected)

    train_loader = _loader(dataset, fold.train_trials, mask, batch_size, "outer_train")
    eval_loader = _loader(dataset, fold.eval_trials, mask, batch_size, "outer_evaluation")
    outer_train_prediction = _predict_dynamics(model, train_loader, device)
    integration_started = time.perf_counter()
    outer_prediction = _predict_dynamics(model, eval_loader, device)
    integration_seconds = time.perf_counter() - integration_started

    all_rates = outer_prediction["rates"]
    expected_shape = (fold.eval_trials.size, EXPECTED_SHAPE[1], EXPECTED_SHAPE[2])
    if all_rates.shape != expected_shape:
        msg = f"outer model output has shape {all_rates.shape}, expected {expected_shape}"
        raise RuntimeError(msg)
    solver = _solver_diagnostics(outer_prediction)
    dynamics = config["dynamics"]
    if bool(dynamics["fail_on_nonfinite_state"]) and (
        solver["nonfinite_state_count"] or not np.isfinite(all_rates).all()
    ):
        msg = "solver produced non-finite latent states or rates"
        raise RuntimeError(msg)
    if np.any(all_rates <= 0.0):
        msg = "outer model rates must be strictly positive"
        raise RuntimeError(msg)
    if solver["maximum_state_norm"] > float(dynamics["maximum_state_norm"]):
        msg = "latent state norm exceeded the configured hard limit"
        raise RuntimeError(msg)
    if solver["maximum_drift_norm"] > float(dynamics["maximum_drift_norm"]):
        msg = "drift norm exceeded the configured hard limit"
        raise RuntimeError(msg)

    outer_counts = _counts(dataset, fold.eval_trials, fold.heldout)
    train_counts = _counts(dataset, fold.train_trials, fold.heldout)
    reference = train_heldout_mean_rate_reference(train_counts, outer_counts.shape, scoring)
    scored = score_heldout_prediction(
        outer_counts,
        all_rates[:, :, fold.heldout],
        reference,
        scoring,
        "neural_ode",
        "outer_evaluation",
        "direct_model",
        True,
    )
    near_peak = _near_peak_scores(dataset, fold, outer_prediction, reference, scoring)
    _behavior_mean_r2(dataset, outer_train_prediction, outer_prediction)
    latent_frame = _latent_diagnostic_rows(
        outer_prediction, int(fold.fold_index), run.initialization_seed
    )
    final = state.history[-1]
    row = {
        "repeat_index": run.repeat_index,
        "fold_index": run.fold_index,
        "split_seed": int(fold.split_seed),
        "neuron_mask_seed": int(fold.neuron_mask_seed),
        "initialization_seed": run.initialization_seed,
        "status": "completed",
        "best_epoch": state.best_epoch,
        "checkpoint_source": "inner_validation",
        "inner_validation_unified_bits_per_spike": state.best_metric,
        "outer_unified_bits_per_spike": float(scored["bits_per_spike"]),
        "outer_poisson_nll": float(scored["poisson_nll"]),
        "baseline_outer_unified_bits_per_spike": baseline_score,
        "paired_difference_vs_baseline": float(scored["bits_per_spike"]) - baseline_score,
        "lfads_outer_mean_reference": lfads_reference_mean,
        "training_seconds": training_seconds,
        "integration_seconds": integration_seconds,
        "peak_cuda_memory_mb": peak_memory,
        "final_train_loss": float(final["train_total_loss"]),
        "final_inner_validation_loss": float(final["validation_total_loss"]),
        "maximum_state_norm": solver["maximum_state_norm"],
        "mean_state_norm": solver["mean_state_norm"],
        "maximum_drift_norm": solver["maximum_drift_norm"],
        "mean_drift_norm": solver["mean_drift_norm"],
        "solver_failure_count": 0,
        "nonfinite_state_count": solver["nonfinite_state_count"],
        "notes": "outer evaluation performed once after inner-validation checkpoint selection",
    }
    resource = {
        "repeat_index": run.repeat_index,
        "fold_index": run.fold_index,
        "initialization_seed": run.initialization_seed,
        "training_seconds": training_seconds,
        "integration_seconds": integration_seconds,
        "best_epoch": state.best_epoch,
        "epochs_completed": len(state.history),
        "peak_cuda_memory_mb": peak_memory,
        "batch_size": batch_size,
        "mixed_precision_enabled": bool(config["training"]["mixed_precision"]),
        "early_stopping_triggered": state.early_stopping_triggered,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
    }
    solver_row = {
        "repeat_index": run.repeat_index,
        "fold_index": run.fold_index,
        "initialization_seed": run.initialization_seed,
        "solver": str(dynamics["solver"]),
        "integration_step_seconds": float(dynamics["integration_step_seconds"]),
        "integration_steps": solver["integration_steps"],
        "solver_failure_count": 0,
        "nonfinite_state_count": solver["nonfinite_state_count"],
        "maximum_state_norm": solver["maximum_state_norm"],
        "mean_state_norm": solver["mean_state_norm"],
        "maximum_drift_norm": solver["maximum_drift_norm"],
        "mean_drift_norm": solver["mean_drift_norm"],
        "terminal_state_norm": solver["terminal_state_norm"],
        "gradient_norm": state.final_gradient_norm,
        "integration_seconds": integration_seconds,
    }
    near_peak_row = {
        "fold_index": int(fold.fold_index),
        "initialization_seed": run.initialization_seed,
        **near_peak,
    }
    return row, selected, resource, latent_frame, solver_row, near_peak_row


NARROW_ACTIONABLE_BAND = 0.05


def build_full_evaluation_recommendation(
    runs: pd.DataFrame, config: dict[str, Any], leakage_checks_passed: bool
) -> dict[str, Any]:
    gates = config["pilot_gates"]
    expected = len(EXPECTED_FOLDS) * len(EXPECTED_SEEDS)
    completed = runs[runs["status"] == "completed"] if "status" in runs else runs.iloc[:0]
    failed_fraction = 1.0 - len(completed) / expected
    scores_finite = _finite(completed, ["outer_unified_bits_per_spike"])
    losses_finite = _finite(completed, ["final_train_loss", "final_inner_validation_loss"])
    checkpoint_valid = bool(
        len(completed) == expected
        and (completed["checkpoint_source"].astype(str) == "inner_validation").all()
    )
    nonfinite_total = (
        float(completed["nonfinite_state_count"].sum())
        if "nonfinite_state_count" in completed and len(completed)
        else 0.0
    )
    solver_failures = (
        float(completed["solver_failure_count"].sum())
        if "solver_failure_count" in completed and len(completed)
        else 0.0
    )
    nonfinite_run_fraction = (
        float((completed["nonfinite_state_count"] > 0).mean())
        if "nonfinite_state_count" in completed and len(completed)
        else 0.0
    )
    solver_stability_passed = bool(
        len(completed) == expected and nonfinite_total == 0.0 and solver_failures == 0.0
    )
    mean_score = (
        float(completed["outer_unified_bits_per_spike"].mean()) if scores_finite else float("nan")
    )
    score_std = (
        float(completed["outer_unified_bits_per_spike"].std(ddof=1))
        if scores_finite
        else float("nan")
    )
    seed_means = (
        completed.groupby("initialization_seed")["outer_unified_bits_per_spike"].mean()
        if scores_finite
        else pd.Series(dtype=float)
    )
    seed_mean_std = float(seed_means.std(ddof=1)) if len(seed_means) > 1 else float("nan")
    positive_seed_fraction = float((seed_means > 0.0).mean()) if len(seed_means) else 0.0
    positive_run_fraction = (
        float((completed["outer_unified_bits_per_spike"] > 0.0).mean()) if scores_finite else 0.0
    )
    mean_difference = (
        float(completed["paired_difference_vs_baseline"].mean())
        if _finite(completed, ["paired_difference_vs_baseline"])
        else float("nan")
    )
    total_seconds = (
        float(completed["training_seconds"].sum())
        if _finite(completed, ["training_seconds"])
        else float("nan")
    )
    peak_memory = (
        float(completed["peak_cuda_memory_mb"].max())
        if _finite(completed, ["peak_cuda_memory_mb"])
        else float("nan")
    )
    margin = float(gates["full_evaluation_margin_over_baseline"])
    checks = {
        "all 25 runs completed": len(completed) == expected,
        "failed run fraction is allowed": failed_fraction
        <= float(gates["maximum_failed_run_fraction"]),
        "all scores are finite": scores_finite,
        "all losses are finite": losses_finite,
        "checkpoint selection uses inner validation": checkpoint_valid,
        "leakage checks passed": bool(leakage_checks_passed),
        "solver stability passed": solver_stability_passed,
        "nonfinite run fraction is allowed": nonfinite_run_fraction
        <= float(gates["maximum_nonfinite_run_fraction"]),
        "mean score is non-negative": bool(
            np.isfinite(mean_score)
            and mean_score >= float(gates["minimum_mean_unified_bits_per_spike"])
        ),
        "positive seed fraction clears gate": positive_seed_fraction
        >= float(gates["minimum_positive_seed_fraction"]),
        "seed-mean standard deviation clears gate": bool(
            np.isfinite(seed_mean_std) and seed_mean_std <= float(gates["maximum_seed_mean_std"])
        ),
        "mean paired difference clears margin": bool(
            np.isfinite(mean_difference) and mean_difference >= margin
        ),
    }
    reasons = [f"failed: {name}" for name, passed in checks.items() if not passed] or [
        "all gates passed"
    ]
    return {
        "proceed": bool(all(checks.values())),
        "all_runs_completed": len(completed) == expected,
        "failed_run_fraction": failed_fraction,
        "mean_unified_bits_per_spike": mean_score,
        "run_level_score_std": score_std,
        "seed_mean_std": seed_mean_std,
        "positive_run_fraction": positive_run_fraction,
        "positive_seed_fraction": positive_seed_fraction,
        "mean_paired_difference_vs_baseline": mean_difference,
        "runs_beating_baseline": int((completed["paired_difference_vs_baseline"] > 0.0).sum())
        if _finite(completed, ["paired_difference_vs_baseline"])
        else 0,
        "checkpoint_selection_valid": checkpoint_valid,
        "leakage_checks_passed": bool(leakage_checks_passed),
        "solver_stability_passed": solver_stability_passed,
        "nonfinite_run_fraction": nonfinite_run_fraction,
        "runtime_estimate_full_evaluation_hours": total_seconds * 5.0 / 3600.0,
        "estimated_peak_cuda_memory_mb": peak_memory,
        "reasons": reasons,
        "_checks": checks,
        "pilot_final_claim_allowed": False,
    }


def build_next_action_recommendation(
    recommendation: dict[str, Any], config: dict[str, Any], near_peak: dict[str, float]
) -> dict[str, Any]:
    gates = config["pilot_gates"]
    checks = dict(recommendation["_checks"])
    integrity_ok = bool(
        checks["all 25 runs completed"]
        and checks["all scores are finite"]
        and checks["all losses are finite"]
        and checks["checkpoint selection uses inner validation"]
        and checks["leakage checks passed"]
        and checks["solver stability passed"]
    )
    mean_score = float(recommendation["mean_unified_bits_per_spike"])
    gap = float(recommendation["mean_paired_difference_vs_baseline"])
    margin = float(gates["full_evaluation_margin_over_baseline"])
    positive_and_stable = bool(
        np.isfinite(mean_score)
        and mean_score >= float(gates["minimum_mean_unified_bits_per_spike"])
        and float(recommendation["positive_seed_fraction"])
        >= float(gates["minimum_positive_seed_fraction"])
        and np.isfinite(recommendation["seed_mean_std"])
        and float(recommendation["seed_mean_std"]) <= float(gates["maximum_seed_mean_std"])
    )
    non_margin_gates = {
        name: value
        for name, value in checks.items()
        if name != "mean paired difference clears margin"
    }
    only_margin_failed = bool(
        all(non_margin_gates.values()) and not checks["mean paired difference clears margin"]
    )
    targeted_available = bool(
        positive_and_stable
        and only_margin_failed
        and np.isfinite(gap)
        and gap >= margin - NARROW_ACTIONABLE_BAND
    )
    if not integrity_ok:
        action = "block_due_to_integrity_issue"
        rationale = "protocol, leakage, checkpoint, or solver-stability checks failed"
    elif bool(recommendation["proceed"]):
        action = "run_full_neural_ode_evaluation"
        rationale = "all predeclared full-evaluation gates passed"
    elif targeted_available:
        action = "run_targeted_neural_ode_diagnostic"
        rationale = "model is positive and stable and one narrow correction could clear the margin"
    else:
        action = "retire_neural_ode_and_close_neural_model_search"
        rationale = (
            "pipeline is correct and results are stable, but the baseline gap is large and no "
            "narrow correction is justified; progress would require broad architecture tuning"
        )
    secondary: list[str] = []
    if np.isfinite(near_peak.get("near_peak", float("nan"))):
        secondary.append(f"near_peak_bits_per_spike={near_peak['near_peak']:.6f}")
    dominant = (
        f"mean unified bits/spike {mean_score:.6f}; mean paired difference vs baseline {gap:.6f}"
    )
    return {
        "recommended_next_action": action,
        "integrity_checks_passed": integrity_ok,
        "pilot_gate_passed": bool(recommendation["proceed"]),
        "dominant_observation": dominant,
        "secondary_observations": secondary,
        "targeted_diagnostic_available": targeted_available,
        "full_evaluation_allowed": bool(recommendation["proceed"]),
        "final_claim_allowed": False,
        "rationale": rationale,
    }


def _aggregate(
    runs: pd.DataFrame,
    checkpoints: pd.DataFrame,
    resources: pd.DataFrame,
    solvers: pd.DataFrame,
    latents: pd.DataFrame,
    near_peaks: pd.DataFrame,
    config: dict[str, Any],
    inputs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, Any], dict[str, Any], dict[str, Any]]:
    completed = runs[runs["status"] == "completed"].copy()
    lfads_reference = float(config["lfads_reference"]["pilot_mean"])
    fold_seed = completed[
        [
            "fold_index",
            "initialization_seed",
            "outer_unified_bits_per_spike",
            "baseline_outer_unified_bits_per_spike",
            "paired_difference_vs_baseline",
        ]
    ].copy()
    fold_seed["lfads_descriptive_reference"] = lfads_reference
    fold_seed["positive_score"] = fold_seed["outer_unified_bits_per_spike"] > 0.0
    fold_seed["beats_baseline"] = fold_seed["paired_difference_vs_baseline"] > 0.0
    fold_seed["beats_lfads_descriptive_reference"] = (
        fold_seed["outer_unified_bits_per_spike"] > lfads_reference
    )

    seed_rows = []
    for seed, group in fold_seed.groupby("initialization_seed", sort=True):
        resource = resources[resources["initialization_seed"] == seed]
        solver = solvers[solvers["initialization_seed"] == seed]
        seed_rows.append(
            {
                "initialization_seed": int(seed),
                "completed_folds": len(group),
                "mean_outer_unified_bits_per_spike": float(
                    group["outer_unified_bits_per_spike"].mean()
                ),
                "std_outer_unified_bits_per_spike": float(
                    group["outer_unified_bits_per_spike"].std(ddof=1)
                ),
                "mean_paired_difference_vs_baseline": float(
                    group["paired_difference_vs_baseline"].mean()
                ),
                "positive_fold_fraction": float(group["positive_score"].mean()),
                "beats_baseline_fold_fraction": float(group["beats_baseline"].mean()),
                "mean_training_seconds": float(resource["training_seconds"].mean()),
                "maximum_peak_cuda_memory_mb": float(resource["peak_cuda_memory_mb"].max()),
                "mean_maximum_state_norm": float(solver["maximum_state_norm"].mean()),
                "mean_maximum_drift_norm": float(solver["maximum_drift_norm"].mean()),
            }
        )
    seed_summary = pd.DataFrame(seed_rows)

    fold_rows = []
    statistics = config["statistics"]
    for fold, group in fold_seed.groupby("fold_index", sort=True):
        values = group["outer_unified_bits_per_spike"].to_numpy(dtype=np.float64)
        low, high = bootstrap_mean_ci(
            values,
            int(statistics["bootstrap_repeats"]),
            float(statistics["confidence_interval"]),
            int(statistics["bootstrap_seed"]) + int(fold),
        )
        fold_rows.append(
            {
                "fold_index": int(fold),
                "completed_seeds": len(group),
                "mean_outer_unified_bits_per_spike": float(np.mean(values)),
                "std_across_seeds": float(np.std(values, ddof=1)),
                "ci95_low": low,
                "ci95_high": high,
                "baseline_outer_unified_bits_per_spike": float(
                    group["baseline_outer_unified_bits_per_spike"].iloc[0]
                ),
                "mean_paired_difference_vs_baseline": float(
                    group["paired_difference_vs_baseline"].mean()
                ),
                "positive_seed_fraction": float(group["positive_score"].mean()),
                "beats_baseline_seed_fraction": float(group["beats_baseline"].mean()),
            }
        )
    fold_summary = pd.DataFrame(fold_rows)
    paired = fold_seed[
        ["fold_index", "initialization_seed", "paired_difference_vs_baseline"]
    ].copy()
    lfads_descriptive = fold_seed[
        [
            "fold_index",
            "initialization_seed",
            "outer_unified_bits_per_spike",
            "lfads_descriptive_reference",
            "beats_lfads_descriptive_reference",
        ]
    ].copy()

    leakage_checks_passed = bool(
        len(completed) == 25 and (completed["checkpoint_source"] == "inner_validation").all()
    )
    recommendation = build_full_evaluation_recommendation(runs, config, leakage_checks_passed)
    near_peak_means = {
        "before_peak": float(near_peaks["before_peak"].mean()) if len(near_peaks) else float("nan"),
        "near_peak": float(near_peaks["near_peak"].mean()) if len(near_peaks) else float("nan"),
        "after_peak": float(near_peaks["after_peak"].mean()) if len(near_peaks) else float("nan"),
    }
    next_action = build_next_action_recommendation(recommendation, config, near_peak_means)

    factor_latents = latents[latents["representation"] == "factor"] if len(latents) else latents
    mean_effective_rank = (
        float(factor_latents["effective_rank"].mean()) if len(factor_latents) else float("nan")
    )
    mean_effective_rank_fraction = (
        float(factor_latents["effective_rank_fraction"].mean())
        if len(factor_latents)
        else float("nan")
    )
    lfads_near_peak = {
        "before_peak": 0.03636918045052086,
        "near_peak": 0.0026472081505686494,
        "after_peak": 0.03198509446884249,
    }
    if not np.isfinite(near_peak_means["near_peak"]):
        near_peak_status = "unresolved"
    elif near_peak_means["near_peak"] > lfads_near_peak["near_peak"] + 0.005:
        near_peak_status = (
            "attenuated"
            if near_peak_means["near_peak"]
            < min(near_peak_means["before_peak"], near_peak_means["after_peak"]) - 0.005
            else "absent"
        )
    else:
        near_peak_status = "reproduced"

    summary = {
        "dataset_name": config["dataset"]["name"],
        "dataset_hash": config["dataset"]["expected_hash"],
        "data_shape": list(EXPECTED_SHAPE),
        "behavior_shape": [500, 64, 4],
        "input_neuron_count": INPUT_NEURON_COUNT,
        "output_neuron_count": EXPECTED_SHAPE[2],
        "heldout_neuron_count": HELDOUT_NEURON_COUNT,
        "repeat_index": EXPECTED_REPEAT,
        "fold_indices": EXPECTED_FOLDS,
        "initialization_seeds": EXPECTED_SEEDS,
        "scheduled_runs": 25,
        "completed_runs": len(completed),
        "failed_runs": int((runs["status"] == "failed").sum()),
        "diffusion_enabled": False,
        "solver": str(config["dynamics"]["solver"]),
        "integration_step_seconds": float(config["dynamics"]["integration_step_seconds"]),
        "mean_unified_bits_per_spike": recommendation["mean_unified_bits_per_spike"],
        "run_level_score_std": recommendation["run_level_score_std"],
        "seed_mean_std": recommendation["seed_mean_std"],
        "positive_run_fraction": recommendation["positive_run_fraction"],
        "positive_seed_fraction": recommendation["positive_seed_fraction"],
        "runs_beating_baseline": recommendation["runs_beating_baseline"],
        "pilot_repeat_baseline_mean": float(np.mean(list(inputs["baseline_by_fold"].values()))),
        "mean_paired_difference_vs_baseline": recommendation["mean_paired_difference_vs_baseline"],
        "lfads_descriptive_reference_mean": lfads_reference,
        "mean_difference_vs_lfads_reference": (
            recommendation["mean_unified_bits_per_spike"] - lfads_reference
            if np.isfinite(recommendation["mean_unified_bits_per_spike"])
            else float("nan")
        ),
        "before_peak_mean_bits_per_spike": near_peak_means["before_peak"],
        "near_peak_mean_bits_per_spike": near_peak_means["near_peak"],
        "after_peak_mean_bits_per_spike": near_peak_means["after_peak"],
        "near_peak_failure_status": near_peak_status,
        "mean_factor_effective_rank": mean_effective_rank,
        "mean_factor_effective_rank_fraction": mean_effective_rank_fraction,
        "lfads_factor_effective_rank": 1.2161114061725022,
        "lfads_factor_effective_rank_fraction": 0.038003481442890695,
        "checkpoint_selection_split": "inner_validation",
        "checkpoint_selection_valid": recommendation["checkpoint_selection_valid"],
        "leakage_checks_passed": recommendation["leakage_checks_passed"],
        "solver_stability_passed": recommendation["solver_stability_passed"],
        "baseline_to_beat": config["baseline"]["method"],
        "full_evaluation_recommended": recommendation["proceed"],
        "recommended_next_action": next_action["recommended_next_action"],
        "single_split_results_reportable": False,
        "official_leaderboard_claim": False,
        "pilot_final_claim_allowed": False,
        "statement": (
            "This pilot assesses feasibility, stability, and dynamics on one held-out-neuron mask. "
            "It is not a final multi-repeat model comparison."
        ),
    }
    protocol = {
        "dataset": config["dataset"],
        "trial_source": config["trial_source"],
        "window": config["window"],
        "binning": config["binning"],
        "outer_protocol": config["outer_protocol"],
        "initialization": config["initialization"],
        "model": config["model"],
        "dynamics": config["dynamics"],
        "training": config["training"],
        "inner_checkpoint_selection": config["inner_checkpoint_selection"],
        "scoring": config["scoring"],
        "baseline": config["baseline"],
        "lfads_reference": config["lfads_reference"],
        "protocol_frozen": True,
        "outer_evaluation_used_for_selection": False,
        "diffusion_enabled": False,
        "pilot_final_claim_allowed": False,
    }
    recommendation.pop("_checks", None)
    tables = {
        "neural_ode_pilot_runs": runs[RUN_COLUMNS],
        "fold_seed_scores": fold_seed[
            [
                "fold_index",
                "initialization_seed",
                "outer_unified_bits_per_spike",
                "baseline_outer_unified_bits_per_spike",
                "paired_difference_vs_baseline",
                "lfads_descriptive_reference",
                "positive_score",
                "beats_baseline",
                "beats_lfads_descriptive_reference",
            ]
        ],
        "seed_summary": seed_summary,
        "fold_summary": fold_summary,
        "paired_baseline_comparison": paired,
        "lfads_descriptive_comparison": lfads_descriptive,
        "checkpoint_manifest": checkpoints.reindex(columns=CHECKPOINT_COLUMNS),
        "solver_diagnostics": solvers.reindex(columns=SOLVER_COLUMNS),
        "latent_diagnostics": latents,
        "training_resource_summary": resources,
    }
    return summary, tables, protocol, recommendation, next_action


def run_neural_ode_pilot(config: dict[str, Any]) -> dict[str, Any]:
    """Run or resume the fixed 25-run repeat-0 deterministic neural-ODE feasibility pilot."""
    validate_neural_ode_pilot_config(config)
    inputs = _load_protocol_inputs(config)
    if not torch.cuda.is_available():
        msg = "CUDA was requested, but torch.cuda.is_available() is False."
        raise RuntimeError(msg)
    device = torch.device("cuda")
    output_dir = _resolve(str(config["reporting"]["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / "neural_ode_pilot_runs.csv"
    checkpoint_path = output_dir / "checkpoint_manifest.csv"
    resource_path = output_dir / "training_resource_summary.csv"
    solver_path = output_dir / "solver_diagnostics.csv"
    latent_path = output_dir / "latent_diagnostics.csv"
    near_peak_path = output_dir / "near_peak_scores.csv"
    runs = pd.read_csv(runs_path) if runs_path.exists() else pd.DataFrame(columns=RUN_COLUMNS)
    checkpoints = (
        pd.read_csv(checkpoint_path)
        if checkpoint_path.exists()
        else pd.DataFrame(columns=CHECKPOINT_COLUMNS)
    )
    resources = pd.read_csv(resource_path) if resource_path.exists() else pd.DataFrame()
    solvers = (
        pd.read_csv(solver_path) if solver_path.exists() else pd.DataFrame(columns=SOLVER_COLUMNS)
    )
    latents = pd.read_csv(latent_path) if latent_path.exists() else pd.DataFrame()
    near_peaks = pd.read_csv(near_peak_path) if near_peak_path.exists() else pd.DataFrame()
    completed: set[tuple[int, int]] = set()
    for row in runs[runs["status"] == "completed"].itertuples(index=False):
        key = (int(row.fold_index), int(row.initialization_seed))
        manifest = checkpoints[
            (checkpoints["fold_index"] == key[0]) & (checkpoints["initialization_seed"] == key[1])
        ]
        if len(manifest) != 1:
            continue
        checkpoint_file = Path(str(manifest.iloc[0]["checkpoint_path"]))
        if checkpoint_file.exists() and checkpoint_sha256(checkpoint_file) == str(
            manifest.iloc[0]["checkpoint_sha256"]
        ):
            completed.add(key)
    folds = {fold.fold_index: fold for fold in inputs["folds"]}
    for run in build_pilot_run_schedule(config):
        if (run.fold_index, run.initialization_seed) in completed:
            continue
        for frame_name in ("runs", "checkpoints", "resources", "solvers"):
            frame = locals()[frame_name]
            if not frame.empty and "fold_index" in frame:
                mask = ~(
                    (frame["fold_index"] == run.fold_index)
                    & (frame["initialization_seed"] == run.initialization_seed)
                )
                if frame_name == "runs":
                    runs = frame[mask]
                elif frame_name == "checkpoints":
                    checkpoints = frame[mask]
                elif frame_name == "resources":
                    resources = frame[mask]
                else:
                    solvers = frame[mask]
        try:
            row, checkpoint, resource, latent_frame, solver_row, near_peak_row = _train_one(
                run,
                folds[run.fold_index],
                inputs["dataset"],
                inputs["assignments"],
                float(inputs["baseline_by_fold"][run.fold_index]),
                float(config["lfads_reference"]["pilot_mean"]),
                config,
                output_dir,
                device,
            )
        except Exception as exc:
            failed = dict.fromkeys(RUN_COLUMNS)
            failed.update(
                {
                    "repeat_index": run.repeat_index,
                    "fold_index": run.fold_index,
                    "initialization_seed": run.initialization_seed,
                    "status": "failed",
                    "notes": str(exc),
                }
            )
            runs = pd.concat([runs, pd.DataFrame([failed])], ignore_index=True)
            runs.to_csv(runs_path, index=False)
            raise
        runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
        checkpoints = pd.concat([checkpoints, pd.DataFrame([checkpoint])], ignore_index=True)
        resources = pd.concat([resources, pd.DataFrame([resource])], ignore_index=True)
        solvers = pd.concat([solvers, pd.DataFrame([solver_row])], ignore_index=True)
        if len(latents):
            latents = latents[
                ~(
                    (latents["fold_index"] == run.fold_index)
                    & (latents["initialization_seed"] == run.initialization_seed)
                )
            ]
        latents = pd.concat([latents, latent_frame], ignore_index=True)
        if len(near_peaks):
            near_peaks = near_peaks[
                ~(
                    (near_peaks["fold_index"] == run.fold_index)
                    & (near_peaks["initialization_seed"] == run.initialization_seed)
                )
            ]
        near_peaks = pd.concat([near_peaks, pd.DataFrame([near_peak_row])], ignore_index=True)
        runs.to_csv(runs_path, index=False)
        checkpoints.to_csv(checkpoint_path, index=False)
        resources.to_csv(resource_path, index=False)
        solvers.to_csv(solver_path, index=False)
        latents.to_csv(latent_path, index=False)
        near_peaks.to_csv(near_peak_path, index=False)

    summary, tables, protocol, recommendation, next_action = _aggregate(
        runs, checkpoints, resources, solvers, latents, near_peaks, config, inputs
    )
    from latentbrain.eval.reporting import write_neural_ode_pilot_outputs  # noqa: PLC0415

    write_neural_ode_pilot_outputs(
        output_dir, summary, tables, protocol, recommendation, next_action
    )
    return {
        "summary": summary,
        "tables": tables,
        "protocol": protocol,
        "recommendation": recommendation,
        "next_action": next_action,
        "output_dir": output_dir,
    }
