from __future__ import annotations

import copy
import csv
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
import yaml
from torch.utils.data import DataLoader

from latentbrain.data.io import compute_dataset_hash, load_neural_dataset
from latentbrain.data.rebinning import rebin_neural_dataset
from latentbrain.data.schemas import NeuralDataset, NeuronMask, TrialSplit
from latentbrain.data.splits import create_neuron_mask, create_trial_split
from latentbrain.data.validation import (
    validate_neural_dataset,
    validate_neuron_mask,
    validate_trial_split,
)
from latentbrain.eval.decoding import (
    apply_standardization,
    fit_ridge_decoder,
    predict_ridge_decoder,
    regression_metrics,
    standardize_train_apply,
)
from latentbrain.eval.lfads_controller_tuning import (
    build_controller_result_row,
    rank_controller_results,
    summarize_controller_tuning,
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
from latentbrain.eval.neural_predictions import (
    flatten_batch_time,
    reshape_flat_predictions,
    summarize_factor_activity,
)
from latentbrain.eval.rebinning import compute_window_bins_for_duration
from latentbrain.eval.scoring import (
    ScoringConfig,
    score_heldout_prediction,
    train_heldout_mean_rate_reference,
)
from latentbrain.eval.windowing import crop_neural_dataset_time
from latentbrain.models.lfads_controller import LFADSController, LFADSControllerConfig
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
from latentbrain.torch.schedules import linear_warmup


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def expand_controller_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        msg = "grid must contain at least one parameter"
        raise ValueError(msg)
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            msg = f"grid entry {key!r} must be a non-empty list"
            raise ValueError(msg)
    keys = list(grid)
    return [
        dict(zip(keys, values, strict=True)) for values in product(*(grid[key] for key in keys))
    ]


def _slug(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def make_controller_run_id(index: int, params: dict[str, Any]) -> str:
    if index < 0:
        msg = "index must be non-negative"
        raise ValueError(msg)
    return (
        f"run_{index:03d}_"
        f"enc{_slug(params.get('encoder_hidden_dim', 'na'))}_"
        f"ctrl{_slug(params.get('controller_hidden_dim', 'na'))}_"
        f"gen{_slug(params.get('generator_hidden_dim', 'na'))}_"
        f"lat{_slug(params.get('latent_dim', 'na'))}_"
        f"fac{_slug(params.get('factor_dim', 'na'))}_"
        f"u{_slug(params.get('inferred_input_dim', 'na'))}_"
        f"idr{_slug(params.get('input_dropout_rate', 'na'))}_"
        f"hw{_slug(params.get('heldout_loss_weight', 'na'))}_"
        f"kl{_slug(params.get('kl_scale', 'na'))}_"
        f"ukl{_slug(params.get('inferred_input_kl_scale', 'na'))}"
    )


def build_controller_train_config(
    base_config: dict[str, Any],
    run_params: dict[str, Any],
    run_output_dir: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    settings = dict(config["model"])
    target_bin = int(config["binning"]["target_bin_size_ms"])
    window_bins = int(config.get("_window_bins", 0)) or compute_window_bins_for_duration(
        float(config["window"]["duration_seconds"]), target_bin
    )
    config["dataset"]["bin_size_ms"] = target_bin
    config["data"] = {
        "input_neuron_group": "heldin",
        "target_neuron_group": "heldout",
        "max_time_bins": window_bins,
        "batch_size": int(settings["batch_size"]),
        "num_workers": 0,
        "drop_last": False,
    }
    config["model"] = {
        "name": "lfads_controller",
        "input_dim": None,
        "output_dim": "all",
        "encoder_hidden_dim": int(run_params["encoder_hidden_dim"]),
        "controller_hidden_dim": int(run_params["controller_hidden_dim"]),
        "generator_hidden_dim": int(run_params["generator_hidden_dim"]),
        "latent_dim": int(run_params["latent_dim"]),
        "factor_dim": int(run_params["factor_dim"]),
        "inferred_input_dim": int(run_params["inferred_input_dim"]),
        "dropout": float(settings["model_dropout"]),
        "min_rate_hz": float(settings["min_rate_hz"]),
        "max_rate_hz": float(settings["max_rate_hz"]),
    }
    config["training"] = {
        "seed": int(config["splits"]["seed"]),
        "epochs": int(run_params["epochs"]),
        "learning_rate": float(settings["learning_rate"]),
        "weight_decay": float(settings["weight_decay"]),
        "gradient_clip_norm": float(settings["gradient_clip_norm"]),
        "heldin_loss_weight": float(settings["heldin_loss_weight"]),
        "heldout_loss_weight": float(run_params["heldout_loss_weight"]),
        "kl_warmup_epochs": int(run_params["kl_warmup_epochs"]),
        "kl_scale": float(run_params["kl_scale"]),
        "inferred_input_kl_scale": float(run_params["inferred_input_kl_scale"]),
        "loss_normalization": str(settings["loss_normalization"]),
        "checkpoint_metric": str(settings["checkpoint_metric"]),
        "checkpoint_mode": str(settings["checkpoint_mode"]),
        "device": str(config["runtime"]["device"]),
    }
    input_dropout_rate = float(run_params["input_dropout_rate"])
    if input_dropout_rate > 0.0:
        config["training"]["input_dropout"] = {
            "enabled": True,
            "rate": input_dropout_rate,
            "apply_to": ["train"],
            "keep_at_least_one_neuron": True,
            "seed": int(config["splits"]["seed"]),
        }
    config["evaluation"] = {
        "evaluate_splits": list(base_config["evaluation"]["evaluate_splits"]),
        "primary_split": str(base_config["scoring"]["primary_split"]),
        "direct_model_primary": bool(base_config["evaluation"]["direct_model_primary"]),
        "also_evaluate_factor_decoder": bool(
            base_config["evaluation"]["also_evaluate_factor_decoder"]
        ),
        "behavior_decoder_enabled": bool(base_config["evaluation"]["behavior_decoder_enabled"]),
    }
    config["reporting"] = {"output_dir": str(run_output_dir)}
    return config


def _resolve_processed_path(config: dict[str, Any]) -> Path:
    repo_root = get_repo_root()
    processed_path = resolve_configured_path(str(config["dataset"]["processed_path"]), repo_root)
    if not processed_path.exists():
        msg = f"Processed dataset is missing: {_relative(processed_path, repo_root)}"
        raise FileNotFoundError(msg)
    return processed_path


def _validate_cuda(config: dict[str, Any]) -> str:
    if str(config["runtime"]["device"]) != "cuda":
        msg = "runtime.device must be cuda for controller-style LFADS-family tuning"
        raise ValueError(msg)
    if (
        bool(config["runtime"].get("fail_if_cuda_unavailable", True))
        and not torch.cuda.is_available()
    ):
        msg = "CUDA was requested, but torch.cuda.is_available() is False."
        raise RuntimeError(msg)
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"


def _load_dataset(config: dict[str, Any]) -> tuple[NeuralDataset, str, int]:
    processed_path = _resolve_processed_path(config)
    dataset = load_neural_dataset(processed_path)
    validate_neural_dataset(dataset)
    dataset_hash = compute_dataset_hash(dataset)
    expected = str(config["dataset"].get("expected_hash", ""))
    if expected and dataset_hash != expected:
        msg = f"Dataset hash mismatch: expected {expected}, got {dataset_hash}"
        raise ValueError(msg)
    target_bin = int(config["binning"]["target_bin_size_ms"])
    rebinned = rebin_neural_dataset(dataset, target_bin)
    window_bins = compute_window_bins_for_duration(
        float(config["window"]["duration_seconds"]), target_bin
    )
    windowed = crop_neural_dataset_time(rebinned, window_bins, str(config["window"]["crop_policy"]))
    return windowed, dataset_hash, window_bins


def _split_and_mask(
    dataset: NeuralDataset, config: dict[str, Any]
) -> tuple[TrialSplit, NeuronMask]:
    split = create_trial_split(
        dataset.trial_ids,
        float(config["splits"]["train_fraction"]),
        float(config["splits"]["validation_fraction"]),
        float(config["splits"]["test_fraction"]),
        seed=int(config["splits"]["seed"]),
    )
    mask = create_neuron_mask(
        dataset.spikes.shape[2],
        float(config["splits"]["heldout_neuron_fraction"]),
        seed=int(config["splits"]["seed"]),
    )
    validate_trial_split(split, dataset.trial_ids)
    validate_neuron_mask(mask, dataset.spikes.shape[2])
    return split, mask


def _trial_mask(dataset: NeuralDataset, trial_ids: np.ndarray) -> np.ndarray:
    return np.isin(dataset.trial_ids, trial_ids)


def _verify_reference_zero(
    dataset: NeuralDataset, split: TrialSplit, mask: NeuronMask, config: dict[str, Any]
) -> float:
    scoring = ScoringConfig(
        bin_size_ms=int(config["binning"]["target_bin_size_ms"]),
        include_poisson_constant=bool(config["scoring"]["include_poisson_constant"]),
        min_rate_hz=float(config["scoring"]["min_rate_hz"]),
        max_rate_hz=float(config["scoring"]["max_rate_hz"]),
        reference_name=str(config["scoring"]["reference_model"]),
    )
    train_counts = dataset.spikes[_trial_mask(dataset, split.train)][:, :, mask.heldout]
    validation_counts = dataset.spikes[_trial_mask(dataset, split.validation)][:, :, mask.heldout]
    reference = train_heldout_mean_rate_reference(train_counts, validation_counts.shape, scoring)
    row = score_heldout_prediction(
        validation_counts,
        reference,
        reference,
        scoring,
        "train_heldout_mean_rate",
        "validation",
        "train_mean_reference_as_model",
        True,
    )
    bits = float(row["bits_per_spike"])
    if abs(bits) > 1e-12:
        msg = "train-heldout mean-rate reference did not score 0.0 bits/spike against itself"
        raise RuntimeError(msg)
    return bits


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


def _loss_for_batch(
    model: LFADSController,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    bin_size_ms: int,
    z0_beta: float,
    input_beta: float,
    training_config: dict[str, Any],
    input_heldin_spikes: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
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
    input_kl = gaussian_kl_standard_normal(
        output["inferred_input_mean"].reshape(-1, model.config.inferred_input_dim),
        output["inferred_input_logvar"].reshape(-1, model.config.inferred_input_dim),
    )
    input_beta_tensor = heldin.new_tensor(float(input_beta))
    base["loss"] = base["loss"] + input_beta_tensor * input_kl
    base["z0_kl_loss"] = base["kl_loss"]
    base["inferred_input_kl_loss"] = input_kl
    base["inferred_input_kl_beta"] = input_beta_tensor
    return base, output["rates_hz"]


def _ensure_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all().item():
        msg = f"{name} is not finite"
        raise RuntimeError(msg)


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _split_metrics(
    model: LFADSController,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    bin_size_ms: int,
    z0_beta: float,
    input_beta: float,
    training_config: dict[str, Any],
) -> dict[str, float]:
    model.eval()
    losses: dict[str, list[float]] = {
        "total_loss": [],
        "heldin_reconstruction_loss": [],
        "heldout_prediction_loss": [],
        "z0_kl_loss": [],
        "inferred_input_kl_loss": [],
        "mean_rate_hz": [],
    }
    with torch.no_grad():
        for batch in loader:
            loss, rates = _loss_for_batch(
                model, batch, device, bin_size_ms, z0_beta, input_beta, training_config
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
            losses["inferred_input_kl_loss"].append(
                float(loss["inferred_input_kl_loss"].detach().cpu())
            )
            losses["mean_rate_hz"].append(float(rates.mean().detach().cpu()))
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


def _write_history(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def _train_controller(
    model: LFADSController,
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
        input_beta = kl_beta * float(training_config.get("inferred_input_kl_scale", 1.0))
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
            loss, _ = _loss_for_batch(
                model,
                batch,
                device,
                int(config["dataset"]["bin_size_ms"]),
                kl_beta,
                input_beta,
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
            "inferred_input_kl_beta": float(input_beta),
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
                input_beta,
                training_config,
            )
            row[f"{split_name}_total_loss"] = metrics["total_loss"]
            row[f"{split_name}_heldin_reconstruction_loss"] = metrics["heldin_reconstruction_loss"]
            row[f"{split_name}_heldout_prediction_loss"] = metrics["heldout_prediction_loss"]
            row[f"{split_name}_z0_kl_loss"] = metrics["z0_kl_loss"]
            row[f"{split_name}_inferred_input_kl_loss"] = metrics["inferred_input_kl_loss"]
            row[f"{split_name}_mean_rate_hz"] = metrics["mean_rate_hz"]
        row["train_total_loss"] = row.get("train_total_loss", row["training_batch_loss"])
        row["z0_kl_loss"] = row.get("validation_z0_kl_loss", float("nan"))
        row["inferred_input_kl_loss"] = row.get("validation_inferred_input_kl_loss", float("nan"))
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


def _load_controller_from_checkpoint(
    checkpoint_path: Path,
    input_dim: int,
    output_dim: int,
    config: dict[str, Any],
    device: torch.device,
) -> LFADSController:
    model_config = dict(config["model"])
    model = LFADSController(
        LFADSControllerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            encoder_hidden_dim=int(model_config["encoder_hidden_dim"]),
            controller_hidden_dim=int(model_config["controller_hidden_dim"]),
            generator_hidden_dim=int(model_config["generator_hidden_dim"]),
            latent_dim=int(model_config["latent_dim"]),
            factor_dim=int(model_config["factor_dim"]),
            inferred_input_dim=int(model_config["inferred_input_dim"]),
            dropout=float(model_config.get("dropout", 0.0)),
            min_rate_hz=float(model_config["min_rate_hz"]),
            max_rate_hz=float(model_config["max_rate_hz"]),
        )
    ).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    return model


def _extract_controller_factors(
    model: LFADSController,
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
                "heldout_spikes": [],
                "trial_ids": [],
            }
            for batch in loader:
                output = model(batch["heldin_spikes"].to(device))
                chunks["factors"].append(output["factors"].detach().cpu().numpy())
                chunks["model_rates_hz"].append(output["rates_hz"].detach().cpu().numpy())
                chunks["heldout_spikes"].append(batch["heldout_spikes"].detach().cpu().numpy())
                chunks["trial_ids"].append(batch["trial_id"].detach().cpu().numpy())
            extracted[split_name] = {
                key: np.concatenate(value, axis=0) for key, value in chunks.items()
            }
    return extracted


def _run_controller_evaluation(
    dataset: NeuralDataset,
    split: TrialSplit,
    mask: NeuronMask,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    input_dim = int(np.count_nonzero(mask.heldin))
    output_dim = int(dataset.spikes.shape[2])
    target_indices = np.flatnonzero(mask.heldout)
    model = _load_controller_from_checkpoint(
        Path(str(config["model"]["checkpoint_path"])), input_dim, output_dim, config, device
    )
    dataloaders = create_dataloaders(
        create_torch_datasets(dataset, split, mask, int(config["data"]["max_time_bins"])),
        batch_size=int(config["data"]["batch_size"]),
        num_workers=0,
        drop_last=False,
        seed=int(config["splits"]["seed"]),
    )
    extracted = _extract_controller_factors(model, dataloaders, device)
    min_rate = float(config["model"]["min_rate_hz"])
    max_rate = float(config["model"]["max_rate_hz"])
    from latentbrain.eval.cosmoothing import (
        _broadcast_reference,
        _rates_from_counts,
        _reference_rates,
    )
    from latentbrain.eval.metrics import safe_clip_rates

    reference = _reference_rates(
        extracted["train"]["heldout_spikes"], dataset.bin_size_ms, min_rate, max_rate
    )
    split_rows: list[dict[str, Any]] = []
    neuron_rows: list[dict[str, Any]] = []
    factor_tables: list[pd.DataFrame] = []
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
        behavior_feature_stats: dict[str, np.ndarray] = {}
        train_behavior_factors = train_factors_raw
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
        "model_name": "lfads_controller",
        "checkpoint_path": str(config["model"]["checkpoint_path"]),
        "checkpoint_epoch": _checkpoint_epoch(Path(str(config["model"]["checkpoint_path"]))),
        "factor_dim": model.config.factor_dim,
        "latent_dim": model.config.latent_dim,
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
    )


def _behavior_mean_r2(path: Path) -> float:
    if not path.exists():
        return float("nan")
    metrics = pd.read_csv(path)
    rows = metrics[metrics["split"] == "validation"] if "split" in metrics else metrics
    return float("nan") if rows.empty else float(rows["r2"].mean())


def _unified_scores(run_dir: Path) -> pd.DataFrame:
    scores = pd.read_csv(run_dir / "evaluation" / "split_metrics.csv")
    scores.insert(0, "reference_name", "train_heldout_mean_rate")
    scores.insert(0, "valid_model", True)
    scores.to_csv(run_dir / "unified_scores.csv", index=False)
    return scores


def _run_metrics(run_dir: Path, device: str) -> dict[str, Any]:
    final = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
    return {
        "status": "completed",
        "device": device,
        "validation_behavior_mean_r2": _behavior_mean_r2(
            run_dir / "evaluation" / "behavior_metrics.csv"
        ),
        "train_total_loss": final.get("train_total_loss", final.get("training_batch_loss")),
        "validation_total_loss": final.get("validation_total_loss"),
        "validation_heldout_prediction_loss": final.get("validation_heldout_prediction_loss"),
        "z0_kl_loss": final.get("z0_kl_loss"),
        "inferred_input_kl_loss": final.get("inferred_input_kl_loss"),
    }


def _train_and_evaluate_run(
    run_config: dict[str, Any],
    run_index: int,
    dataset: NeuralDataset,
    split: TrialSplit,
    mask: NeuronMask,
) -> None:
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
    model = LFADSController(
        LFADSControllerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            encoder_hidden_dim=int(run_config["model"]["encoder_hidden_dim"]),
            controller_hidden_dim=int(run_config["model"]["controller_hidden_dim"]),
            generator_hidden_dim=int(run_config["model"]["generator_hidden_dim"]),
            latent_dim=int(run_config["model"]["latent_dim"]),
            factor_dim=int(run_config["model"]["factor_dim"]),
            inferred_input_dim=int(run_config["model"]["inferred_input_dim"]),
            dropout=float(run_config["model"].get("dropout", 0.0)),
            min_rate_hz=float(run_config["model"]["min_rate_hz"]),
            max_rate_hz=float(run_config["model"]["max_rate_hz"]),
        )
    )
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
    _train_controller(model, dataloaders, snapshot, output_dir, device)
    eval_config = copy.deepcopy(snapshot)
    eval_config["model"]["checkpoint_path"] = str(output_dir / "checkpoints" / "best_validation.pt")
    split_metrics, neuron_metrics, behavior_metrics, factor_summary, metadata = (
        _run_controller_evaluation(dataset, split, mask, eval_config, device)
    )
    from latentbrain.eval.reporting import write_lfads_gru_evaluation_outputs

    validation = split_metrics[
        (split_metrics["split"] == "validation")
        & (split_metrics["prediction_source"] == "direct_model")
    ].iloc[0]
    write_lfads_gru_evaluation_outputs(
        output_dir / "evaluation",
        {
            "dataset_name": run_config["dataset"]["name"],
            "dataset_hash": run_config["dataset"].get("expected_hash"),
            "checkpoint_path": _relative(
                output_dir / "checkpoints" / "best_validation.pt", get_repo_root()
            ),
            "checkpoint_epoch": metadata.get("checkpoint_epoch"),
            "model_name": "lfads_controller",
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
            "factor_decoder_validation_bits_per_spike": None,
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


def run_lfads_controller_tuning(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    _resolve_processed_path(config)
    gpu_name = _validate_cuda(config)
    dataset, dataset_hash, window_bins = _load_dataset(config)
    split, mask = _split_and_mask(dataset, config)
    reference_zero = _verify_reference_zero(dataset, split, mask, config)
    repo_root = get_repo_root()
    output_dir = resolve_configured_path(str(config["reporting"]["output_dir"]), repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    refs = dict(config["references"])
    refs["train_mean_validation_bits_per_spike"] = reference_zero
    grid = expand_controller_grid(dict(config["grid"]))[: int(config["search"]["max_runs"])]
    rows: list[dict[str, Any]] = []
    base = copy.deepcopy(config)
    base["_window_bins"] = window_bins
    for run_index, params in enumerate(grid):
        run_id = make_controller_run_id(run_index, params)
        run_dir = output_dir / "runs" / run_id
        run_config = build_controller_train_config(base, params, run_dir)
        _train_and_evaluate_run(run_config, run_index, dataset, split, mask)
        scores = _unified_scores(run_dir)
        rows.append(
            build_controller_result_row(
                run_id,
                run_index,
                params,
                scores,
                _run_metrics(run_dir, str(config["runtime"]["device"])),
                refs,
                run_dir,
            )
        )
    results = pd.DataFrame(rows)
    leaderboard = rank_controller_results(results)
    summary = summarize_controller_tuning(results, refs)
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
    best_config: dict[str, Any] = {}
    if not leaderboard.empty:
        best = leaderboard.iloc[0]
        best_result = results.loc[results["run_id"] == best["run_id"]].iloc[0]
        best_params = {
            key: best_result[key]
            for key in config["grid"]
            if key in best_result and pd.notna(best_result[key])
        }
        best_config = build_controller_train_config(
            base, best_params, Path(str(best_result["output_dir"]))
        )
    (output_dir / "best_config.yaml").write_text(
        yaml.safe_dump(json.loads(json.dumps(best_config, default=_json_default)), sort_keys=False),
        encoding="utf-8",
    )
    return results, summary
