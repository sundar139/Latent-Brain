from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from latentbrain.data.schemas import NeuralDataset, TrialSplit
from latentbrain.data.splits import create_neuron_mask, create_trial_split
from latentbrain.eval.windowing import crop_neural_dataset_time
from latentbrain.models.lfads_gru import LFADSGRU, LFADSGRUConfig
from latentbrain.randomness import seed_everything
from latentbrain.torch.datasets import create_dataloaders, create_torch_datasets
from latentbrain.torch.device import resolve_device
from latentbrain.train.lfads_trainer import train_lfads_gru
from latentbrain.train.lfads_tuning import _load_windowed_dataset

TINY_OVERFIT_COLUMNS = [
    "epoch",
    "train_total_loss",
    "train_heldin_reconstruction_loss",
    "train_heldout_prediction_loss",
    "train_kl_loss",
    "validation_total_loss",
    "kl_beta",
    "mean_predicted_rate",
]


def loss_drop_fraction(initial_loss: float, final_loss: float) -> float:
    """Return fractional drop from initial to final loss, clipped at zero for invalid baselines."""
    if initial_loss <= 0.0:
        return 0.0
    return max(0.0, float((initial_loss - final_loss) / initial_loss))


def _default_model_config(config: dict[str, Any]) -> dict[str, Any]:
    model = dict(config.get("model", {}))
    model.setdefault("encoder_hidden_dim", 64)
    model.setdefault("generator_hidden_dim", 96)
    model.setdefault("latent_dim", 16)
    model.setdefault("factor_dim", 32)
    model.setdefault("dropout", 0.0)
    model.setdefault("min_rate_hz", 1.0e-4)
    model.setdefault("max_rate_hz", 500.0)
    model.setdefault("output_dim", "all")
    return model


def _tiny_training_config(config: dict[str, Any]) -> dict[str, Any]:
    training = dict(config.get("training", {}))
    audit = dict(config.get("audit", {}))
    training.update(
        {
            "seed": int(training.get("seed", config.get("splits", {}).get("seed", 2027))),
            "epochs": int(audit["tiny_subset_epochs"]),
            "learning_rate": float(audit["tiny_subset_learning_rate"]),
            "weight_decay": float(training.get("weight_decay", 1.0e-5)),
            "gradient_clip_norm": float(training.get("gradient_clip_norm", 5.0)),
            "heldin_loss_weight": float(training.get("heldin_loss_weight", 1.0)),
            "heldout_loss_weight": float(training.get("heldout_loss_weight", 1.0)),
            "loss_normalization": str(training.get("loss_normalization", "per_observed_spike_bin")),
            "kl_warmup_epochs": int(training.get("kl_warmup_epochs", 5)),
            "checkpoint_metric": "validation_total_loss",
            "checkpoint_mode": "min",
            "device": str(config.get("runtime", {}).get("device", "cuda")),
        }
    )
    return training


def _run_real_tiny_subset_overfit(base_config: dict[str, Any], output_dir: Path) -> pd.DataFrame:
    audit = dict(base_config["audit"])
    seed = int(base_config.get("splits", {}).get("seed", 2027))
    seed_everything(seed)
    load_config = copy.deepcopy(base_config)
    load_config["data"] = {"max_time_bins": int(base_config["window"]["max_time_bins"])}
    dataset, _, _ = _load_windowed_dataset(load_config)
    dataset = crop_neural_dataset_time(
        dataset, int(audit["tiny_subset_max_time_bins"]), "from_start"
    )
    full_split = create_trial_split(
        dataset.trial_ids,
        float(base_config["splits"]["train_fraction"]),
        float(base_config["splits"]["validation_fraction"]),
        float(base_config["splits"]["test_fraction"]),
        seed=seed,
    )
    tiny_train = full_split.train[: int(audit["tiny_subset_trials"])]
    keep_trial_ids = np.concatenate([tiny_train, full_split.validation, full_split.test])
    index_by_trial = {int(trial_id): index for index, trial_id in enumerate(dataset.trial_ids)}
    keep_indices = np.asarray([index_by_trial[int(trial_id)] for trial_id in keep_trial_ids])
    dataset = NeuralDataset(
        spikes=dataset.spikes[keep_indices],
        rates=None if dataset.rates is None else dataset.rates[keep_indices],
        latents=None if dataset.latents is None else dataset.latents[keep_indices],
        trial_ids=dataset.trial_ids[keep_indices],
        time_ms=dataset.time_ms,
        bin_size_ms=dataset.bin_size_ms,
        metadata=dataset.metadata,
        behavior=None if dataset.behavior is None else dataset.behavior[keep_indices],
        behavior_names=dataset.behavior_names,
    )
    split = TrialSplit(train=tiny_train, validation=full_split.validation, test=full_split.test)
    neuron_mask = create_neuron_mask(
        dataset.spikes.shape[2],
        float(base_config["splits"]["heldout_neuron_fraction"]),
        seed=seed,
    )
    dataloaders = create_dataloaders(
        create_torch_datasets(dataset, split, neuron_mask, int(audit["tiny_subset_max_time_bins"])),
        batch_size=max(1, min(4, len(tiny_train))),
        num_workers=0,
        drop_last=False,
        seed=seed,
    )
    model_config = _default_model_config(base_config)
    model_config["input_dim"] = int(neuron_mask.heldin.sum())
    output_dim = (
        int(dataset.spikes.shape[2])
        if model_config.get("output_dim") == "all"
        else int(model_config["output_dim"])
    )
    model = LFADSGRU(
        LFADSGRUConfig(
            input_dim=int(model_config["input_dim"]),
            output_dim=output_dim,
            encoder_hidden_dim=int(model_config["encoder_hidden_dim"]),
            generator_hidden_dim=int(model_config["generator_hidden_dim"]),
            latent_dim=int(model_config["latent_dim"]),
            factor_dim=int(model_config["factor_dim"]),
            dropout=float(model_config.get("dropout", 0.0)),
            min_rate_hz=float(model_config["min_rate_hz"]),
            max_rate_hz=float(model_config["max_rate_hz"]),
        )
    )
    train_config = copy.deepcopy(base_config)
    train_config["dataset"]["bin_size_ms"] = dataset.bin_size_ms
    train_config["model"] = model_config
    train_config["training"] = _tiny_training_config(base_config)
    train_config["evaluation"] = {"evaluate_splits": ["train", "validation"]}
    device = resolve_device(str(train_config["training"]["device"]))
    state = train_lfads_gru(model, dataloaders, train_config, output_dir / "checkpoints", device)
    rows = []
    for row in state.history:
        rows.append(
            {
                "epoch": int(row["epoch"]),
                "train_total_loss": float(row["train_total_loss"]),
                "train_heldin_reconstruction_loss": float(row["train_heldin_reconstruction_loss"]),
                "train_heldout_prediction_loss": float(row["train_heldout_prediction_loss"]),
                "train_kl_loss": float(row["train_kl_loss"]),
                "validation_total_loss": float(row["validation_total_loss"]),
                "kl_beta": float(row["kl_beta"]),
                "mean_predicted_rate": float(row["train_mean_rate_hz"]),
            }
        )
    return pd.DataFrame(rows, columns=TINY_OVERFIT_COLUMNS)


def run_tiny_subset_overfit(base_config: dict[str, Any], output_dir: Path) -> pd.DataFrame:
    """Run or dispatch the local tiny-subset LFADS-style overfit diagnostic."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _run_real_tiny_subset_overfit(base_config, output_dir)
    return frame[TINY_OVERFIT_COLUMNS]
