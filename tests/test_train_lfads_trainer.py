from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch

from latentbrain.data.schemas import NeuralDataset, NeuronMask, TrialSplit
from latentbrain.models.lfads_gru import LFADSGRU, LFADSGRUConfig
from latentbrain.torch.datasets import create_dataloaders, create_torch_datasets
from latentbrain.train.lfads_trainer import _loss_for_batch, train_lfads_gru


def _loaders() -> dict[str, torch.utils.data.DataLoader]:
    rng = np.random.default_rng(3)
    dataset = NeuralDataset(
        spikes=rng.poisson(0.2, size=(6, 8, 4)).astype(np.int64),
        rates=None,
        latents=None,
        trial_ids=np.arange(6, dtype=np.int64),
        time_ms=np.arange(8, dtype=np.float64) * 10.0,
        bin_size_ms=10,
        metadata={},
    )
    split = TrialSplit(
        train=np.array([0, 1, 2, 3], dtype=np.int64),
        validation=np.array([4], dtype=np.int64),
        test=np.array([5], dtype=np.int64),
    )
    mask = NeuronMask(
        heldin=np.array([True, True, True, False]),
        heldout=np.array([False, False, False, True]),
    )
    return create_dataloaders(
        create_torch_datasets(dataset, split, mask, max_time_bins=6),
        batch_size=2,
        num_workers=0,
        drop_last=False,
        seed=3,
    )


def test_tiny_dataset_trains_and_writes_outputs(tmp_path: Path) -> None:
    model = LFADSGRU(LFADSGRUConfig(3, 3, 8, 8, 3, 4, 0.0, 1.0e-4, 500.0))
    state = train_lfads_gru(
        model,
        _loaders(),
        config={
            "dataset": {"bin_size_ms": 10},
            "model": {"output_dim": None},
            "training": {
                "epochs": 1,
                "learning_rate": 1.0e-3,
                "weight_decay": 0.0,
                "gradient_clip_norm": 5.0,
                "kl_warmup_epochs": 1,
                "checkpoint_metric": "validation_loss",
                "checkpoint_mode": "min",
            },
            "evaluation": {"evaluate_splits": ["train", "validation"]},
        },
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )

    assert len(state.history) == 1
    assert np.isfinite(state.history[0]["validation_loss"])
    assert (tmp_path / "metrics_history.csv").exists()
    assert (tmp_path / "final_metrics.json").exists()
    assert (tmp_path / "checkpoints" / "latest.pt").exists()
    assert (tmp_path / "checkpoints" / "best_validation.pt").exists()
    assert np.isfinite(pd.read_csv(tmp_path / "metrics_history.csv")["validation_loss"]).all()


def test_tiny_cosmoothing_training_writes_heldout_metrics(tmp_path: Path) -> None:
    model = LFADSGRU(LFADSGRUConfig(3, 4, 8, 8, 3, 4, 0.0, 1.0e-4, 500.0))
    state = train_lfads_gru(
        model,
        _loaders(),
        config={
            "dataset": {"bin_size_ms": 10},
            "model": {"output_dim": "all"},
            "training": {
                "epochs": 1,
                "learning_rate": 1.0e-3,
                "weight_decay": 0.0,
                "gradient_clip_norm": 5.0,
                "kl_warmup_epochs": 1,
                "heldin_loss_weight": 1.0,
                "heldout_loss_weight": 1.0,
                "loss_normalization": "mean",
                "checkpoint_metric": "validation_total_loss",
                "checkpoint_mode": "min",
            },
            "evaluation": {"evaluate_splits": ["train", "validation"]},
        },
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )

    row = state.history[0]
    assert np.isfinite(row["validation_total_loss"])
    assert np.isfinite(row["validation_heldout_prediction_loss"])
    assert row["validation_heldout_prediction_loss"] > 0.0
    assert (tmp_path / "checkpoints" / "latest.pt").exists()
    assert (tmp_path / "checkpoints" / "best_validation.pt").exists()


def test_trainer_initializes_readout_bias_from_train_rates_only(tmp_path: Path) -> None:
    model = LFADSGRU(LFADSGRUConfig(3, 4, 8, 8, 3, 4, 0.0, 1.0e-4, 500.0))
    loaders = _loaders()
    validation_before = loaders["validation"].dataset[0]["all_spikes"].numpy().copy()
    train_lfads_gru(
        model,
        loaders,
        config={
            "dataset": {"bin_size_ms": 10},
            "model": {"output_dim": "all"},
            "training": {
                "epochs": 1,
                "learning_rate": 1.0e-3,
                "weight_decay": 0.0,
                "gradient_clip_norm": 5.0,
                "kl_warmup_epochs": 1,
                "heldin_loss_weight": 1.0,
                "heldout_loss_weight": 1.0,
                "loss_normalization": "mean",
                "checkpoint_metric": "validation_total_loss",
                "checkpoint_mode": "min",
                "initialize_readout_bias_from_train_rates": True,
            },
            "evaluation": {"evaluate_splits": ["train", "validation"]},
        },
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )

    assert np.array_equal(validation_before, loaders["validation"].dataset[0]["all_spikes"].numpy())
    assert torch.isfinite(model.rate_readout.bias).all()


def test_trainer_logs_training_input_dropout_without_mutating_targets(tmp_path: Path) -> None:
    model = LFADSGRU(LFADSGRUConfig(3, 4, 8, 8, 3, 4, 0.0, 1.0e-4, 500.0))
    loaders = _loaders()
    validation_before = loaders["validation"].dataset[0]["heldin_spikes"].clone()
    state = train_lfads_gru(
        model,
        loaders,
        config={
            "dataset": {"bin_size_ms": 10},
            "model": {"output_dim": "all"},
            "training": {
                "seed": 9,
                "epochs": 1,
                "learning_rate": 1.0e-3,
                "weight_decay": 0.0,
                "gradient_clip_norm": 5.0,
                "kl_warmup_epochs": 1,
                "heldin_loss_weight": 1.0,
                "heldout_loss_weight": 1.0,
                "loss_normalization": "mean",
                "checkpoint_metric": "validation_total_loss",
                "checkpoint_mode": "min",
                "input_dropout": {
                    "enabled": True,
                    "rate": 0.5,
                    "apply_to": ["train"],
                    "keep_at_least_one_neuron": True,
                    "seed": 9,
                },
            },
            "evaluation": {"evaluate_splits": ["train", "validation"]},
        },
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )

    row = state.history[0]
    assert row["configured_input_dropout_rate"] == 0.5
    assert row["realized_input_dropout_fraction"] > 0.0
    assert torch.equal(validation_before, loaders["validation"].dataset[0]["heldin_spikes"])


def test_loss_uses_masked_input_but_original_targets() -> None:
    model = LFADSGRU(LFADSGRUConfig(3, 4, 8, 8, 3, 4, 0.0, 1.0e-4, 500.0))
    batch = next(iter(_loaders()["train"]))
    original_heldin = batch["heldin_spikes"].clone()
    masked = torch.zeros_like(original_heldin)

    loss, _ = _loss_for_batch(
        model,
        batch,
        torch.device("cpu"),
        10,
        1.0,
        {"heldin_loss_weight": 1.0, "heldout_loss_weight": 1.0, "loss_normalization": "mean"},
        "cosmoothing",
        input_heldin_spikes=masked,
    )

    assert torch.equal(batch["heldin_spikes"], original_heldin)
    assert torch.isfinite(loss["heldin_reconstruction_loss"])
