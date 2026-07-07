from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch

from latentbrain.data.schemas import NeuralDataset, NeuronMask, TrialSplit
from latentbrain.models.lfads_gru import LFADSGRU, LFADSGRUConfig
from latentbrain.torch.datasets import create_dataloaders, create_torch_datasets
from latentbrain.train.lfads_trainer import train_lfads_gru


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
