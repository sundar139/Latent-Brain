from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from latentbrain.train import lfads_diagnostics
from latentbrain.train.lfads_diagnostics import loss_drop_fraction, run_tiny_subset_overfit


def _config() -> dict[str, Any]:
    return {
        "audit": {
            "tiny_subset_trials": 4,
            "tiny_subset_epochs": 3,
            "tiny_subset_max_time_bins": 8,
            "tiny_subset_learning_rate": 1.0e-3,
        },
        "runtime": {"device": "cuda"},
        "training": {"seed": 1},
    }


def test_loss_drop_fraction_is_computed() -> None:
    assert loss_drop_fraction(10.0, 7.0) == 0.3
    assert loss_drop_fraction(0.0, 1.0) == 0.0


def test_tiny_subset_overfit_can_be_simulated_without_cuda(
    tmp_path: Path, monkeypatch: object
) -> None:
    frame = pd.DataFrame(
        {
            "epoch": [0, 1],
            "train_total_loss": [10.0, 5.0],
            "train_heldin_reconstruction_loss": [6.0, 3.0],
            "train_heldout_prediction_loss": [3.0, 1.5],
            "train_kl_loss": [1.0, 0.5],
            "validation_total_loss": [11.0, 8.0],
            "kl_beta": [0.0, 1.0],
            "mean_predicted_rate": [2.0, 3.0],
        }
    )

    monkeypatch.setattr(lfads_diagnostics, "_run_real_tiny_subset_overfit", lambda *_: frame)

    result = run_tiny_subset_overfit(_config(), tmp_path)

    assert result.columns.tolist() == list(frame.columns)
    assert result.iloc[-1]["train_total_loss"] == 5.0
