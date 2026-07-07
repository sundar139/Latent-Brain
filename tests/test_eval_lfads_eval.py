from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from latentbrain.data.schemas import NeuralDataset, NeuronMask, TrialSplit
from latentbrain.eval.lfads_eval import (
    extract_lfads_factors,
    load_lfads_gru_from_checkpoint,
    run_lfads_gru_evaluation,
)
from latentbrain.models.lfads_gru import LFADSGRU, LFADSGRUConfig
from latentbrain.torch.checkpoints import save_checkpoint
from latentbrain.torch.datasets import create_dataloaders, create_torch_datasets


def _dataset(with_behavior: bool = True, validation_offset: int = 0) -> NeuralDataset:
    rng = np.random.default_rng(8)
    heldin = rng.poisson(0.3, size=(8, 7, 4)).astype(np.int64)
    heldout = np.stack([heldin[:, :, 0] + 1, heldin[:, :, 1] + 1], axis=2).astype(np.int64)
    heldout[4:6] += validation_offset
    spikes = np.concatenate([heldin, heldout], axis=2)
    behavior = None
    names = None
    if with_behavior:
        t = np.arange(7, dtype=np.float64)[None, :, None]
        trial = np.arange(8, dtype=np.float64)[:, None, None]
        t_all = np.broadcast_to(t, (8, 7, 1))
        behavior = np.concatenate([t_all + trial, t_all - trial, 2 * t_all, -t_all], axis=2)
        names = ["hand_pos_x", "hand_pos_y", "cursor_pos_x", "cursor_pos_y"]
    return NeuralDataset(
        spikes=spikes,
        rates=None,
        latents=None,
        trial_ids=np.arange(8, dtype=np.int64),
        time_ms=np.arange(7, dtype=np.float64) * 10,
        bin_size_ms=10,
        metadata={},
        behavior=behavior,
        behavior_names=names,
    )


def _split() -> TrialSplit:
    return TrialSplit(
        train=np.array([0, 1, 2, 3], dtype=np.int64),
        validation=np.array([4, 5], dtype=np.int64),
        test=np.array([6, 7], dtype=np.int64),
    )


def _mask() -> NeuronMask:
    return NeuronMask(
        heldin=np.array([True, True, True, True, False, False]),
        heldout=np.array([False, False, False, False, True, True]),
    )


def _checkpoint(path: Path) -> Path:
    model = LFADSGRU(LFADSGRUConfig(4, 4, 6, 6, 3, 5, 0.0, 1.0e-4, 500.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    return save_checkpoint(path, model, optimizer, 2, {"validation_loss": 1.0}, {"unit": True})


def _config(checkpoint_path: Path, behavior_enabled: bool = True) -> dict[str, object]:
    return {
        "dataset": {"bin_size_ms": 10},
        "data": {"max_time_bins": 6, "batch_size": 2, "num_workers": 0, "drop_last": False},
        "model": {
            "checkpoint_path": str(checkpoint_path),
            "encoder_hidden_dim": 6,
            "generator_hidden_dim": 6,
            "latent_dim": 3,
            "factor_dim": 5,
            "dropout": 0.0,
            "min_rate_hz": 1.0e-4,
            "max_rate_hz": 500.0,
        },
        "heldout_decoder": {
            "alpha": 1.0,
            "fit_intercept": True,
            "min_rate_hz": 1.0e-4,
            "max_rate_hz": 500.0,
            "standardize_factors": True,
        },
        "behavior_decoder": {
            "enabled": behavior_enabled,
            "alpha": 1.0,
            "fit_intercept": True,
            "standardize_factors": True,
            "standardize_targets": True,
            "target_prefixes": ["hand_pos", "cursor_pos"],
            "derive_velocity": True,
            "velocity_method": "central_difference",
        },
        "evaluation": {
            "evaluate_splits": ["train", "validation", "test"],
            "primary_split": "validation",
        },
    }


def test_checkpointed_toy_lfads_model_can_be_loaded(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path / "model.pt")

    model = load_lfads_gru_from_checkpoint(path, 4, 4, _config(path), torch.device("cpu"))

    assert not model.training
    assert model.config.factor_dim == 5


def test_factor_extraction_preserves_split_order_and_ignores_heldout_spikes(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path / "model.pt")
    model = load_lfads_gru_from_checkpoint(path, 4, 4, _config(path), torch.device("cpu"))
    datasets = create_torch_datasets(_dataset(), _split(), _mask(), max_time_bins=6)
    loaders = create_dataloaders(datasets, 2, 0, False, seed=1)

    extracted = extract_lfads_factors(model, loaders, torch.device("cpu"))
    changed = _dataset()
    changed.spikes[:, :, 4:] += 1000
    changed_loaders = create_dataloaders(
        create_torch_datasets(changed, _split(), _mask(), max_time_bins=6), 2, 0, False, seed=1
    )
    extracted_changed = extract_lfads_factors(model, changed_loaders, torch.device("cpu"))

    np.testing.assert_array_equal(extracted["train"]["trial_ids"], np.array([0, 1, 2, 3]))
    np.testing.assert_allclose(
        extracted["validation"]["factors"], extracted_changed["validation"]["factors"]
    )


def test_lfads_evaluation_outputs_metrics_and_uses_train_only_decoder(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path / "model.pt")
    split_metrics, neuron_metrics, behavior_metrics, factor_summary, metadata = (
        run_lfads_gru_evaluation(_dataset(), _split(), _mask(), _config(path), torch.device("cpu"))
    )
    _, _, _, _, changed_metadata = run_lfads_gru_evaluation(
        _dataset(validation_offset=100), _split(), _mask(), _config(path), torch.device("cpu")
    )

    assert {"split", "bits_per_spike", "poisson_nll", "factor_dim"}.issubset(split_metrics.columns)
    assert {"split", "target_neuron_index", "mean_predicted_rate_hz"}.issubset(
        neuron_metrics.columns
    )
    assert {"split", "target_name", "r2", "mse", "mae", "target_variance"}.issubset(
        behavior_metrics.columns
    )
    assert {"split", "factor_index", "mean", "variance"}.issubset(factor_summary.columns)
    assert np.isfinite(split_metrics["bits_per_spike"]).all()
    assert (split_metrics["mean_predicted_rate_hz"] > 0).all()
    np.testing.assert_allclose(
        metadata["heldout_decoder_coefficients"], changed_metadata["heldout_decoder_coefficients"]
    )


def test_lfads_evaluation_works_with_behavior_decoder_disabled(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path / "model.pt")

    _, _, behavior_metrics, _, metadata = run_lfads_gru_evaluation(
        _dataset(with_behavior=False),
        _split(),
        _mask(),
        _config(path, behavior_enabled=False),
        torch.device("cpu"),
    )

    assert behavior_metrics.empty
    assert metadata["behavior_decoder_enabled"] is False
