from __future__ import annotations

import numpy as np
import torch

from latentbrain.data.schemas import NeuralDataset, NeuronMask, TrialSplit
from latentbrain.torch.datasets import create_dataloaders, create_torch_datasets


def _dataset() -> NeuralDataset:
    spikes = np.arange(4 * 6 * 5, dtype=np.int64).reshape(4, 6, 5)
    return NeuralDataset(
        spikes=spikes,
        rates=None,
        latents=None,
        trial_ids=np.array([10, 11, 12, 13], dtype=np.int64),
        time_ms=np.arange(6, dtype=np.float64) * 5.0,
        bin_size_ms=5,
        metadata={},
    )


def _split() -> TrialSplit:
    return TrialSplit(
        train=np.array([12, 10], dtype=np.int64),
        validation=np.array([11], dtype=np.int64),
        test=np.array([13], dtype=np.int64),
    )


def _mask() -> NeuronMask:
    return NeuronMask(
        heldin=np.array([True, False, True, True, False]),
        heldout=np.array([False, True, False, False, True]),
    )


def test_dataset_returns_heldin_and_heldout_shapes_without_leakage() -> None:
    datasets = create_torch_datasets(_dataset(), _split(), _mask(), max_time_bins=4)

    item = datasets["train"][0]

    assert item["heldin_spikes"].shape == (4, 3)
    assert item["heldout_spikes"].shape == (4, 2)
    assert item["all_spikes"].shape == (4, 5)
    torch.testing.assert_close(item["heldin_indices"], torch.tensor([0, 2, 3]))
    torch.testing.assert_close(item["heldout_indices"], torch.tensor([1, 4]))
    torch.testing.assert_close(item["heldin_spikes"], item["all_spikes"][:, [0, 2, 3]])
    torch.testing.assert_close(item["heldout_spikes"], item["all_spikes"][:, [1, 4]])
    assert set(item["heldin_indices"].tolist()).isdisjoint(item["heldout_indices"].tolist())
    assert not torch.equal(item["heldin_spikes"][:, :2], item["heldout_spikes"])


def test_max_time_bin_cropping_uses_start_and_records_metadata() -> None:
    datasets = create_torch_datasets(_dataset(), _split(), _mask(), max_time_bins=3)

    item = datasets["validation"][0]

    assert datasets["validation"].metadata["max_time_bins"] == 3
    assert datasets["validation"].metadata["crop_start"] == 0
    torch.testing.assert_close(
        item["all_spikes"], torch.as_tensor(_dataset().spikes[1, :3], dtype=torch.float32)
    )


def test_dataloaders_are_deterministic() -> None:
    datasets = create_torch_datasets(_dataset(), _split(), _mask(), max_time_bins=2)

    first = create_dataloaders(datasets, batch_size=1, num_workers=0, drop_last=False, seed=7)
    second = create_dataloaders(datasets, batch_size=1, num_workers=0, drop_last=False, seed=7)

    first_ids = [int(batch["trial_id"][0]) for batch in first["train"]]
    second_ids = [int(batch["trial_id"][0]) for batch in second["train"]]
    assert first_ids == [12, 10]
    assert first_ids == second_ids


def test_dataset_keeps_unmasked_original_tensors_available() -> None:
    datasets = create_torch_datasets(_dataset(), _split(), _mask(), max_time_bins=4)

    first = datasets["train"][0]
    second = datasets["train"][0]

    torch.testing.assert_close(first["heldin_spikes"], second["heldin_spikes"])
    torch.testing.assert_close(first["heldout_spikes"], second["heldout_spikes"])
    torch.testing.assert_close(first["all_spikes"], second["all_spikes"])
