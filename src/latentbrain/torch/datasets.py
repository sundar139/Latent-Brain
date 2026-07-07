from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from latentbrain.data.schemas import NeuralDataset, NeuronMask, TrialSplit
from latentbrain.data.validation import (
    validate_neural_dataset,
    validate_neuron_mask,
    validate_trial_split,
)

SplitName = Literal["train", "validation", "test"]


class NeuralTrialDataset(Dataset[dict[str, torch.Tensor]]):
    """Trial-major torch dataset exposing held-in inputs and held-out targets."""

    def __init__(
        self,
        dataset: NeuralDataset,
        trial_ids: np.ndarray,
        neuron_mask: NeuronMask,
        max_time_bins: int | None,
        split_name: str,
    ) -> None:
        validate_neural_dataset(dataset)
        validate_neuron_mask(neuron_mask, dataset.spikes.shape[2])
        if max_time_bins is not None and max_time_bins <= 0:
            msg = "max_time_bins must be positive when provided"
            raise ValueError(msg)
        self._dataset = dataset
        self._trial_ids = np.asarray(trial_ids, dtype=dataset.trial_ids.dtype)
        trial_to_index = {int(trial_id): index for index, trial_id in enumerate(dataset.trial_ids)}
        self._indices = np.asarray([trial_to_index[int(trial_id)] for trial_id in self._trial_ids])
        time_bins = dataset.spikes.shape[1]
        self._time_bins = time_bins if max_time_bins is None else min(max_time_bins, time_bins)
        self._heldin_indices = np.flatnonzero(neuron_mask.heldin)
        self._heldout_indices = np.flatnonzero(neuron_mask.heldout)
        self.metadata = {
            "split": split_name,
            "max_time_bins": max_time_bins,
            "crop_start": 0,
            "n_trials": int(len(self._trial_ids)),
            "n_time_bins": int(self._time_bins),
            "heldin_neurons": int(len(self._heldin_indices)),
            "heldout_neurons": int(len(self._heldout_indices)),
        }

    def __len__(self) -> int:
        return int(len(self._indices))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        trial_index = int(self._indices[index])
        spikes = self._dataset.spikes[trial_index, : self._time_bins, :]
        all_spikes = torch.as_tensor(spikes, dtype=torch.float32)
        return {
            "heldin_spikes": all_spikes[:, self._heldin_indices],
            "heldout_spikes": all_spikes[:, self._heldout_indices],
            "all_spikes": all_spikes,
            "trial_id": torch.tensor(int(self._dataset.trial_ids[trial_index]), dtype=torch.int64),
        }


def create_torch_datasets(
    dataset: NeuralDataset,
    split: TrialSplit,
    neuron_mask: NeuronMask,
    max_time_bins: int | None,
) -> dict[str, NeuralTrialDataset]:
    """Create deterministic train/validation/test torch datasets."""
    validate_trial_split(split, dataset.trial_ids)
    return {
        "train": NeuralTrialDataset(dataset, split.train, neuron_mask, max_time_bins, "train"),
        "validation": NeuralTrialDataset(
            dataset, split.validation, neuron_mask, max_time_bins, "validation"
        ),
        "test": NeuralTrialDataset(dataset, split.test, neuron_mask, max_time_bins, "test"),
    }


def create_dataloaders(
    datasets: dict[str, NeuralTrialDataset],
    batch_size: int,
    num_workers: int,
    drop_last: bool,
    seed: int,
) -> dict[str, DataLoader[dict[str, torch.Tensor]]]:
    """Create deterministic dataloaders; ordering is preserved for all splits."""
    if batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)
    if num_workers < 0:
        msg = "num_workers must be non-negative"
        raise ValueError(msg)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=drop_last if name == "train" else False,
            generator=generator,
        )
        for name, dataset in datasets.items()
    }
