from __future__ import annotations

from typing import Any

import numpy as np

from latentbrain.data.schemas import NeuralDataset


def crop_time_window(
    values: np.ndarray, max_time_bins: int, policy: str = "from_start"
) -> np.ndarray:
    """Crop a trial-major array along its time axis."""
    array = np.asarray(values)
    if array.ndim < 2:
        msg = f"values must have at least trial and time axes; got shape {array.shape}"
        raise ValueError(msg)
    if max_time_bins <= 0:
        msg = "max_time_bins must be positive"
        raise ValueError(msg)
    if policy != "from_start":
        msg = f"unsupported crop policy: {policy}"
        raise ValueError(msg)
    cropped_time_bins = min(max_time_bins, array.shape[1])
    return np.array(array[:, :cropped_time_bins, ...], copy=True)


def _crop_optional(values: np.ndarray | None, max_time_bins: int, policy: str) -> np.ndarray | None:
    if values is None:
        return None
    return crop_time_window(values, max_time_bins, policy)


def describe_time_window(
    original_time_bins: int,
    cropped_time_bins: int,
    bin_size_ms: int,
) -> dict[str, Any]:
    """Describe a time crop in bins, milliseconds, and seconds."""
    if original_time_bins <= 0 or cropped_time_bins <= 0:
        msg = "time bin counts must be positive"
        raise ValueError(msg)
    if bin_size_ms <= 0:
        msg = "bin_size_ms must be positive"
        raise ValueError(msg)
    return {
        "original_time_bins": int(original_time_bins),
        "cropped_time_bins": int(cropped_time_bins),
        "bin_size_ms": int(bin_size_ms),
        "window_ms": int(cropped_time_bins * bin_size_ms),
        "window_seconds": float(cropped_time_bins * bin_size_ms / 1000.0),
    }


def crop_neural_dataset_time(
    dataset: NeuralDataset,
    max_time_bins: int,
    policy: str = "from_start",
) -> NeuralDataset:
    """Crop all time-indexed arrays in a NeuralDataset consistently."""
    original_time_bins = int(dataset.spikes.shape[1])
    cropped_spikes = crop_time_window(dataset.spikes, max_time_bins, policy)
    cropped_time_bins = int(cropped_spikes.shape[1])
    metadata = dict(dataset.metadata)
    metadata["time_window"] = describe_time_window(
        original_time_bins,
        cropped_time_bins,
        dataset.bin_size_ms,
    ) | {"crop_policy": policy}
    return NeuralDataset(
        spikes=cropped_spikes,
        rates=_crop_optional(dataset.rates, max_time_bins, policy),
        latents=_crop_optional(dataset.latents, max_time_bins, policy),
        trial_ids=np.array(dataset.trial_ids, copy=True),
        time_ms=np.array(dataset.time_ms[:cropped_time_bins], copy=True),
        bin_size_ms=int(dataset.bin_size_ms),
        metadata=metadata,
        behavior=_crop_optional(dataset.behavior, max_time_bins, policy),
        behavior_names=None if dataset.behavior_names is None else list(dataset.behavior_names),
    )
