from __future__ import annotations

import numpy as np

from latentbrain.data.schemas import NeuralDataset


def validate_rebin_factor(original_bin_size_ms: int, target_bin_size_ms: int) -> int:
    """Return integer temporal aggregation factor from original to target bin size."""
    if original_bin_size_ms <= 0 or target_bin_size_ms <= 0:
        msg = "bin sizes must be positive"
        raise ValueError(msg)
    if target_bin_size_ms % original_bin_size_ms != 0:
        msg = "target_bin_size_ms must be a multiple of original_bin_size_ms"
        raise ValueError(msg)
    return target_bin_size_ms // original_bin_size_ms


def _trimmed(values: np.ndarray, factor: int, trim: bool) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 3:
        msg = f"values must have shape [trials, time, features]; got {array.shape}"
        raise ValueError(msg)
    usable = (array.shape[1] // factor) * factor
    if usable != array.shape[1] and not trim:
        msg = f"time dimension {array.shape[1]} is not divisible by rebin factor {factor}"
        raise ValueError(msg)
    if usable == 0:
        msg = "rebin factor leaves no complete time bins"
        raise ValueError(msg)
    return array[:, :usable, :]


def rebin_spike_counts(spikes: np.ndarray, factor: int, trim: bool = True) -> np.ndarray:
    """Sum spike counts across adjacent time bins."""
    if factor <= 0:
        msg = "factor must be positive"
        raise ValueError(msg)
    values = _trimmed(spikes, factor, trim)
    shape = (values.shape[0], values.shape[1] // factor, factor, values.shape[2])
    rebinned = values.reshape(shape).sum(axis=2).astype(values.dtype, copy=False)
    return np.asarray(rebinned)


def rebin_behavior(
    behavior: np.ndarray, factor: int, method: str = "mean", trim: bool = True
) -> np.ndarray:
    """Aggregate behavior samples across adjacent time bins."""
    if factor <= 0:
        msg = "factor must be positive"
        raise ValueError(msg)
    values = _trimmed(behavior, factor, trim)
    shape = (values.shape[0], values.shape[1] // factor, factor, values.shape[2])
    grouped = values.reshape(shape)
    if method == "mean":
        return np.asarray(grouped.mean(axis=2))
    if method == "sum":
        return np.asarray(grouped.sum(axis=2))
    msg = f"unsupported behavior rebin method: {method}"
    raise ValueError(msg)


def _rebin_optional_rates(values: np.ndarray | None, factor: int, trim: bool) -> np.ndarray | None:
    if values is None:
        return None
    return rebin_behavior(values, factor, "mean", trim)


def rebin_neural_dataset(
    dataset: NeuralDataset,
    target_bin_size_ms: int,
    trim: bool = True,
) -> NeuralDataset:
    """Return a NeuralDataset rebinned to a coarser target bin size."""
    factor = validate_rebin_factor(dataset.bin_size_ms, target_bin_size_ms)
    if factor == 1:
        return NeuralDataset(
            spikes=np.array(dataset.spikes, copy=True),
            rates=None if dataset.rates is None else np.array(dataset.rates, copy=True),
            latents=None if dataset.latents is None else np.array(dataset.latents, copy=True),
            trial_ids=np.array(dataset.trial_ids, copy=True),
            time_ms=np.array(dataset.time_ms, copy=True),
            bin_size_ms=int(dataset.bin_size_ms),
            metadata=dict(dataset.metadata),
            behavior=None if dataset.behavior is None else np.array(dataset.behavior, copy=True),
            behavior_names=None if dataset.behavior_names is None else list(dataset.behavior_names),
        )
    spikes = rebin_spike_counts(dataset.spikes, factor, trim)
    usable = spikes.shape[1] * factor
    metadata = dict(dataset.metadata)
    metadata["rebinning"] = {
        "source_bin_size_ms": int(dataset.bin_size_ms),
        "target_bin_size_ms": int(target_bin_size_ms),
        "factor": int(factor),
        "trimmed_time_bins": int(dataset.spikes.shape[1] - usable),
    }
    return NeuralDataset(
        spikes=spikes,
        rates=_rebin_optional_rates(dataset.rates, factor, trim),
        latents=None
        if dataset.latents is None
        else rebin_behavior(dataset.latents, factor, "mean", trim),
        trial_ids=np.array(dataset.trial_ids, copy=True),
        time_ms=np.array(dataset.time_ms[:usable:factor], copy=True),
        bin_size_ms=int(target_bin_size_ms),
        metadata=metadata,
        behavior=None
        if dataset.behavior is None
        else rebin_behavior(dataset.behavior, factor, "mean", trim),
        behavior_names=None if dataset.behavior_names is None else list(dataset.behavior_names),
    )
