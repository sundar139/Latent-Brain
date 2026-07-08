from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from latentbrain.data.schemas import NeuralDataset, NeuronMask, TrialSplit

SPARSITY_COLUMNS = [
    "bin_size_ms",
    "split",
    "time_bins",
    "window_seconds",
    "n_trials",
    "n_heldout_neurons",
    "spike_count",
    "total_observations",
    "zero_fraction",
    "observed_rate_hz",
    "mean_spikes_per_bin",
]

BASELINE_COLUMNS = [
    "bin_size_ms",
    "method_name",
    "split",
    "prediction_source",
    "time_bins",
    "window_seconds",
    "spike_count",
    "zero_fraction",
    "observed_rate_hz",
    "poisson_nll",
    "poisson_log_likelihood",
    "reference_log_likelihood",
    "bits_per_spike",
    "mse_rate_hz",
    "mae_rate_hz",
]

LFADS_COLUMNS = [
    "bin_size_ms",
    "run_id",
    "split",
    "prediction_source",
    "time_bins",
    "window_seconds",
    "spike_count",
    "zero_fraction",
    "observed_rate_hz",
    "validation_total_loss",
    "heldout_prediction_loss",
    "poisson_nll",
    "poisson_log_likelihood",
    "reference_log_likelihood",
    "bits_per_spike",
    "behavior_mean_r2",
    "checkpoint_path",
]


def compute_window_bins_for_duration(duration_seconds: float, bin_size_ms: int) -> int:
    """Convert a fixed duration to an integer number of temporal bins."""
    if duration_seconds <= 0.0 or bin_size_ms <= 0:
        msg = "duration_seconds and bin_size_ms must be positive"
        raise ValueError(msg)
    bins = duration_seconds * 1000.0 / bin_size_ms
    rounded = round(bins)
    if abs(bins - rounded) > 1e-8:
        msg = "window duration must convert to an integer number of bins"
        raise ValueError(msg)
    return int(rounded)


def _split_ids(split: TrialSplit, name: str) -> np.ndarray:
    if name == "train":
        return split.train
    if name == "validation":
        return split.validation
    if name == "test":
        return split.test
    msg = f"unknown split: {name}"
    raise ValueError(msg)


def compute_sparsity_summary(
    dataset: NeuralDataset,
    split: TrialSplit,
    neuron_mask: NeuronMask,
    bin_size_ms: int,
    window_bins: int,
) -> pd.DataFrame:
    """Summarize held-out sparsity by split for a fixed cropped window."""
    heldout = np.flatnonzero(neuron_mask.heldout)
    if heldout.size == 0:
        msg = "heldout neuron mask is empty"
        raise ValueError(msg)
    index_by_trial = {int(trial_id): index for index, trial_id in enumerate(dataset.trial_ids)}
    rows = []
    time_bins = min(window_bins, dataset.spikes.shape[1])
    for split_name in ("train", "validation", "test"):
        ids = _split_ids(split, split_name)
        trial_indices = np.asarray([index_by_trial[int(trial_id)] for trial_id in ids], dtype=int)
        counts = dataset.spikes[trial_indices, :time_bins, :][:, :, heldout]
        total = int(counts.size)
        spikes = float(np.sum(counts))
        seconds = total * (bin_size_ms / 1000.0)
        rows.append(
            {
                "bin_size_ms": int(bin_size_ms),
                "split": split_name,
                "time_bins": int(time_bins),
                "window_seconds": float(time_bins * bin_size_ms / 1000.0),
                "n_trials": int(len(ids)),
                "n_heldout_neurons": int(heldout.size),
                "spike_count": spikes,
                "total_observations": total,
                "zero_fraction": float(np.mean(counts == 0.0)) if total else float("nan"),
                "observed_rate_hz": float(spikes / seconds) if seconds > 0 else float("nan"),
                "mean_spikes_per_bin": float(np.mean(counts)) if total else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=SPARSITY_COLUMNS)


def build_binning_comparison_row(
    method_name: str,
    bin_size_ms: int,
    split: str,
    metrics: dict[str, float],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Attach bin-size metadata to one baseline metric row."""
    return {
        "bin_size_ms": int(bin_size_ms),
        "method_name": method_name,
        "split": split,
        "prediction_source": metadata.get("prediction_source", method_name),
        "time_bins": int(metadata.get("time_bins", 0)),
        "window_seconds": float(metadata.get("window_seconds", 0.0)),
        "spike_count": float(metrics.get("spike_count", float("nan"))),
        "zero_fraction": float(metadata.get("zero_fraction", float("nan"))),
        "observed_rate_hz": float(metadata.get("observed_rate_hz", float("nan"))),
        "poisson_nll": float(metrics.get("poisson_nll", float("nan"))),
        "poisson_log_likelihood": float(metrics.get("poisson_log_likelihood", float("nan"))),
        "reference_log_likelihood": float(metrics.get("reference_log_likelihood", float("nan"))),
        "bits_per_spike": float(metrics.get("bits_per_spike", float("nan"))),
        "mse_rate_hz": float(metrics.get("mse_rate_hz", float("nan"))),
        "mae_rate_hz": float(metrics.get("mae_rate_hz", float("nan"))),
    }
