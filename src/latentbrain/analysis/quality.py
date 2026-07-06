from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from latentbrain.data.schemas import NeuralDataset


def _bin_seconds(dataset: NeuralDataset) -> float:
    return dataset.bin_size_ms / 1000.0


def _zero_fraction(array: np.ndarray) -> float:
    return float(np.mean(array == 0)) if array.size else 0.0


def _rate(total_spikes: float, bins: int, bin_seconds: float) -> float:
    return 0.0 if bins == 0 else float(total_spikes / (bins * bin_seconds))


def compute_dataset_summary(
    dataset: NeuralDataset,
    dataset_hash: str | None = None,
) -> dict[str, Any]:
    """Compute scalar quality metrics for a trial-major neural dataset."""
    spikes = dataset.spikes
    n_trials, n_time_bins, n_neurons = spikes.shape
    bin_seconds = _bin_seconds(dataset)
    neuron_rates = spikes.sum(axis=(0, 1)) / (n_trials * n_time_bins * bin_seconds)
    total_spikes = float(np.nansum(spikes))
    summary: dict[str, Any] = {
        "n_trials": int(n_trials),
        "n_time_bins": int(n_time_bins),
        "n_neurons": int(n_neurons),
        "bin_size_ms": int(dataset.bin_size_ms),
        "duration_seconds": float(n_time_bins * bin_seconds),
        "total_spikes": int(total_spikes) if np.isfinite(total_spikes) else total_spikes,
        "mean_spikes_per_trial": float(total_spikes / n_trials),
        "mean_spikes_per_neuron": float(total_spikes / n_neurons),
        "mean_population_rate_hz": _rate(total_spikes, n_trials * n_time_bins, bin_seconds),
        "median_neuron_rate_hz": float(np.nanmedian(neuron_rates)),
        "min_neuron_rate_hz": float(np.nanmin(neuron_rates)),
        "max_neuron_rate_hz": float(np.nanmax(neuron_rates)),
        "zero_fraction": _zero_fraction(spikes),
        "dataset_hash": dataset_hash,
        "has_rates": dataset.rates is not None,
        "has_latents": dataset.latents is not None,
    }
    return summary


def compute_neuron_activity(dataset: NeuralDataset) -> pd.DataFrame:
    """Compute per-neuron spike totals, rates, and sparsity."""
    spikes = dataset.spikes
    n_trials, n_time_bins, n_neurons = spikes.shape
    totals = spikes.sum(axis=(0, 1))
    rates = totals / (n_trials * n_time_bins * _bin_seconds(dataset))
    zero_fraction = np.mean(spikes == 0, axis=(0, 1))
    ranks = pd.Series(-totals).rank(method="first").astype(int).to_numpy()
    return pd.DataFrame(
        {
            "neuron_index": np.arange(n_neurons, dtype=np.int64),
            "total_spikes": totals.astype(np.int64, copy=False),
            "mean_rate_hz": rates.astype(float, copy=False),
            "zero_fraction": zero_fraction.astype(float, copy=False),
            "active": totals > 0,
            "activity_rank": ranks,
        }
    )


def compute_trial_activity(dataset: NeuralDataset) -> pd.DataFrame:
    """Compute per-trial spike totals, population rates, and sparsity."""
    spikes = dataset.spikes
    n_time_bins = spikes.shape[1]
    totals = spikes.sum(axis=(1, 2))
    rates = totals / (n_time_bins * _bin_seconds(dataset))
    return pd.DataFrame(
        {
            "trial_id": dataset.trial_ids.astype(np.int64, copy=False),
            "total_spikes": totals.astype(np.int64, copy=False),
            "mean_population_rate_hz": rates.astype(float, copy=False),
            "zero_fraction": np.mean(spikes == 0, axis=(1, 2)).astype(float, copy=False),
        }
    )


def compute_time_activity(dataset: NeuralDataset) -> pd.DataFrame:
    """Compute per-time-bin population activity across trials and neurons."""
    spikes = dataset.spikes
    n_trials = spikes.shape[0]
    totals = spikes.sum(axis=(0, 2))
    rates = totals / (n_trials * _bin_seconds(dataset))
    return pd.DataFrame(
        {
            "time_bin": np.arange(spikes.shape[1], dtype=np.int64),
            "time_ms": dataset.time_ms.astype(float, copy=False),
            "total_spikes": totals.astype(np.int64, copy=False),
            "mean_population_rate_hz": rates.astype(float, copy=False),
            "zero_fraction": np.mean(spikes == 0, axis=(0, 2)).astype(float, copy=False),
        }
    )


def compute_split_activity_summary(
    dataset: NeuralDataset,
    train_ids: np.ndarray,
    validation_ids: np.ndarray,
    test_ids: np.ndarray,
) -> pd.DataFrame:
    """Summarize activity for deterministic train, validation, and test trial splits."""
    rows: list[dict[str, Any]] = []
    for name, ids in (
        ("train", train_ids),
        ("validation", validation_ids),
        ("test", test_ids),
    ):
        mask = np.isin(dataset.trial_ids, ids)
        spikes = dataset.spikes[mask]
        total = float(spikes.sum()) if spikes.size else 0.0
        bins = int(mask.sum()) * dataset.spikes.shape[1]
        rows.append(
            {
                "split": name,
                "trial_count": int(mask.sum()),
                "total_spikes": int(total),
                "mean_population_rate_hz": _rate(total, bins, _bin_seconds(dataset)),
                "zero_fraction": _zero_fraction(spikes),
            }
        )
    return pd.DataFrame(rows)


def compute_quality_flags(
    dataset: NeuralDataset,
    summary: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[dict[str, str]]:
    """Compute error and warning flags without raising."""
    flags: list[dict[str, str]] = []
    spikes = dataset.spikes
    nan_count = int(np.isnan(spikes).sum())
    inf_count = int(np.isinf(spikes).sum())
    if nan_count > int(thresholds["max_nan_count"]):
        flags.append(
            {"severity": "error", "code": "nan_spikes", "message": f"NaN spikes: {nan_count}"}
        )
    if inf_count > int(thresholds["max_inf_count"]):
        flags.append(
            {"severity": "error", "code": "inf_spikes", "message": f"Inf spikes: {inf_count}"}
        )
    if float(summary["total_spikes"]) < float(thresholds["min_total_spikes"]):
        flags.append(
            {"severity": "error", "code": "zero_total_spikes", "message": "Total spikes are zero"}
        )
    if float(summary["zero_fraction"]) > float(thresholds["max_zero_fraction_warning"]):
        flags.append(
            {
                "severity": "warning",
                "code": "high_zero_fraction",
                "message": f"Zero fraction is {summary['zero_fraction']:.4f}",
            }
        )
    if np.all(np.isfinite(spikes)):
        neuron_activity = compute_neuron_activity(dataset)
        inactive_count = int(
            (
                neuron_activity["mean_rate_hz"]
                < float(thresholds["inactive_neuron_rate_hz_threshold"])
            ).sum()
        )
        if inactive_count:
            flags.append(
                {
                    "severity": "warning",
                    "code": "inactive_neurons",
                    "message": f"Inactive neurons below threshold: {inactive_count}",
                }
            )
    max_rate = float(summary.get("max_neuron_rate_hz", 0.0))
    if np.isfinite(max_rate) and max_rate > float(thresholds["high_rate_warning_hz"]):
        flags.append(
            {
                "severity": "warning",
                "code": "high_neuron_rate",
                "message": f"Max neuron rate is {summary['max_neuron_rate_hz']:.3f} Hz",
            }
        )
    return flags
