from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
from matplotlib import pyplot as plt


def _save(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_neuron_firing_rates(neuron_activity: pd.DataFrame, output_path: Path) -> Path:
    plt.figure()
    plt.hist(neuron_activity["mean_rate_hz"], bins=30)
    plt.title("Neuron firing-rate distribution")
    plt.xlabel("Mean rate (Hz)")
    plt.ylabel("Neuron count")
    return _save(output_path)


def plot_trial_spike_counts(trial_activity: pd.DataFrame, output_path: Path) -> Path:
    plt.figure()
    plt.hist(trial_activity["total_spikes"], bins=30)
    plt.title("Trial spike-count distribution")
    plt.xlabel("Total spikes per trial")
    plt.ylabel("Trial count")
    return _save(output_path)


def plot_population_rate_over_time(time_activity: pd.DataFrame, output_path: Path) -> Path:
    plt.figure()
    plt.plot(time_activity["time_ms"], time_activity["mean_population_rate_hz"])
    plt.title("Population rate over time")
    plt.xlabel("Time (ms)")
    plt.ylabel("Population rate (Hz)")
    return _save(output_path)


def plot_zero_fraction_by_neuron(neuron_activity: pd.DataFrame, output_path: Path) -> Path:
    plt.figure()
    plt.plot(
        neuron_activity["neuron_index"],
        neuron_activity["zero_fraction"],
        marker=".",
        linestyle="none",
    )
    plt.title("Zero-spike fraction by neuron")
    plt.xlabel("Neuron index")
    plt.ylabel("Zero fraction")
    return _save(output_path)


def plot_split_activity_summary(split_summary: pd.DataFrame, output_path: Path) -> Path:
    plt.figure()
    plt.bar(split_summary["split"], split_summary["mean_population_rate_hz"])
    plt.title("Split activity summary")
    plt.xlabel("Split")
    plt.ylabel("Mean population rate (Hz)")
    return _save(output_path)
