from __future__ import annotations

import json
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from latentbrain.eval.reporting import (
    write_baseline_markdown_report,
    write_baseline_outputs,
    write_cosmoothing_markdown_report,
)


def _split_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["validation"],
            "neuron_group": ["heldout"],
            "n_trials": [1],
            "n_neurons": [2],
            "n_time_bins": [3],
            "spike_count": [4.0],
            "poisson_nll": [5.0],
            "poisson_log_likelihood": [-5.0],
            "reference_log_likelihood": [-6.0],
            "bits_per_spike": [0.1],
            "mean_predicted_rate_hz": [7.0],
        }
    )


def _neuron_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "neuron_index": [0],
            "neuron_group": ["heldout"],
            "train_mean_rate_hz": [1.0],
            "total_spikes_all_trials": [2],
            "validation_spikes": [1],
            "test_spikes": [1],
        }
    )


def test_baseline_outputs_write_json_csv_and_markdown(tmp_path: Path) -> None:
    summary = {
        "dataset_name": "mc_maze_small",
        "dataset_hash": "abc",
        "baseline_name": "mean_rate",
        "train_only_fit": True,
        "primary_split": "validation",
        "primary_neuron_group": "heldout",
        "primary_bits_per_spike": 0.1,
        "primary_poisson_nll": 5.0,
    }

    outputs = write_baseline_outputs(tmp_path, summary, _split_metrics(), _neuron_metrics())

    assert json.loads(outputs["metrics_summary"].read_text())["baseline_name"] == "mean_rate"
    assert outputs["split_metrics"].read_text().startswith("split,neuron_group")
    report = outputs["baseline_report"].read_text(encoding="utf-8")
    assert "mc_maze_small" in report
    assert "local sanity baseline, not an official NLB leaderboard result" in report
    assert "No neural network model was trained" in report


def test_markdown_report_mentions_primary_metric_and_fit_policy(tmp_path: Path) -> None:
    report_path = write_baseline_markdown_report(
        tmp_path / "baseline_report.md",
        dataset_name="mc_maze_small",
        metrics_summary={
            "dataset_hash": "abc",
            "baseline_name": "mean_rate",
            "train_only_fit": True,
            "primary_split": "validation",
            "primary_neuron_group": "heldout",
            "primary_bits_per_spike": 0.1,
        },
        split_metrics=_split_metrics(),
        neuron_metrics=_neuron_metrics(),
    )

    report = report_path.read_text(encoding="utf-8")
    assert "Primary metric: validation heldout bits/spike" in report
    assert "Fit policy: train trials only" in report


def test_cosmoothing_report_includes_disclaimers(tmp_path: Path) -> None:
    report_path = write_cosmoothing_markdown_report(
        tmp_path / "cosmoothing_report.md",
        metrics_summary={
            "dataset_name": "mc_maze_small",
            "dataset_hash": "abc",
            "input_neuron_group": "heldin",
            "target_neuron_group": "heldout",
            "smoothing": {"method": "gaussian", "sigma_ms": 50.0},
            "decoder_name": "ridge",
            "decoder_alpha": 100.0,
            "fit_policy": "train trials only",
            "standardization_policy": "train-only held-in features",
            "reference_policy": "train-only held-out mean rates",
            "primary_split": "validation",
            "primary_bits_per_spike": 0.1,
            "primary_poisson_nll": 5.0,
        },
        split_metrics=_split_metrics(),
    )

    report = report_path.read_text(encoding="utf-8")
    assert "local co-smoothing sanity baseline, not an official NLB leaderboard result" in report
    assert "No neural network model was trained" in report
