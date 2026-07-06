from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from latentbrain.analysis.figures import (
    plot_neuron_firing_rates,
    plot_population_rate_over_time,
    plot_split_activity_summary,
    plot_trial_spike_counts,
    plot_zero_fraction_by_neuron,
)
from latentbrain.analysis.reporting import write_json_report, write_markdown_validation_report


def test_json_report_writes_valid_json(tmp_path: Path) -> None:
    output_path = tmp_path / "summary.json"

    result = write_json_report({"n_trials": 2}, output_path)

    assert result == output_path
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"n_trials": 2}


def test_markdown_report_contains_required_statements(tmp_path: Path) -> None:
    output_path = tmp_path / "report.md"

    write_markdown_validation_report(
        output_path=output_path,
        dataset_name="mc_maze_small",
        summary={"dataset_hash": "abc", "n_trials": 2, "n_time_bins": 3, "n_neurons": 4},
        quality_flags=[],
        generated_tables={"neurons": "neuron_activity.csv"},
        generated_figures={"rates": "figures/neuron_firing_rates.png"},
        metadata={"neuron_mask_counts": {"heldin": 3, "heldout": 1}},
        provenance={"train_file_used": "train.nwb"},
    )

    text = output_path.read_text(encoding="utf-8")
    assert "mc_maze_small" in text
    assert "No model training or benchmark evaluation was performed in this report." in text
    assert "heldin" in text
    assert "train.nwb" in text


def test_markdown_report_includes_behavior_section(tmp_path: Path) -> None:
    output_path = tmp_path / "report.md"

    write_markdown_validation_report(
        output_path=output_path,
        dataset_name="mc_maze_small",
        summary={
            "dataset_hash": "abc",
            "n_trials": 2,
            "n_time_bins": 3,
            "n_neurons": 4,
            "behavior": {
                "has_behavior": True,
                "n_behavior_dims": 2,
                "behavior_names": ["hand_pos_x", "hand_pos_y"],
                "behavior_nan_count": 0,
                "behavior_inf_count": 0,
            },
        },
        quality_flags=[],
        generated_tables={"behavior_activity": "behavior_activity.csv"},
        generated_figures={},
        metadata=None,
        provenance=None,
    )

    text = output_path.read_text(encoding="utf-8")
    assert "## Behavior" in text
    assert "hand_pos_x" in text
    assert "behavior_activity.csv" in text


def test_figure_functions_create_png_files(tmp_path: Path) -> None:
    neuron_activity = pd.DataFrame(
        {"neuron_index": [0, 1], "mean_rate_hz": [1.0, 2.0], "zero_fraction": [0.0, 0.5]}
    )
    trial_activity = pd.DataFrame({"trial_id": [1, 2], "total_spikes": [3, 4]})
    time_activity = pd.DataFrame({"time_ms": [0.0, 5.0], "mean_population_rate_hz": [10.0, 20.0]})
    split_summary = pd.DataFrame(
        {"split": ["train", "validation"], "mean_population_rate_hz": [1.0, 2.0]}
    )

    paths = [
        plot_neuron_firing_rates(neuron_activity, tmp_path / "neuron.png"),
        plot_trial_spike_counts(trial_activity, tmp_path / "trial.png"),
        plot_population_rate_over_time(time_activity, tmp_path / "time.png"),
        plot_zero_fraction_by_neuron(neuron_activity, tmp_path / "zero.png"),
        plot_split_activity_summary(split_summary, tmp_path / "split.png"),
    ]

    for path in paths:
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0
