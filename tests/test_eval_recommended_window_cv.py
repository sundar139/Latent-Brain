from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from latentbrain.data.io import save_neural_dataset
from latentbrain.data.schemas import NeuralDataset
from latentbrain.eval.recommended_window_cv import (
    build_recommended_window_dataset,
    build_recommended_window_protocol,
    summarize_recommended_window_cv,
)


def _write_dataset(path: Path, *, behavior: bool = True) -> None:
    trials, time_bins, neurons = 20, 40, 12
    behavior_values = None
    behavior_names = None
    if behavior:
        behavior_values = np.zeros((trials, time_bins, 4), dtype=np.float64)
        angles = np.linspace(-np.pi, np.pi, trials, endpoint=False)
        ramp = np.zeros(time_bins)
        ramp[24:] = np.linspace(0.0, 1.0, time_bins - 24)
        for trial, angle in enumerate(angles):
            behavior_values[trial, :, 0] = ramp * np.cos(angle)
            behavior_values[trial, :, 1] = ramp * np.sin(angle)
            behavior_values[trial, :, 2:] = behavior_values[trial, :, :2]
        behavior_names = ["hand_pos_x", "hand_pos_y", "cursor_pos_x", "cursor_pos_y"]
    save_neural_dataset(
        NeuralDataset(
            spikes=np.random.default_rng(11)
            .poisson(0.2, size=(trials, time_bins, neurons))
            .astype(np.int64),
            rates=None,
            latents=None,
            trial_ids=np.arange(trials),
            time_ms=np.arange(time_bins) * 5.0,
            bin_size_ms=5,
            metadata={"name": "unit"},
            behavior=behavior_values,
            behavior_names=behavior_names,
        ),
        path,
    )


def _config(path: Path) -> dict[str, Any]:
    return {
        "dataset": {"name": "unit", "processed_path": str(path), "original_bin_size_ms": 5},
        "binning": {"target_bin_size_ms": 20},
        "window": {
            "name": "behavior_speed_peak_centered_1p28s",
            "crop_policy": "behavior_speed_peak_centered",
            "duration_seconds": 0.08,
            "report_label": "Peak-speed-centered reach window",
        },
        "cross_validation": {
            "fold_count": 4,
            "repeats": 2,
            "base_seed": 2027,
            "heldout_neuron_fraction": 0.25,
            "assignment_method": "greedy_balanced",
            "min_trials_per_stratum": 2,
        },
        "stratification": {
            "use_endpoint_direction": True,
            "endpoint_direction_bins": 4,
            "use_endpoint_distance": True,
            "endpoint_distance_bins": 2,
            "use_mean_speed": True,
            "mean_speed_bins": 2,
            "use_population_rate": True,
            "population_rate_bins": 2,
            "use_heldout_rate": True,
            "heldout_rate_bins": 2,
            "fallback_when_behavior_missing": "fail",
        },
        "scoring": {
            "reference_model": "train_heldout_mean_rate",
            "include_poisson_constant": True,
            "min_rate_hz": 1.0e-4,
            "max_rate_hz": 500.0,
            "primary_metric": "unified_bits_per_spike",
        },
        "methods": [],
        "statistics": {
            "confidence_interval": 0.95,
            "bootstrap_repeats": 50,
            "bootstrap_seed": 1337,
        },
        "references": {},
        "reporting": {"output_dir": str(path.parent / "out")},
    }


def test_recommended_window_dataset_builder_applies_peak_speed_centered_window(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dataset.npz"
    _write_dataset(path)

    built = build_recommended_window_dataset(_config(path))

    dataset = built["dataset"]
    slices = built["window_slices"]
    assert dataset.bin_size_ms == 20
    assert dataset.spikes.shape[1] == 4
    assert bool((slices["start_bin"] > 0).any())
    assert built["behavior_statistics"]["moving_bin_fraction"].mean() > 0.0


def test_summary_is_claim_safe_and_detects_disappearing_leakage_dominance() -> None:
    scores = pd.DataFrame(
        {
            "method_name": ["factor_latent", "factor_latent", "split_mean_rate_invalid"] * 2,
            "method_type": ["factor_latent", "factor_latent", "invalid_control"] * 2,
            "valid_model": [True, True, False] * 2,
            "reportable_as_model_performance": [True, True, False] * 2,
            "unified_bits_per_spike": [0.08, 0.06, 0.05, 0.09, 0.07, 0.06],
            "notes": ["valid", "valid", "invalid"] * 2,
        }
    )
    behavior = pd.DataFrame(
        {"moving_bin_fraction": [0.5, 0.7], "endpoint_angle_rad": [0.0, np.pi / 2.0]}
    )
    balance = pd.DataFrame(
        {"endpoint_direction_entropy": [1.9, 2.0], "fold_balance_warning": ["none", "none"]}
    )

    summary = summarize_recommended_window_cv(scores, behavior, balance, {})

    for key in (
        "recommended_window_name",
        "recommended_reporting_mode",
        "factor_latent_mean",
        "factor_latent_ci95_low",
        "factor_latent_positive_fraction",
        "split_mean_invalid_mean",
        "factor_latent_minus_split_mean_invalid",
        "moving_bin_fraction_mean",
        "endpoint_direction_entropy_mean",
        "fold_balance_warning",
    ):
        assert key in summary
    assert summary["factor_latent_beats_invalid_control_mean"] is True
    assert summary["leakage_dominance_persists"] is False
    assert summary["single_split_results_reportable"] is False
    assert summary["official_leaderboard_claim"] is False
    assert summary["old_mean_rate_values_used_as_targets"] is False
    assert summary["invalid_controls_excluded_from_model_selection"] is True


def test_protocol_freezes_recommended_window_settings(tmp_path: Path) -> None:
    config = _config(tmp_path / "dataset.npz")
    config["methods"] = [
        {"name": "factor_latent", "valid_model": True},
        {"name": "split_mean_rate_invalid", "valid_model": False},
    ]
    summary = {"protocol_frozen": True, "factor_latent_beats_invalid_control_mean": True}

    protocol = build_recommended_window_protocol(config, summary)

    assert protocol["window"]["name"] == "behavior_speed_peak_centered_1p28s"
    assert protocol["window"]["crop_policy"] == "behavior_speed_peak_centered"
    assert protocol["cross_validation"]["fold_count"] == 4
    assert protocol["stratification"]["fallback_when_behavior_missing"] == "fail"
    assert protocol["claim_safety"]["invalid_controls_excluded_from_model_selection"] is True
    assert protocol["protocol_frozen"] is True
