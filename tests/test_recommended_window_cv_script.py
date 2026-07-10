from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml

from latentbrain.data.io import save_neural_dataset
from latentbrain.data.schemas import NeuralDataset
from latentbrain.eval.recommended_window_cv import (
    BEHAVIOR_STATISTICS_COLUMNS,
    FOLD_BALANCE_COLUMNS,
    LEAKAGE_DIAGNOSTIC_COLUMNS,
    SCORE_COLUMNS,
)
from latentbrain.eval.stratified_cv import METHOD_SUMMARY_COLUMNS

EXPECTED_OUTPUTS = (
    "recommended_window_cv_summary.json",
    "recommended_window_scores.csv",
    "recommended_window_method_summary.csv",
    "recommended_window_fold_assignments.csv",
    "recommended_window_behavior_statistics.csv",
    "recommended_window_fold_balance.csv",
    "recommended_window_leakage_diagnostics.csv",
    "recommended_window_protocol.yaml",
    "recommended_window_cv_report.md",
)


def _script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_recommended_window_cv", Path("scripts/run_recommended_window_cv.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_dataset(path: Path, *, behavior: bool = True) -> None:
    generator = np.random.default_rng(13)
    trials, time_bins, neurons = 24, 48, 12
    behavior_values = None
    names = None
    if behavior:
        behavior_values = np.zeros((trials, time_bins, 4), dtype=np.float64)
        angles = np.linspace(-np.pi, np.pi, trials, endpoint=False)
        ramp = np.zeros(time_bins)
        ramp[20:36] = np.linspace(0.0, 1.0, 16)
        ramp[36:] = 1.0
        for trial, angle in enumerate(angles):
            behavior_values[trial, :, 0] = ramp * np.cos(angle)
            behavior_values[trial, :, 1] = ramp * np.sin(angle)
            behavior_values[trial, :, 2:] = behavior_values[trial, :, :2]
        names = ["hand_pos_x", "hand_pos_y", "cursor_pos_x", "cursor_pos_y"]
    save_neural_dataset(
        NeuralDataset(
            spikes=generator.poisson(0.25, size=(trials, time_bins, neurons)).astype(np.int64),
            rates=None,
            latents=None,
            trial_ids=np.arange(trials),
            time_ms=np.arange(time_bins) * 5.0,
            bin_size_ms=5,
            metadata={"name": "unit"},
            behavior=behavior_values,
            behavior_names=names,
        ),
        path,
    )


def _config(tmp_path: Path, processed: Path) -> dict[str, Any]:
    return {
        "dataset": {"name": "unit", "processed_path": str(processed), "original_bin_size_ms": 5},
        "binning": {"target_bin_size_ms": 20},
        "window": {
            "name": "behavior_speed_peak_centered_1p28s",
            "crop_policy": "behavior_speed_peak_centered",
            "duration_seconds": 0.08,
            "report_label": "Peak-speed-centered reach window",
        },
        "scoring": {
            "reference_model": "train_heldout_mean_rate",
            "include_poisson_constant": True,
            "min_rate_hz": 1.0e-4,
            "max_rate_hz": 500.0,
            "primary_metric": "unified_bits_per_spike",
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
        "methods": [
            {
                "name": "train_mean_rate",
                "type": "rate_control",
                "valid_model": False,
                "reportable_as_model_performance": False,
                "notes": "Canonical reference; not a competitor.",
            },
            {
                "name": "factor_latent",
                "type": "factor_latent",
                "valid_model": True,
                "reportable_as_model_performance": True,
                "latent_dim": 3,
                "smoothing_sigma_ms": 40.0,
                "heldout_decoder_alpha": 100.0,
                "standardize_features": True,
                "fit_intercept": True,
                "factor_analysis_random_state": 0,
                "notes": "Carried-forward valid baseline.",
            },
            {
                "name": "split_mean_rate_invalid",
                "type": "invalid_control",
                "valid_model": False,
                "reportable_as_model_performance": False,
                "invalid_reason": "Uses evaluation fold target counts.",
                "notes": "Leakage diagnostic only.",
            },
        ],
        "statistics": {
            "confidence_interval": 0.95,
            "bootstrap_repeats": 50,
            "bootstrap_seed": 1337,
        },
        "inputs": {},
        "references": {
            "previous_from_start_factor_latent_mean": 0.025,
            "previous_from_start_split_mean_invalid_mean": 0.080,
        },
        "reporting": {"output_dir": str(tmp_path / "out")},
    }


def _write_config(tmp_path: Path, config: dict[str, Any]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_missing_processed_data_fails_clearly(tmp_path: Path) -> None:
    module = _script_module()
    config = _config(tmp_path, tmp_path / "missing.npz")
    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 2


def test_missing_behavior_fails_clearly(tmp_path: Path) -> None:
    module = _script_module()
    processed = tmp_path / "dataset.npz"
    _write_dataset(processed, behavior=False)
    config = _config(tmp_path, processed)
    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 2


def test_script_run_writes_expected_outputs_and_claim_safe_report(tmp_path: Path) -> None:
    module = _script_module()
    processed = tmp_path / "dataset.npz"
    _write_dataset(processed)
    config = _config(tmp_path, processed)

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 0

    output_dir = Path(config["reporting"]["output_dir"])
    for name in EXPECTED_OUTPUTS:
        assert (output_dir / name).exists(), name
    for name in (
        "recommended_window_score_distribution.png",
        "valid_vs_invalid_by_fold.png",
        "movement_coverage_summary.png",
        "fold_balance_summary.png",
    ):
        assert (output_dir / "figures" / name).exists(), name

    behavior = pd.read_csv(output_dir / "recommended_window_behavior_statistics.csv")
    assert list(behavior.columns) == BEHAVIOR_STATISTICS_COLUMNS
    assert list(pd.read_csv(output_dir / "recommended_window_scores.csv").columns) == SCORE_COLUMNS
    assert (
        list(pd.read_csv(output_dir / "recommended_window_method_summary.csv").columns)
        == METHOD_SUMMARY_COLUMNS
    )
    assert (
        list(pd.read_csv(output_dir / "recommended_window_fold_balance.csv").columns)
        == FOLD_BALANCE_COLUMNS
    )
    assert (
        list(pd.read_csv(output_dir / "recommended_window_leakage_diagnostics.csv").columns)
        == LEAKAGE_DIAGNOSTIC_COLUMNS
    )
    summary = json.loads(
        (output_dir / "recommended_window_cv_summary.json").read_text(encoding="utf-8")
    )
    assert summary["single_split_results_reportable"] is False
    assert summary["invalid_controls_excluded_from_model_selection"] is True

    report = (output_dir / "recommended_window_cv_report.md").read_text(encoding="utf-8")
    assert "not performance improvements over from-start" in report
    assert (
        "Invalid controls use evaluation fold targets and cannot be reported as model performance."
        in report
    )
    assert "not an official NLB leaderboard result" in report
