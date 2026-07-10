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
from latentbrain.eval.window_audit import SCORE_COLUMNS


def _script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_window_audit", Path("scripts/run_window_audit.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_dataset(
    path: Path, trials: int = 40, time_bins: int = 64, neurons: int = 12, behavior: bool = True
) -> None:
    generator = np.random.default_rng(7)
    spikes = generator.poisson(0.4, size=(trials, time_bins, neurons)).astype(np.int64)
    behavior_array = None
    names = None
    if behavior:
        behavior_array = np.zeros((trials, time_bins, 4))
        angles = np.linspace(-np.pi, np.pi, trials, endpoint=False)
        for trial in range(trials):
            radius = 1.0 + generator.random()
            ramp = np.zeros(time_bins)
            ramp[20:] = np.linspace(0.0, 1.0, time_bins - 20)
            behavior_array[trial, :, 0] = ramp * radius * np.cos(angles[trial])
            behavior_array[trial, :, 1] = ramp * radius * np.sin(angles[trial])
            behavior_array[trial, :, 2] = behavior_array[trial, :, 0]
            behavior_array[trial, :, 3] = behavior_array[trial, :, 1]
        names = ["hand_pos_x", "hand_pos_y", "cursor_pos_x", "cursor_pos_y"]
    dataset = NeuralDataset(
        spikes=spikes,
        rates=None,
        latents=None,
        trial_ids=np.arange(trials),
        time_ms=np.arange(time_bins) * 5.0,
        bin_size_ms=5,
        metadata={"name": "unit"},
        behavior=behavior_array,
        behavior_names=names,
    )
    save_neural_dataset(dataset, path)


def _config(tmp_path: Path, processed: Path) -> dict[str, Any]:
    return {
        "dataset": {"name": "unit", "processed_path": str(processed), "original_bin_size_ms": 5},
        "binning": {"target_bin_size_ms": 20},
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
        "window_candidates": [
            {
                "name": "from_start_1p28s",
                "crop_policy": "from_start",
                "duration_seconds": 0.08,
                "start_seconds": 0.0,
                "report_label": "Current accepted early window",
            },
            {
                "name": "behavior_speed_peak_centered_1p28s",
                "crop_policy": "behavior_speed_peak_centered",
                "duration_seconds": 0.08,
                "report_label": "Centered on peak hand speed",
            },
            {
                "name": "behavior_movement_onset_1p28s",
                "crop_policy": "behavior_movement_onset",
                "duration_seconds": 0.08,
                "pre_event_seconds": 0.02,
                "speed_threshold_quantile": 0.7,
                "report_label": "Movement-onset aligned window",
            },
        ],
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
                "smoothing_sigma_ms": 200.0,
                "heldout_decoder_alpha": 10000.0,
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
        "stratification": {
            "use_endpoint_direction": True,
            "endpoint_direction_bins": 4,
            "use_endpoint_distance": True,
            "endpoint_distance_bins": 2,
            "use_mean_speed": False,
            "mean_speed_bins": 2,
            "use_population_rate": True,
            "population_rate_bins": 2,
            "use_heldout_rate": False,
            "heldout_rate_bins": 2,
            "fallback_when_behavior_missing": "rate_only",
        },
        "statistics": {
            "confidence_interval": 0.95,
            "bootstrap_repeats": 50,
            "bootstrap_seed": 1337,
        },
        "inputs": {},
        "references": {
            "current_from_start_factor_latent_ci95_low": -1.0,
            "train_mean_validation_bits_per_spike": 0.0,
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


def test_missing_behavior_for_aligned_windows_fails_clearly(tmp_path: Path) -> None:
    module = _script_module()
    processed = tmp_path / "data.npz"
    _write_dataset(processed, behavior=False)
    config = _config(tmp_path, processed)

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 2


def test_from_start_only_config_runs_without_behavior(tmp_path: Path) -> None:
    module = _script_module()
    processed = tmp_path / "data.npz"
    _write_dataset(processed, behavior=False)
    config = _config(tmp_path, processed)
    config["window_candidates"] = [config["window_candidates"][0]]
    config["stratification"]["use_endpoint_direction"] = False
    config["stratification"]["use_endpoint_distance"] = False

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 0


def test_duplicate_window_names_fail_validation(tmp_path: Path) -> None:
    module = _script_module()
    processed = tmp_path / "data.npz"
    _write_dataset(processed)
    config = _config(tmp_path, processed)
    config["window_candidates"][1]["name"] = config["window_candidates"][0]["name"]

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 2


def test_invalid_control_marked_reportable_fails_validation(tmp_path: Path) -> None:
    module = _script_module()
    processed = tmp_path / "data.npz"
    _write_dataset(processed)
    config = _config(tmp_path, processed)
    config["methods"][2]["reportable_as_model_performance"] = True

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 2


def test_script_run_writes_expected_outputs(tmp_path: Path) -> None:
    module = _script_module()
    processed = tmp_path / "data.npz"
    _write_dataset(processed)
    config = _config(tmp_path, processed)

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 0

    out = Path(config["reporting"]["output_dir"])
    for name in (
        "window_audit_summary.json",
        "window_candidate_scores.csv",
        "window_behavior_statistics.csv",
        "window_balance_statistics.csv",
        "window_recommendations.json",
        "window_audit_report.md",
    ):
        assert (out / name).exists(), name
    for name in (
        "factor_latent_by_window.png",
        "behavior_coverage_by_window.png",
        "endpoint_direction_entropy_by_window.png",
        "speed_profile_windows.png",
        "invalid_control_gap_by_window.png",
    ):
        assert (out / "figures" / name).exists(), name

    scores = pd.read_csv(out / "window_candidate_scores.csv")
    assert list(scores.columns) == SCORE_COLUMNS
    assert set(scores["window_name"]) == {
        "from_start_1p28s",
        "behavior_speed_peak_centered_1p28s",
        "behavior_movement_onset_1p28s",
    }
    # The canonical reference scored against itself must be exactly zero on every window and fold.
    reference = scores[scores["method_name"] == "train_mean_rate"]
    assert np.allclose(reference["unified_bits_per_spike"], 0.0, atol=1e-12)

    behavior = pd.read_csv(out / "window_behavior_statistics.csv")
    assert set(behavior["behavior_source"]) == {"hand_pos"}


def test_summary_excludes_invalid_controls_from_selection(tmp_path: Path) -> None:
    module = _script_module()
    processed = tmp_path / "data.npz"
    _write_dataset(processed)
    config = _config(tmp_path, processed)

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 0

    out = Path(config["reporting"]["output_dir"])
    summary = json.loads((out / "window_audit_summary.json").read_text(encoding="utf-8"))

    assert summary["best_valid_method"] == "factor_latent"
    assert summary["invalid_controls_excluded_from_window_selection"] is True
    assert summary["invalid_controls_excluded_from_valid_model_selection"] is True
    assert summary["recommended_reporting_mode"] == "stratified_cross_validation"
    assert summary["single_split_results_reportable"] is False
    assert summary["official_benchmark_claim"] is False

    recommendations = json.loads((out / "window_recommendations.json").read_text(encoding="utf-8"))
    assert recommendations["recommended_window_name"] == summary["recommended_window_name"]
    assert recommendations["official_benchmark_claim"] is False


def test_report_states_invalid_controls_and_not_official(tmp_path: Path) -> None:
    module = _script_module()
    processed = tmp_path / "data.npz"
    _write_dataset(processed)
    config = _config(tmp_path, processed)

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 0

    report = (Path(config["reporting"]["output_dir"]) / "window_audit_report.md").read_text(
        encoding="utf-8"
    )
    assert (
        "Invalid controls use evaluation fold targets and cannot be reported as model performance."
        in report
    )
    assert "not an official NLB leaderboard result" in report
    assert "Old incompatible mean-rate values are not used as tuning targets" in report
    assert "Endpoint direction entropy by window" in report
    assert "Recommended window:" in report
    assert "never on invalid-control gains" in report
