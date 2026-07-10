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
from latentbrain.eval.stratified_cv import (
    FACTOR_LATENT,
    FOLD_ASSIGNMENT_COLUMNS,
    SCORE_COLUMNS,
    SPLIT_MEAN_RATE_INVALID,
    TRAIN_MEAN_RATE,
)


def _script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_stratified_cv", Path("scripts/run_stratified_cv.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_dataset(path: Path, trials: int = 40, time_bins: int = 32, neurons: int = 16) -> None:
    generator = np.random.default_rng(9)
    spikes = generator.poisson(0.4, size=(trials, time_bins, neurons)).astype(np.int64)
    behavior = np.zeros((trials, time_bins, 4))
    angles = np.linspace(-np.pi, np.pi, trials, endpoint=False)
    for trial in range(trials):
        radius = 1.0 + generator.random()
        behavior[trial, :, 0] = np.linspace(0.0, radius * np.cos(angles[trial]), time_bins)
        behavior[trial, :, 1] = np.linspace(0.0, radius * np.sin(angles[trial]), time_bins)
        behavior[trial, :, 2] = behavior[trial, :, 0]
        behavior[trial, :, 3] = behavior[trial, :, 1]
    dataset = NeuralDataset(
        spikes=spikes,
        rates=None,
        latents=None,
        trial_ids=np.arange(trials),
        time_ms=np.arange(time_bins) * 5.0,
        bin_size_ms=5,
        metadata={"name": "unit"},
        behavior=behavior,
        behavior_names=["hand_pos_x", "hand_pos_y", "cursor_pos_x", "cursor_pos_y"],
    )
    save_neural_dataset(dataset, path)


def _config(tmp_path: Path, processed: Path) -> dict[str, Any]:
    return {
        "dataset": {"name": "unit", "processed_path": str(processed), "original_bin_size_ms": 5},
        "window": {"duration_seconds": 0.08, "crop_policy": "from_start"},
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
            },
            "fallback_when_behavior_missing": "rate_only",
            "min_trials_per_stratum": 2,
            "assignment_method": "greedy_balanced",
            "compare_random_splits": True,
            "random_split_repeats": 8,
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
        "statistics": {
            "confidence_interval": 0.95,
            "bootstrap_repeats": 50,
            "bootstrap_seed": 1337,
        },
        "inputs": {},
        "references": {
            "repeated_split_factor_latent_test_mean": 0.00897,
            "repeated_split_factor_latent_test_positive_fraction": 0.76,
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


def test_fold_count_below_three_fails_validation(tmp_path: Path) -> None:
    module = _script_module()
    processed = tmp_path / "data.npz"
    _write_dataset(processed)
    config = _config(tmp_path, processed)
    config["cross_validation"]["fold_count"] = 2

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 2


def test_invalid_control_marked_reportable_fails_validation(tmp_path: Path) -> None:
    module = _script_module()
    processed = tmp_path / "data.npz"
    _write_dataset(processed)
    config = _config(tmp_path, processed)
    config["methods"][2]["reportable_as_model_performance"] = True

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 2


def test_train_mean_rate_marked_reportable_fails_validation(tmp_path: Path) -> None:
    module = _script_module()
    processed = tmp_path / "data.npz"
    _write_dataset(processed)
    config = _config(tmp_path, processed)
    config["methods"][0]["reportable_as_model_performance"] = True

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 2


def test_script_run_writes_expected_outputs(tmp_path: Path) -> None:
    module = _script_module()
    processed = tmp_path / "data.npz"
    _write_dataset(processed)
    config = _config(tmp_path, processed)

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 0

    out = Path(config["reporting"]["output_dir"])
    for name in (
        "stratified_cv_summary.json",
        "stratified_cv_scores.csv",
        "stratified_fold_assignments.csv",
        "fold_balance_statistics.csv",
        "fold_balance_comparisons.csv",
        "stratified_cv_method_summary.csv",
        "stratified_cv_report.md",
    ):
        assert (out / name).exists(), name
    for name in (
        "stratified_cv_score_distribution.png",
        "fold_balance_endpoint_direction.png",
        "fold_balance_rate_distributions.png",
        "fold_balance_distance_speed.png",
        "random_vs_stratified_variance.png",
    ):
        assert (out / "figures" / name).exists(), name

    assignments = pd.read_csv(out / "stratified_fold_assignments.csv")
    assert list(assignments.columns) == FOLD_ASSIGNMENT_COLUMNS
    for _, group in assignments.groupby("repeat_index"):
        assert sorted(group["trial_index"]) == list(range(40))

    scores = pd.read_csv(out / "stratified_cv_scores.csv")
    assert list(scores.columns) == SCORE_COLUMNS
    assert len(scores) == 4 * 2 * 3
    # The canonical reference scored against itself must be exactly zero on every fold.
    reference = scores[scores["method_name"] == TRAIN_MEAN_RATE]
    assert np.allclose(reference["unified_bits_per_spike"], 0.0, atol=1e-12)


def test_score_summary_excludes_invalid_controls_from_best_valid_method(tmp_path: Path) -> None:
    module = _script_module()
    processed = tmp_path / "data.npz"
    _write_dataset(processed)
    config = _config(tmp_path, processed)

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 0

    out = Path(config["reporting"]["output_dir"])
    summary = json.loads((out / "stratified_cv_summary.json").read_text(encoding="utf-8"))

    assert summary["best_valid_method"] == FACTOR_LATENT
    assert summary["carried_forward_method"] == FACTOR_LATENT
    assert summary["invalid_controls_excluded_from_valid_model_selection"] is True
    assert SPLIT_MEAN_RATE_INVALID in summary["invalid_control_methods"]
    assert summary["recommended_reporting_mode"] == "stratified_cross_validation"
    assert summary["single_split_results_reportable"] is False

    method_summary = pd.read_csv(out / "stratified_cv_method_summary.csv")
    invalid = method_summary[method_summary["method_name"] == SPLIT_MEAN_RATE_INVALID].iloc[0]
    assert bool(invalid["valid_model"]) is False
    assert bool(invalid["reportable_as_model_performance"]) is False


def test_report_states_invalid_controls_and_not_official(tmp_path: Path) -> None:
    module = _script_module()
    processed = tmp_path / "data.npz"
    _write_dataset(processed)
    config = _config(tmp_path, processed)

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 0

    report = (Path(config["reporting"]["output_dir"]) / "stratified_cv_report.md").read_text(
        encoding="utf-8"
    )
    assert (
        "Invalid controls use evaluation fold targets and cannot be reported as model performance."
        in report
    )
    assert "not an official NLB leaderboard result" in report
    assert "Old incompatible mean-rate values are not used as tuning targets" in report
    assert "Stratified cross-validation is preferred over single-split reporting." in report
    assert "Random versus stratified comparison" in report
