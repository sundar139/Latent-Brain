from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from test_eval_large_recommended_window_cv import _config

EXPECTED_OUTPUTS = (
    "recommended_window_cv_summary.json",
    "recommended_window_scores.csv",
    "recommended_window_method_summary.csv",
    "recommended_window_fold_assignments.csv",
    "recommended_window_behavior_statistics.csv",
    "recommended_window_fold_balance.csv",
    "recommended_window_leakage_diagnostics.csv",
    "factor_analysis_random_state_sensitivity.csv",
    "small_large_protocol_comparison.csv",
    "recommended_window_protocol.yaml",
    "recommended_window_cv_report.md",
)
EXPECTED_FIGURES = (
    "recommended_window_score_distribution.png",
    "valid_vs_invalid_by_fold.png",
    "movement_coverage_summary.png",
    "fold_balance_summary.png",
    "factor_analysis_random_state_sensitivity.png",
    "small_large_stability_comparison.png",
)


def _script_module(name: str, path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_config(tmp_path: Path, config: dict[str, Any]) -> Path:
    path = tmp_path / "large_cv.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_large_script_writes_every_output_and_prints_required_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    module = _script_module("run_recommended_window_cv", "scripts/run_recommended_window_cv.py")
    config = _config(tmp_path, monkeypatch)

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 0

    out = tmp_path / "out"
    for name in EXPECTED_OUTPUTS:
        assert (out / name).exists()
    for name in EXPECTED_FIGURES:
        assert (out / "figures" / name).exists()

    printed = capsys.readouterr().out
    for key in (
        "dataset_name",
        "dataset_hash",
        "window_name",
        "trial_source",
        "target_bin_size_ms",
        "trial_count",
        "time_bins",
        "neuron_count",
        "fold_count",
        "repeats",
        "total_folds",
        "train_trials_per_fold",
        "eval_trials_per_fold",
        "factor_latent_mean",
        "factor_latent_ci95",
        "factor_latent_positive_fraction",
        "split_mean_invalid_mean",
        "factor_latent_minus_split_mean_invalid",
        "factor_latent_beats_invalid_control_mean",
        "factor_latent_beats_invalid_control_fraction",
        "leakage_dominance_persists",
        "factor_analysis_random_state_range",
        "factor_analysis_random_state_warning",
        "fold_balance_warning",
        "recommended_reporting_mode",
        "invalid_controls_excluded_from_model_selection",
        "protocol_frozen",
        "output_directory",
    ):
        assert f"{key}:" in printed

    report = (out / "recommended_window_cv_report.md").read_text("utf-8")
    assert "extracted from the trial-aware raw representation" in report
    assert "cannot be reported as model performance" in report
    assert "not interpreted as directly comparable" in report
    assert "not an official NLB leaderboard result" in report
    assert "Old incompatible mean-rate values were not used as tuning targets." in report

    protocol = yaml.safe_load((out / "recommended_window_protocol.yaml").read_text("utf-8"))
    assert protocol["trial_source"]["allow_global_crop_to_min"] is False
    assert protocol["binning"]["extract_before_rebin"] is True
    assert protocol["claim_safety"]["official_leaderboard_claim"] is False
    assert protocol["protocol_frozen"] is True


def test_large_script_rejects_global_crop_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    module = _script_module("run_recommended_window_cv", "scripts/run_recommended_window_cv.py")
    config = _config(tmp_path, monkeypatch)
    config["trial_source"]["allow_global_crop_to_min"] = True

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 2
    assert "allow_global_crop_to_min must be false" in capsys.readouterr().out


def test_large_script_rejects_duplicate_random_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    module = _script_module("run_recommended_window_cv", "scripts/run_recommended_window_cv.py")
    config = _config(tmp_path, monkeypatch)
    config["factor_analysis_sensitivity"]["random_states"] = [0, 0]

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 2
    assert "random_states must be unique" in capsys.readouterr().out


def _scoreboard_config(tmp_path: Path, summary_path: Path) -> Path:
    config = {
        "dataset": {"name": "toy_large"},
        "inputs": {
            "window_audit_summary_path": str(tmp_path / "absent_audit.json"),
            "recommended_window_cv_summary_path": str(summary_path),
        },
        "reporting": {"output_dir": str(tmp_path / "scoreboard")},
    }
    path = tmp_path / "scoreboard.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _valid_summary() -> dict[str, Any]:
    return {
        "recommended_window_name": "behavior_speed_peak_centered_1p28s",
        "recommended_reporting_mode": "recommended_window_stratified_cross_validation",
        "factor_latent_mean": 0.05,
        "factor_latent_ci95_low": 0.04,
        "factor_latent_ci95_high": 0.06,
        "factor_latent_positive_fraction": 1.0,
        "factor_latent_beats_invalid_control_mean": True,
        "leakage_dominance_persists": False,
        "single_split_results_reportable": False,
        "official_leaderboard_claim": False,
    }


def test_scoreboard_loads_large_summary(tmp_path: Path, capsys: Any) -> None:
    module = _script_module("run_unified_scoreboard", "scripts/run_unified_scoreboard.py")
    summary_path = tmp_path / "cv_summary.json"
    summary_path.write_text(json.dumps(_valid_summary()), encoding="utf-8")

    assert module.main(["--config", str(_scoreboard_config(tmp_path, summary_path))]) == 0

    printed = capsys.readouterr().out
    assert "recommended_window_cv_available: True" in printed
    assert "best_valid_method: factor_latent" in printed
    assert "official_leaderboard_claim: False" in printed
    written = json.loads((tmp_path / "scoreboard" / "unified_scoreboard_summary.json").read_text())
    assert written["best_valid_method"] == "factor_latent"
    assert written["window_audit_available"] is False


def test_scoreboard_missing_summary_falls_back_cleanly(tmp_path: Path, capsys: Any) -> None:
    module = _script_module("run_unified_scoreboard", "scripts/run_unified_scoreboard.py")

    assert (
        module.main(["--config", str(_scoreboard_config(tmp_path, tmp_path / "absent.json"))]) == 0
    )

    printed = capsys.readouterr().out
    assert "recommended_window_cv_available: False" in printed
    assert "best_valid_method: None" in printed


def test_scoreboard_malformed_summary_fails_clearly(tmp_path: Path) -> None:
    module = _script_module("run_unified_scoreboard", "scripts/run_unified_scoreboard.py")
    summary_path = tmp_path / "cv_summary.json"
    broken = _valid_summary()
    del broken["factor_latent_positive_fraction"]
    summary_path.write_text(json.dumps(broken), encoding="utf-8")

    with pytest.raises(ValueError, match="missing factor_latent_positive_fraction"):
        module.main(["--config", str(_scoreboard_config(tmp_path, summary_path))])


def test_scoreboard_rejects_leaderboard_claim(tmp_path: Path) -> None:
    module = _script_module("run_unified_scoreboard", "scripts/run_unified_scoreboard.py")
    summary_path = tmp_path / "cv_summary.json"
    summary_path.write_text(
        json.dumps({**_valid_summary(), "official_leaderboard_claim": True}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="no leaderboard claim is allowed"):
        module.main(["--config", str(_scoreboard_config(tmp_path, summary_path))])
