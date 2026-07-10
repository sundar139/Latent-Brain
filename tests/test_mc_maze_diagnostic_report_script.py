from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import yaml

from latentbrain.reporting.report_validation import (
    INVALID_CONTROL_STATEMENT,
    NOT_OFFICIAL_STATEMENT,
)

INVALID_METHODS = ["oracle_split_scaled_factor_latent_invalid", "split_mean_rate_invalid"]


def _script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_mc_maze_diagnostic_report", Path("scripts/build_mc_maze_diagnostic_report.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_inputs(tmp_path: Path) -> dict[str, str]:
    (tmp_path / "quality.json").write_text(json.dumps({"n_trials": 100}), encoding="utf-8")
    (tmp_path / "scoreboard.json").write_text(
        json.dumps({"best_valid_model": "factor_latent"}), "utf-8"
    )
    (tmp_path / "seed.json").write_text(
        json.dumps({"carried_forward_method": "factor_latent"}), encoding="utf-8"
    )
    (tmp_path / "split.json").write_text(
        json.dumps(
            {
                "generalization_risk": "high",
                "validation_trial_count": 15,
                "test_trial_count": 15,
                "validation_positive_test_negative_persists": False,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "cv.json").write_text(
        json.dumps(
            {
                "factor_latent_repeated_split_validation_mean": 0.0269,
                "factor_latent_repeated_split_test_mean": 0.0090,
                "factor_latent_test_positive_fraction": 0.76,
                "invalid_split_mean_advantage_over_factor_latent": 0.0842,
                "rate_offset_explains_split_mean_advantage": False,
                "train_only_rate_calibration_test_gain": 1.3e-6,
                "invalid_control_methods": INVALID_METHODS,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "method_name": "factor_latent",
                "valid_model": True,
                "mean_test_unified_bits_per_spike": 0.0090,
                "notes": "train-only",
            },
            {
                "method_name": "split_mean_rate_invalid",
                "valid_model": False,
                "mean_test_unified_bits_per_spike": 0.0924,
                "notes": "leaks",
            },
        ]
    ).to_csv(tmp_path / "method_summary.csv", index=False)
    (tmp_path / "recommended.json").write_text(
        json.dumps(
            {
                "recommended_window_name": "behavior_speed_peak_centered_1p28s",
                "bin_size_ms": 20,
                "fold_count": 5,
                "repeats": 5,
                "total_folds": 25,
                "factor_latent_mean": 0.07707984048489147,
                "factor_latent_ci95_low": 0.07143536625695274,
                "factor_latent_ci95_high": 0.08251744011449201,
                "factor_latent_positive_fraction": 1.0,
                "split_mean_invalid_mean": 0.07110368937717054,
                "factor_latent_minus_split_mean_invalid": 0.005976151107720928,
                "leakage_dominance_persists": False,
                "moving_bin_fraction_mean": 0.576875,
                "endpoint_direction_entropy_mean": 2.0283893834346562,
                "fold_balance_warning": "none",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame([{"method_name": "factor_latent"}]).to_csv(
        tmp_path / "recommended_scores.csv", index=False
    )
    pd.DataFrame([{"method_name": "factor_latent"}]).to_csv(
        tmp_path / "recommended_method_summary.csv", index=False
    )
    (tmp_path / "recommended_protocol.yaml").write_text("protocol_frozen: true\n", encoding="utf-8")
    return {
        "data_quality_summary_path": str(tmp_path / "quality.json"),
        "unified_scoreboard_summary_path": str(tmp_path / "scoreboard.json"),
        "seed_robustness_summary_path": str(tmp_path / "seed.json"),
        "split_audit_summary_path": str(tmp_path / "split.json"),
        "cv_rate_audit_summary_path": str(tmp_path / "cv.json"),
        "method_summary_path": str(tmp_path / "method_summary.csv"),
        "recommended_window_cv_summary_path": str(tmp_path / "recommended.json"),
        "recommended_window_scores_path": str(tmp_path / "recommended_scores.csv"),
        "recommended_window_method_summary_path": str(tmp_path / "recommended_method_summary.csv"),
        "recommended_window_protocol_path": str(tmp_path / "recommended_protocol.yaml"),
        "window_audit_summary_path": str(tmp_path / "recommended.json"),
    }


def _config(tmp_path: Path) -> dict[str, Any]:
    return {
        "dataset": {
            "name": "mc_maze_small",
            "processed_path": str(tmp_path / "missing.npz"),
            "expected_hash": "abc",
            "original_bin_size_ms": 5,
        },
        "analysis": {
            "bin_size_ms": 20,
            "window_seconds": 1.28,
            "canonical_reference_model": "train_heldout_mean_rate",
            "canonical_metric": "unified_bits_per_spike",
            "official_leaderboard_claim": False,
        },
        "inputs": _write_inputs(tmp_path),
        "accepted_findings": {
            "carried_forward_valid_method": "factor_latent",
            "carried_forward_window": "behavior_speed_peak_centered_1p28s",
            "previous_from_start_window_status": "early_premovement_diagnostic",
            "single_split_results_reportable": False,
            "recommended_reporting_mode": "recommended_window_stratified_cross_validation",
            "invalid_rate_controls_present": True,
            "invalid_controls_excluded_from_model_performance": True,
            "neural_ode_near_win_seed_specific": True,
            "split_instability_disclosed": True,
            "split_mean_advantage_is_rate_offset": False,
            "split_mean_advantage_is_target_leakage": True,
            "leakage_dominance_persists_on_recommended_window": False,
            "no_official_benchmark_claim": True,
        },
        "reporting": {
            "output_dir": str(tmp_path / "out"),
            "fail_if_required_inputs_missing": True,
            "include_claim_safety_checklist": True,
            "include_invalid_control_warning": True,
            "include_negative_results": True,
            "include_next_steps": True,
        },
    }


def _write_config(tmp_path: Path, config: dict[str, Any]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_missing_required_input_fails_clearly(tmp_path: Path) -> None:
    module = _script_module()
    config = _config(tmp_path)
    config["inputs"]["cv_rate_audit_summary_path"] = str(tmp_path / "absent.json")

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 2


def test_missing_recommended_window_summary_fails_clearly(tmp_path: Path) -> None:
    module = _script_module()
    config = _config(tmp_path)
    config["inputs"]["recommended_window_cv_summary_path"] = str(tmp_path / "absent.json")

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 2


def test_official_leaderboard_claim_fails_config_validation(tmp_path: Path) -> None:
    module = _script_module()
    config = _config(tmp_path)
    config["analysis"]["official_leaderboard_claim"] = True

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 2


def test_single_split_reporting_mode_fails_config_validation(tmp_path: Path) -> None:
    module = _script_module()
    config = _config(tmp_path)
    config["accepted_findings"]["recommended_reporting_mode"] = "single_split"

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 2


def test_wrong_carried_forward_window_fails_config_validation(tmp_path: Path) -> None:
    module = _script_module()
    config = _config(tmp_path)
    config["accepted_findings"]["carried_forward_window"] = "from_start_1p28s"

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 2


def test_invalid_carried_forward_method_fails_config_validation(tmp_path: Path) -> None:
    module = _script_module()
    config = _config(tmp_path)
    config["accepted_findings"]["carried_forward_valid_method"] = "split_mean_rate_invalid"

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 2


def test_script_run_writes_report_summary_registry_and_checklist(tmp_path: Path) -> None:
    module = _script_module()
    config = _config(tmp_path)

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 0

    out = Path(config["reporting"]["output_dir"])
    assert (out / "mc_maze_small_diagnostic_report.md").exists()
    assert (out / "mc_maze_small_diagnostic_summary.json").exists()
    assert (out / "method_registry.csv").exists()
    assert (out / "claim_safety_checklist.md").exists()
    assert (out / "accepted_findings.json").exists()

    registry = pd.read_csv(out / "method_registry.csv")
    invalid = registry[registry["method_name"].isin(INVALID_METHODS)]
    assert not bool(invalid["valid_model"].any())
    assert not bool(invalid["reportable_as_model_performance"].any())


def test_script_prints_carried_forward_method_and_checklist_status(
    tmp_path: Path, capsys: Any
) -> None:
    module = _script_module()
    config = _config(tmp_path)

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 0

    printed = capsys.readouterr().out
    assert "carried_forward_method: factor_latent" in printed
    assert "claim_safety_checklist_passed: True" in printed
    assert "recommended_reporting_mode: recommended_window_stratified_cross_validation" in printed
    assert "carried_forward_window: behavior_speed_peak_centered_1p28s" in printed
    assert "official_leaderboard_claim: False" in printed


def test_generated_report_says_not_official_leaderboard_result(tmp_path: Path) -> None:
    module = _script_module()
    config = _config(tmp_path)

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 0

    report = (
        Path(config["reporting"]["output_dir"]) / "mc_maze_small_diagnostic_report.md"
    ).read_text(encoding="utf-8")
    assert NOT_OFFICIAL_STATEMENT in report
    assert INVALID_CONTROL_STATEMENT in report

    summary = json.loads(
        (
            Path(config["reporting"]["output_dir"]) / "mc_maze_small_diagnostic_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["claim_safety_checklist_passed"] is True
    assert summary["invalid_controls_excluded_from_model_performance"] is True
