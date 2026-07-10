from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pytest

from latentbrain.reporting.mc_maze_diagnostic import (
    build_accepted_findings,
    build_claim_safety_checklist,
    checklist_passed,
    load_diagnostic_inputs,
    render_mc_maze_diagnostic_report,
    write_mc_maze_diagnostic_bundle,
)
from latentbrain.reporting.report_tables import build_diagnostic_tables, build_method_registry
from latentbrain.reporting.report_validation import (
    INVALID_CONTROL_STATEMENT,
    NOT_OFFICIAL_STATEMENT,
    REQUIRED_REPORT_SECTIONS,
    validate_report_text,
)

INVALID_METHODS = ["oracle_split_scaled_factor_latent_invalid", "split_mean_rate_invalid"]

RECOMMENDED_SUMMARY = {
    "bin_size_ms": 20,
    "fold_count": 5,
    "repeats": 5,
    "total_folds": 25,
    "recommended_window_name": "behavior_speed_peak_centered_1p28s",
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


def _write_toy_inputs(tmp_path: Path) -> dict[str, str]:
    (tmp_path / "quality.json").write_text(
        json.dumps({"n_trials": 100, "n_neurons": 142}), encoding="utf-8"
    )
    (tmp_path / "scoreboard.json").write_text(
        json.dumps({"best_valid_model": "factor_latent"}), encoding="utf-8"
    )
    (tmp_path / "seed.json").write_text(
        json.dumps(
            {
                "best_mean_method": "factor_latent",
                "carried_forward_method": "factor_latent",
                "any_neural_beats_factor_latent_mean": False,
                "any_neural_beats_factor_latent_lower_ci": False,
                "paired_mean_difference_best_neural_minus_factor_latent": -0.0232,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "split.json").write_text(
        json.dumps(
            {
                "generalization_risk": "high",
                "validation_trial_count": 15,
                "test_trial_count": 15,
                "train_trial_count": 70,
                "heldin_neuron_count": 106,
                "heldout_neuron_count": 36,
                "accepted_split_seed": 2027,
                "validation_positive_test_negative_persists": False,
                "factor_latent_validation_mean": 0.029,
                "factor_latent_test_mean": -0.0083,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "cv.json").write_text(
        json.dumps(
            {
                "factor_latent_repeated_split_validation_mean": 0.0269,
                "factor_latent_repeated_split_validation_std": 0.0182,
                "factor_latent_repeated_split_test_mean": 0.0090,
                "factor_latent_repeated_split_test_std": 0.0129,
                "factor_latent_test_positive_fraction": 0.76,
                "invalid_split_mean_advantage_over_factor_latent": 0.0842,
                "rate_offset_explains_split_mean_advantage": False,
                "train_only_rate_calibration_test_gain": 1.3e-6,
                "train_only_rate_calibration_gain_is_negligible": True,
                "invalid_control_methods": INVALID_METHODS,
                "recommended_reporting_mode": "repeated_split",
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
    (tmp_path / "recommended_summary.json").write_text(
        json.dumps(RECOMMENDED_SUMMARY), encoding="utf-8"
    )
    pd.DataFrame([{"repeat_index": 0, "fold_index": 0, "method_name": "factor_latent"}]).to_csv(
        tmp_path / "recommended_scores.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "method_name": "factor_latent",
                "valid_model": True,
                "reportable_as_model_performance": True,
                "mean_unified_bits_per_spike": RECOMMENDED_SUMMARY["factor_latent_mean"],
            },
            {
                "method_name": "split_mean_rate_invalid",
                "valid_model": False,
                "reportable_as_model_performance": False,
                "mean_unified_bits_per_spike": RECOMMENDED_SUMMARY["split_mean_invalid_mean"],
            },
        ]
    ).to_csv(tmp_path / "recommended_method_summary.csv", index=False)
    (tmp_path / "recommended_protocol.yaml").write_text(
        "recommended_reporting_mode: recommended_window_stratified_cross_validation\n",
        encoding="utf-8",
    )
    return {
        "data_quality_summary_path": str(tmp_path / "quality.json"),
        "unified_scoreboard_summary_path": str(tmp_path / "scoreboard.json"),
        "seed_robustness_summary_path": str(tmp_path / "seed.json"),
        "split_audit_summary_path": str(tmp_path / "split.json"),
        "cv_rate_audit_summary_path": str(tmp_path / "cv.json"),
        "method_summary_path": str(tmp_path / "method_summary.csv"),
        "recommended_window_cv_summary_path": str(tmp_path / "recommended_summary.json"),
        "recommended_window_scores_path": str(tmp_path / "recommended_scores.csv"),
        "recommended_window_method_summary_path": str(tmp_path / "recommended_method_summary.csv"),
        "recommended_window_protocol_path": str(tmp_path / "recommended_protocol.yaml"),
        "window_audit_summary_path": str(tmp_path / "recommended_summary.json"),
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
        "inputs": _write_toy_inputs(tmp_path),
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


def _render(tmp_path: Path) -> str:
    config = _config(tmp_path)
    inputs = load_diagnostic_inputs(config)
    findings = build_accepted_findings(inputs, config)
    registry = build_method_registry(inputs, config)
    tables = build_diagnostic_tables(inputs, config)
    return render_mc_maze_diagnostic_report(findings, registry, tables, config)


def test_diagnostic_inputs_load_from_toy_files(tmp_path: Path) -> None:
    inputs = load_diagnostic_inputs(_config(tmp_path))

    assert inputs["data_quality_summary"]["n_trials"] == 100
    assert inputs["cv_rate_audit_summary"]["factor_latent_test_positive_fraction"] == 0.76
    assert isinstance(inputs["method_summary"], pd.DataFrame)
    assert inputs["recommended_window_cv_summary"]["factor_latent_mean"] == pytest.approx(
        0.07707984048489147
    )
    assert isinstance(inputs["recommended_window_method_summary"], pd.DataFrame)
    # Optional CSV inputs were never configured, so they degrade cleanly.
    assert inputs["repeated_split_scores"] is None
    assert "repeated_split_scores_path" in inputs["missing_inputs"]


def test_missing_required_input_fails_clearly(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["inputs"]["cv_rate_audit_summary_path"] = str(tmp_path / "absent.json")

    with pytest.raises(FileNotFoundError, match="Required diagnostic input is missing"):
        load_diagnostic_inputs(config)


def test_missing_recommended_window_summary_fails_clearly(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["inputs"]["recommended_window_cv_summary_path"] = str(tmp_path / "absent.json")

    with pytest.raises(FileNotFoundError, match="Required diagnostic input is missing"):
        load_diagnostic_inputs(config)


def test_malformed_required_input_fails_clearly(tmp_path: Path) -> None:
    config = _config(tmp_path)
    Path(config["inputs"]["cv_rate_audit_summary_path"]).write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="malformed diagnostic input"):
        load_diagnostic_inputs(config)


def test_accepted_findings_include_carried_forward_factor_latent(tmp_path: Path) -> None:
    config = _config(tmp_path)
    findings = build_accepted_findings(load_diagnostic_inputs(config), config)

    assert findings["carried_forward_valid_method"] == "factor_latent"
    assert findings["carried_forward_window"] == "behavior_speed_peak_centered_1p28s"
    assert findings["recommended_reporting_mode"] == (
        "recommended_window_stratified_cross_validation"
    )
    assert findings["single_split_results_reportable"] is False
    assert findings["split_mean_advantage_is_target_leakage"] is True
    assert findings["rate_offset_explains_split_mean_advantage"] is False
    assert findings["factor_latent_test_positive_fraction"] == 0.76
    assert findings["factor_latent_recommended_window_mean"] == pytest.approx(0.07707984048489147)
    assert findings["factor_latent_beats_invalid_control_mean"] is True


def test_method_registry_marks_invalid_controls_invalid(tmp_path: Path) -> None:
    config = _config(tmp_path)
    registry = build_method_registry(load_diagnostic_inputs(config), config)

    invalid = registry[registry["method_name"].isin(INVALID_METHODS)]
    assert not bool(invalid["valid_model"].any())
    assert not bool(invalid["reportable_as_model_performance"].any())


def test_report_includes_all_required_sections(tmp_path: Path) -> None:
    report = _render(tmp_path)

    for section in REQUIRED_REPORT_SECTIONS:
        assert section in report
    assert validate_report_text(report) == []


def test_report_discloses_recommended_window_and_old_target_status(tmp_path: Path) -> None:
    report = _render(tmp_path)

    assert "## Recommended movement-window protocol" in report
    assert (
        "The carried-forward MC_Maze Small window is behavior_speed_peak_centered_1p28s." in report
    )
    assert "## Previous early-window diagnostics" in report
    assert (
        "Previous from_start_1p28s results describe an early/pre-movement window and should not "
        "be described as reach-dynamics performance." in report
    )
    assert "different prediction targets, not direct performance improvements" in report


def test_report_includes_negative_neural_model_findings(tmp_path: Path) -> None:
    report = _render(tmp_path)

    assert "LFADS-family models did not beat factor-latent" in report
    assert "Switching latent dynamics collapsed to one dominant regime" in report
    assert "Objective variants did not beat factor-latent" in report
    assert "were not robust" in report


def test_report_includes_target_leakage_conclusion(tmp_path: Path) -> None:
    report = _render(tmp_path)

    assert INVALID_CONTROL_STATEMENT in report
    assert "per-neuron evaluation-target leakage" in report
    assert "not a" in report and "global rate-offset correction" in report


def test_report_states_it_is_not_an_official_result(tmp_path: Path) -> None:
    assert NOT_OFFICIAL_STATEMENT in _render(tmp_path)


def test_report_is_deterministic(tmp_path: Path) -> None:
    first = _render(tmp_path)
    second = _render(tmp_path)

    assert first == second


def test_claim_safety_checklist_passes_and_lists_every_item(tmp_path: Path) -> None:
    config = _config(tmp_path)
    findings = build_accepted_findings(load_diagnostic_inputs(config), config)

    checklist = build_claim_safety_checklist(findings)

    assert "All items passed: yes" in checklist
    assert checklist_passed(findings) is True
    assert "| no |" not in checklist


def test_claim_safety_checklist_fails_when_single_split_is_reportable(tmp_path: Path) -> None:
    config = _config(tmp_path)
    findings = build_accepted_findings(load_diagnostic_inputs(config), config)
    findings["single_split_results_reportable"] = True

    assert checklist_passed(findings) is False


def test_bundle_writes_every_artifact_and_passes_safety(tmp_path: Path) -> None:
    config = _config(tmp_path)

    summary = write_mc_maze_diagnostic_bundle(config)

    out = Path(config["reporting"]["output_dir"])
    for name in (
        "mc_maze_small_diagnostic_report.md",
        "mc_maze_small_diagnostic_summary.json",
        "accepted_findings.json",
        "method_registry.csv",
        "claim_safety_checklist.md",
    ):
        assert (out / name).exists(), name
    for table in (
        "dataset_summary",
        "accepted_results",
        "valid_model_summary",
        "invalid_control_summary",
        "seed_robustness_summary",
        "split_generalization_summary",
        "cv_rate_audit_summary",
    ):
        assert (out / "tables" / f"{table}.csv").exists(), table

    assert summary["claim_safety_checklist_passed"] is True
    assert summary["carried_forward_method"] == "factor_latent"
    assert summary["official_leaderboard_claim"] is False
    assert summary["split_mean_advantage_is_target_leakage"] is True
    assert summary["carried_forward_window"] == "behavior_speed_peak_centered_1p28s"
    assert summary["recommended_window_cv_available"] is True
    assert summary["factor_latent_recommended_window_mean"] == pytest.approx(0.07707984048489147)
    assert summary["leakage_dominance_persists_on_recommended_window"] is False


def test_bundle_refuses_to_write_when_claim_safety_fails(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["accepted_findings"]["split_mean_advantage_is_rate_offset"] = True

    with pytest.raises(ValueError, match="claim safety validation failed"):
        write_mc_maze_diagnostic_bundle(config)

    assert not (
        Path(config["reporting"]["output_dir"]) / "mc_maze_small_diagnostic_report.md"
    ).exists()
