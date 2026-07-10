from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from test_eval_baseline_suite import _config, _write_accepted_mean

EXPECTED_OUTPUTS = (
    "baseline_suite_summary.json",
    "outer_fold_scores.csv",
    "inner_selection_results.csv",
    "selected_hyperparameters.csv",
    "method_summary.csv",
    "paired_method_comparisons.csv",
    "repeat_level_scores.csv",
    "baseline_protocol.yaml",
    "neural_reevaluation_readiness.json",
    "baseline_suite_report.md",
)
EXPECTED_FIGURES = (
    "baseline_score_distribution.png",
    "paired_differences_by_repeat.png",
    "heldout_mask_variability.png",
    "selected_hyperparameters.png",
)


def _script_module(name: str, path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_config(tmp_path: Path, config: dict[str, Any]) -> Path:
    path = tmp_path / "baseline_suite.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    module = _script_module("run_baseline_suite", "scripts/run_baseline_suite.py")
    config = _config(tmp_path, monkeypatch)
    _write_accepted_mean(config, 0.0)
    # Run once to learn the reproduced mean, then pin it so reproduction passes.
    from latentbrain.eval.baseline_suite import FACTOR_LATENT_FIXED, run_baseline_suite

    outer = run_baseline_suite(config)["outer_scores"]
    fixed = outer[outer["method_name"] == FACTOR_LATENT_FIXED]["unified_bits_per_spike"].mean()
    _write_accepted_mean(config, float(fixed))
    return {"module": module, "config": config}


def test_script_writes_every_output_and_prints_required_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    prepared = _prepared(tmp_path, monkeypatch)
    capsys.readouterr()

    assert (
        prepared["module"].main(["--config", str(_write_config(tmp_path, prepared["config"]))]) == 0
    )

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
        "outer_fold_count",
        "outer_repeats",
        "total_outer_evaluations",
        "inner_fold_count",
        "factor_latent_fixed_mean",
        "factor_latent_reproduction_difference",
        "valid_methods",
        "best_valid_method",
        "best_valid_method_mean",
        "baseline_to_beat",
        "baseline_replaced",
        "baseline_replacement_supported",
        "paired_difference_against_factor_latent",
        "paired_ci_against_factor_latent",
        "positive_repeat_fraction_against_factor_latent",
        "invalid_controls_excluded",
        "neural_reevaluation_ready",
        "readiness_blockers",
        "output_directory",
    ):
        assert f"{key}:" in printed

    report = (out / "baseline_suite_report.md").read_text("utf-8")
    assert "extracted from the trial-aware source before rebinning" in report
    assert "selected using only outer-training data" in report
    assert "not treated as statistically independent" in report
    assert "cannot be reported as model performance" in report
    assert "not interpreted as directly comparable" in report
    assert "not an official NLB leaderboard result" in report
    for forbidden in ("better than Small", "worse than Small", "outperforms Small"):
        assert forbidden not in report

    readiness = json.loads((out / "neural_reevaluation_readiness.json").read_text("utf-8"))
    assert readiness["neural_experiment_run_during_this_milestone"] is False
    assert readiness["required_neural_seeds"] >= 5
    assert "from_start_1p28s" in readiness["forbidden_old_protocols"]
    assert readiness["ready"] is True

    summary = json.loads((out / "baseline_suite_summary.json").read_text("utf-8"))
    assert summary["baseline_to_beat"] not in ("split_mean_rate_invalid", "train_mean_rate")
    assert summary["official_leaderboard_claim"] is False
    assert summary["single_split_results_reportable"] is False
    assert summary["old_mean_rate_values_used_as_targets"] is False

    assert summary["baseline_to_beat_ci95_low"] <= summary["baseline_to_beat_ci95_high"]

    protocol = yaml.safe_load((out / "baseline_protocol.yaml").read_text("utf-8"))
    assert protocol["trial_source"]["allow_global_crop_to_min"] is False
    assert protocol["claim_safety"]["official_leaderboard_claim"] is False
    assert protocol["protocol_frozen"] is True


def test_script_rejects_global_crop_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    module = _script_module("run_baseline_suite", "scripts/run_baseline_suite.py")
    config = _config(tmp_path, monkeypatch)
    config["trial_source"]["allow_global_crop_to_min"] = True

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 2
    assert "allow_global_crop_to_min must be false" in capsys.readouterr().out


def test_script_rejects_reportable_invalid_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    module = _script_module("run_baseline_suite", "scripts/run_baseline_suite.py")
    config = _config(tmp_path, monkeypatch)
    for method in config["methods"]:
        if method["name"] == "split_mean_rate_invalid":
            method["reportable_as_model_performance"] = True

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 2
    assert "must not be reportable as model performance" in capsys.readouterr().out


def test_script_rejects_fold_level_comparison_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    module = _script_module("run_baseline_suite", "scripts/run_baseline_suite.py")
    config = _config(tmp_path, monkeypatch)
    config["statistics"]["comparison_unit"] = "fold"

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 2
    assert "comparison_unit must be repeat" in capsys.readouterr().out


def _scoreboard_config(tmp_path: Path, baseline_path: Path) -> Path:
    config = {
        "dataset": {"name": "toy_large"},
        "inputs": {
            "window_audit_summary_path": str(tmp_path / "absent_audit.json"),
            "recommended_window_cv_summary_path": str(tmp_path / "absent_cv.json"),
            "baseline_suite_summary_path": str(baseline_path),
        },
        "reporting": {"output_dir": str(tmp_path / "scoreboard")},
    }
    path = tmp_path / "scoreboard.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _baseline_summary(**overrides: Any) -> dict[str, Any]:
    base = {
        "baseline_to_beat": "factor_latent_fixed",
        "baseline_replaced": False,
        "baseline_replacement_supported": False,
        "neural_reevaluation_ready": True,
        "invalid_controls_excluded": True,
        "official_leaderboard_claim": False,
        "factor_latent_fixed_mean": 0.1227,
        "best_valid_method_mean": 0.1227,
    }
    return {**base, **overrides}


def test_scoreboard_loads_baseline_suite(tmp_path: Path, capsys: Any) -> None:
    module = _script_module("run_unified_scoreboard", "scripts/run_unified_scoreboard.py")
    path = tmp_path / "baseline_summary.json"
    path.write_text(json.dumps(_baseline_summary()), encoding="utf-8")

    assert module.main(["--config", str(_scoreboard_config(tmp_path, path))]) == 0

    printed = capsys.readouterr().out
    assert "baseline_suite_available: True" in printed
    assert "baseline_to_beat: factor_latent_fixed" in printed
    assert "neural_reevaluation_ready: True" in printed
    written = json.loads((tmp_path / "scoreboard" / "unified_scoreboard_summary.json").read_text())
    assert written["baseline_to_beat_mean"] == pytest.approx(0.1227)


def test_scoreboard_missing_baseline_summary_falls_back_cleanly(
    tmp_path: Path, capsys: Any
) -> None:
    module = _script_module("run_unified_scoreboard", "scripts/run_unified_scoreboard.py")

    assert (
        module.main(["--config", str(_scoreboard_config(tmp_path, tmp_path / "absent.json"))]) == 0
    )

    printed = capsys.readouterr().out
    assert "baseline_suite_available: False" in printed
    assert "neural_reevaluation_ready: False" in printed


def test_scoreboard_malformed_baseline_summary_fails_clearly(tmp_path: Path) -> None:
    module = _script_module("run_unified_scoreboard", "scripts/run_unified_scoreboard.py")
    path = tmp_path / "baseline_summary.json"
    broken = _baseline_summary()
    del broken["neural_reevaluation_ready"]
    path.write_text(json.dumps(broken), encoding="utf-8")

    with pytest.raises(ValueError, match="missing neural_reevaluation_ready"):
        module.main(["--config", str(_scoreboard_config(tmp_path, path))])


def test_scoreboard_rejects_invalid_control_as_baseline(tmp_path: Path) -> None:
    module = _script_module("run_unified_scoreboard", "scripts/run_unified_scoreboard.py")
    path = tmp_path / "baseline_summary.json"
    path.write_text(
        json.dumps(_baseline_summary(baseline_to_beat="split_mean_rate_invalid")), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="cannot be the baseline to beat"):
        module.main(["--config", str(_scoreboard_config(tmp_path, path))])
