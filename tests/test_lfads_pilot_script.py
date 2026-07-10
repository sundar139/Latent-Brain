from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pytest
import yaml

from latentbrain.eval.reporting import write_lfads_pilot_outputs

EXPECTED_OUTPUTS = (
    "lfads_pilot_summary.json",
    "lfads_pilot_runs.csv",
    "fold_seed_scores.csv",
    "seed_summary.csv",
    "fold_summary.csv",
    "paired_baseline_comparison.csv",
    "checkpoint_manifest.csv",
    "training_resource_summary.csv",
    "lfads_pilot_protocol.yaml",
    "full_evaluation_recommendation.json",
    "lfads_pilot_report.md",
)


def _script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_lfads_pilot", Path("scripts/run_lfads_pilot.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _summary() -> dict[str, Any]:
    return {
        "dataset_name": "mc_maze_large",
        "dataset_hash": "0" * 64,
        "data_shape": [500, 64, 162],
        "repeat_index": 0,
        "completed_runs": 25,
        "failed_runs": 0,
        "mean_unified_bits_per_spike": 0.1,
        "score_std": 0.01,
        "positive_seed_fraction": 1.0,
        "mean_paired_difference_vs_baseline": -0.01,
        "pilot_repeat_baseline_mean": 0.11,
        "checkpoint_selection_valid": True,
        "leakage_checks_passed": True,
        "baseline_to_beat": "factor_latent_train_selected",
        "single_split_results_reportable": False,
        "official_leaderboard_claim": False,
        "pilot_final_claim_allowed": False,
    }


def test_development_dependency_pins_validated_mypy_version() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert '"mypy==2.2.0"' in pyproject


def test_quality_workflow_installs_project_development_dependencies() -> None:
    workflow = Path(".github/workflows/quality.yml").read_text(encoding="utf-8")

    assert 'pip install -e ".[dev]"' in workflow
    assert "pip install mypy" not in workflow


def test_no_second_dependency_system_is_introduced() -> None:
    tracked_candidates = {
        path.name
        for path in Path(".").iterdir()
        if path.name in {"requirements.txt", "Pipfile", "poetry.lock"}
    }

    assert tracked_candidates == set()


def test_report_writer_emits_claim_safe_pilot_bundle(tmp_path: Path) -> None:
    empty = pd.DataFrame()
    tables = {
        "lfads_pilot_runs": empty,
        "fold_seed_scores": empty,
        "seed_summary": empty,
        "fold_summary": empty,
        "paired_baseline_comparison": empty,
        "checkpoint_manifest": empty,
        "training_resource_summary": empty,
    }
    recommendation = {
        "proceed": True,
        "runtime_estimate_full_evaluation_hours": 1.0,
        "estimated_peak_cuda_memory_mb": 100.0,
        "reasons": ["all gates passed"],
    }

    paths = write_lfads_pilot_outputs(
        tmp_path, _summary(), tables, {"protocol_frozen": True}, recommendation
    )

    assert set(paths) == {
        "summary",
        "lfads_pilot_runs",
        "fold_seed_scores",
        "seed_summary",
        "fold_summary",
        "paired_baseline_comparison",
        "checkpoint_manifest",
        "training_resource_summary",
        "protocol",
        "recommendation",
        "report",
    }
    for name in EXPECTED_OUTPUTS:
        assert (tmp_path / name).exists()
    report = (tmp_path / "lfads_pilot_report.md").read_text(encoding="utf-8")
    assert "one held-out-neuron mask" in report
    assert "not a final multi-repeat model comparison" in report
    assert "Outer-evaluation data were not used for checkpoint selection" in report
    assert "factor_latent_train_selected" in report
    assert "not an official NLB leaderboard result" in report


def test_script_rejects_nonzero_repeat(tmp_path: Path, capsys: Any) -> None:
    module = _script_module()
    config = yaml.safe_load(Path("configs/mc_maze_large_lfads_pilot.yaml").read_text("utf-8"))
    config["outer_protocol"]["repeat_index"] = 1
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert module.main(["--config", str(path)]) == 2
    assert "repeat_index" in capsys.readouterr().out


def test_generated_summary_cannot_enable_final_claim(tmp_path: Path) -> None:
    summary = _summary()
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["pilot_final_claim_allowed"] is False


def test_development_toolchain_pins_mypy_and_quality_uses_dev_dependencies() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    development = project["project"]["optional-dependencies"]["dev"]
    workflow = (root / ".github" / "workflows" / "quality.yml").read_text(encoding="utf-8")

    assert "mypy==2.2.0" in development
    assert 'pip install -e ".[dev]"' in workflow
    assert not list(root.glob("*requirements*"))


def test_small_dataset_summary_does_not_load_lfads_pilot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_unified_scoreboard.py"
    spec = importlib.util.spec_from_file_location(
        "run_unified_scoreboard_for_lfads_pilot_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module, "load_dataset_cv_scoreboard", lambda _config: {"dataset_name": "mc_maze_small"}
    )
    monkeypatch.setattr(module, "load_baseline_suite_scoreboard", lambda _config: {})
    monkeypatch.setattr(
        module,
        "load_lfads_pilot_scoreboard",
        lambda _config: pytest.fail("Small must not load the Large LFADS pilot"),
    )
    monkeypatch.setattr(
        module,
        "load_lfads_diagnostics_scoreboard",
        lambda _config: pytest.fail("Small must not load the Large LFADS diagnostics"),
    )
    monkeypatch.setattr(
        module,
        "load_neural_ode_pilot_scoreboard",
        lambda _config: pytest.fail("Small must not load the Large neural-ODE pilot"),
    )
    monkeypatch.setattr(
        module,
        "load_neural_ode_diagnostics_scoreboard",
        lambda _config: pytest.fail("Small must not load the Large neural-ODE diagnostics"),
    )
    monkeypatch.setattr(module, "write_dataset_scoreboard_outputs", lambda *_args: None)

    result = module.run_dataset_scoreboard(
        {"dataset": {"name": "mc_maze_small"}, "reporting": {"output_dir": str(tmp_path)}}
    )

    assert result["summary"] == {"dataset_name": "mc_maze_small"}
