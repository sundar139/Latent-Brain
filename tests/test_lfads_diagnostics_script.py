from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import yaml

from latentbrain.eval.reporting import write_lfads_diagnostics_outputs

EXPECTED_OUTPUTS = (
    "lfads_diagnostics_summary.json",
    "run_diagnostics.csv",
    "checkpoint_diagnostics.csv",
    "neuron_diagnostics.csv",
    "time_bin_diagnostics.csv",
    "latent_diagnostics.csv",
    "rate_diagnostics.csv",
    "objective_diagnostics.csv",
    "baseline_gap_decomposition.csv",
    "next_action_recommendation.json",
    "lfads_diagnostics_report.md",
)


def _script_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_lfads_diagnostics.py"
    spec = importlib.util.spec_from_file_location("run_lfads_diagnostics_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary() -> dict[str, Any]:
    return {
        "dataset_name": "mc_maze_large",
        "dataset_hash": "0" * 64,
        "integrity_checks_passed": True,
        "accepted_checkpoints": 25,
        "excluded_preflight_artifacts": 1,
        "outer_mean_unified_bits_per_spike": 0.029,
        "pilot_repeat_baseline_mean": 0.174,
        "mean_baseline_gap": 0.145,
        "train_mean_unified_bits_per_spike": 0.04,
        "inner_mean_unified_bits_per_spike": 0.031,
        "mean_train_to_inner_gap": 0.009,
        "mean_inner_to_outer_gap": 0.002,
        "positive_neuron_fraction": 0.6,
        "negative_neuron_fraction": 0.4,
        "mean_effective_rank": 3.0,
        "mean_effective_rank_fraction": 0.094,
        "posterior_collapse_detected": True,
        "dominant_failure_mode": "posterior_or_latent_collapse",
        "estimated_recoverable_gap": 0.05,
        "recommended_next_action": "targeted_lfads_repair_pilot",
        "full_lfads_evaluation_allowed": False,
        "no_training_performed": True,
    }


def _tables() -> dict[str, pd.DataFrame]:
    return {
        "run_diagnostics": pd.DataFrame([{"split_name": "outer_evaluation"}]),
        "checkpoint_diagnostics": pd.DataFrame([{"fold_index": 0}]),
        "neuron_diagnostics": pd.DataFrame([{"unified_bits_per_spike": 0.0}]),
        "time_bin_diagnostics": pd.DataFrame([{"time_bin": 0}]),
        "latent_diagnostics": pd.DataFrame([{"effective_rank": 1.0}]),
        "rate_diagnostics": pd.DataFrame([{"global_rate_ratio": 1.0}]),
        "objective_diagnostics": pd.DataFrame([{"kl_loss": 0.0}]),
        "baseline_gap_decomposition": pd.DataFrame(
            [
                {
                    "component": "latent underutilization",
                    "estimated_recoverable_bits_per_spike": 0.05,
                    "evidence": "collapsed",
                    "diagnostic_only": True,
                }
            ]
        ),
    }


def test_writer_creates_required_outputs_and_claim_safe_report(tmp_path: Path) -> None:
    recommendation = {
        "recommended_next_action": "targeted_lfads_repair_pilot",
        "integrity_checks_passed": True,
        "dominant_failure_mode": "posterior_or_latent_collapse",
        "secondary_failure_modes": ["insufficient_latent_utilization"],
        "estimated_recoverable_gap": 0.05,
        "targeted_repair_available": True,
        "full_lfads_evaluation_allowed": False,
        "rationale": "one repair",
        "required_next_protocol": "targeted_lfads_repair_pilot",
    }

    paths = write_lfads_diagnostics_outputs(tmp_path, _summary(), _tables(), recommendation)

    assert set(paths) == {
        "summary",
        "run_diagnostics",
        "checkpoint_diagnostics",
        "neuron_diagnostics",
        "time_bin_diagnostics",
        "latent_diagnostics",
        "rate_diagnostics",
        "objective_diagnostics",
        "baseline_gap_decomposition",
        "recommendation",
        "report",
    }
    for name in EXPECTED_OUTPUTS:
        assert (tmp_path / name).is_file()
    report = (tmp_path / "lfads_diagnostics_report.md").read_text(encoding="utf-8")
    assert "does not train or select a new model" in report
    assert "Outer-evaluation diagnostics were not used to change checkpoint selection" in report
    assert "one held-out-neuron mask" in report
    assert "Full multi-repeat LFADS evaluation remains disallowed" in report


def test_script_main_uses_existing_checkpoints_without_training(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    module = _script_module()
    config_path = tmp_path / "diagnostics.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {"name": "mc_maze_large", "expected_hash": "0" * 64},
                "protocol": {
                    "repeat_index": 0,
                    "fold_indices": [0, 1, 2, 3, 4],
                    "initialization_seeds": [2027, 2028, 2029, 2030, 2031],
                },
                "decision": {"allow_full_lfads_evaluation": False},
                "reporting": {"output_dir": str(tmp_path / "out")},
            }
        ),
        encoding="utf-8",
    )
    result = {
        "summary": _summary(),
        "tables": _tables(),
        "recommendation": {
            "recommended_next_action": "targeted_lfads_repair_pilot",
            "full_lfads_evaluation_allowed": False,
        },
        "output_dir": tmp_path / "out",
    }
    monkeypatch.setattr(module, "run_lfads_diagnostics", lambda _config: result)
    monkeypatch.setattr(module, "_write_figures", lambda *_args: None)

    assert module.main(["--config", str(config_path)]) == 0
    output = capsys.readouterr().out
    assert "recommended_next_action: targeted_lfads_repair_pilot" in output
    assert json.loads(output.strip().splitlines()[-1]) == {"status": "lfads_diagnostics_complete"}
