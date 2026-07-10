from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import yaml

from latentbrain.eval.reporting import write_neural_ode_pilot_outputs

EXPECTED_OUTPUTS = (
    "neural_ode_pilot_summary.json",
    "neural_ode_pilot_runs.csv",
    "fold_seed_scores.csv",
    "seed_summary.csv",
    "fold_summary.csv",
    "paired_baseline_comparison.csv",
    "lfads_descriptive_comparison.csv",
    "checkpoint_manifest.csv",
    "solver_diagnostics.csv",
    "latent_diagnostics.csv",
    "training_resource_summary.csv",
    "neural_ode_pilot_protocol.yaml",
    "full_evaluation_recommendation.json",
    "next_action_recommendation.json",
    "neural_ode_pilot_report.md",
)


def _script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_neural_ode_pilot", Path("scripts/run_neural_ode_pilot.py")
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
        "fold_indices": [0, 1, 2, 3, 4],
        "initialization_seeds": [2027, 2028, 2029, 2030, 2031],
        "input_neuron_count": 122,
        "output_neuron_count": 162,
        "heldout_neuron_count": 40,
        "completed_runs": 25,
        "failed_runs": 0,
        "diffusion_enabled": False,
        "solver": "euler",
        "integration_step_seconds": 0.02,
        "mean_unified_bits_per_spike": 0.03,
        "run_level_score_std": 0.01,
        "seed_mean_std": 0.001,
        "positive_run_fraction": 1.0,
        "positive_seed_fraction": 1.0,
        "runs_beating_baseline": 0,
        "pilot_repeat_baseline_mean": 0.174,
        "mean_paired_difference_vs_baseline": -0.14,
        "lfads_descriptive_reference_mean": 0.0293,
        "mean_difference_vs_lfads_reference": 0.0,
        "before_peak_mean_bits_per_spike": 0.04,
        "near_peak_mean_bits_per_spike": 0.03,
        "after_peak_mean_bits_per_spike": 0.035,
        "near_peak_failure_status": "absent",
        "mean_factor_effective_rank": 3.0,
        "mean_factor_effective_rank_fraction": 0.1,
        "lfads_factor_effective_rank": 1.2161114061725022,
        "lfads_factor_effective_rank_fraction": 0.038003481442890695,
        "checkpoint_selection_split": "inner_validation",
        "checkpoint_selection_valid": True,
        "leakage_checks_passed": True,
        "solver_stability_passed": True,
        "baseline_to_beat": "factor_latent_train_selected",
        "full_evaluation_recommended": False,
        "recommended_next_action": "retire_neural_ode_and_close_neural_model_search",
        "single_split_results_reportable": False,
        "official_leaderboard_claim": False,
        "pilot_final_claim_allowed": False,
    }


def test_report_writer_emits_claim_safe_pilot_bundle(tmp_path: Path) -> None:
    empty = pd.DataFrame()
    tables = dict.fromkeys(
        (
            "neural_ode_pilot_runs",
            "fold_seed_scores",
            "seed_summary",
            "fold_summary",
            "paired_baseline_comparison",
            "lfads_descriptive_comparison",
            "checkpoint_manifest",
            "solver_diagnostics",
            "latent_diagnostics",
            "training_resource_summary",
        ),
        empty,
    )
    recommendation = {
        "proceed": False,
        "runtime_estimate_full_evaluation_hours": 2.0,
        "estimated_peak_cuda_memory_mb": 60.0,
        "reasons": ["failed: mean paired difference clears margin"],
    }
    next_action = {
        "recommended_next_action": "retire_neural_ode_and_close_neural_model_search",
        "rationale": "stable but large gap",
    }

    paths = write_neural_ode_pilot_outputs(
        tmp_path, _summary(), tables, {"protocol_frozen": True}, recommendation, next_action
    )

    assert "next_action" in paths
    for name in EXPECTED_OUTPUTS:
        assert (tmp_path / name).exists()
    report = (tmp_path / "neural_ode_pilot_report.md").read_text(encoding="utf-8")
    assert "deterministic neural-ODE pilot" in report
    assert "Diffusion is disabled" in report
    assert "one held-out-neuron mask" in report
    assert "descriptive reference" in report
    assert (
        "Event-centered inputs were extracted from the trial-aware source before rebinning"
        in report
    )
    assert "Outer-evaluation data were not used for checkpoint selection" in report
    assert "factor_latent_train_selected" in report
    assert "not an official NLB leaderboard result" in report


def test_script_rejects_nonzero_repeat(tmp_path: Path, capsys: Any) -> None:
    module = _script_module()
    config = yaml.safe_load(Path("configs/mc_maze_large_neural_ode_pilot.yaml").read_text("utf-8"))
    config["outer_protocol"]["repeat_index"] = 1
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert module.main(["--config", str(path)]) == 2
    assert "repeat_index" in capsys.readouterr().out


def test_committed_config_contains_no_placeholders() -> None:
    text = Path("configs/mc_maze_large_neural_ode_pilot.yaml").read_text("utf-8")
    assert "reuse accepted" not in text
    assert "<" not in text
    config = yaml.safe_load(text)
    assert config["model"]["diffusion_enabled"] is False
    assert config["initialization"]["seeds"] == [2027, 2028, 2029, 2030, 2031]
    assert config["outer_protocol"]["repeat_index"] == 0
    assert config["training"]["checkpoint_metric"] == "inner_validation_unified_bits_per_spike"
