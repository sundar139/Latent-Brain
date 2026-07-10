from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import yaml

from latentbrain.eval.reporting import write_neural_ode_diagnostics_outputs

EXPECTED_OUTPUTS = (
    "neural_ode_diagnostics_summary.json",
    "checkpoint_integrity.csv",
    "split_diagnostics.csv",
    "neuron_diagnostics.csv",
    "time_bin_diagnostics.csv",
    "latent_diagnostics.csv",
    "dynamics_diagnostics.csv",
    "decoder_diagnostics.csv",
    "objective_diagnostics.csv",
    "counterfactual_diagnostics.csv",
    "baseline_gap_decomposition.csv",
    "next_action_recommendation.json",
    "neural_ode_diagnostics_report.md",
)


def _script_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_neural_ode_diagnostics.py"
    spec = importlib.util.spec_from_file_location("run_neural_ode_diagnostics_for_test", path)
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
        "excluded_preflight_artifacts": 0,
        "accepted_outer_scores_reproduced": True,
        "outer_training_mean_unified_bits_per_spike": 0.20,
        "inner_validation_mean_unified_bits_per_spike": 0.16,
        "outer_evaluation_mean_unified_bits_per_spike": 0.141,
        "pilot_repeat_baseline_mean": 0.174,
        "mean_baseline_gap": 0.033,
        "mean_train_to_inner_gap": 0.04,
        "mean_inner_to_outer_gap": 0.019,
        "positive_neuron_fraction": 0.6,
        "negative_neuron_fraction": 0.4,
        "median_neuron_unified_bits_per_spike": 0.01,
        "fraction_neurons_beating_factor_latent": 0.3,
        "mean_effective_rank": 3.5,
        "mean_effective_rank_fraction": 0.11,
        "mean_z0_effective_rank_fraction": 0.2,
        "mean_decoder_condition_number": 500.0,
        "decoder_ill_conditioned": False,
        "decoder_low_rank": False,
        "static_state_recovery_vs_accepted": 0.0,
        "frozen_readout_recovery_vs_accepted": 0.005,
        "scalar_calibration_recovery_vs_accepted": 0.001,
        "mean_first_difference_variance_ratio": 0.4,
        "temporal_oversmoothing_detected": True,
        "mean_drift_jacobian_norm": 1.2,
        "solver_discretization_directly_testable": False,
        "solver_discretization_note": "not directly testable",
        "exact_required_recovery": 0.0126,
        "estimated_recoverable_gap": 0.005,
        "dominant_failure_mode": "trained decoder limitation",
        "secondary_failure_modes": [],
        "targeted_repair_available": False,
        "proposed_single_repair": None,
        "recommended_next_action": "retire_neural_ode_and_close_neural_model_search",
        "full_evaluation_allowed": False,
        "broad_sweep_allowed": False,
        "lfads_remains_retired": True,
    }


def _tables() -> dict[str, pd.DataFrame]:
    return {
        "checkpoint_integrity": pd.DataFrame([{"fold_index": 0}]),
        "split_diagnostics": pd.DataFrame(
            [
                {"split_name": "outer_evaluation", "unified_bits_per_spike": 0.14},
                {"split_name": "outer_training", "unified_bits_per_spike": 0.20},
            ]
        ),
        "neuron_diagnostics": pd.DataFrame([{"unified_bits_per_spike": 0.0}]),
        "time_bin_diagnostics": pd.DataFrame(
            [{"relative_time_seconds": 0.0, "unified_bits_per_spike": 0.1}]
        ),
        "latent_diagnostics": pd.DataFrame(
            [{"representation": "factor", "dimension": 0, "covariance_eigenvalue": 1.0}]
        ),
        "dynamics_diagnostics": pd.DataFrame([{"mean_drift_jacobian_norm": 1.0}]),
        "decoder_diagnostics": pd.DataFrame([{"decoder_condition_number": 500.0}]),
        "objective_diagnostics": pd.DataFrame(
            [
                {
                    "fold_index": 0,
                    "initialization_seed": 2027,
                    "inner_validation_reconstruction_loss": 0.1,
                    "inner_validation_heldout_prediction_loss": 0.2,
                    "inner_validation_z0_kl_loss": 0.0,
                }
            ]
        ),
        "counterfactual_diagnostics": pd.DataFrame(
            [
                {
                    "fold_index": 0,
                    "initialization_seed": 2027,
                    "method": "static_encoder_only_no_dynamics",
                    "outer_unified_bits_per_spike": 0.10,
                    "accepted_outer_unified_bits_per_spike": 0.14,
                    "recovery_vs_accepted": -0.04,
                    "fit_policy": "frozen weights",
                    "diagnostic_only": True,
                }
            ]
        ),
        "baseline_gap_decomposition": pd.DataFrame(
            [
                {
                    "component": "unexplained remainder",
                    "estimated_recoverable_bits_per_spike": 0.03,
                    "evidence": "no single component clears the gap",
                    "diagnostic_only": True,
                    "overlaps_other_components": False,
                }
            ]
        ),
    }


def test_writer_creates_required_outputs_and_claim_safe_report(tmp_path: Path) -> None:
    recommendation = {
        "recommended_next_action": "retire_neural_ode_and_close_neural_model_search",
        "integrity_checks_passed": True,
        "dominant_failure_mode": "trained decoder limitation",
        "secondary_failure_modes": [],
        "exact_required_recovery": 0.0126,
        "estimated_recoverable_gap": 0.005,
        "targeted_repair_available": False,
        "proposed_single_repair": None,
        "full_evaluation_allowed": False,
        "broad_sweep_allowed": False,
        "rationale": "no single repair clears the required recovery",
        "required_next_protocol": "retire_neural_ode_and_close_neural_model_search",
    }

    paths = write_neural_ode_diagnostics_outputs(tmp_path, _summary(), _tables(), recommendation)

    assert "next_action" in paths
    for name in EXPECTED_OUTPUTS:
        assert (tmp_path / name).is_file()
    report = (tmp_path / "neural_ode_diagnostics_report.md").read_text(encoding="utf-8")
    assert "does not train or select a" in report
    assert "diagnostic only and are not reported as" in report
    assert "one held-out-neuron mask" in report
    assert "LFADS remains retired." in report
    assert "not an official NLB leaderboard result" in report


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
                "decision": {"full_evaluation_currently_allowed": False},
                "reporting": {"output_dir": str(tmp_path / "out")},
            }
        ),
        encoding="utf-8",
    )
    result = {
        "summary": _summary(),
        "tables": _tables(),
        "recommendation": {
            "recommended_next_action": "retire_neural_ode_and_close_neural_model_search",
            "full_evaluation_allowed": False,
            "rationale": "no single repair clears the required recovery",
        },
        "output_dir": tmp_path / "out",
    }
    monkeypatch.setattr(module, "run_neural_ode_diagnostics", lambda _config: result)
    monkeypatch.setattr(module, "_write_figures", lambda *_args: None)

    assert module.main(["--config", str(config_path)]) == 0
    output = capsys.readouterr().out
    assert "recommended_next_action: retire_neural_ode_and_close_neural_model_search" in output
    assert json.loads(output.strip().splitlines()[-1]) == {
        "status": "neural_ode_diagnostics_complete"
    }
