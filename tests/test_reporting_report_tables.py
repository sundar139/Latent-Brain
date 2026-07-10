from __future__ import annotations

from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pytest

from latentbrain.reporting.report_tables import (
    METHOD_REGISTRY_COLUMNS,
    TABLE_NAMES,
    build_diagnostic_tables,
    build_method_registry,
)

INVALID_METHODS = {"split_mean_rate_invalid", "oracle_split_scaled_factor_latent_invalid"}


def _config() -> dict[str, Any]:
    return {
        "dataset": {
            "name": "mc_maze_small",
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
    }


def _method_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method_name": "factor_latent",
                "valid_model": True,
                "mean_test_unified_bits_per_spike": 0.0082,
                "notes": "train-only",
            },
            {
                "method_name": "split_mean_rate_invalid",
                "valid_model": False,
                "mean_test_unified_bits_per_spike": 0.0924,
                "notes": "leaks",
            },
        ]
    )


def _inputs() -> dict[str, Any]:
    return {
        "data_quality_summary": {"n_trials": 100, "n_neurons": 142},
        "seed_robustness_summary": {
            "best_mean_method": "factor_latent",
            "carried_forward_method": "factor_latent",
            "any_neural_beats_factor_latent_mean": False,
            "any_neural_beats_factor_latent_lower_ci": False,
        },
        "split_audit_summary": {
            "generalization_risk": "high",
            "validation_trial_count": 15,
            "test_trial_count": 15,
            "accepted_split_seed": 2027,
            "heldin_neuron_count": 106,
            "heldout_neuron_count": 36,
            "train_trial_count": 70,
            "validation_positive_test_negative_persists": False,
        },
        "cv_rate_audit_summary": {
            "factor_latent_repeated_split_validation_mean": 0.0269,
            "factor_latent_repeated_split_test_mean": 0.0090,
            "factor_latent_test_positive_fraction": 0.76,
            "invalid_split_mean_advantage_over_factor_latent": 0.0842,
            "rate_offset_explains_split_mean_advantage": False,
            "invalid_control_methods": sorted(INVALID_METHODS),
        },
        "recommended_window_cv_summary": {
            "factor_latent_mean": 0.07707984048489147,
            "factor_latent_ci95_low": 0.07143536625695274,
            "factor_latent_ci95_high": 0.08251744011449201,
            "split_mean_invalid_mean": 0.07110368937717054,
            "factor_latent_minus_split_mean_invalid": 0.005976151107720928,
            "leakage_dominance_persists": False,
        },
        "recommended_window_method_summary": pd.DataFrame(),
        "method_summary": _method_summary(),
        "missing_inputs": [],
    }


def test_method_registry_has_required_columns() -> None:
    registry = build_method_registry()

    assert list(registry.columns) == METHOD_REGISTRY_COLUMNS
    assert not registry.empty


def test_method_registry_marks_invalid_controls_invalid() -> None:
    registry = build_method_registry(_inputs(), _config())

    invalid = registry[registry["method_name"].isin(INVALID_METHODS)]
    assert len(invalid) == len(INVALID_METHODS)
    assert not bool(invalid["valid_model"].any())
    assert not bool(invalid["reportable_as_model_performance"].any())
    assert invalid["invalid_reason"].astype(str).ne("").all()


def test_method_registry_marks_factor_latent_carried_forward() -> None:
    registry = build_method_registry()

    carried = registry[registry["carried_forward"].astype(bool)]
    assert len(carried) == 1
    assert carried.iloc[0]["method_name"] == "factor_latent"
    assert bool(carried.iloc[0]["valid_model"]) is True
    assert carried.iloc[0]["evaluated_window"] == "behavior_speed_peak_centered_1p28s"
    assert carried.iloc[0]["reporting_protocol"] == (
        "recommended_window_stratified_cross_validation"
    )
    assert carried.iloc[0]["current_protocol_status"] == "carried_forward_recommended_window"


def test_neural_methods_are_marked_negative_or_historical_diagnostics() -> None:
    registry = build_method_registry().set_index("method_name")

    for method in (
        "neural_ode_refinement",
        "neural_ode_objective_low_dropout_high_heldout",
        "switching_ode_tuning",
    ):
        assert registry.loc[method, "status"] == "negative_diagnostic"
        assert bool(registry.loc[method, "reportable_as_model_performance"]) is False
        assert registry.loc[method, "current_protocol_status"] == "early_premovement_diagnostic"
        assert "negative_diagnostic_under_old_window_or_unstable_protocol" in str(
            registry.loc[method, "notes"]
        )
    for method in ("lfads_unified_tuning", "lfads_controller_tuning", "neural_sde_tuning"):
        assert registry.loc[method, "status"] == "historical_diagnostic"
        assert bool(registry.loc[method, "reportable_as_model_performance"]) is False
        assert registry.loc[method, "current_protocol_status"] == "historical_diagnostic"


def test_audit_flagged_invalid_methods_override_the_static_table() -> None:
    inputs = _inputs()
    inputs["cv_rate_audit_summary"]["invalid_control_methods"] = ["neural_ode_refinement"]

    registry = build_method_registry(inputs, _config()).set_index("method_name")

    assert bool(registry.loc["neural_ode_refinement", "valid_model"]) is False
    assert registry.loc["neural_ode_refinement", "invalid_reason"] != ""


def test_tables_are_built_with_required_names() -> None:
    tables = build_diagnostic_tables(_inputs(), _config())

    assert set(tables) == set(TABLE_NAMES)
    for frame in tables.values():
        assert isinstance(frame, pd.DataFrame)


def test_invalid_controls_are_excluded_from_the_valid_model_table() -> None:
    tables = build_diagnostic_tables(_inputs(), _config())

    valid = tables["valid_model_summary"]
    invalid = tables["invalid_control_summary"]

    assert set(valid["method_name"]).isdisjoint(INVALID_METHODS)
    assert set(invalid["method_name"]) == {"split_mean_rate_invalid"}
    assert "invalid_reason" in invalid.columns
    assert invalid.iloc[0]["invalid_reason"] != ""


def test_empty_optional_inputs_are_handled_clearly() -> None:
    inputs: dict[str, Any] = {
        "data_quality_summary": None,
        "seed_robustness_summary": None,
        "split_audit_summary": None,
        "cv_rate_audit_summary": None,
        "method_summary": None,
        "missing_inputs": ["data_quality_summary_path"],
    }

    tables = build_diagnostic_tables(inputs, _config())

    assert set(tables) == set(TABLE_NAMES)
    dataset = tables["dataset_summary"].set_index("field")["value"]
    assert dataset["n_trials"] == "unavailable"
    assert dataset["validation_trial_count"] == "unavailable"
    assert tables["valid_model_summary"].empty
    assert tables["invalid_control_summary"].empty
    seed = tables["seed_robustness_summary"].set_index("field")["value"]
    assert seed["best_mean_method"] == "unavailable"


def test_dataset_summary_reports_configured_hash_and_binning() -> None:
    dataset = build_diagnostic_tables(_inputs(), _config())["dataset_summary"].set_index("field")

    assert dataset.loc["dataset_hash", "value"] == "abc"
    assert dataset.loc["bin_size_ms", "value"] == 20
    assert dataset.loc["window_seconds", "value"] == pytest.approx(1.28)
    assert dataset.loc["validation_trial_count", "value"] == 15
