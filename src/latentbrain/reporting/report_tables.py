from __future__ import annotations

from typing import Any

import pandas as pd  # type: ignore[import-untyped]

METHOD_REGISTRY_COLUMNS = [
    "method_name",
    "method_family",
    "valid_model",
    "reportable_as_model_performance",
    "invalid_reason",
    "status",
    "carried_forward",
    "evaluated_window",
    "reporting_protocol",
    "current_protocol_status",
    "notes",
]

TABLE_NAMES = (
    "dataset_summary",
    "accepted_results",
    "valid_model_summary",
    "invalid_control_summary",
    "seed_robustness_summary",
    "split_generalization_summary",
    "cv_rate_audit_summary",
    "recommended_window_summary",
)

CARRIED_FORWARD_METHOD = "factor_latent"

_EVALUATION_TARGET_LEAK = (
    "Uses evaluation split target information; cannot be reported as model performance."
)

# Every method the project has scored, with an explicit verdict. A method absent from this table
# has no accepted status and must not appear in the report as a result.
_REGISTRY_ROWS: tuple[dict[str, Any], ...] = (
    {
        "method_name": "train_mean_rate",
        "method_family": "reference",
        "valid_model": True,
        "reportable_as_model_performance": False,
        "invalid_reason": "",
        "status": "canonical_reference",
        "carried_forward": False,
        "notes": (
            "Canonical train-held-out mean rate. Scores exactly 0.0 bits/spike against itself; "
            "it is the reference, not a competitor."
        ),
    },
    {
        "method_name": CARRIED_FORWARD_METHOD,
        "method_family": "non_neural_latent",
        "valid_model": True,
        "reportable_as_model_performance": True,
        "invalid_reason": "",
        "status": "carried_forward_baseline",
        "carried_forward": True,
        "notes": (
            "Train-only factor-analysis latents decoded to held-out rates. Carried forward under "
            "recommended-window stratified cross-validation, never as a single-split number."
        ),
    },
    {
        "method_name": "neural_ode_refinement",
        "method_family": "deterministic_latent_dynamics",
        "valid_model": True,
        "reportable_as_model_performance": False,
        "invalid_reason": "",
        "status": "negative_diagnostic",
        "carried_forward": False,
        "notes": (
            "Its apparent near-win against factor-latent was seed-specific and did not survive "
            "multi-seed robustness; the CI lower bound does not clear zero."
        ),
    },
    {
        "method_name": "neural_ode_objective_low_dropout_high_heldout",
        "method_family": "deterministic_latent_dynamics",
        "valid_model": True,
        "reportable_as_model_performance": False,
        "invalid_reason": "",
        "status": "negative_diagnostic",
        "carried_forward": False,
        "notes": (
            "Best controlled objective variant. Did not beat factor-latent under a shared seed."
        ),
    },
    {
        "method_name": "lfads_unified_tuning",
        "method_family": "lfads_style",
        "valid_model": True,
        "reportable_as_model_performance": False,
        "invalid_reason": "",
        "status": "historical_diagnostic",
        "carried_forward": False,
        "notes": "Single-split tuning record; seeded per run index, so its spread is uncontrolled.",
    },
    {
        "method_name": "lfads_controller_tuning",
        "method_family": "lfads_style",
        "valid_model": True,
        "reportable_as_model_performance": False,
        "invalid_reason": "",
        "status": "historical_diagnostic",
        "carried_forward": False,
        "notes": "Single-split tuning record; seeded per run index, so its spread is uncontrolled.",
    },
    {
        "method_name": "neural_sde_tuning",
        "method_family": "stochastic_latent_dynamics",
        "valid_model": True,
        "reportable_as_model_performance": False,
        "invalid_reason": "",
        "status": "historical_diagnostic",
        "carried_forward": False,
        "notes": "Single-split tuning record; selected a zero-diffusion run.",
    },
    {
        "method_name": "switching_ode_tuning",
        "method_family": "switching_latent_dynamics",
        "valid_model": True,
        "reportable_as_model_performance": False,
        "invalid_reason": "",
        "status": "negative_diagnostic",
        "carried_forward": False,
        "notes": "Collapsed to one dominant regime and did not improve on deterministic dynamics.",
    },
    {
        "method_name": "split_mean_rate_invalid",
        "method_family": "invalid_control",
        "valid_model": False,
        "reportable_as_model_performance": False,
        "invalid_reason": _EVALUATION_TARGET_LEAK,
        "status": "invalid_control",
        "carried_forward": False,
        "notes": (
            "Predicts each evaluation split from that split's own held-out mean rate. Its "
            "advantage is per-neuron evaluation-target leakage, not a correctable rate offset."
        ),
    },
    {
        "method_name": "oracle_split_scaled_factor_latent_invalid",
        "method_family": "invalid_control",
        "valid_model": False,
        "reportable_as_model_performance": False,
        "invalid_reason": _EVALUATION_TARGET_LEAK,
        "status": "invalid_control",
        "carried_forward": False,
        "notes": (
            "Rescales factor-latent to the evaluation split's observed mean rate. Recovers only a "
            "tiny fraction of the split-mean advantage, which is why that advantage is leakage."
        ),
    },
)


def build_method_registry(
    inputs: dict[str, Any] | None = None, config: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Static registry of every scored method with its accepted validity verdict."""
    registry = pd.DataFrame(list(_REGISTRY_ROWS))
    registry["evaluated_window"] = "from_start_1p28s"
    registry["reporting_protocol"] = "historical_diagnostic"
    registry["current_protocol_status"] = "historical_diagnostic"
    recommended = registry["method_name"].isin(
        {"train_mean_rate", "factor_latent", "split_mean_rate_invalid"}
    )
    registry.loc[recommended, "evaluated_window"] = "behavior_speed_peak_centered_1p28s"
    registry.loc[recommended, "reporting_protocol"] = (
        "recommended_window_stratified_cross_validation"
    )
    registry.loc[
        registry["method_name"].eq("factor_latent"),
        "current_protocol_status",
    ] = "carried_forward_recommended_window"
    registry.loc[registry["method_family"].eq("invalid_control"), "current_protocol_status"] = (
        "invalid_control"
    )
    negative_neural = registry["status"].eq("negative_diagnostic")
    registry.loc[negative_neural, "current_protocol_status"] = "early_premovement_diagnostic"
    neural = ~registry["method_family"].isin({"reference", "non_neural_latent", "invalid_control"})
    registry.loc[neural, "notes"] = registry.loc[neural, "notes"].map(
        lambda value: f"negative_diagnostic_under_old_window_or_unstable_protocol; {value}"
    )
    registry = registry[METHOD_REGISTRY_COLUMNS]
    invalid_methods = set()
    if inputs:
        cv_summary = inputs.get("cv_rate_audit_summary") or {}
        invalid_methods = {str(name) for name in cv_summary.get("invalid_control_methods", [])}
    if invalid_methods:
        # Any method the audit flagged invalid must be invalid here too, even if the static
        # table drifted.
        mask = registry["method_name"].isin(invalid_methods)
        registry.loc[mask, "valid_model"] = False
        registry.loc[mask, "reportable_as_model_performance"] = False
        registry.loc[registry["invalid_reason"].eq("") & mask, "invalid_reason"] = (
            _EVALUATION_TARGET_LEAK
        )
    return registry


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _rows_to_frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns)


def _dataset_summary(inputs: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    quality = inputs.get("data_quality_summary") or {}
    split_audit = inputs.get("split_audit_summary") or {}
    analysis = config["analysis"]
    rows = [
        {"field": "dataset_name", "value": config["dataset"]["name"]},
        {"field": "dataset_hash", "value": config["dataset"].get("expected_hash", "")},
        {"field": "bin_size_ms", "value": analysis["bin_size_ms"]},
        {"field": "window_seconds", "value": analysis["window_seconds"]},
        {"field": "original_bin_size_ms", "value": config["dataset"]["original_bin_size_ms"]},
        {"field": "n_trials", "value": quality.get("n_trials", "unavailable")},
        {"field": "n_neurons", "value": quality.get("n_neurons", "unavailable")},
        {
            "field": "train_trial_count",
            "value": split_audit.get("train_trial_count", "unavailable"),
        },
        {
            "field": "validation_trial_count",
            "value": split_audit.get("validation_trial_count", "unavailable"),
        },
        {"field": "test_trial_count", "value": split_audit.get("test_trial_count", "unavailable")},
        {
            "field": "heldin_neuron_count",
            "value": split_audit.get("heldin_neuron_count", "unavailable"),
        },
        {
            "field": "heldout_neuron_count",
            "value": split_audit.get("heldout_neuron_count", "unavailable"),
        },
        {
            "field": "accepted_split_seed",
            "value": split_audit.get("accepted_split_seed", "unavailable"),
        },
    ]
    return _rows_to_frame(rows, ["field", "value"])


def _accepted_results(inputs: dict[str, Any]) -> pd.DataFrame:
    cv = inputs.get("cv_rate_audit_summary") or {}
    seed = inputs.get("seed_robustness_summary") or {}
    split = inputs.get("split_audit_summary") or {}
    rows = [
        {
            "finding": "factor_latent_repeated_split_validation_mean",
            "value": _float(cv.get("factor_latent_repeated_split_validation_mean")),
            "source": "cv_rate_audit",
        },
        {
            "finding": "factor_latent_repeated_split_test_mean",
            "value": _float(cv.get("factor_latent_repeated_split_test_mean")),
            "source": "cv_rate_audit",
        },
        {
            "finding": "factor_latent_test_positive_fraction",
            "value": _float(cv.get("factor_latent_test_positive_fraction")),
            "source": "cv_rate_audit",
        },
        {
            "finding": "invalid_split_mean_advantage_over_factor_latent",
            "value": _float(cv.get("invalid_split_mean_advantage_over_factor_latent")),
            "source": "cv_rate_audit",
        },
        {
            "finding": "any_neural_beats_factor_latent_mean",
            "value": seed.get("any_neural_beats_factor_latent_mean"),
            "source": "seed_robustness",
        },
        {
            "finding": "any_neural_beats_factor_latent_lower_ci",
            "value": seed.get("any_neural_beats_factor_latent_lower_ci"),
            "source": "seed_robustness",
        },
        {
            "finding": "generalization_risk",
            "value": split.get("generalization_risk"),
            "source": "split_audit",
        },
        {
            "finding": "validation_positive_test_negative_persists",
            "value": split.get("validation_positive_test_negative_persists"),
            "source": "split_audit",
        },
        {
            "finding": "rate_offset_explains_split_mean_advantage",
            "value": cv.get("rate_offset_explains_split_mean_advantage"),
            "source": "cv_rate_audit",
        },
    ]
    return _rows_to_frame(rows, ["finding", "value", "source"])


def _method_summary_frames(
    inputs: dict[str, Any], registry: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    method_summary = inputs.get("method_summary")
    columns = ["method_name", "valid_model", "mean_test_unified_bits_per_spike", "notes"]
    if method_summary is None or method_summary.empty:
        empty = _rows_to_frame([], columns)
        return empty, empty
    merged = method_summary.copy()
    valid = merged[merged["valid_model"].astype(bool)]
    invalid = merged[~merged["valid_model"].astype(bool)]
    invalid = invalid.merge(
        registry[["method_name", "invalid_reason"]], on="method_name", how="left"
    )
    return valid.reset_index(drop=True), invalid.reset_index(drop=True)


def _seed_robustness_summary(inputs: dict[str, Any]) -> pd.DataFrame:
    summary = inputs.get("seed_robustness_summary") or {}
    keys = (
        "best_mean_method",
        "best_mean_validation_unified_bits_per_spike",
        "best_lower_ci_method",
        "best_lower_ci_validation_unified_bits_per_spike",
        "best_neural_method",
        "best_neural_method_mean_validation_unified_bits_per_spike",
        "paired_mean_difference_best_neural_minus_factor_latent",
        "any_neural_beats_factor_latent_mean",
        "any_neural_beats_factor_latent_lower_ci",
        "carried_forward_method",
    )
    rows = [{"field": key, "value": summary.get(key, "unavailable")} for key in keys]
    return _rows_to_frame(rows, ["field", "value"])


def _split_generalization_summary(inputs: dict[str, Any]) -> pd.DataFrame:
    summary = inputs.get("split_audit_summary") or {}
    keys = (
        "accepted_split_seed",
        "validation_trial_count",
        "test_trial_count",
        "factor_latent_validation_mean",
        "factor_latent_test_mean",
        "factor_latent_validation_test_gap",
        "generalization_risk",
        "repeated_split_test_mean",
        "repeated_split_test_positive_fraction",
        "validation_positive_test_negative_persists",
    )
    rows = [{"field": key, "value": summary.get(key, "unavailable")} for key in keys]
    return _rows_to_frame(rows, ["field", "value"])


def _cv_rate_audit_summary(inputs: dict[str, Any]) -> pd.DataFrame:
    summary = inputs.get("cv_rate_audit_summary") or {}
    keys = (
        "factor_latent_repeated_split_validation_mean",
        "factor_latent_repeated_split_validation_std",
        "factor_latent_repeated_split_test_mean",
        "factor_latent_repeated_split_test_std",
        "factor_latent_test_positive_fraction",
        "between_split_test_variance",
        "within_split_random_state_test_variance",
        "factor_analysis_random_state_validation_range",
        "factor_analysis_random_state_test_range",
        "split_mean_rate_invalid_test_mean",
        "invalid_split_mean_advantage_over_factor_latent",
        "rate_offset_explains_split_mean_advantage",
        "train_only_rate_calibration_test_gain",
        "train_only_rate_calibration_gain_is_negligible",
        "best_valid_rate_control_method",
        "recommended_reporting_mode",
    )
    rows = [{"field": key, "value": summary.get(key, "unavailable")} for key in keys]
    return _rows_to_frame(rows, ["field", "value"])


def _recommended_window_summary(inputs: dict[str, Any]) -> pd.DataFrame:
    summary = inputs.get("recommended_window_cv_summary") or {}
    keys = (
        "recommended_window_name",
        "bin_size_ms",
        "fold_count",
        "repeats",
        "total_folds",
        "factor_latent_mean",
        "factor_latent_ci95_low",
        "factor_latent_ci95_high",
        "factor_latent_positive_fraction",
        "split_mean_invalid_mean",
        "factor_latent_minus_split_mean_invalid",
        "leakage_dominance_persists",
        "moving_bin_fraction_mean",
        "endpoint_direction_entropy_mean",
        "fold_balance_warning",
    )
    rows = [{"field": key, "value": summary.get(key, "unavailable")} for key in keys]
    return _rows_to_frame(rows, ["field", "value"])


def build_diagnostic_tables(
    inputs: dict[str, Any], config: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    registry = build_method_registry(inputs, config)
    valid_summary, invalid_summary = _method_summary_frames(inputs, registry)
    tables = {
        "dataset_summary": _dataset_summary(inputs, config),
        "accepted_results": _accepted_results(inputs),
        "valid_model_summary": valid_summary,
        "invalid_control_summary": invalid_summary,
        "seed_robustness_summary": _seed_robustness_summary(inputs),
        "split_generalization_summary": _split_generalization_summary(inputs),
        "cv_rate_audit_summary": _cv_rate_audit_summary(inputs),
        "recommended_window_summary": _recommended_window_summary(inputs),
    }
    missing = set(TABLE_NAMES) - set(tables)
    if missing:
        msg = f"diagnostic tables are missing: {sorted(missing)}"
        raise ValueError(msg)
    return tables
