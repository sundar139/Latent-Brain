from __future__ import annotations

from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from latentbrain.reporting.report_tables import METHOD_REGISTRY_COLUMNS

NOT_OFFICIAL_STATEMENT = (
    "This is a local diagnostic report, not an official NLB leaderboard result."
)
INVALID_CONTROL_STATEMENT = (
    "Invalid controls use evaluation split target information and cannot be reported as "
    "model performance."
)
SEED_CONFOUND_STATEMENT = (
    "The neural-ODE near-win was seed-specific and did not survive multi-seed robustness."
)
SPLIT_INSTABILITY_STATEMENT = (
    "The 15-trial validation/test splits are unstable, so single-split numbers are not final "
    "performance."
)
OLD_MEAN_RATE_STATEMENT = (
    "Old incompatible mean-rate values are historical-only and are not used as tuning targets."
)
RECOMMENDED_WINDOW_STATEMENT = (
    "The carried-forward MC_Maze Small window is behavior_speed_peak_centered_1p28s."
)
EARLY_WINDOW_STATEMENT = (
    "Previous from_start_1p28s results describe an early/pre-movement window and should not be "
    "described as reach-dynamics performance."
)
DIFFERENT_TARGET_STATEMENT = (
    "Recommended-window scores and from-start scores use different prediction targets, not "
    "direct performance improvements."
)

REQUIRED_REPORT_PHRASES = (
    NOT_OFFICIAL_STATEMENT,
    INVALID_CONTROL_STATEMENT,
    SEED_CONFOUND_STATEMENT,
    SPLIT_INSTABILITY_STATEMENT,
    OLD_MEAN_RATE_STATEMENT,
    RECOMMENDED_WINDOW_STATEMENT,
    EARLY_WINDOW_STATEMENT,
    DIFFERENT_TARGET_STATEMENT,
)

FORBIDDEN_REPORT_PHRASES = (
    "recommended-window scores are performance improvements over from-start scores",
    "recommended-window scores improve performance over from-start scores",
    "recommended-window scores outperform from-start scores",
    "recommended-window performance improvement over from-start",
)

REQUIRED_REPORT_SECTIONS = (
    "# MC_Maze Small Diagnostic Report",
    "## Scope",
    "## Dataset and preprocessing",
    "## Canonical metric",
    "## Recommended movement-window protocol",
    "## Previous early-window diagnostics",
    "## Method registry",
    "## Accepted results",
    "## Multi-seed robustness",
    "## Split generalization audit",
    "## Cross-validated rate audit",
    "## Invalid controls and leakage diagnostics",
    "## Negative neural-model findings",
    "## Reporting recommendation",
    "## Claim safety checklist",
    "## Next research actions",
)

CHECKLIST_ITEMS = (
    "Recommended movement window disclosed",
    "Previous from-start window labeled early/pre-movement diagnostic",
    "Recommended-window scores not compared as performance improvements over from-start scores",
    "Invalid controls excluded from model performance",
    "No official leaderboard claim",
    "Single-split results not reported as final performance",
    "Canonical unified metric used",
    "Old incompatible mean-rate values excluded from current targets",
    "Generated outputs not committed",
    "Negative neural results included",
    "Seed confound disclosed",
    "Split instability disclosed",
)


def validate_method_registry(method_registry: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    if method_registry.empty:
        return ["method registry is empty"]
    missing = [column for column in METHOD_REGISTRY_COLUMNS if column not in method_registry]
    if missing:
        failures.append(f"method registry is missing columns: {missing}")
        return failures
    invalid = method_registry[~method_registry["valid_model"].astype(bool)]
    if invalid["reportable_as_model_performance"].astype(bool).any():
        failures.append("an invalid control is marked reportable as model performance")
    if invalid["invalid_reason"].astype(str).eq("").any():
        failures.append("an invalid control has no invalid_reason")
    if invalid["carried_forward"].astype(bool).any():
        failures.append("an invalid control is marked carried forward")
    carried = method_registry[method_registry["carried_forward"].astype(bool)]
    if carried.empty:
        failures.append("no method is marked carried forward")
    elif len(carried) > 1:
        failures.append("more than one method is marked carried forward")
    elif not bool(carried.iloc[0]["valid_model"]):
        failures.append("the carried-forward method is not a valid model")
    if method_registry["method_name"].duplicated().any():
        failures.append("method registry contains duplicate method names")
    return failures


def validate_claim_safety(findings: dict[str, Any], method_registry: pd.DataFrame) -> list[str]:
    failures: list[str] = list(validate_method_registry(method_registry))
    if bool(findings.get("official_leaderboard_claim", False)):
        failures.append("official leaderboard claim is true")
    if not bool(findings.get("no_official_benchmark_claim", True)):
        failures.append("no_official_benchmark_claim is false")
    if bool(findings.get("single_split_results_reportable", True)):
        failures.append("single-split results are marked reportable")
    if not bool(findings.get("factor_latent_beats_invalid_control_mean", False)):
        failures.append("factor-latent does not beat the invalid split-mean control by mean")
    mode = str(findings.get("recommended_reporting_mode", ""))
    expected_mode = "recommended_window_stratified_cross_validation"
    if mode != expected_mode:
        failures.append(f"recommended reporting mode must be {expected_mode}; got {mode!r}")
    window = str(findings.get("carried_forward_window", ""))
    expected_window = "behavior_speed_peak_centered_1p28s"
    if not window:
        failures.append("carried-forward window is missing")
    elif window != expected_window:
        failures.append(f"carried-forward window must be {expected_window}; got {window!r}")
    if str(findings.get("previous_from_start_window_status", "")) != (
        "early_premovement_diagnostic"
    ):
        failures.append("from_start_1p28s is not labelled early/pre-movement diagnostic")
    if not bool(findings.get("invalid_controls_excluded_from_model_performance", False)):
        failures.append("invalid controls are not excluded from model performance")
    carried = str(findings.get("carried_forward_valid_method", ""))
    if not carried:
        failures.append("no carried-forward method recorded")
    elif not method_registry.empty and "method_name" in method_registry:
        rows = method_registry[method_registry["method_name"] == carried]
        if rows.empty:
            failures.append(f"carried-forward method {carried!r} is absent from the registry")
        elif not bool(rows.iloc[0]["valid_model"]):
            failures.append(f"carried-forward method {carried!r} is an invalid control")
    if findings.get("split_mean_advantage_is_rate_offset"):
        failures.append("split-mean advantage is incorrectly attributed to a global rate offset")
    if not bool(findings.get("split_mean_advantage_is_target_leakage", False)):
        failures.append("split-mean advantage is not labelled as target leakage")
    if not bool(findings.get("neural_ode_near_win_seed_specific", False)):
        failures.append("neural-ODE near-win is not labelled seed-specific")
    if str(findings.get("canonical_metric", "")) != "unified_bits_per_spike":
        failures.append("canonical metric is not unified_bits_per_spike")
    return failures


def validate_report_text(report_text: str) -> list[str]:
    failures: list[str] = []
    for section in REQUIRED_REPORT_SECTIONS:
        if section not in report_text:
            failures.append(f"report is missing section: {section}")
    for phrase in REQUIRED_REPORT_PHRASES:
        if phrase not in report_text:
            failures.append(f"report is missing required statement: {phrase}")
    lowered = report_text.lower()
    for phrase in FORBIDDEN_REPORT_PHRASES:
        if phrase in lowered:
            failures.append(f"report implies direct performance improvements: {phrase}")
    return failures
