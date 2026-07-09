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

REQUIRED_REPORT_PHRASES = (
    NOT_OFFICIAL_STATEMENT,
    INVALID_CONTROL_STATEMENT,
    SEED_CONFOUND_STATEMENT,
    SPLIT_INSTABILITY_STATEMENT,
    OLD_MEAN_RATE_STATEMENT,
)

REQUIRED_REPORT_SECTIONS = (
    "# MC_Maze Small Diagnostic Report",
    "## Scope",
    "## Dataset and preprocessing",
    "## Canonical metric",
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
    "No official leaderboard claim",
    "No invalid control reported as model performance",
    "No single-split result reported as final performance",
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
    if bool(findings.get("single_split_results_reportable", False)):
        failures.append("single-split results are marked reportable")
    mode = str(findings.get("recommended_reporting_mode", ""))
    if mode == "single_split":
        failures.append("recommended reporting mode is single_split")
    if mode != "repeated_split":
        failures.append(f"recommended reporting mode must be repeated_split; got {mode!r}")
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
    return failures
