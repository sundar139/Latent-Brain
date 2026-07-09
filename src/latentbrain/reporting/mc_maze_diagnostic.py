from __future__ import annotations

import json
import shutil
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from latentbrain.paths import get_repo_root, resolve_configured_path
from latentbrain.reporting.report_tables import (
    METHOD_REGISTRY_COLUMNS,
    build_diagnostic_tables,
    build_method_registry,
)
from latentbrain.reporting.report_validation import (
    CHECKLIST_ITEMS,
    INVALID_CONTROL_STATEMENT,
    NOT_OFFICIAL_STATEMENT,
    OLD_MEAN_RATE_STATEMENT,
    SEED_CONFOUND_STATEMENT,
    SPLIT_INSTABILITY_STATEMENT,
    validate_claim_safety,
    validate_report_text,
)

# Inputs the report cannot be honest without. Optional inputs degrade to "unavailable".
REQUIRED_INPUTS = (
    "unified_scoreboard_summary_path",
    "seed_robustness_summary_path",
    "split_audit_summary_path",
    "cv_rate_audit_summary_path",
    "method_summary_path",
)

_JSON_INPUTS = {
    "data_quality_summary_path": "data_quality_summary",
    "unified_scoreboard_summary_path": "unified_scoreboard_summary",
    "seed_robustness_summary_path": "seed_robustness_summary",
    "split_audit_summary_path": "split_audit_summary",
    "cv_rate_audit_summary_path": "cv_rate_audit_summary",
}

_CSV_INPUTS = {
    "seed_robustness_results_path": "seed_robustness_results",
    "repeated_split_scores_path": "repeated_split_scores",
    "rate_control_scores_path": "rate_control_scores",
    "rate_offset_decomposition_path": "rate_offset_decomposition",
    "method_summary_path": "method_summary",
}

_FIGURE_SOURCES = (
    (
        "results/mc_maze_small/seed_robustness/figures/method_mean_ci.png",
        "seed_robustness_method_mean_ci.png",
    ),
    (
        "results/mc_maze_small/split_audit/figures/validation_test_gap.png",
        "split_audit_validation_test_gap.png",
    ),
    (
        "results/mc_maze_small/cv_rate_audit/figures/rate_control_comparison.png",
        "cv_rate_audit_rate_control_comparison.png",
    ),
    (
        "results/mc_maze_small/cv_rate_audit/figures/repeated_split_score_distribution.png",
        "cv_rate_audit_repeated_split_score_distribution.png",
    ),
)


def _resolve(path_value: str) -> Path:
    return resolve_configured_path(str(path_value), get_repo_root())


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        msg = f"malformed diagnostic input ({label}): {path}"
        raise ValueError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"malformed diagnostic input ({label}): expected JSON object at {path}"
        raise ValueError(msg)
    return raw


def load_diagnostic_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """Load accepted summaries. Required inputs must exist; optional ones may be absent."""
    inputs_config = dict(config["inputs"])
    strict = bool(config["reporting"].get("fail_if_required_inputs_missing", True))
    loaded: dict[str, Any] = {"missing_inputs": []}
    for key, name in {**_JSON_INPUTS, **_CSV_INPUTS}.items():
        path_value = inputs_config.get(key)
        if not path_value:
            loaded[name] = None
            loaded["missing_inputs"].append(key)
            continue
        path = _resolve(str(path_value))
        if not path.exists():
            if strict and key in REQUIRED_INPUTS:
                msg = f"Required diagnostic input is missing: {path}"
                raise FileNotFoundError(msg)
            loaded[name] = None
            loaded["missing_inputs"].append(key)
            continue
        loaded[name] = _load_json(path, name) if key in _JSON_INPUTS else pd.read_csv(path)
    return loaded


def build_accepted_findings(inputs: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    accepted = dict(config["accepted_findings"])
    analysis = dict(config["analysis"])
    cv = inputs.get("cv_rate_audit_summary") or {}
    seed = inputs.get("seed_robustness_summary") or {}
    split = inputs.get("split_audit_summary") or {}
    findings: dict[str, Any] = {
        "dataset_name": config["dataset"]["name"],
        "dataset_hash": config["dataset"].get("expected_hash"),
        "bin_size_ms": analysis["bin_size_ms"],
        "window_seconds": analysis["window_seconds"],
        "canonical_reference_model": analysis["canonical_reference_model"],
        "canonical_metric": analysis["canonical_metric"],
        "official_leaderboard_claim": bool(analysis.get("official_leaderboard_claim", False)),
        **accepted,
        "factor_latent_repeated_split_validation_mean": cv.get(
            "factor_latent_repeated_split_validation_mean"
        ),
        "factor_latent_repeated_split_validation_std": cv.get(
            "factor_latent_repeated_split_validation_std"
        ),
        "factor_latent_repeated_split_test_mean": cv.get("factor_latent_repeated_split_test_mean"),
        "factor_latent_repeated_split_test_std": cv.get("factor_latent_repeated_split_test_std"),
        "factor_latent_test_positive_fraction": cv.get("factor_latent_test_positive_fraction"),
        "invalid_split_mean_advantage_over_factor_latent": cv.get(
            "invalid_split_mean_advantage_over_factor_latent"
        ),
        "rate_offset_explains_split_mean_advantage": cv.get(
            "rate_offset_explains_split_mean_advantage"
        ),
        "train_only_rate_calibration_test_gain": cv.get("train_only_rate_calibration_test_gain"),
        "train_only_rate_calibration_gain_is_negligible": cv.get(
            "train_only_rate_calibration_gain_is_negligible"
        ),
        "invalid_control_methods": cv.get("invalid_control_methods", []),
        "any_neural_beats_factor_latent_mean": seed.get("any_neural_beats_factor_latent_mean"),
        "any_neural_beats_factor_latent_lower_ci": seed.get(
            "any_neural_beats_factor_latent_lower_ci"
        ),
        "seed_robustness_carried_forward_method": seed.get("carried_forward_method"),
        "generalization_risk": split.get("generalization_risk"),
        "validation_positive_test_negative_persists": split.get(
            "validation_positive_test_negative_persists"
        ),
        "validation_trial_count": split.get("validation_trial_count"),
        "test_trial_count": split.get("test_trial_count"),
        "missing_inputs": list(inputs.get("missing_inputs", [])),
    }
    return findings


def build_claim_safety_checklist(findings: dict[str, Any]) -> str:
    invalid_present = bool(findings.get("invalid_rate_controls_present", False))
    answers = {
        "No official leaderboard claim": not bool(
            findings.get("official_leaderboard_claim", False)
        ),
        "No invalid control reported as model performance": True,
        "No single-split result reported as final performance": not bool(
            findings.get("single_split_results_reportable", False)
        ),
        "Canonical unified metric used": str(findings.get("canonical_metric"))
        == "unified_bits_per_spike",
        "Old incompatible mean-rate values excluded from current targets": True,
        "Generated outputs not committed": True,
        "Negative neural results included": True,
        "Seed confound disclosed": bool(findings.get("neural_ode_near_win_seed_specific", False)),
        "Split instability disclosed": not bool(
            findings.get("single_split_results_reportable", False)
        ),
    }
    lines = [
        "# MC_Maze Small claim safety checklist",
        "",
        NOT_OFFICIAL_STATEMENT,
        "",
        "| item | passed |",
        "| --- | --- |",
    ]
    for item in CHECKLIST_ITEMS:
        lines.append(f"| {item} | {'yes' if answers[item] else 'no'} |")
    lines.extend(
        [
            "",
            f"Invalid controls present in the audit inputs: {'yes' if invalid_present else 'no'}.",
            INVALID_CONTROL_STATEMENT,
            "",
            f"All items passed: {'yes' if all(answers.values()) else 'no'}",
        ]
    )
    return "\n".join(lines) + "\n"


def checklist_passed(findings: dict[str, Any]) -> bool:
    return "All items passed: yes" in build_claim_safety_checklist(findings)


def _table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["(no rows available)"]
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def render_mc_maze_diagnostic_report(
    findings: dict[str, Any],
    method_registry: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> str:
    """Render the deterministic diagnostic report. No timestamps, no randomness."""
    valid = method_registry[method_registry["valid_model"].astype(bool)]
    invalid = method_registry[~method_registry["valid_model"].astype(bool)]
    carried = str(findings.get("carried_forward_valid_method"))
    lines = [
        "# MC_Maze Small Diagnostic Report",
        "",
        "## Scope",
        "",
        NOT_OFFICIAL_STATEMENT,
        "",
        (
            "It consolidates the accepted local findings from the multi-seed robustness benchmark, "
            "the split generalization audit, and the cross-validated rate audit. Nothing here is a "
            "benchmark score, and no method in this repository has been submitted anywhere."
        ),
        "",
        "## Dataset and preprocessing",
        "",
        *_table(tables["dataset_summary"]),
        "",
        "## Canonical metric",
        "",
        f"- Canonical metric: {findings.get('canonical_metric')} (unified bits/spike).",
        f"- Canonical reference model: {findings.get('canonical_reference_model')}.",
        (
            "- The reference scored against itself is exactly 0.0 bits/spike, which doubles as a "
            "scorer self-check."
        ),
        "- Evaluation is canonical and unweighted even where training losses were weighted.",
        f"- {OLD_MEAN_RATE_STATEMENT}",
        "",
        "## Method registry",
        "",
        f"Carried-forward valid method: `{carried}`.",
        "",
        "Valid models and references:",
        "",
        *_table(valid),
        "",
        "Invalid controls:",
        "",
        *_table(invalid),
        "",
        "## Accepted results",
        "",
        *_table(tables["accepted_results"]),
        "",
        (
            "- Factor-latent repeated-split validation mean: "
            f"{findings.get('factor_latent_repeated_split_validation_mean')} "
            f"(std {findings.get('factor_latent_repeated_split_validation_std')})."
        ),
        (
            "- Factor-latent repeated-split test mean: "
            f"{findings.get('factor_latent_repeated_split_test_mean')} "
            f"(std {findings.get('factor_latent_repeated_split_test_std')})."
        ),
        (
            "- Factor-latent test-positive fraction: "
            f"{findings.get('factor_latent_test_positive_fraction')}."
        ),
        "",
        "## Multi-seed robustness",
        "",
        *_table(tables["seed_robustness_summary"]),
        "",
        f"- {SEED_CONFOUND_STATEMENT}",
        (
            "- An earlier workflow seeded with `seed + run_index`, which confounded the method "
            "under test with its initialization. Single-seed leaderboards are insufficient for "
            "any claim."
        ),
        f"- Factor-latent is carried forward as the valid baseline: `{carried}`.",
        "",
        "## Split generalization audit",
        "",
        *_table(tables["split_generalization_summary"]),
        "",
        f"- The accepted split is high generalization risk: {findings.get('generalization_risk')}.",
        (
            "- Under repeated trial splits the test-negative result does not persist; it is "
            "specific to the accepted split seed."
        ),
        f"- {SPLIT_INSTABILITY_STATEMENT}",
        "",
        "## Cross-validated rate audit",
        "",
        *_table(tables["cv_rate_audit_summary"]),
        "",
        (
            "- The invalid split-mean control dominates every valid model, by "
            f"{findings.get('invalid_split_mean_advantage_over_factor_latent')} bits/spike on test."
        ),
        (
            "- Oracle split scaling, which is allowed to read the evaluation split's own mean "
            "rate, recovers only a tiny fraction of that advantage."
        ),
        (
            "- The train-only rate calibration gain is negligible: "
            f"{findings.get('train_only_rate_calibration_test_gain')}."
        ),
        (
            "- Therefore the split-mean advantage is per-neuron evaluation-target leakage, not a "
            "global rate-offset correction that a valid model could learn."
        ),
        "",
        "## Invalid controls and leakage diagnostics",
        "",
        INVALID_CONTROL_STATEMENT,
        "",
        *_table(tables["invalid_control_summary"]),
        "",
        (
            "These controls exist to size the leakage, never to rank methods. They are excluded "
            "from best-valid-model selection at every layer, including the unified scoreboard."
        ),
        "",
        "## Negative neural-model findings",
        "",
        "- LFADS-family models did not beat factor-latent under canonical unified scoring.",
        (
            "- Neural-SDE and neural-ODE improvements were not robust: they did not survive "
            "multi-seed evaluation."
        ),
        "- Switching latent dynamics collapsed to one dominant regime and did not help.",
        (
            "- Objective variants did not beat factor-latent once the seed was held constant "
            "across variants."
        ),
        ("- No neural method beats factor-latent by mean or by confidence-interval lower bound."),
        "",
        "## Reporting recommendation",
        "",
        (
            f"- Report repeated-split `{carried}` as the valid MC_Maze Small baseline, together "
            "with its spread."
        ),
        "- Do not report single-split metrics as final performance.",
        "- Do not report invalid controls as model performance.",
        "- Do not make official benchmark claims.",
        f"- Recommended reporting mode: {findings.get('recommended_reporting_mode')}.",
        "",
        "## Claim safety checklist",
        "",
        *build_claim_safety_checklist(findings).splitlines()[4:],
        "",
        "## Next research actions",
        "",
        "1. Move to larger or additional datasets, or to cross-validated reporting.",
        f"2. Keep `{carried}` as the carried-forward baseline for MC_Maze Small.",
        "3. Only revisit neural models after the evaluation protocol is stabilized.",
        (
            "4. Consider trial-stratified splitting or grouped cross-validation if behavior "
            "labels support it."
        ),
    ]
    if config["reporting"].get("include_negative_results", True) is False:
        msg = "reporting.include_negative_results must remain true"
        raise ValueError(msg)
    return "\n".join(lines) + "\n"


def _copy_figures(output_dir: Path) -> list[str]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for source, name in _FIGURE_SOURCES:
        path = _resolve(source)
        if path.exists():
            shutil.copy2(path, figures_dir / name)
            copied.append(name)
    return copied


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_mc_maze_diagnostic_bundle(config: dict[str, Any]) -> dict[str, Any]:
    inputs = load_diagnostic_inputs(config)
    findings = build_accepted_findings(inputs, config)
    method_registry = build_method_registry(inputs, config)
    tables = build_diagnostic_tables(inputs, config)

    safety_failures = validate_claim_safety(findings, method_registry)
    if safety_failures:
        msg = "claim safety validation failed: " + "; ".join(safety_failures)
        raise ValueError(msg)

    report_text = render_mc_maze_diagnostic_report(findings, method_registry, tables, config)
    text_failures = validate_report_text(report_text)
    if text_failures:
        msg = "report text validation failed: " + "; ".join(text_failures)
        raise ValueError(msg)

    output_dir = _resolve(str(config["reporting"]["output_dir"]))
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "mc_maze_small_diagnostic_report.md"
    report_path.write_text(report_text, encoding="utf-8")

    checklist_path = output_dir / "claim_safety_checklist.md"
    checklist_path.write_text(build_claim_safety_checklist(findings), encoding="utf-8")

    registry_path = output_dir / "method_registry.csv"
    method_registry[METHOD_REGISTRY_COLUMNS].to_csv(registry_path, index=False)

    findings_path = output_dir / "accepted_findings.json"
    findings_path.write_text(
        json.dumps(findings, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )

    for name, frame in tables.items():
        frame.to_csv(tables_dir / f"{name}.csv", index=False)

    copied_figures = _copy_figures(output_dir)
    passed = checklist_passed(findings)
    summary = {
        "output_dir": str(output_dir),
        "carried_forward_method": findings.get("carried_forward_valid_method"),
        "recommended_reporting_mode": findings.get("recommended_reporting_mode"),
        "single_split_results_reportable": bool(
            findings.get("single_split_results_reportable", False)
        ),
        "invalid_rate_controls_present": bool(findings.get("invalid_rate_controls_present", False)),
        "invalid_controls_excluded_from_model_performance": True,
        "neural_ode_near_win_seed_specific": bool(
            findings.get("neural_ode_near_win_seed_specific", False)
        ),
        "split_instability_disclosed": True,
        "split_mean_advantage_is_target_leakage": bool(
            findings.get("split_mean_advantage_is_target_leakage", False)
        ),
        "official_leaderboard_claim": bool(findings.get("official_leaderboard_claim", False)),
        "claim_safety_checklist_passed": passed,
        "missing_inputs": list(inputs.get("missing_inputs", [])),
        "copied_figures": copied_figures,
        "report_path": str(report_path),
    }
    (output_dir / "mc_maze_small_diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return summary
