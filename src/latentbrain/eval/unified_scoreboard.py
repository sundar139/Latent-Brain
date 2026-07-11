from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from latentbrain.paths import get_repo_root, resolve_configured_path

HISTORICAL_STATUS = "historical_only_not_directly_comparable"
FACTOR_LATENT_METHOD = "factor_latent"
SPLIT_SCORE_COLUMNS = [
    "method_name",
    "split",
    "prediction_source",
    "reference_name",
    "valid_model",
    "bits_per_spike",
    "poisson_nll",
    "rank_scope",
    "notes",
    "source_summary_path",
]
LEADERBOARD_COLUMNS = [
    "rank",
    "method_name",
    "prediction_source",
    "valid_model",
    "validation_bits_per_spike",
    "validation_poisson_nll",
    "reference_name",
    "beats_train_mean_reference",
    "beats_factor_latent_reference",
    "is_oracle_control",
    "notes",
    "source_summary_path",
]
HISTORICAL_NOTE_COLUMNS = ["metric_name", "value", "status", "reason"]


def build_unified_score_row(
    method_name: str,
    prediction_source: str,
    split: str,
    bits_per_spike: float,
    poisson_nll: float | None,
    valid_model: bool,
    reference_name: str,
    notes: str,
    source_summary_path: str | None = None,
) -> dict[str, Any]:
    return {
        "method_name": method_name,
        "split": split,
        "prediction_source": prediction_source,
        "reference_name": reference_name,
        "valid_model": bool(valid_model),
        "bits_per_spike": float(bits_per_spike),
        "poisson_nll": poisson_nll,
        "rank_scope": "valid_model" if valid_model else "diagnostic_control",
        "notes": notes,
        "source_summary_path": source_summary_path,
    }


def build_lfads_tuning_score_row(
    tuning_summary: dict[str, Any], reference_name: str, source_summary_path: str | None = None
) -> dict[str, Any]:
    return build_unified_score_row(
        "lfads_unified_tuning",
        "canonical_tuning_direct_model",
        "validation",
        float(tuning_summary["best_validation_unified_bits_per_spike"]),
        float(tuning_summary["best_validation_poisson_nll"]),
        True,
        reference_name,
        f"Latest canonical LFADS-style tuning run: {tuning_summary['best_run_id']}",
        source_summary_path,
    )


def _summary_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    return resolve_configured_path(path_value, get_repo_root())


def _load_summary(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        msg = f"malformed LFADS-family summary ({label}): {path}"
        raise ValueError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"malformed LFADS-family summary ({label}): expected JSON object at {path}"
        raise ValueError(msg)
    return raw


def _required_float(summary: dict[str, Any], key: str, path: Path, label: str) -> float | None:
    if key not in summary:
        msg = f"malformed LFADS-family summary ({label}): missing {key} at {path}"
        raise ValueError(msg)
    if summary[key] is None:
        return None
    try:
        return float(summary[key])
    except (TypeError, ValueError) as exc:
        msg = f"malformed LFADS-family summary ({label}): {key} must be numeric at {path}"
        raise ValueError(msg) from exc


def _optional_float(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    return None if value is None else float(value)


def _summary_candidate(
    *,
    method_name: str,
    prediction_source: str,
    label: str,
    path: Path,
    summary: dict[str, Any],
    bits_key: str,
    nll_key: str,
    reference_name: str,
) -> dict[str, Any] | None:
    bits = _required_float(summary, bits_key, path, label)
    if bits is None:
        return None
    run_id = summary.get("best_run_id")
    note = f"Loaded from local summary: {path}"
    if run_id:
        note = f"Latest local run {run_id}. {note}"
    return build_unified_score_row(
        method_name,
        prediction_source,
        "validation",
        bits,
        _optional_float(summary, nll_key),
        True,
        reference_name,
        note,
        str(path),
    )


def _static_lfads_rows(
    config: dict[str, Any], reference_name: str, skip_methods: set[str]
) -> list[dict[str, Any]]:
    values = config["known_unified_values"]
    rows: list[dict[str, Any]] = []
    if "raw_lfads" not in skip_methods:
        rows.append(
            build_unified_score_row(
                "raw_lfads",
                "configured_known_value",
                "validation",
                float(values["lfads_unified_validation_bits_per_spike"]),
                None,
                True,
                reference_name,
                "Static fallback from config; local raw LFADS-family summary missing.",
            )
        )
    if "coordinated_dropout_lfads" not in skip_methods:
        rows.append(
            build_unified_score_row(
                "coordinated_dropout_lfads",
                "configured_known_value",
                "validation",
                float(values["coordinated_dropout_unified_validation_bits_per_spike"]),
                None,
                True,
                reference_name,
                "Static fallback from config; local coordinated-dropout summary missing.",
            )
        )
    return rows


def load_lfads_family_candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    reference_name = str(
        config.get("scoring", {}).get("reference_model", "train_heldout_mean_rate")
    )
    inputs = config["inputs"]
    candidates: list[dict[str, Any]] = []
    loaded_methods: set[str] = set()
    direct_summaries = [
        (
            "neural_ode_objectives",
            "neural_ode_objectives_direct_model",
            "deterministic neural-ODE objective diagnostics",
            inputs.get("neural_ode_objective_summary_path"),
            "best_validation_unified_bits_per_spike",
            "best_validation_poisson_nll",
        ),
        (
            "neural_ode_refinement",
            "neural_ode_refinement_direct_model",
            "deterministic neural-ODE refinement",
            inputs.get("neural_ode_refinement_summary_path"),
            "best_validation_unified_bits_per_spike",
            "best_validation_poisson_nll",
        ),
        (
            "switching_ode_tuning",
            "switching_ode_tuning_direct_model",
            "switching neural-ODE-style tuning",
            inputs.get("switching_ode_tuning_summary_path"),
            "best_validation_unified_bits_per_spike",
            "best_validation_poisson_nll",
        ),
        (
            "lfads_unified_tuning",
            "canonical_tuning_direct_model",
            "canonical LFADS tuning",
            inputs.get("lfads_unified_tuning_summary_path"),
            "best_validation_unified_bits_per_spike",
            "best_validation_poisson_nll",
        ),
        (
            "lfads_controller_tuning",
            "controller_tuning_direct_model",
            "controller LFADS tuning",
            inputs.get("lfads_controller_tuning_summary_path"),
            "best_validation_unified_bits_per_spike",
            "best_validation_poisson_nll",
        ),
        (
            "neural_sde_tuning",
            "neural_sde_tuning_direct_model",
            "neural-SDE-style tuning",
            inputs.get("neural_sde_tuning_summary_path"),
            "best_validation_unified_bits_per_spike",
            "best_validation_poisson_nll",
        ),
        (
            "neural_ode_tuning",
            "neural_ode_tuning_direct_model",
            "deterministic neural-ODE-style tuning",
            inputs.get("neural_ode_tuning_summary_path"),
            "best_validation_unified_bits_per_spike",
            "best_validation_poisson_nll",
        ),
    ]
    for method, source, label, path_value, bits_key, nll_key in direct_summaries:
        path = _summary_path(None if path_value is None else str(path_value))
        if path is None or not path.exists():
            continue
        row = _summary_candidate(
            method_name=method,
            prediction_source=source,
            label=label,
            path=path,
            summary=_load_summary(path, label),
            bits_key=bits_key,
            nll_key=nll_key,
            reference_name=reference_name,
        )
        if row is not None:
            candidates.append(row)
            loaded_methods.add(method)

    derived_summaries = [
        (
            "coordinated_dropout_lfads",
            "coordinated_dropout_direct_model",
            "coordinated dropout LFADS",
            Path(str(inputs["coordinated_dropout_dir"])) / "coordinated_dropout_summary.json",
            "best_validation_bits_per_spike",
            "best_validation_poisson_nll",
        ),
        (
            "raw_lfads",
            "rate_calibration_raw_lfads",
            "raw LFADS",
            Path(str(inputs["rate_calibration_dir"])) / "rate_calibration_summary.json",
            "raw_lfads_validation_bits_per_spike",
            "raw_lfads_validation_poisson_nll",
        ),
    ]
    for method, source, label, relative_path, bits_key, nll_key in derived_summaries:
        path = resolve_configured_path(str(relative_path), get_repo_root())
        if not path.exists():
            continue
        row = _summary_candidate(
            method_name=method,
            prediction_source=source,
            label=label,
            path=path,
            summary=_load_summary(path, label),
            bits_key=bits_key,
            nll_key=nll_key,
            reference_name=reference_name,
        )
        if row is not None:
            candidates.append(row)
            loaded_methods.add(method)

    candidates.extend(_static_lfads_rows(config, reference_name, loaded_methods))
    return candidates


MULTI_SEED_NOTE = "Multi-seed local summary (aggregate row, not a single run)."


def load_seed_robustness_candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Aggregate multi-seed rows. Absent summary falls back to no rows; malformed fails."""
    reference_name = str(
        config.get("scoring", {}).get("reference_model", "train_heldout_mean_rate")
    )
    path = _summary_path(config["inputs"].get("seed_robustness_summary_path"))
    if path is None or not path.exists():
        return []
    label = "seed robustness"
    summary = _load_summary(path, label)
    aggregates = [
        (
            "seed_robustness_best_mean",
            "multi_seed_mean",
            "best_mean_method",
            "best_mean_validation_unified_bits_per_spike",
        ),
        (
            "seed_robustness_best_lower_ci",
            "multi_seed_ci95_low",
            "best_lower_ci_method",
            "best_lower_ci_validation_unified_bits_per_spike",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for method_name, prediction_source, method_key, bits_key in aggregates:
        bits = _required_float(summary, bits_key, path, label)
        if bits is None:
            continue
        winner = summary.get(method_key)
        seeds = summary.get("seeds_evaluated") or []
        note = f"{MULTI_SEED_NOTE} Winner {winner} over {len(seeds)} seeds. Loaded from {path}"
        rows.append(
            build_unified_score_row(
                method_name,
                prediction_source,
                "validation",
                bits,
                None,
                True,
                reference_name,
                note,
                str(path),
            )
        )
    return rows


def _is_oracle(row: pd.Series) -> bool:
    text = " ".join(
        [
            str(row.get("method_name", "")),
            str(row.get("prediction_source", "")),
            str(row.get("notes", "")),
        ]
    ).lower()
    return "oracle" in text


def rank_unified_validation_scores(
    scores: pd.DataFrame,
    primary_split: str = "validation",
) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    validation = scores[scores["split"] == primary_split].copy()
    if validation.empty:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    valid = validation[validation["valid_model"].astype(bool)].copy()
    controls = validation[~validation["valid_model"].astype(bool)].copy()
    valid = valid.sort_values("bits_per_spike", ascending=False, kind="mergesort")
    controls = controls.sort_values("bits_per_spike", ascending=False, kind="mergesort")
    ranked = pd.concat([valid, controls], ignore_index=True)
    factor_rows = validation[validation["method_name"].astype(str).str.contains("factor_latent")]
    factor_reference = float(factor_rows["bits_per_spike"].max()) if not factor_rows.empty else 0.0
    rows: list[dict[str, Any]] = []
    for index, row in ranked.iterrows():
        bits = float(row["bits_per_spike"])
        rows.append(
            {
                "rank": int(index) + 1,
                "method_name": row["method_name"],
                "prediction_source": row["prediction_source"],
                "valid_model": bool(row["valid_model"]),
                "validation_bits_per_spike": bits,
                "validation_poisson_nll": row.get("poisson_nll"),
                "reference_name": row["reference_name"],
                "beats_train_mean_reference": bits > 0.0,
                "beats_factor_latent_reference": bits > factor_reference,
                "is_oracle_control": _is_oracle(row),
                "notes": row.get("notes", ""),
                "source_summary_path": row.get("source_summary_path"),
            }
        )
    return pd.DataFrame(rows, columns=LEADERBOARD_COLUMNS)


def build_historical_metric_notes(historical_values: dict[str, float]) -> pd.DataFrame:
    rows = [
        {
            "metric_name": name,
            "value": value,
            "status": HISTORICAL_STATUS,
            "reason": (
                "Old mean-rate value used an incompatible reference convention and is not "
                "directly comparable."
            ),
        }
        for name, value in historical_values.items()
    ]
    return pd.DataFrame(rows, columns=HISTORICAL_NOTE_COLUMNS)


HIGH_RISK_SCOREBOARD_NOTE = (
    "Split audit reports high generalization risk. Current results should be interpreted as "
    "validation-only diagnostics."
)


def load_split_audit_warning(config: dict[str, Any]) -> dict[str, Any]:
    """Split-audit warning fields. Absent summary falls back cleanly; malformed fails."""
    path = _summary_path(config["inputs"].get("split_audit_summary_path"))
    if path is None or not path.exists():
        return {
            "split_audit_available": False,
            "generalization_risk": None,
            "validation_test_instability_detected": False,
            "split_audit_summary_path": None,
        }
    label = "split audit"
    summary = _load_summary(path, label)
    if "generalization_risk" not in summary:
        msg = f"malformed split audit summary ({label}): missing generalization_risk at {path}"
        raise ValueError(msg)
    risk = summary["generalization_risk"]
    if not isinstance(risk, str):
        msg = f"malformed split audit summary ({label}): generalization_risk must be a string"
        raise ValueError(msg)
    return {
        "split_audit_available": True,
        "generalization_risk": risk,
        "validation_test_instability_detected": bool(
            summary.get("validation_test_instability_detected", risk in {"high", "moderate"})
        ),
        "validation_only_diagnostics": risk == "high",
        "split_audit_summary_path": str(path),
    }


def load_cv_rate_audit_warning(config: dict[str, Any]) -> dict[str, Any]:
    """Cross-validated rate-audit warnings. Absent summary falls back cleanly; malformed fails."""
    path = _summary_path(config["inputs"].get("cv_rate_audit_summary_path"))
    if path is None or not path.exists():
        return {
            "cv_rate_audit_available": False,
            "single_split_results_reportable": None,
            "recommended_reporting_mode": None,
            "invalid_rate_controls_present": False,
            "rate_offset_warning": None,
        }
    label = "cv rate audit"
    summary = _load_summary(path, label)
    for key in ("single_split_results_reportable", "recommended_reporting_mode"):
        if key not in summary:
            msg = f"malformed cv rate audit summary ({label}): missing {key} at {path}"
            raise ValueError(msg)
    invalid_methods = list(summary.get("invalid_control_methods", []))
    warning = None
    if summary.get("invalid_controls_dominate_valid_models"):
        warning = (
            "An invalid rate control that reads evaluation targets outscores every valid model; "
            "an unmodeled split-level rate offset remains."
        )
    return {
        "cv_rate_audit_available": True,
        "single_split_results_reportable": bool(summary["single_split_results_reportable"]),
        "recommended_reporting_mode": str(summary["recommended_reporting_mode"]),
        "invalid_rate_controls_present": bool(invalid_methods),
        "invalid_rate_control_methods": invalid_methods,
        "invalid_controls_dominate_valid_models": bool(
            summary.get("invalid_controls_dominate_valid_models", False)
        ),
        "rate_offset_warning": warning,
        "cv_rate_audit_summary_path": str(path),
    }


def load_stratified_cv_warning(config: dict[str, Any]) -> dict[str, Any]:
    """Stratified cross-validation recommendation. Absent falls back cleanly; malformed fails."""
    path = _summary_path(config["inputs"].get("stratified_cv_summary_path"))
    if path is None or not path.exists():
        return {
            "stratified_cv_available": False,
            "factor_latent_stratified_cv_mean": None,
            "factor_latent_stratified_cv_ci95_low": None,
        }
    label = "stratified cross-validation"
    summary = _load_summary(path, label)
    for key in ("recommended_reporting_mode", "factor_latent_mean_unified_bits_per_spike"):
        if key not in summary:
            msg = f"malformed stratified cv summary ({label}): missing {key} at {path}"
            raise ValueError(msg)
    mode = summary["recommended_reporting_mode"]
    if not isinstance(mode, str):
        msg = (
            f"malformed stratified cv summary ({label}): "
            "recommended_reporting_mode must be a string"
        )
        raise ValueError(msg)
    return {
        "stratified_cv_available": True,
        # The stratified protocol supersedes repeated random splits as the reporting mode.
        "recommended_reporting_mode": mode,
        "single_split_results_reportable": False,
        "factor_latent_stratified_cv_mean": _required_float(
            summary, "factor_latent_mean_unified_bits_per_spike", path, label
        ),
        "factor_latent_stratified_cv_ci95_low": _optional_float(summary, "factor_latent_ci95_low"),
        "factor_latent_stratified_cv_ci95_high": _optional_float(
            summary, "factor_latent_ci95_high"
        ),
        "stratified_cv_summary_path": str(path),
    }


def load_window_audit_warning(config: dict[str, Any]) -> dict[str, Any]:
    """Movement-window recommendation. Absent falls back cleanly; malformed fails clearly."""
    path = _summary_path(config["inputs"].get("window_audit_summary_path"))
    if path is None or not path.exists():
        return {
            "window_audit_available": False,
            "recommended_window_name": None,
            "current_window_still_supported": None,
        }
    label = "movement window audit"
    summary = _load_summary(path, label)
    for key in ("recommended_window_name", "recommended_reporting_mode"):
        if key not in summary:
            msg = f"malformed window audit summary ({label}): missing {key} at {path}"
            raise ValueError(msg)
    window_name = summary["recommended_window_name"]
    if not isinstance(window_name, str):
        msg = f"malformed window audit summary ({label}): recommended_window_name must be a string"
        raise ValueError(msg)
    return {
        "window_audit_available": True,
        "recommended_window_name": window_name,
        "recommended_reporting_mode": str(summary["recommended_reporting_mode"]),
        "current_window_still_supported": summary.get("current_window_still_supported"),
        "window_audit_summary_path": str(path),
    }


def load_recommended_window_cv_warning(config: dict[str, Any]) -> dict[str, Any]:
    """Recommended-window CV fields. Absent falls back cleanly; malformed fails clearly."""
    path = _summary_path(config["inputs"].get("recommended_window_cv_summary_path"))
    if path is None or not path.exists():
        return {
            "recommended_window_cv_available": False,
            "recommended_window_name": None,
            "recommended_reporting_mode": None,
            "factor_latent_recommended_window_mean": None,
            "factor_latent_recommended_window_ci95_low": None,
            "factor_latent_beats_invalid_control_mean": None,
            "single_split_results_reportable": False,
        }
    label = "recommended window cross-validation"
    summary = _load_summary(path, label)
    required = (
        "recommended_window_name",
        "recommended_reporting_mode",
        "factor_latent_mean",
        "factor_latent_ci95_low",
        "factor_latent_beats_invalid_control_mean",
        "single_split_results_reportable",
    )
    for key in required:
        if key not in summary:
            msg = f"malformed recommended window cv summary ({label}): missing {key} at {path}"
            raise ValueError(msg)
    if not isinstance(summary["recommended_window_name"], str) or not isinstance(
        summary["recommended_reporting_mode"], str
    ):
        msg = f"malformed recommended window cv summary ({label}): names must be strings"
        raise ValueError(msg)
    if bool(summary["single_split_results_reportable"]):
        msg = (
            f"malformed recommended window cv summary ({label}): "
            "single_split_results_reportable must be false"
        )
        raise ValueError(msg)
    return {
        "recommended_window_cv_available": True,
        "recommended_window_name": summary["recommended_window_name"],
        "recommended_reporting_mode": summary["recommended_reporting_mode"],
        "factor_latent_recommended_window_mean": _required_float(
            summary, "factor_latent_mean", path, label
        ),
        "factor_latent_recommended_window_ci95_low": _required_float(
            summary, "factor_latent_ci95_low", path, label
        ),
        "factor_latent_beats_invalid_control_mean": bool(
            summary["factor_latent_beats_invalid_control_mean"]
        ),
        "single_split_results_reportable": False,
        "recommended_window_cv_summary_path": str(path),
    }


def _is_seed_robustness_aggregate(method_name: object) -> bool:
    return str(method_name).startswith("seed_robustness_")


def summarize_unified_scoreboard(
    leaderboard: pd.DataFrame,
    known_values: dict[str, float],
) -> dict[str, Any]:
    valid = (
        leaderboard[leaderboard["valid_model"].astype(bool)]
        if not leaderboard.empty
        else leaderboard
    )
    aggregates = (
        valid[valid["method_name"].map(_is_seed_robustness_aggregate)] if not valid.empty else valid
    )
    # An aggregate over methods is not itself a method, so it never competes for
    # best-valid-model or best-dynamics-family.
    valid = (
        valid[~valid["method_name"].map(_is_seed_robustness_aggregate)]
        if not valid.empty
        else valid
    )
    best_valid = valid.iloc[0] if not valid.empty else None
    family_mask = (
        valid["method_name"]
        .astype(str)
        .str.lower()
        .str.contains("lfads|neural_sde|neural_ode|switching_ode")
    )
    lfads = valid[family_mask] if not valid.empty else valid
    best_lfads = (
        lfads.sort_values("validation_bits_per_spike", ascending=False).iloc[0]
        if not lfads.empty
        else None
    )
    factor_value = float(
        known_values.get("factor_latent_unified_validation_bits_per_spike", float("nan"))
    )
    oracle_value = float(known_values.get("best_oracle_validation_bits_per_spike", float("nan")))
    return {
        "train_mean_validation_bits_per_spike": float(
            known_values.get("train_mean_as_model_validation_bits_per_spike", 0.0)
        ),
        "best_valid_model": None if best_valid is None else str(best_valid["method_name"]),
        "best_valid_model_validation_bits_per_spike": None
        if best_valid is None
        else float(best_valid["validation_bits_per_spike"]),
        "factor_latent_validation_bits_per_spike": factor_value,
        "best_lfads_family_method": None if best_lfads is None else str(best_lfads["method_name"]),
        "best_lfads_family_validation_bits_per_spike": None
        if best_lfads is None
        else float(best_lfads["validation_bits_per_spike"]),
        "best_lfads_family_source_summary_path": None
        if best_lfads is None
        else best_lfads.get("source_summary_path"),
        "lfads_family_beats_factor_latent": False
        if best_lfads is None
        else float(best_lfads["validation_bits_per_spike"]) > factor_value,
        "oracle_validation_bits_per_spike": oracle_value,
        "old_mean_rate_values_historical_only": True,
        "seed_robustness_aggregate_methods": []
        if aggregates.empty
        else [str(name) for name in aggregates["method_name"]],
        "seed_robustness_ingested": not aggregates.empty,
    }


DATASET_SCOREBOARD_KEYS = (
    "recommended_window_name",
    "recommended_reporting_mode",
    "factor_latent_mean",
    "factor_latent_ci95_low",
    "factor_latent_ci95_high",
    "factor_latent_positive_fraction",
    "factor_latent_beats_invalid_control_mean",
    "leakage_dominance_persists",
    "single_split_results_reportable",
    "official_leaderboard_claim",
)


def load_dataset_cv_scoreboard(config: dict[str, Any]) -> dict[str, Any]:
    """Summary-only scoreboard for a dataset whose evidence is recommended-window CV.

    Missing summaries fall back cleanly; malformed summaries fail clearly. An invalid control
    can never surface as the best valid method here because only factor-latent is exposed.
    """
    dataset_name = str(config["dataset"]["name"])
    path = _summary_path(config["inputs"].get("recommended_window_cv_summary_path"))
    window_audit = _summary_path(config["inputs"].get("window_audit_summary_path"))
    if path is None or not path.exists():
        return {
            "dataset_name": dataset_name,
            "recommended_window_cv_available": False,
            "recommended_window_name": None,
            "recommended_reporting_mode": None,
            "factor_latent_recommended_window_mean": None,
            "factor_latent_recommended_window_ci95_low": None,
            "factor_latent_recommended_window_ci95_high": None,
            "factor_latent_positive_fraction": None,
            "factor_latent_beats_invalid_control_mean": None,
            "leakage_dominance_persists": None,
            "best_valid_method": None,
            "window_audit_available": bool(window_audit and window_audit.exists()),
            "single_split_results_reportable": False,
            "official_leaderboard_claim": False,
        }
    label = f"{dataset_name} recommended window cross-validation"
    summary = _load_summary(path, label)
    for key in DATASET_SCOREBOARD_KEYS:
        if key not in summary:
            msg = f"malformed recommended window cv summary ({label}): missing {key} at {path}"
            raise ValueError(msg)
    if bool(summary["single_split_results_reportable"]):
        msg = (
            f"malformed recommended window cv summary ({label}): "
            "single_split_results_reportable must be false"
        )
        raise ValueError(msg)
    if bool(summary["official_leaderboard_claim"]):
        msg = f"malformed recommended window cv summary ({label}): no leaderboard claim is allowed"
        raise ValueError(msg)
    return {
        "dataset_name": dataset_name,
        "recommended_window_cv_available": True,
        "recommended_window_name": str(summary["recommended_window_name"]),
        "recommended_reporting_mode": str(summary["recommended_reporting_mode"]),
        "factor_latent_recommended_window_mean": _required_float(
            summary, "factor_latent_mean", path, label
        ),
        "factor_latent_recommended_window_ci95_low": _required_float(
            summary, "factor_latent_ci95_low", path, label
        ),
        "factor_latent_recommended_window_ci95_high": _required_float(
            summary, "factor_latent_ci95_high", path, label
        ),
        "factor_latent_positive_fraction": _required_float(
            summary, "factor_latent_positive_fraction", path, label
        ),
        "factor_latent_beats_invalid_control_mean": bool(
            summary["factor_latent_beats_invalid_control_mean"]
        ),
        "leakage_dominance_persists": bool(summary["leakage_dominance_persists"]),
        # Only reportable valid models are eligible; invalid controls are never surfaced.
        "best_valid_method": FACTOR_LATENT_METHOD,
        "window_audit_available": bool(window_audit and window_audit.exists()),
        "single_split_results_reportable": False,
        "official_leaderboard_claim": False,
        "recommended_window_cv_summary_path": str(path),
    }


BASELINE_SUITE_KEYS = (
    "baseline_to_beat",
    "baseline_replaced",
    "baseline_replacement_supported",
    "neural_reevaluation_ready",
    "invalid_controls_excluded",
    "official_leaderboard_claim",
)


def load_baseline_suite_scoreboard(config: dict[str, Any]) -> dict[str, Any]:
    """Baseline suite fields. Missing falls back cleanly; malformed fails clearly."""
    path = _summary_path(config["inputs"].get("baseline_suite_summary_path"))
    if path is None or not path.exists():
        return {
            "baseline_suite_available": False,
            "baseline_to_beat": None,
            "baseline_to_beat_mean": None,
            "baseline_to_beat_ci95_low": None,
            "baseline_to_beat_ci95_high": None,
            "baseline_replaced": None,
            "baseline_replacement_supported": None,
            "neural_reevaluation_ready": False,
            "invalid_controls_excluded": True,
        }
    label = "baseline suite"
    summary = _load_summary(path, label)
    for key in BASELINE_SUITE_KEYS:
        if key not in summary:
            msg = f"malformed baseline suite summary ({label}): missing {key} at {path}"
            raise ValueError(msg)
    if bool(summary["official_leaderboard_claim"]):
        msg = f"malformed baseline suite summary ({label}): no leaderboard claim is allowed"
        raise ValueError(msg)
    if not bool(summary["invalid_controls_excluded"]):
        msg = f"malformed baseline suite summary ({label}): invalid controls must be excluded"
        raise ValueError(msg)
    baseline = str(summary["baseline_to_beat"])
    invalid_names = {"split_mean_rate_invalid", "train_mean_rate"}
    if baseline in invalid_names:
        msg = (
            f"malformed baseline suite summary ({label}): an invalid control or reference "
            f"cannot be the baseline to beat ({baseline})"
        )
        raise ValueError(msg)
    mean = (
        summary.get("best_valid_method_mean")
        if bool(summary["baseline_replaced"])
        else summary.get("factor_latent_fixed_mean")
    )
    return {
        "baseline_suite_available": True,
        "baseline_to_beat": baseline,
        "baseline_to_beat_mean": None if mean is None else float(mean),
        "baseline_to_beat_ci95_low": _optional_float(summary, "baseline_to_beat_ci95_low"),
        "baseline_to_beat_ci95_high": _optional_float(summary, "baseline_to_beat_ci95_high"),
        "baseline_replaced": bool(summary["baseline_replaced"]),
        "baseline_replacement_supported": bool(summary["baseline_replacement_supported"]),
        "neural_reevaluation_ready": bool(summary["neural_reevaluation_ready"]),
        "invalid_controls_excluded": True,
        "baseline_suite_summary_path": str(path),
    }


LFADS_PILOT_KEYS = (
    "completed_runs",
    "failed_runs",
    "mean_unified_bits_per_spike",
    "seed_level_std",
    "positive_seed_fraction",
    "mean_paired_difference_vs_baseline",
    "full_evaluation_recommended",
    "pilot_final_claim_allowed",
)


def load_lfads_pilot_scoreboard(config: dict[str, Any]) -> dict[str, Any]:
    """Optional one-repeat LFADS feasibility fields; never a ranked baseline candidate."""
    path = _summary_path(config.get("inputs", {}).get("lfads_pilot_summary_path"))
    fallback = {
        "lfads_pilot_available": False,
        "lfads_pilot_complete": False,
        "lfads_pilot_mean": None,
        "lfads_pilot_seed_std": None,
        "lfads_pilot_positive_seed_fraction": None,
        "lfads_pilot_mean_difference_vs_baseline": None,
        "lfads_full_evaluation_recommended": False,
        "lfads_pilot_final_claim_allowed": False,
    }
    if path is None or not path.exists():
        return fallback
    label = "LFADS pilot"
    summary = _load_summary(path, label)
    for key in LFADS_PILOT_KEYS:
        if key not in summary:
            msg = f"malformed LFADS pilot summary: missing {key} at {path}"
            raise ValueError(msg)
    if bool(summary["pilot_final_claim_allowed"]):
        msg = f"malformed LFADS pilot summary: final claim must remain false at {path}"
        raise ValueError(msg)
    completed = int(summary["completed_runs"])
    failed = int(summary["failed_runs"])
    return {
        "lfads_pilot_available": True,
        "lfads_pilot_complete": completed == 25 and failed == 0,
        "lfads_pilot_mean": _required_float(summary, "mean_unified_bits_per_spike", path, label),
        "lfads_pilot_seed_std": _required_float(summary, "seed_level_std", path, label),
        "lfads_pilot_positive_seed_fraction": _required_float(
            summary, "positive_seed_fraction", path, label
        ),
        "lfads_pilot_mean_difference_vs_baseline": _required_float(
            summary, "mean_paired_difference_vs_baseline", path, label
        ),
        "lfads_full_evaluation_recommended": bool(summary["full_evaluation_recommended"]),
        "lfads_pilot_final_claim_allowed": False,
        "lfads_pilot_summary_path": str(path),
    }


LFADS_DIAGNOSTIC_KEYS = (
    "integrity_checks_passed",
    "dominant_failure_mode",
    "estimated_recoverable_gap",
    "recommended_next_action",
    "full_lfads_evaluation_allowed",
)


def load_lfads_diagnostics_scoreboard(config: dict[str, Any]) -> dict[str, Any]:
    """Optional post-hoc LFADS audit fields; never a ranked baseline candidate."""
    path = _summary_path(config.get("inputs", {}).get("lfads_diagnostics_summary_path"))
    fallback = {
        "lfads_diagnostics_available": False,
        "lfads_integrity_checks_passed": None,
        "lfads_dominant_failure_mode": None,
        "lfads_estimated_recoverable_gap": None,
        "lfads_recommended_next_action": None,
        "lfads_full_evaluation_allowed": False,
    }
    if path is None or not path.exists():
        return fallback
    summary = _load_summary(path, "LFADS diagnostics")
    for key in LFADS_DIAGNOSTIC_KEYS:
        if key not in summary:
            msg = f"malformed LFADS diagnostics summary: missing {key} at {path}"
            raise ValueError(msg)
    if bool(summary["full_lfads_evaluation_allowed"]):
        msg = (
            "malformed LFADS diagnostics summary: full LFADS evaluation must remain false at "
            f"{path}"
        )
        raise ValueError(msg)
    allowed = {
        "targeted_lfads_repair_pilot",
        "retire_lfads_and_start_neural_ode_pilot",
        "block_due_to_integrity_issue",
    }
    action = str(summary["recommended_next_action"])
    if action not in allowed:
        msg = f"malformed LFADS diagnostics summary: invalid next action {action} at {path}"
        raise ValueError(msg)
    return {
        "lfads_diagnostics_available": True,
        "lfads_integrity_checks_passed": bool(summary["integrity_checks_passed"]),
        "lfads_dominant_failure_mode": str(summary["dominant_failure_mode"]),
        "lfads_estimated_recoverable_gap": _required_float(
            summary, "estimated_recoverable_gap", path, "LFADS diagnostics"
        ),
        "lfads_recommended_next_action": action,
        "lfads_full_evaluation_allowed": False,
        "lfads_diagnostics_summary_path": str(path),
    }


NEURAL_ODE_PILOT_KEYS = (
    "completed_runs",
    "failed_runs",
    "mean_unified_bits_per_spike",
    "seed_mean_std",
    "positive_seed_fraction",
    "mean_paired_difference_vs_baseline",
    "solver_stability_passed",
    "full_evaluation_recommended",
    "recommended_next_action",
    "pilot_final_claim_allowed",
)

NEURAL_ODE_NEXT_ACTIONS = (
    "run_full_neural_ode_evaluation",
    "run_targeted_neural_ode_diagnostic",
    "retire_neural_ode_and_close_neural_model_search",
    "block_due_to_integrity_issue",
)


def load_neural_ode_pilot_scoreboard(config: dict[str, Any]) -> dict[str, Any]:
    """Optional one-repeat deterministic neural-ODE feasibility fields; never a baseline."""
    path = _summary_path(config.get("inputs", {}).get("neural_ode_pilot_summary_path"))
    fallback = {
        "neural_ode_pilot_available": False,
        "neural_ode_pilot_complete": False,
        "neural_ode_pilot_mean": None,
        "neural_ode_pilot_seed_std": None,
        "neural_ode_pilot_positive_seed_fraction": None,
        "neural_ode_pilot_mean_difference_vs_baseline": None,
        "neural_ode_solver_stability_passed": None,
        "neural_ode_full_evaluation_recommended": False,
        "neural_ode_recommended_next_action": None,
        "neural_ode_pilot_final_claim_allowed": False,
    }
    if path is None or not path.exists():
        return fallback
    label = "neural-ODE pilot"
    summary = _load_summary(path, label)
    for key in NEURAL_ODE_PILOT_KEYS:
        if key not in summary:
            msg = f"malformed neural-ODE pilot summary: missing {key} at {path}"
            raise ValueError(msg)
    if bool(summary["pilot_final_claim_allowed"]):
        msg = f"malformed neural-ODE pilot summary: final claim must remain false at {path}"
        raise ValueError(msg)
    action = str(summary["recommended_next_action"])
    if action not in NEURAL_ODE_NEXT_ACTIONS:
        msg = f"malformed neural-ODE pilot summary: invalid next action {action} at {path}"
        raise ValueError(msg)
    completed = int(summary["completed_runs"])
    failed = int(summary["failed_runs"])
    return {
        "neural_ode_pilot_available": True,
        "neural_ode_pilot_complete": completed == 25 and failed == 0,
        "neural_ode_pilot_mean": _required_float(
            summary, "mean_unified_bits_per_spike", path, label
        ),
        "neural_ode_pilot_seed_std": _required_float(summary, "seed_mean_std", path, label),
        "neural_ode_pilot_positive_seed_fraction": _required_float(
            summary, "positive_seed_fraction", path, label
        ),
        "neural_ode_pilot_mean_difference_vs_baseline": _required_float(
            summary, "mean_paired_difference_vs_baseline", path, label
        ),
        "neural_ode_solver_stability_passed": bool(summary["solver_stability_passed"]),
        "neural_ode_full_evaluation_recommended": bool(summary["full_evaluation_recommended"]),
        "neural_ode_recommended_next_action": action,
        "neural_ode_pilot_final_claim_allowed": False,
        "neural_ode_pilot_summary_path": str(path),
    }


NEURAL_ODE_DIAGNOSTICS_KEYS = (
    "integrity_checks_passed",
    "dominant_failure_mode",
    "exact_required_recovery",
    "estimated_recoverable_gap",
    "targeted_repair_available",
    "proposed_single_repair",
    "recommended_next_action",
    "full_evaluation_allowed",
)

NEURAL_ODE_DIAGNOSTICS_ACTIONS = (
    "run_targeted_neural_ode_repair_pilot",
    "run_full_neural_ode_evaluation",
    "retire_neural_ode_and_close_neural_model_search",
    "block_due_to_integrity_issue",
)


def load_neural_ode_diagnostics_scoreboard(config: dict[str, Any]) -> dict[str, Any]:
    """Optional post-hoc neural-ODE audit fields; never a ranked baseline candidate."""
    path = _summary_path(config.get("inputs", {}).get("neural_ode_diagnostics_summary_path"))
    fallback = {
        "neural_ode_diagnostics_available": False,
        "neural_ode_integrity_checks_passed": None,
        "neural_ode_dominant_failure_mode": None,
        "neural_ode_exact_required_recovery": None,
        "neural_ode_estimated_recoverable_gap": None,
        "neural_ode_targeted_repair_available": None,
        "neural_ode_proposed_single_repair": None,
        "neural_ode_diagnostics_recommended_next_action": None,
        "neural_ode_diagnostics_full_evaluation_allowed": False,
    }
    if path is None or not path.exists():
        return fallback
    label = "neural-ODE diagnostics"
    summary = _load_summary(path, label)
    for key in NEURAL_ODE_DIAGNOSTICS_KEYS:
        if key not in summary:
            msg = f"malformed neural-ODE diagnostics summary: missing {key} at {path}"
            raise ValueError(msg)
    if bool(summary["full_evaluation_allowed"]):
        msg = (
            f"malformed neural-ODE diagnostics summary: full evaluation must remain false at {path}"
        )
        raise ValueError(msg)
    action = str(summary["recommended_next_action"])
    if action not in NEURAL_ODE_DIAGNOSTICS_ACTIONS:
        msg = f"malformed neural-ODE diagnostics summary: invalid next action {action} at {path}"
        raise ValueError(msg)
    return {
        "neural_ode_diagnostics_available": True,
        "neural_ode_integrity_checks_passed": bool(summary["integrity_checks_passed"]),
        "neural_ode_dominant_failure_mode": str(summary["dominant_failure_mode"]),
        "neural_ode_exact_required_recovery": _required_float(
            summary, "exact_required_recovery", path, label
        ),
        "neural_ode_estimated_recoverable_gap": _required_float(
            summary, "estimated_recoverable_gap", path, label
        ),
        "neural_ode_targeted_repair_available": bool(summary["targeted_repair_available"]),
        "neural_ode_proposed_single_repair": summary["proposed_single_repair"],
        "neural_ode_diagnostics_recommended_next_action": action,
        "neural_ode_diagnostics_full_evaluation_allowed": False,
        "neural_ode_diagnostics_summary_path": str(path),
    }


LATENT_INTERPRETABILITY_KEYS = (
    "latent_interpretability_complete",
    "out_of_fold_latents_used",
    "behavior_decoding_complete",
    "direction_decoding_complete",
    "shuffle_controls_complete",
    "representation_stability_complete",
    "supported_claim_count",
    "descriptive_claim_count",
    "unsupported_claim_count",
    "primary_neuroscience_finding",
    "ready_for_final_report",
    "official_leaderboard_claim",
    "causal_claim_allowed",
)


def load_latent_interpretability_scoreboard(config: dict[str, Any]) -> dict[str, Any]:
    """Optional claim-safe interpretability fields; never changes model ranking."""
    path = _summary_path(config.get("inputs", {}).get("latent_interpretability_summary_path"))
    fallback = {
        "latent_interpretability_available": False,
        "latent_interpretability_complete": False,
        "out_of_fold_latents_used": False,
        "behavior_decoding_complete": False,
        "direction_decoding_complete": False,
        "shuffle_controls_complete": False,
        "representation_stability_complete": False,
        "supported_claim_count": 0,
        "descriptive_claim_count": 0,
        "unsupported_claim_count": 0,
        "primary_neuroscience_finding": None,
        "ready_for_final_report": False,
        "official_leaderboard_claim": False,
        "causal_claim_allowed": False,
    }
    if path is None or not path.exists():
        return fallback
    label = "latent interpretability"
    summary = _load_summary(path, label)
    for key in LATENT_INTERPRETABILITY_KEYS:
        if key not in summary:
            msg = f"malformed latent interpretability summary: missing {key} at {path}"
            raise ValueError(msg)
    if bool(summary["official_leaderboard_claim"]) or bool(summary["causal_claim_allowed"]):
        msg = f"claim-unsafe latent interpretability summary at {path}"
        raise ValueError(msg)
    return {
        "latent_interpretability_available": True,
        **{key: summary[key] for key in LATENT_INTERPRETABILITY_KEYS},
        "official_leaderboard_claim": False,
        "causal_claim_allowed": False,
        "latent_interpretability_summary_path": str(path),
    }
