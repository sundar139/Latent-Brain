from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from latentbrain.paths import get_repo_root, resolve_configured_path

HISTORICAL_STATUS = "historical_only_not_directly_comparable"
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


def summarize_unified_scoreboard(
    leaderboard: pd.DataFrame,
    known_values: dict[str, float],
) -> dict[str, Any]:
    valid = (
        leaderboard[leaderboard["valid_model"].astype(bool)]
        if not leaderboard.empty
        else leaderboard
    )
    best_valid = valid.iloc[0] if not valid.empty else None
    family_mask = valid["method_name"].astype(str).str.lower().str.contains("lfads|neural_sde")
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
    }
