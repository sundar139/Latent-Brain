from __future__ import annotations

from typing import Any

import pandas as pd  # type: ignore[import-untyped]

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
    }


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
    lfads = (
        valid[valid["method_name"].astype(str).str.lower().str.contains("lfads")]
        if not valid.empty
        else valid
    )
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
        "oracle_validation_bits_per_spike": oracle_value,
        "old_mean_rate_values_historical_only": True,
    }
