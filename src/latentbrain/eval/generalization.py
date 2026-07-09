from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

GAP_COLUMNS = [
    "method_name",
    "seed",
    "validation_unified_bits_per_spike",
    "test_unified_bits_per_spike",
    "gap_validation_minus_test",
]

GAP_SUMMARY_COLUMNS = [
    "method_name",
    "mean_validation",
    "mean_test",
    "mean_gap",
    "gap_ci95_low",
    "gap_ci95_high",
    "test_positive_fraction",
    "generalization_risk",
]

RISK_LOW = "low"
RISK_MODERATE = "moderate"
RISK_HIGH = "high"
RISK_UNRESOLVED = "unresolved_missing_data"


def validation_test_gap_table(seed_robustness_results: pd.DataFrame) -> pd.DataFrame:
    if seed_robustness_results.empty:
        return pd.DataFrame(columns=GAP_COLUMNS)
    results = seed_robustness_results
    if "status" in results:
        results = results[results["status"] == "completed"]
    if results.empty:
        return pd.DataFrame(columns=GAP_COLUMNS)
    table = results[
        [
            "method_name",
            "seed",
            "validation_unified_bits_per_spike",
            "test_unified_bits_per_spike",
        ]
    ].copy()
    table["gap_validation_minus_test"] = (
        table["validation_unified_bits_per_spike"] - table["test_unified_bits_per_spike"]
    )
    return table.sort_values(["method_name", "seed"], kind="mergesort").reset_index(drop=True)[
        GAP_COLUMNS
    ]


def bootstrap_gap_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    repeats: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    """Paired percentile bootstrap CI for mean(values_a - values_b). Deterministic given seed."""
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape:
        msg = "values_a and values_b must have the same shape"
        raise ValueError(msg)
    if repeats <= 0:
        msg = "bootstrap repeats must be positive"
        raise ValueError(msg)
    if not 0.0 < confidence < 1.0:
        msg = "confidence must be in (0, 1)"
        raise ValueError(msg)
    # Resample seeds, not the two splits independently: the pairing is the point.
    differences = a - b
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        return (float("nan"), float("nan"))
    if differences.size == 1:
        only = float(differences[0])
        return (only, only)
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, differences.size, size=(repeats, differences.size))
    means = differences[draws].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return (float(low), float(high))


def flag_generalization_risk(
    validation_mean: float,
    test_mean: float,
    gap_ci_low: float,
    gap_ci_high: float,
) -> str:
    values = (validation_mean, test_mean, gap_ci_low, gap_ci_high)
    if any(not np.isfinite(value) for value in values):
        return RISK_UNRESOLVED
    # Positive on validation but negative on test means nothing generalizes to held-out trials.
    if validation_mean > 0.0 and test_mean < 0.0:
        return RISK_HIGH
    if gap_ci_low > 0.0:
        return RISK_MODERATE
    return RISK_LOW


def summarize_validation_test_gap(
    gap_table: pd.DataFrame,
    repeats: int = 10000,
    confidence: float = 0.95,
    seed: int = 1337,
) -> pd.DataFrame:
    if gap_table.empty:
        return pd.DataFrame(columns=GAP_SUMMARY_COLUMNS)
    rows: list[dict[str, Any]] = []
    for method_name, group in gap_table.groupby("method_name", sort=True):
        validation = group["validation_unified_bits_per_spike"].to_numpy(dtype=np.float64)
        test = group["test_unified_bits_per_spike"].to_numpy(dtype=np.float64)
        ci_low, ci_high = bootstrap_gap_ci(validation, test, repeats, confidence, seed)
        validation_mean = float(np.mean(validation))
        test_mean = float(np.mean(test))
        rows.append(
            {
                "method_name": str(method_name),
                "mean_validation": validation_mean,
                "mean_test": test_mean,
                "mean_gap": validation_mean - test_mean,
                "gap_ci95_low": ci_low,
                "gap_ci95_high": ci_high,
                "test_positive_fraction": float(np.mean(test > 0.0)),
                "generalization_risk": flag_generalization_risk(
                    validation_mean, test_mean, ci_low, ci_high
                ),
            }
        )
    return pd.DataFrame(rows, columns=GAP_SUMMARY_COLUMNS)


def overall_generalization_risk(gap_summary: pd.DataFrame) -> str:
    if gap_summary.empty:
        return RISK_UNRESOLVED
    risks = set(gap_summary["generalization_risk"])
    for level in (RISK_HIGH, RISK_MODERATE, RISK_UNRESOLVED):
        if level in risks:
            return level
    return RISK_LOW


def summarize_gap_dictionary(gap_summary: pd.DataFrame) -> dict[str, Any]:
    risk = overall_generalization_risk(gap_summary)
    return {
        "generalization_risk": risk,
        "validation_test_instability_detected": risk in {RISK_HIGH, RISK_MODERATE},
        "model_gap_diagnostics_available": not gap_summary.empty,
        "methods_with_negative_test_mean": []
        if gap_summary.empty
        else [
            str(row["method_name"])
            for _, row in gap_summary.iterrows()
            if float(row["mean_test"]) < 0.0
        ],
    }
