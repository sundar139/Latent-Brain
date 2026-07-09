from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

RESULT_COLUMNS = [
    "method_name",
    "method_type",
    "seed",
    "split_seed",
    "initialization_seed",
    "config_hash",
    "valid_model",
    "status",
    "validation_unified_bits_per_spike",
    "validation_poisson_nll",
    "validation_behavior_mean_r2",
    "validation_factor_decoder_unified_bits_per_spike",
    "train_unified_bits_per_spike",
    "test_unified_bits_per_spike",
    "beats_train_mean_reference",
    "beats_factor_latent_single_seed_reference",
    "beats_neural_ode_refinement_single_seed_reference",
    "output_dir",
    "notes",
]

METHOD_SUMMARY_COLUMNS = [
    "method_name",
    "method_type",
    "valid_model",
    "n_seeds",
    "mean_validation_unified_bits_per_spike",
    "std_validation_unified_bits_per_spike",
    "median_validation_unified_bits_per_spike",
    "min_validation_unified_bits_per_spike",
    "max_validation_unified_bits_per_spike",
    "ci95_low",
    "ci95_high",
    "mean_validation_poisson_nll",
    "mean_test_unified_bits_per_spike",
    "beats_factor_latent_mean",
    "beats_factor_latent_lower_ci",
    "notes",
]

SEED_EFFECT_COLUMNS = [
    "seed",
    "method_a",
    "method_b",
    "metric",
    "method_a_value",
    "method_b_value",
    "difference",
]

LEADERBOARD_COLUMNS = [
    "rank",
    "method_name",
    "method_type",
    "mean_validation_unified_bits_per_spike",
    "std_validation_unified_bits_per_spike",
    "ci95_low",
    "ci95_high",
    "mean_test_unified_bits_per_spike",
    "beats_factor_latent_mean",
    "beats_factor_latent_lower_ci",
    "valid_model",
    "notes",
]

FACTOR_LATENT_METHOD = "factor_latent"

# Lower is simpler. Used only as a near-tie tie-breaker, never to override the metric.
METHOD_TYPE_SIMPLICITY = {
    "factor_latent": 0,
    "neural_ode": 1,
    "neural_ode_objective": 2,
}

PRIMARY_METRIC = "validation_unified_bits_per_spike"


def bootstrap_mean_ci(
    values: np.ndarray,
    repeats: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean. Deterministic given `seed`."""
    samples = np.asarray(values, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return (float("nan"), float("nan"))
    if repeats <= 0:
        msg = "bootstrap repeats must be positive"
        raise ValueError(msg)
    if not 0.0 < confidence < 1.0:
        msg = "confidence must be in (0, 1)"
        raise ValueError(msg)
    if samples.size == 1:
        only = float(samples[0])
        return (only, only)
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, samples.size, size=(repeats, samples.size))
    means = samples[draws].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return (float(low), float(high))


def _completed(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty or "status" not in results:
        return results
    return results[results["status"] == "completed"]


def summarize_method_scores(
    results: pd.DataFrame,
    bootstrap_repeats: int = 10000,
    confidence: float = 0.95,
    bootstrap_seed: int = 1337,
) -> pd.DataFrame:
    """Aggregate per-seed scores into one row per method, excluding failed runs."""
    completed = _completed(results)
    if completed.empty:
        return pd.DataFrame(columns=METHOD_SUMMARY_COLUMNS)
    rows: list[dict[str, Any]] = []
    for method_name, group in completed.groupby("method_name", sort=True):
        values = group[PRIMARY_METRIC].to_numpy(dtype=np.float64)
        ci_low, ci_high = bootstrap_mean_ci(values, bootstrap_repeats, confidence, bootstrap_seed)
        rows.append(
            {
                "method_name": str(method_name),
                "method_type": str(group.iloc[0]["method_type"]),
                "valid_model": bool(group.iloc[0]["valid_model"]),
                "n_seeds": int(group["seed"].nunique()),
                "mean_validation_unified_bits_per_spike": float(np.mean(values)),
                "std_validation_unified_bits_per_spike": float(np.std(values, ddof=1))
                if values.size > 1
                else 0.0,
                "median_validation_unified_bits_per_spike": float(np.median(values)),
                "min_validation_unified_bits_per_spike": float(np.min(values)),
                "max_validation_unified_bits_per_spike": float(np.max(values)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "mean_validation_poisson_nll": float(group["validation_poisson_nll"].mean()),
                "mean_test_unified_bits_per_spike": float(
                    group["test_unified_bits_per_spike"].mean()
                ),
                "beats_factor_latent_mean": False,
                "beats_factor_latent_lower_ci": False,
                "notes": str(group.iloc[0].get("notes", "")),
            }
        )
    summary = pd.DataFrame(rows, columns=METHOD_SUMMARY_COLUMNS)
    factor_rows = summary[summary["method_name"] == FACTOR_LATENT_METHOD]
    if not factor_rows.empty:
        factor_mean = float(factor_rows.iloc[0]["mean_validation_unified_bits_per_spike"])
        summary["beats_factor_latent_mean"] = (
            summary["mean_validation_unified_bits_per_spike"] > factor_mean
        )
        # A method clears factor-latent by lower CI when even the pessimistic end of its
        # own mean interval sits above the factor-latent mean.
        summary["beats_factor_latent_lower_ci"] = summary["ci95_low"] > factor_mean
    return summary


def paired_seed_differences(
    results: pd.DataFrame,
    method_a: str,
    method_b: str,
    metric: str = PRIMARY_METRIC,
) -> pd.DataFrame:
    """Per-seed `method_a - method_b` differences over seeds both methods completed."""
    completed = _completed(results)
    if completed.empty:
        return pd.DataFrame(columns=SEED_EFFECT_COLUMNS)
    a = completed[completed["method_name"] == method_a].set_index("seed")[metric]
    b = completed[completed["method_name"] == method_b].set_index("seed")[metric]
    shared = sorted(set(a.index) & set(b.index))
    rows = [
        {
            "seed": int(seed),
            "method_a": method_a,
            "method_b": method_b,
            "metric": metric,
            "method_a_value": float(a.loc[seed]),
            "method_b_value": float(b.loc[seed]),
            "difference": float(a.loc[seed]) - float(b.loc[seed]),
        }
        for seed in shared
    ]
    return pd.DataFrame(rows, columns=SEED_EFFECT_COLUMNS)


def build_seed_robustness_leaderboard(method_summary: pd.DataFrame) -> pd.DataFrame:
    if method_summary.empty:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    ranked = method_summary.copy()
    ranked["_simplicity"] = (
        ranked["method_type"].map(METHOD_TYPE_SIMPLICITY).fillna(len(METHOD_TYPE_SIMPLICITY))
    )
    ranked = ranked.sort_values(
        [
            "mean_validation_unified_bits_per_spike",
            "ci95_low",
            "std_validation_unified_bits_per_spike",
            "mean_test_unified_bits_per_spike",
            "_simplicity",
            "method_name",
        ],
        ascending=[False, False, True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked[LEADERBOARD_COLUMNS]


def summarize_seed_robustness(
    results: pd.DataFrame,
    method_summary: pd.DataFrame,
    references: dict[str, float],
) -> dict[str, Any]:
    completed = _completed(results)
    summary: dict[str, Any] = {
        "methods_evaluated": sorted(results["method_name"].unique().tolist())
        if not results.empty
        else [],
        "seeds_evaluated": sorted(int(seed) for seed in results["seed"].unique())
        if not results.empty
        else [],
        "total_jobs": int(len(results)),
        "successful_jobs": int(len(completed)),
        "primary_metric": PRIMARY_METRIC,
        "reference_model": "train_heldout_mean_rate",
        "evaluation_metric_is_unweighted": True,
        "single_seed_leaderboards_are_insufficient": True,
        "train_mean_validation_bits_per_spike": float(
            references["train_mean_validation_bits_per_spike"]
        ),
        "factor_latent_single_seed_reference": float(
            references["factor_latent_single_seed_reference"]
        ),
        "neural_ode_refinement_single_seed_reference": float(
            references["neural_ode_refinement_single_seed_reference"]
        ),
        "neural_ode_objective_single_seed_reference": float(
            references["neural_ode_objective_single_seed_reference"]
        ),
        "oracle_validation_bits_per_spike": float(references["oracle_validation_bits_per_spike"]),
        "old_incompatible_mean_rate_values_used_as_targets": False,
        "official_benchmark_claim": False,
    }
    if method_summary.empty:
        return summary | {
            "best_mean_method": None,
            "best_mean_validation_unified_bits_per_spike": None,
            "best_lower_ci_method": None,
            "best_lower_ci_validation_unified_bits_per_spike": None,
            "factor_latent_mean_validation_unified_bits_per_spike": None,
            "best_neural_method": None,
            "best_neural_method_mean_validation_unified_bits_per_spike": None,
            "paired_mean_difference_best_neural_minus_factor_latent": None,
            "any_neural_beats_factor_latent_mean": None,
            "any_neural_beats_factor_latent_lower_ci": None,
            "carried_forward_method": None,
            "carried_forward_reason": "no successful runs",
        }
    leaderboard = build_seed_robustness_leaderboard(method_summary)
    best_mean = leaderboard.iloc[0]
    best_lower_ci = method_summary.sort_values(
        ["ci95_low", "mean_validation_unified_bits_per_spike"],
        ascending=[False, False],
        kind="mergesort",
    ).iloc[0]
    factor_rows = method_summary[method_summary["method_name"] == FACTOR_LATENT_METHOD]
    factor_mean = (
        float(factor_rows.iloc[0]["mean_validation_unified_bits_per_spike"])
        if not factor_rows.empty
        else float("nan")
    )
    neural = method_summary[method_summary["method_name"] != FACTOR_LATENT_METHOD]
    best_neural = (
        neural.sort_values(
            "mean_validation_unified_bits_per_spike", ascending=False, kind="mergesort"
        ).iloc[0]
        if not neural.empty
        else None
    )
    paired_difference = float("nan")
    if best_neural is not None and not factor_rows.empty:
        differences = paired_seed_differences(
            completed, str(best_neural["method_name"]), FACTOR_LATENT_METHOD
        )
        if not differences.empty:
            paired_difference = float(differences["difference"].mean())
    beats_mean = bool(neural["beats_factor_latent_mean"].any()) if not neural.empty else False
    beats_lower_ci = (
        bool(neural["beats_factor_latent_lower_ci"].any()) if not neural.empty else False
    )
    carried_forward: str | None
    if beats_lower_ci:
        carried_forward = str(best_lower_ci["method_name"])
        reason = (
            "A neural method clears the factor-latent mean at its 95% CI lower bound; "
            "move to held-out test reporting and additional datasets."
        )
    elif beats_mean:
        carried_forward = str(best_mean["method_name"])
        reason = (
            "A neural method beats the factor-latent mean but not at the CI lower bound; "
            "run more seeds before any claims."
        )
    else:
        carried_forward = FACTOR_LATENT_METHOD if not factor_rows.empty else None
        reason = (
            "No neural method beats factor-latent across seeds; stop adding architecture on "
            "this dataset/window and invest in rigorous reporting or additional datasets."
        )
    return summary | {
        "best_mean_method": str(best_mean["method_name"]),
        "best_mean_validation_unified_bits_per_spike": float(
            best_mean["mean_validation_unified_bits_per_spike"]
        ),
        "best_lower_ci_method": str(best_lower_ci["method_name"]),
        "best_lower_ci_validation_unified_bits_per_spike": float(best_lower_ci["ci95_low"]),
        "factor_latent_mean_validation_unified_bits_per_spike": factor_mean,
        "best_neural_method": None if best_neural is None else str(best_neural["method_name"]),
        "best_neural_method_mean_validation_unified_bits_per_spike": None
        if best_neural is None
        else float(best_neural["mean_validation_unified_bits_per_spike"]),
        "paired_mean_difference_best_neural_minus_factor_latent": paired_difference,
        "any_neural_beats_factor_latent_mean": beats_mean,
        "any_neural_beats_factor_latent_lower_ci": beats_lower_ci,
        "carried_forward_method": carried_forward,
        "carried_forward_reason": reason,
    }
