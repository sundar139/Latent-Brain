from __future__ import annotations

from itertools import product
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

TUNING_RESULT_COLUMNS = [
    "run_id",
    "run_index",
    "status",
    "encoder_hidden_dim",
    "generator_hidden_dim",
    "latent_dim",
    "factor_dim",
    "dropout",
    "learning_rate",
    "weight_decay",
    "heldout_loss_weight",
    "kl_warmup_epochs",
    "epochs",
    "device",
    "validation_bits_per_spike",
    "validation_poisson_nll",
    "validation_behavior_mean_r2",
    "validation_total_loss",
    "validation_heldout_prediction_loss",
    "beats_window_matched_mean_rate",
    "beats_window_matched_factor_latent",
    "beats_previous_lfads_masked_direct",
    "output_dir",
    "notes",
]


def expand_tuning_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand a tuning grid in deterministic insertion-key order."""
    if not grid:
        msg = "grid must contain at least one parameter"
        raise ValueError(msg)
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            msg = f"grid entry {key!r} must be a non-empty list"
            raise ValueError(msg)
    keys = list(grid)
    return [
        dict(zip(keys, values, strict=True)) for values in product(*(grid[key] for key in keys))
    ]


def _slug(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m").replace("None", "na")


def make_run_id(index: int, params: dict[str, Any]) -> str:
    """Create a stable, readable run id from the deterministic grid index."""
    if index < 0:
        msg = "index must be non-negative"
        raise ValueError(msg)
    return (
        f"run_{index:03d}_"
        f"enc{_slug(params.get('encoder_hidden_dim', 'na'))}_"
        f"gen{_slug(params.get('generator_hidden_dim', 'na'))}_"
        f"lat{_slug(params.get('latent_dim', 'na'))}_"
        f"fac{_slug(params.get('factor_dim', 'na'))}_"
        f"drop{_slug(params.get('dropout', 'na'))}_"
        f"hw{_slug(params.get('heldout_loss_weight', 'na'))}"
    )


def _require_columns(results: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns - set(results.columns))
    if missing:
        msg = f"tuning results are missing required columns: {', '.join(missing)}"
        raise ValueError(msg)


def rank_tuning_results(
    results: pd.DataFrame,
    metric: str,
    mode: str,
) -> pd.DataFrame:
    """Rank successful tuning runs with deterministic tie-breakers."""
    if mode not in {"min", "max"}:
        msg = "mode must be 'min' or 'max'"
        raise ValueError(msg)
    required = {"run_id", "run_index", "status", metric}
    _require_columns(results, required)
    ranked = results[results["status"] == "completed"].copy()
    if ranked.empty:
        return ranked.assign(rank=pd.Series(dtype="int64"))
    if "parameter_count_estimate" not in ranked.columns:
        ranked["parameter_count_estimate"] = 0
    for column, default in (
        ("validation_poisson_nll", float("inf")),
        ("validation_behavior_mean_r2", float("-inf")),
    ):
        if column not in ranked.columns:
            ranked[column] = default
    sort_columns = [
        metric,
        "validation_poisson_nll",
        "validation_behavior_mean_r2",
        "parameter_count_estimate",
        "run_index",
    ]
    ascending = [mode == "min", True, False, True, True]
    ranked = ranked.sort_values(sort_columns, ascending=ascending, kind="mergesort").reset_index(
        drop=True
    )
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def summarize_tuning_results(
    results: pd.DataFrame,
    baseline_references: dict[str, float],
    selection_metric: str,
) -> dict[str, Any]:
    """Summarize local validation tuning and baseline comparisons."""
    ranked = rank_tuning_results(results, selection_metric, "max")
    successful = int((results.get("status", pd.Series(dtype=str)) == "completed").sum())
    summary: dict[str, Any] = {
        "runs_attempted": int(len(results)),
        "successful_runs": successful,
        "selection_metric": selection_metric,
        "selection_mode": "max",
        "baseline_references": dict(baseline_references),
        "official_benchmark_claim": False,
        "full_lfads_claim": False,
    }
    if ranked.empty:
        return summary | {
            "best_run_id": None,
            "best_validation_bits_per_spike": None,
            "best_validation_poisson_nll": None,
            "best_validation_behavior_mean_r2": None,
            "beats_window_matched_mean_rate": None,
            "beats_window_matched_factor_latent": None,
            "beats_previous_lfads_masked_direct": None,
        }
    best = ranked.iloc[0]
    bits = float(best["validation_bits_per_spike"])
    mean_ref = baseline_references["window_matched_mean_rate_validation_bits_per_spike"]
    factor_ref = baseline_references["window_matched_factor_latent_validation_bits_per_spike"]
    previous_ref = baseline_references["previous_lfads_masked_direct_validation_bits_per_spike"]
    return summary | {
        "best_run_id": str(best["run_id"]),
        "best_run_index": int(best["run_index"]),
        "best_run_params": {
            key: best[key]
            for key in (
                "encoder_hidden_dim",
                "generator_hidden_dim",
                "latent_dim",
                "factor_dim",
                "dropout",
                "learning_rate",
                "weight_decay",
                "heldout_loss_weight",
                "kl_warmup_epochs",
                "epochs",
            )
            if key in best
        },
        "best_validation_bits_per_spike": bits,
        "best_validation_poisson_nll": float(best["validation_poisson_nll"]),
        "best_validation_behavior_mean_r2": float(best["validation_behavior_mean_r2"]),
        "best_validation_total_loss": float(best.get("validation_total_loss", float("nan"))),
        "best_validation_heldout_prediction_loss": float(
            best.get("validation_heldout_prediction_loss", float("nan"))
        ),
        "beats_window_matched_mean_rate": bits > float(mean_ref),
        "beats_window_matched_factor_latent": bits > float(factor_ref),
        "beats_previous_lfads_masked_direct": bits > float(previous_ref),
        "best_output_dir": str(best.get("output_dir", "")),
    }
