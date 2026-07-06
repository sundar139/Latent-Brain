from __future__ import annotations

from itertools import product
from typing import Any

import pandas as pd  # type: ignore[import-untyped]


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand a parameter grid in deterministic insertion-key order."""
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            msg = f"grid entry {key!r} must be a non-empty list"
            raise ValueError(msg)
    keys = list(grid)
    return [
        dict(zip(keys, values, strict=True)) for values in product(*(grid[key] for key in keys))
    ]


def _require_columns(dataframe: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns - set(dataframe.columns))
    if missing:
        msg = f"sweep results are missing required columns: {', '.join(missing)}"
        raise ValueError(msg)


def rank_sweep_results(
    results: pd.DataFrame,
    primary_split: str,
    primary_metric: str,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """Rank sweep rows for one split with deterministic tie-breakers."""
    _require_columns(results, {"run_id", "split", primary_metric})
    primary = results[results["split"] == primary_split].copy()
    if primary.empty:
        msg = f"no sweep rows found for primary split {primary_split!r}"
        raise ValueError(msg)

    sort_columns = [primary_metric]
    ascending = [not higher_is_better]
    if "poisson_nll" in primary.columns:
        sort_columns.append("poisson_nll")
        ascending.append(True)
    for column in ("ridge_alpha", "smoothing_sigma_ms", "run_id"):
        if column in primary.columns:
            sort_columns.append(column)
            ascending.append(True)

    ranked = primary.sort_values(sort_columns, ascending=ascending, kind="mergesort").reset_index(
        drop=True
    )
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def select_best_config(ranked_results: pd.DataFrame) -> dict[str, Any]:
    """Return parameter values for the top-ranked sweep configuration."""
    required = {
        "run_id",
        "smoothing_sigma_ms",
        "ridge_alpha",
        "standardize_features",
        "fit_intercept",
    }
    _require_columns(ranked_results, required)
    if ranked_results.empty:
        msg = "ranked sweep results are empty"
        raise ValueError(msg)
    row = ranked_results.iloc[0]
    return {
        "run_id": row["run_id"],
        "smoothing_sigma_ms": float(row["smoothing_sigma_ms"]),
        "ridge_alpha": float(row["ridge_alpha"]),
        "standardize_features": bool(row["standardize_features"]),
        "fit_intercept": bool(row["fit_intercept"]),
    }
