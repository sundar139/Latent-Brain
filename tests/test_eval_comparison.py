from __future__ import annotations

import math

import pandas as pd  # type: ignore[import-untyped]

from latentbrain.eval.comparison import (
    COMPARISON_COLUMNS,
    build_comparison_row,
    rank_validation_methods,
    summarize_comparison,
)


def test_comparison_row_has_required_columns_and_nan_behavior() -> None:
    row = build_comparison_row(
        "mean_rate_windowed",
        "validation",
        "constant_rate",
        {"bits_per_spike": 0.1, "poisson_nll": 5.0},
        {"time_bins": 3, "window_seconds": 0.015},
    )

    assert set(COMPARISON_COLUMNS).issubset(row)
    assert math.isnan(float(row["behavior_mean_r2"]))
    assert row["official_benchmark_claim"] is False


def test_validation_ranking_sorts_by_bits_then_poisson_then_non_neural() -> None:
    metrics = pd.DataFrame(
        [
            build_comparison_row(
                "neural",
                "validation",
                "direct",
                {"bits_per_spike": 0.2, "poisson_nll": 4.0},
                {"uses_neural_network": True},
            ),
            build_comparison_row(
                "non_neural",
                "validation",
                "ridge",
                {"bits_per_spike": 0.2, "poisson_nll": 4.0},
                {"uses_neural_network": False},
            ),
            build_comparison_row(
                "worse_nll",
                "validation",
                "ridge",
                {"bits_per_spike": 0.2, "poisson_nll": 5.0},
                {"uses_neural_network": False},
            ),
            build_comparison_row(
                "lower_bits",
                "validation",
                "ridge",
                {"bits_per_spike": 0.1, "poisson_nll": 1.0},
                {"uses_neural_network": False},
            ),
        ]
    )

    ranked = rank_validation_methods(metrics, "validation", "bits_per_spike")

    assert ranked["method_name"].tolist() == ["non_neural", "neural", "worse_nll", "lower_bits"]


def test_summary_selects_best_method() -> None:
    metrics = pd.DataFrame(
        [
            build_comparison_row(
                "a", "validation", "x", {"bits_per_spike": 0.1, "poisson_nll": 2.0}, {}
            ),
            build_comparison_row(
                "b",
                "validation",
                "y",
                {"bits_per_spike": 0.3, "poisson_nll": 3.0, "behavior_mean_r2": 0.4},
                {},
            ),
        ]
    )

    summary = summarize_comparison(metrics, "validation", "bits_per_spike")

    assert summary["best_method_name"] == "b"
    assert summary["best_prediction_source"] == "y"
    assert summary["best_validation_bits_per_spike"] == 0.3
    assert summary["best_behavior_mean_r2"] == 0.4
