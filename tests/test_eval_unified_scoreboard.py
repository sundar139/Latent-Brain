from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]

from latentbrain.eval.unified_scoreboard import (
    HISTORICAL_STATUS,
    LEADERBOARD_COLUMNS,
    SPLIT_SCORE_COLUMNS,
    build_historical_metric_notes,
    build_unified_score_row,
    rank_unified_validation_scores,
    summarize_unified_scoreboard,
)


def test_score_row_contains_required_fields() -> None:
    row = build_unified_score_row(
        "factor_latent",
        "factor_decoder",
        "validation",
        0.03,
        10.0,
        True,
        "train_heldout_mean_rate",
        "unified",
    )

    assert set(SPLIT_SCORE_COLUMNS).issubset(row)


def test_validation_ranking_puts_valid_models_in_descending_bits() -> None:
    scores = pd.DataFrame(
        [
            build_unified_score_row("lfads", "direct", "validation", 0.01, 2.0, True, "ref", ""),
            build_unified_score_row(
                "factor_latent", "decoder", "validation", 0.03, 1.0, True, "ref", ""
            ),
            build_unified_score_row("oracle", "oracle", "validation", 3.0, 0.1, False, "ref", ""),
        ]
    )

    ranked = rank_unified_validation_scores(scores)

    assert list(ranked.columns) == LEADERBOARD_COLUMNS
    assert ranked.iloc[0]["method_name"] == "factor_latent"
    assert ranked.iloc[1]["method_name"] == "lfads"
    assert bool(ranked.iloc[2]["is_oracle_control"])
    assert not bool(ranked.iloc[2]["valid_model"])


def test_historical_notes_mark_incompatible_values_as_historical_only() -> None:
    notes = build_historical_metric_notes({"old_mean": 0.7})

    assert notes.iloc[0]["status"] == HISTORICAL_STATUS
    assert "not directly comparable" in notes.iloc[0]["reason"]


def test_summary_identifies_best_valid_and_lfads_family_methods() -> None:
    leaderboard = pd.DataFrame(
        {
            "rank": [1, 2, 3],
            "method_name": ["factor_latent", "coordinated_dropout_lfads", "oracle_smoothed"],
            "prediction_source": ["factor_decoder", "direct_model", "oracle"],
            "valid_model": [True, True, False],
            "validation_bits_per_spike": [0.03, 0.01, 3.0],
            "validation_poisson_nll": [1.0, 2.0, 0.1],
            "reference_name": ["ref", "ref", "ref"],
            "beats_train_mean_reference": [True, True, True],
            "beats_factor_latent_reference": [False, False, True],
            "is_oracle_control": [False, False, True],
            "notes": ["", "", "invalid model"],
        }
    )

    summary = summarize_unified_scoreboard(
        leaderboard,
        {
            "factor_latent_unified_validation_bits_per_spike": 0.03,
            "best_oracle_validation_bits_per_spike": 3.0,
        },
    )

    assert summary["best_valid_model"] == "factor_latent"
    assert summary["best_lfads_family_method"] == "coordinated_dropout_lfads"
    assert summary["old_mean_rate_values_historical_only"] is True
