from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from latentbrain.eval.window_audit import (
    MIN_MOVING_BIN_FRACTION,
    WINDOW_SLICE_COLUMNS,
    apply_window_candidate,
    build_window_recommendations,
    build_window_slices,
    summarize_window_candidates,
    window_entropy_table,
)

HAND_NAMES = ["hand_pos_x", "hand_pos_y", "cursor_pos_x", "cursor_pos_y"]
BIN = 0.02
CURRENT = "from_start_1p28s"
CHALLENGER = "behavior_speed_peak_centered_1p28s"


def _spikes(trials: int = 4, time: int = 40, neurons: int = 6) -> np.ndarray:
    generator = np.random.default_rng(1)
    return generator.poisson(0.4, size=(trials, time, neurons)).astype(np.float64)


def _behavior_with_late_peak(trials: int = 4, time: int = 40) -> np.ndarray:
    """Position is static until bin 25, then jumps: peak speed lands at index 26."""
    behavior = np.zeros((trials, time, 4))
    for trial in range(trials):
        x = np.zeros(time)
        x[26:] = 5.0
        behavior[trial, :, 0] = x
        behavior[trial, :, 1] = 0.0
        behavior[trial, :, 2] = x
    return behavior


def _candidate(name: str, policy: str, duration: float, **extra: Any) -> dict[str, Any]:
    return {"name": name, "crop_policy": policy, "duration_seconds": duration, **extra}


def test_from_start_window_slices_start_at_zero() -> None:
    slices = build_window_slices(_spikes(), None, None, _candidate("w", "from_start", 0.2), BIN)

    assert list(slices.columns) == WINDOW_SLICE_COLUMNS
    assert (slices["start_bin"] == 0).all()
    assert (slices["end_bin"] == 10).all()
    assert not bool(slices["clipped"].any())


def test_from_start_window_honours_start_seconds() -> None:
    slices = build_window_slices(
        _spikes(), None, None, _candidate("w", "from_start", 0.2, start_seconds=0.1), BIN
    )

    assert (slices["start_bin"] == 5).all()


def test_window_slices_are_deterministic() -> None:
    behavior = _behavior_with_late_peak()
    candidate = _candidate("w", "behavior_speed_peak_centered", 0.2)

    first = build_window_slices(_spikes(), behavior, HAND_NAMES, candidate, BIN)
    second = build_window_slices(_spikes(), behavior, HAND_NAMES, candidate, BIN)

    assert first.equals(second)


def test_peak_speed_centered_window_contains_the_peak() -> None:
    behavior = _behavior_with_late_peak()
    slices = build_window_slices(
        _spikes(), behavior, HAND_NAMES, _candidate("w", "behavior_speed_peak_centered", 0.2), BIN
    )

    # Peak speed is at bin 26; a 10-bin window centred there spans [21, 31).
    assert (slices["start_bin"] == 21).all()
    assert (slices["end_bin"] == 31).all()
    assert ((slices["start_bin"] <= 26) & (slices["end_bin"] > 26)).all()


def test_movement_onset_window_includes_configured_pre_event_bins() -> None:
    behavior = _behavior_with_late_peak()
    slices = build_window_slices(
        _spikes(),
        behavior,
        HAND_NAMES,
        _candidate(
            "w",
            "behavior_movement_onset",
            0.2,
            pre_event_seconds=0.06,
            speed_threshold_quantile=0.7,
        ),
        BIN,
    )

    # The trial is static until bin 26, so the 70th-percentile speed is degenerate and onset
    # falls back to a fraction of peak speed, landing at bin 26. Three pre-event bins -> start 23.
    assert (slices["start_bin"] == 23).all()
    assert (slices["end_bin"] == 33).all()


def test_clipped_windows_are_marked_clipped() -> None:
    behavior = np.zeros((2, 12, 4))
    behavior[:, 1:, 0] = 9.0  # peak speed at bin 1, so a centred window runs off the left edge

    slices = build_window_slices(
        _spikes(trials=2, time=12),
        behavior,
        HAND_NAMES,
        _candidate("w", "behavior_speed_peak_centered", 0.16),
        BIN,
    )

    assert bool(slices["clipped"].all())
    assert (slices["start_bin"] == 0).all()
    assert (slices["end_bin"] == 8).all()


def test_window_longer_than_trial_is_rejected() -> None:
    with pytest.raises(ValueError, match="requests"):
        build_window_slices(_spikes(time=10), None, None, _candidate("w", "from_start", 1.0), BIN)


def test_behavior_aligned_window_without_behavior_fails_clearly() -> None:
    with pytest.raises(ValueError, match="requires behavior data"):
        build_window_slices(
            _spikes(), None, None, _candidate("w", "behavior_speed_peak_centered", 0.2), BIN
        )


def test_apply_window_candidate_crops_spikes_and_behavior_per_trial() -> None:
    spikes = _spikes(trials=2, time=20, neurons=3)
    behavior = _behavior_with_late_peak(trials=2, time=20)
    slices = pd.DataFrame(
        {
            "trial_index": [0, 1],
            "start_bin": [0, 5],
            "end_bin": [4, 9],
            "clipped": [False, False],
        }
    )

    cropped_spikes, cropped_behavior = apply_window_candidate(spikes, behavior, slices)

    assert cropped_spikes.shape == (2, 4, 3)
    assert cropped_behavior is not None and cropped_behavior.shape == (2, 4, 4)
    assert np.array_equal(cropped_spikes[0], spikes[0, 0:4])
    assert np.array_equal(cropped_spikes[1], spikes[1, 5:9])
    assert np.array_equal(cropped_behavior[1], behavior[1, 5:9])


def _method_summary(current_mean: float, challenger_mean: float) -> pd.DataFrame:
    rows = []
    for window, factor_mean in ((CURRENT, current_mean), (CHALLENGER, challenger_mean)):
        rows.append(
            {
                "window_name": window,
                "method_name": "factor_latent",
                "valid_model": True,
                "reportable_as_model_performance": True,
                "mean_unified_bits_per_spike": factor_mean,
                "ci95_low": factor_mean - 0.005,
                "ci95_high": factor_mean + 0.005,
            }
        )
        rows.append(
            {
                "window_name": window,
                "method_name": "split_mean_rate_invalid",
                "valid_model": False,
                "reportable_as_model_performance": False,
                "mean_unified_bits_per_spike": 0.09,
                "ci95_low": 0.08,
                "ci95_high": 0.10,
            }
        )
    return pd.DataFrame(rows)


def _diagnostics(
    current_entropy: float, challenger_entropy: float, challenger_moving: float = 0.6
) -> list[dict]:
    return [
        {
            "window_name": CURRENT,
            "report_label": "current",
            "behavior_source": "hand_pos",
            "endpoint_direction_entropy": current_entropy,
            "moving_bin_fraction": 0.3,
            "fold_balance_warning": "none",
        },
        {
            "window_name": CHALLENGER,
            "report_label": "challenger",
            "behavior_source": "hand_pos",
            "endpoint_direction_entropy": challenger_entropy,
            "moving_bin_fraction": challenger_moving,
            "fold_balance_warning": "none",
        },
    ]


def _references() -> dict[str, Any]:
    return {"current_from_start_factor_latent_ci95_low": 0.019}


def _summarize(method_summary: pd.DataFrame, diagnostics: list[dict]) -> dict[str, Any]:
    return summarize_window_candidates(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        _references(),
        diagnostics,
        method_summary,
        CURRENT,
    )


def test_summary_recommends_a_challenger_that_improves_coverage_and_preserves_performance() -> None:
    summary = _summarize(_method_summary(0.025, 0.026), _diagnostics(0.8, 1.5))

    assert summary["recommended_window_name"] == CHALLENGER
    assert summary["current_window_still_supported"] is False
    assert summary["endpoint_direction_entropy_best_window"] == 1.5
    assert "invalid controls were ignored" in summary["window_selection_rationale"]


def test_summary_keeps_a_well_covered_current_window_on_its_own_merits() -> None:
    summary = _summarize(_method_summary(0.025, 0.030), _diagnostics(1.5, 0.8))

    assert summary["recommended_window_name"] == CURRENT
    assert summary["current_window_still_supported"] is True
    # Coverage is above the floor, so the retained window is not an early-window diagnostic and
    # the rationale must not claim otherwise.
    assert summary["current_window_is_early_window_diagnostic"] is False
    assert "early-window diagnostic" not in summary["window_selection_rationale"]
    assert "retained on its own merits" in summary["window_selection_rationale"]


def test_displaced_current_window_is_still_labelled_an_early_window_diagnostic() -> None:
    summary = _summarize(_method_summary(0.025, 0.026), _diagnostics(0.8, 1.5))

    assert summary["recommended_window_name"] == CHALLENGER
    assert summary["current_window_still_supported"] is False
    # Losing the recommendation does not make the current window a movement window.
    assert summary["current_window_is_early_window_diagnostic"] is True
    assert "early-window diagnostic" in summary["window_selection_rationale"]


def test_current_window_without_moving_bins_is_an_early_window_diagnostic() -> None:
    diagnostics = _diagnostics(1.5, 0.8)
    diagnostics[0]["moving_bin_fraction"] = 0.0

    summary = _summarize(_method_summary(0.025, 0.030), diagnostics)

    assert summary["recommended_window_name"] == CURRENT
    assert summary["current_window_is_early_window_diagnostic"] is True
    assert summary["current_window_moving_bin_fraction"] == 0.0
    assert "early-window diagnostic" in summary["window_selection_rationale"]


def test_summary_rejects_a_challenger_that_degrades_factor_latent() -> None:
    # Better entropy and coverage, but factor-latent collapses below the current CI lower bound.
    summary = _summarize(_method_summary(0.025, 0.001), _diagnostics(0.8, 1.9))

    assert summary["recommended_window_name"] == CURRENT
    assert CHALLENGER not in summary["eligible_windows"]


def test_summary_rejects_a_challenger_with_a_fold_balance_warning() -> None:
    diagnostics = _diagnostics(0.8, 1.9)
    diagnostics[1]["fold_balance_warning"] = "fold trial counts are imbalanced"

    summary = _summarize(_method_summary(0.025, 0.030), diagnostics)

    assert summary["recommended_window_name"] == CURRENT
    assert CHALLENGER not in summary["eligible_windows"]


def test_summary_rejects_a_challenger_without_movement_coverage() -> None:
    diagnostics = _diagnostics(0.8, 1.9, challenger_moving=MIN_MOVING_BIN_FRACTION / 2.0)

    summary = _summarize(_method_summary(0.025, 0.030), diagnostics)

    assert summary["recommended_window_name"] == CURRENT
    assert CHALLENGER in summary["behavior_coverage_warning"]


def test_invalid_controls_are_excluded_from_window_recommendation() -> None:
    # The invalid control is far stronger on the challenger, yet must not sway the choice.
    method_summary = _method_summary(0.025, 0.010)
    method_summary.loc[
        (method_summary["window_name"] == CHALLENGER)
        & (method_summary["method_name"] == "split_mean_rate_invalid"),
        "mean_unified_bits_per_spike",
    ] = 0.5

    summary = _summarize(method_summary, _diagnostics(0.8, 1.9))

    assert summary["recommended_window_name"] == CURRENT
    assert summary["invalid_controls_excluded_from_window_selection"] is True
    assert summary["best_valid_method"] == "factor_latent"


def test_recommendations_carry_the_rationale_and_never_claim_a_benchmark() -> None:
    summary = _summarize(_method_summary(0.025, 0.026), _diagnostics(0.8, 1.5))

    recommendations = build_window_recommendations(summary)

    assert recommendations["recommended_window_name"] == CHALLENGER
    assert recommendations["recommended_reporting_mode"] == "stratified_cross_validation"
    assert recommendations["official_benchmark_claim"] is False
    assert recommendations["invalid_controls_excluded_from_window_selection"] is True
    assert recommendations["carried_forward_method"] == "factor_latent"


def test_window_entropy_table_lists_every_candidate() -> None:
    table = window_entropy_table(_diagnostics(0.8, 1.5))

    assert list(table["window_name"]) == [CURRENT, CHALLENGER]
    assert "endpoint_direction_entropy" in table.columns
    assert "moving_bin_fraction" in table.columns
