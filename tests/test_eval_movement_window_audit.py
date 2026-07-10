from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from latentbrain.data.schemas import TrialSequences
from latentbrain.data.validation import validate_trial_sequences
from latentbrain.eval.window_audit import (
    apply_window_gates,
    crop_to_min_impact,
    evaluate_window_coverage,
    recommend_movement_window,
    reference_peak_speed,
    trial_movement_table,
    window_bounds,
)

NAMES = ["hand_pos_x", "hand_pos_y", "cursor_pos_x", "cursor_pos_y"]
SELECTION = {
    "maximum_clipped_trial_fraction": 0.05,
    "minimum_moving_bin_fraction": 0.25,
    "minimum_endpoint_direction_entropy_fraction": 0.70,
    "prioritize_behavior_coverage": True,
    "use_model_scores_for_selection": False,
}


def _trial(length: int, angle: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """One reach whose speed peaks late, at 0.7 of the trial, as real reaches often do."""
    generator = np.random.default_rng(seed)
    spikes = generator.poisson(0.3, size=(length, 5)).astype(np.int64)
    times = np.arange(length, dtype=np.float64)
    speed = np.exp(-(((times - 0.7 * length) / (0.08 * length)) ** 2))
    position = np.cumsum(speed)
    behavior = np.zeros((length, 4))
    behavior[:, 0] = position * np.cos(angle)
    behavior[:, 1] = position * np.sin(angle)
    behavior[:, 2] = behavior[:, 0]
    behavior[:, 3] = behavior[:, 1]
    return spikes, behavior


def _sequences(lengths: tuple[int, ...] = (40, 48, 56, 64)) -> TrialSequences:
    angles = np.linspace(-np.pi, np.pi, len(lengths), endpoint=False)
    spikes, behavior = [], []
    for index, length in enumerate(lengths):
        trial_spikes, trial_behavior = _trial(length, angle=angles[index], seed=index)
        spikes.append(trial_spikes)
        behavior.append(trial_behavior)
    return TrialSequences(
        spikes=spikes,
        behavior=behavior,
        behavior_names=NAMES,
        trial_ids=np.arange(len(lengths), dtype=np.int64),
        trial_lengths=np.asarray(lengths, dtype=np.int64),
        bin_size_ms=5,
        metadata={"spikes_conserved": True},
    )


def _candidate(name: str, policy: str, duration: float, **extra: Any) -> dict[str, Any]:
    return {"name": name, "crop_policy": policy, "duration_seconds": duration, **extra}


def test_trial_sequences_preserve_variable_lengths_and_alignment() -> None:
    sequences = _sequences()
    validate_trial_sequences(sequences)

    assert [trial.shape[0] for trial in sequences.spikes] == [40, 48, 56, 64]
    assert len({trial.shape[0] for trial in sequences.spikes}) > 1
    for spikes, behavior in zip(sequences.spikes, sequences.behavior or [], strict=True):
        assert spikes.shape[0] == behavior.shape[0]
    np.testing.assert_array_equal(sequences.trial_ids, np.arange(4))


def test_validate_trial_sequences_rejects_misaligned_behavior() -> None:
    sequences = _sequences()
    assert sequences.behavior is not None
    sequences.behavior[1] = sequences.behavior[1][:-1]

    with pytest.raises(ValueError, match="behavior and spikes are misaligned"):
        validate_trial_sequences(sequences)


def test_trial_movement_table_is_deterministic_and_per_trial() -> None:
    sequences = _sequences()

    first = trial_movement_table(sequences)
    second = trial_movement_table(sequences)

    pd.testing.assert_frame_equal(first, second)
    assert list(first["trial_length_bins"]) == [40, 48, 56, 64]
    assert (first["peak_speed_bin"] > 0).all()
    assert first["behavior_source"].eq("hand_pos").all()


def test_crop_to_min_impact_counts_bins_spikes_and_events() -> None:
    sequences = _sequences()
    movement = trial_movement_table(sequences)

    table, summary = crop_to_min_impact(sequences, movement, global_time_bins=40)

    assert list(table["global_cropped_time_bins"]) == [40, 40, 40, 40]
    assert list(table["excluded_time_bins"]) == [0, 8, 16, 24]
    raw = sum(int(trial.sum()) for trial in sequences.spikes)
    kept = sum(int(trial[:40].sum()) for trial in sequences.spikes)
    assert summary["raw_spike_count"] == raw
    assert summary["global_crop_retained_spike_count"] == kept
    assert summary["fraction_raw_spikes_excluded"] == pytest.approx((raw - kept) / raw)
    assert summary["fraction_raw_bins_excluded"] == pytest.approx(48 / 208)
    # Longer trials peak late, so the global crop drops their peak: exactly what the audit is for.
    assert summary["fraction_trials_peak_inside_global_crop"] < 1.0
    assert summary["crop_to_min_removes_peak_for_any_trial"] is True
    assert summary["global_crop_suitable_for_movement_window_audit"] is False


def test_crop_to_min_impact_reports_suitable_when_events_survive() -> None:
    sequences = _sequences(lengths=(40, 40, 40, 40))
    movement = trial_movement_table(sequences)

    _, summary = crop_to_min_impact(sequences, movement, global_time_bins=40)

    assert summary["fraction_trials_peak_inside_global_crop"] == 1.0
    assert summary["fraction_trials_onset_inside_global_crop"] == 1.0
    assert summary["global_crop_suitable_for_movement_window_audit"] is True
    assert summary["fraction_raw_spikes_excluded"] == 0.0


def test_window_bounds_distinguishes_left_and_right_clipping() -> None:
    left = window_bounds(
        "behavior_speed_peak_centered", 40, 20, peak_bin=2, onset_bin=0, pre_bins=0, start_bin=0
    )
    right = window_bounds(
        "behavior_speed_peak_centered", 40, 20, peak_bin=38, onset_bin=0, pre_bins=0, start_bin=0
    )
    interior = window_bounds(
        "behavior_speed_peak_centered", 40, 20, peak_bin=20, onset_bin=0, pre_bins=0, start_bin=0
    )

    assert left["left_clipped"] and not left["right_clipped"]
    assert right["right_clipped"] and not right["left_clipped"]
    assert not interior["left_clipped"] and not interior["right_clipped"]
    assert interior["padded_bins"] == 0


def test_window_bounds_pads_when_trial_is_shorter_than_window() -> None:
    bounds = window_bounds(
        "behavior_speed_peak_centered", 12, 20, peak_bin=6, onset_bin=0, pre_bins=0, start_bin=0
    )

    assert bounds["available_bins"] == 12
    assert bounds["padded_bins"] == 8
    assert bounds["left_clipped"] and bounds["right_clipped"]


def test_movement_onset_window_includes_configured_pre_event_time() -> None:
    bounds = window_bounds(
        "behavior_movement_onset", 100, 40, peak_bin=50, onset_bin=30, pre_bins=4, start_bin=0
    )

    assert bounds["start_bin"] == 26
    assert bounds["end_bin"] == 66


def test_peak_centered_window_contains_the_peak_for_every_trial() -> None:
    sequences = _sequences()
    movement = trial_movement_table(sequences)
    reference = reference_peak_speed(sequences, 20)

    coverage, summary, _ = evaluate_window_coverage(
        sequences, movement, _candidate("peak", "behavior_speed_peak_centered", 0.08), 20, reference
    )

    assert coverage["peak_speed_in_window"].all()
    assert summary["peak_speed_coverage_fraction"] == 1.0
    assert (coverage["requested_bins"] == 16).all()


def test_window_extraction_is_deterministic_and_rebinning_preserves_spikes() -> None:
    sequences = _sequences()
    movement = trial_movement_table(sequences)
    reference = reference_peak_speed(sequences, 20)
    candidate = _candidate("peak", "behavior_speed_peak_centered", 0.08)

    first_coverage, _, first_extras = evaluate_window_coverage(
        sequences, movement, candidate, 20, reference
    )
    second_coverage, _, second_extras = evaluate_window_coverage(
        sequences, movement, candidate, 20, reference
    )

    pd.testing.assert_frame_equal(first_coverage, second_coverage)
    np.testing.assert_array_equal(first_extras["windowed_spikes"], second_extras["windowed_spikes"])

    windowed = first_extras["windowed_spikes"]
    assert windowed.shape[1] == 4  # 16 source bins at 5 ms -> 4 bins at 20 ms
    expected = 0
    for index, trial in enumerate(sequences.spikes):
        start = int(first_coverage.iloc[index]["start_bin"])
        end = int(first_coverage.iloc[index]["end_bin"])
        expected += int(trial[start:end].sum())
    assert int(windowed.sum()) == expected


def _summary(name: str, duration: float, **overrides: Any) -> dict[str, Any]:
    base = {
        "window_name": name,
        "duration_seconds": duration,
        "crop_policy": "behavior_speed_peak_centered",
        "clipped_trial_fraction": 0.0,
        "moving_bin_fraction_mean": 0.6,
        "endpoint_direction_entropy": 2.02,
        "peak_speed_coverage_fraction": 1.0,
        "movement_onset_coverage_fraction": 1.0,
    }
    return {**base, **overrides}


def test_gates_reject_clipped_and_static_windows() -> None:
    clipped = apply_window_gates(_summary("a", 1.28, clipped_trial_fraction=0.4), SELECTION)
    static = apply_window_gates(_summary("b", 1.28, moving_bin_fraction_mean=0.01), SELECTION)
    passing = apply_window_gates(_summary("c", 1.28), SELECTION)

    assert clipped["usable_for_reporting"] is False
    assert "clipped_trial_fraction" in clipped["rejection_reasons"]
    assert static["usable_for_reporting"] is False
    assert "moving_bin_fraction_mean" in static["rejection_reasons"]
    assert passing["usable_for_reporting"] is True
    assert passing["rejection_reasons"] == "none"


def test_transferred_small_window_wins_when_all_gates_pass() -> None:
    summaries = [
        _summary("behavior_speed_peak_centered_1p28s", 1.28),
        _summary("behavior_speed_peak_centered_2p56s", 2.56, moving_bin_fraction_mean=0.9),
    ]

    result = recommend_movement_window(
        summaries, SELECTION, {"small_recommended_window": "behavior_speed_peak_centered_1p28s"}
    )

    assert result["recommended_window_name"] == "behavior_speed_peak_centered_1p28s"
    assert result["small_window_transfers"] is True
    assert result["selection_used_model_scores"] is False


def test_shorter_valid_window_wins_when_transfer_fails() -> None:
    summaries = [
        _summary("behavior_speed_peak_centered_1p28s", 1.28, clipped_trial_fraction=0.5),
        _summary("behavior_movement_onset_1p28s", 1.28, crop_policy="behavior_movement_onset"),
        _summary("behavior_movement_onset_2p56s", 2.56, crop_policy="behavior_movement_onset"),
    ]

    result = recommend_movement_window(
        summaries, SELECTION, {"small_recommended_window": "behavior_speed_peak_centered_1p28s"}
    )

    assert result["recommended_window_name"] == "behavior_movement_onset_1p28s"
    assert result["small_window_transfers"] is False
    assert "failed transfer" in result["window_selection_rationale"]
    assert "behavior_speed_peak_centered_1p28s" in result["rejected_windows"]


def test_model_scores_cannot_drive_selection() -> None:
    selection = {**SELECTION, "use_model_scores_for_selection": True}

    with pytest.raises(ValueError, match="must be false"):
        recommend_movement_window(
            [_summary("a", 1.28)], selection, {"small_recommended_window": "a"}
        )


def test_recommendation_ignores_any_model_score_field() -> None:
    summaries = [
        _summary("behavior_speed_peak_centered_1p28s", 1.28, factor_latent_mean=0.001),
        _summary("behavior_speed_peak_centered_2p56s", 2.56, factor_latent_mean=999.0),
    ]

    result = recommend_movement_window(
        summaries, SELECTION, {"small_recommended_window": "behavior_speed_peak_centered_1p28s"}
    )

    assert result["recommended_window_name"] == "behavior_speed_peak_centered_1p28s"
