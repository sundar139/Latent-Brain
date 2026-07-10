from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from latentbrain.eval.movement_features import (
    COVERAGE_KEYS,
    CURSOR_SOURCE,
    ENDPOINT_COLUMNS,
    HAND_SOURCE,
    compute_endpoint_features,
    compute_hand_speed,
    compute_window_behavior_coverage,
    endpoint_direction_entropy,
    find_movement_onset_index,
    find_peak_speed_index,
    global_peak_speed,
    resolve_behavior_source,
)

HAND_NAMES = ["hand_pos_x", "hand_pos_y", "cursor_pos_x", "cursor_pos_y"]
CURSOR_ONLY_NAMES = ["cursor_pos_x", "cursor_pos_y"]
BIN = 0.02


def _ramp_behavior(trials: int = 3, time: int = 10) -> np.ndarray:
    """Position ramps outward; speed is constant except for a spike at bin 6."""
    behavior = np.zeros((trials, time, 4))
    for trial in range(trials):
        x = np.arange(time, dtype=float) * 0.1
        x[7:] += 1.0  # a large jump between bins 6 and 7 -> peak speed at index 7
        behavior[trial, :, 0] = x
        behavior[trial, :, 1] = 0.0
        behavior[trial, :, 2] = x
        behavior[trial, :, 3] = 0.0
    return behavior


def test_resolve_behavior_source_prefers_hand_then_cursor() -> None:
    assert resolve_behavior_source(HAND_NAMES)[0] == HAND_SOURCE
    assert resolve_behavior_source(CURSOR_ONLY_NAMES)[0] == CURSOR_SOURCE
    assert resolve_behavior_source(["a", "b"]) is None
    assert resolve_behavior_source(None) is None


def test_hand_speed_is_finite_and_bin_aligned() -> None:
    speed = compute_hand_speed(_ramp_behavior(), HAND_NAMES, BIN)

    assert speed.shape == (3, 10)
    assert np.isfinite(speed).all()
    assert (speed >= 0.0).all()
    # Constant 0.1 displacement per 0.02 s bin -> 5.0 units/s away from the jump.
    assert speed[0, 0] == pytest.approx(5.0)
    assert speed[0, 2] == pytest.approx(5.0)


def test_hand_speed_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="bin_size_seconds must be positive"):
        compute_hand_speed(_ramp_behavior(), HAND_NAMES, 0.0)
    with pytest.raises(ValueError, match="shape"):
        compute_hand_speed(np.zeros((3, 4)), HAND_NAMES, BIN)
    with pytest.raises(ValueError, match="hand_pos or cursor_pos"):
        compute_hand_speed(_ramp_behavior(), ["a", "b", "c", "d"], BIN)


def test_peak_speed_index_is_computed_correctly() -> None:
    speed = compute_hand_speed(_ramp_behavior(), HAND_NAMES, BIN)

    peaks = find_peak_speed_index(speed)

    assert peaks.tolist() == [7, 7, 7]
    assert speed[0, 7] > speed[0, 6]


def test_movement_onset_index_respects_threshold() -> None:
    # Speed rises monotonically, so a higher quantile must not select an earlier bin.
    speed = np.tile(np.arange(10, dtype=float), (2, 1))

    low = find_movement_onset_index(speed, 0.1)
    mid = find_movement_onset_index(speed, 0.5)
    high = find_movement_onset_index(speed, 0.9)

    assert (mid >= low).all()
    assert (high >= mid).all()
    assert speed[0, low[0]] >= np.quantile(speed[0], 0.1)
    assert speed[0, high[0]] >= np.quantile(speed[0], 0.9)


def test_movement_onset_falls_back_when_the_quantile_is_degenerate() -> None:
    # Static until bin 6, then a jump. The 70th-percentile speed is 0, which would otherwise
    # place onset at bin 0 instead of where movement actually begins.
    behavior = np.zeros((2, 12, 4))
    behavior[:, 7:, 0] = 5.0
    speed = compute_hand_speed(behavior, HAND_NAMES, BIN)

    onset = find_movement_onset_index(speed, 0.7)

    assert np.quantile(speed[0], 0.7) == pytest.approx(0.0)
    assert onset.tolist() == [7, 7]


def test_movement_onset_rejects_bad_quantile() -> None:
    with pytest.raises(ValueError, match="threshold_quantile"):
        find_movement_onset_index(np.zeros((2, 4)), 1.5)


def test_endpoint_features_use_hand_position_when_available() -> None:
    behavior = _ramp_behavior()
    behavior[:, :, 2] = 0.0  # zero out the cursor channel entirely
    behavior[:, :, 3] = 0.0

    features = compute_endpoint_features(behavior, HAND_NAMES, BIN)

    assert list(features.columns) == ENDPOINT_COLUMNS
    assert (features["behavior_source"] == HAND_SOURCE).all()
    # Hand moved along +x, so a zeroed cursor cannot have produced this displacement.
    assert (features["endpoint_dx"] > 0.0).all()
    assert features["endpoint_angle_rad"].iloc[0] == pytest.approx(0.0)
    assert (features["peak_speed"] > features["mean_speed"]).all()
    assert features["peak_speed_time_seconds"].iloc[0] == pytest.approx(7 * BIN)


def test_endpoint_features_fall_back_to_cursor_position() -> None:
    behavior = _ramp_behavior()[:, :, 2:]

    features = compute_endpoint_features(behavior, CURSOR_ONLY_NAMES, BIN)

    assert (features["behavior_source"] == CURSOR_SOURCE).all()
    assert np.isfinite(features["endpoint_distance"]).all()


def test_endpoint_features_require_known_behavior_names() -> None:
    with pytest.raises(ValueError, match="hand_pos or cursor_pos"):
        compute_endpoint_features(_ramp_behavior(), ["a", "b", "c", "d"], BIN)


def test_behavior_coverage_summary_has_required_keys() -> None:
    coverage = compute_window_behavior_coverage(_ramp_behavior(), HAND_NAMES, BIN)

    assert set(COVERAGE_KEYS).issubset(coverage)
    assert coverage["behavior_source"] == HAND_SOURCE
    assert 0.0 <= coverage["moving_bin_fraction"] <= 1.0
    assert coverage["trials_with_movement_fraction"] == pytest.approx(1.0)
    assert np.isfinite(coverage["mean_peak_speed"])


def test_behavior_coverage_reports_no_movement_for_static_trials() -> None:
    static = np.zeros((3, 10, 4))

    coverage = compute_window_behavior_coverage(static, HAND_NAMES, BIN)

    assert coverage["mean_peak_speed"] == pytest.approx(0.0)
    assert coverage["trials_with_movement_fraction"] == pytest.approx(0.0)


def test_endpoint_direction_entropy_matches_uniform_and_concentrated_cases() -> None:
    uniform = np.linspace(-np.pi, np.pi, 64, endpoint=False)
    concentrated = np.full(16, 0.1)

    assert endpoint_direction_entropy(uniform) == pytest.approx(np.log(8.0))
    assert endpoint_direction_entropy(concentrated) == pytest.approx(0.0)
    assert np.isnan(endpoint_direction_entropy(np.array([np.nan])))


def test_endpoint_features_return_dataframe_of_expected_length() -> None:
    features = compute_endpoint_features(_ramp_behavior(trials=5), HAND_NAMES, BIN)

    assert isinstance(features, pd.DataFrame)
    assert len(features) == 5


def _static_window() -> np.ndarray:
    """Tiny pre-movement jitter only."""
    behavior = np.zeros((3, 10, 4))
    behavior[:, :, 0] = np.arange(10) * 0.001
    return behavior


def _reach_window() -> np.ndarray:
    behavior = np.zeros((3, 10, 4))
    behavior[:, :, 0] = np.arange(10) * 1.0
    return behavior


def test_global_peak_speed_uses_the_whole_recording() -> None:
    peak = global_peak_speed(_reach_window(), HAND_NAMES, BIN)

    assert peak == pytest.approx(50.0)


def test_per_window_coverage_cannot_distinguish_a_static_window() -> None:
    # Without a reference scale, both windows look equally "moving" because each is thresholded
    # against its own peak. This is why coverage needs a window-independent scale.
    static = compute_window_behavior_coverage(_static_window(), HAND_NAMES, BIN)
    reach = compute_window_behavior_coverage(_reach_window(), HAND_NAMES, BIN)

    assert static["moving_bin_fraction"] == pytest.approx(reach["moving_bin_fraction"])


def test_reference_peak_speed_makes_coverage_comparable_across_windows() -> None:
    reference = global_peak_speed(_reach_window(), HAND_NAMES, BIN)

    static = compute_window_behavior_coverage(_static_window(), HAND_NAMES, BIN, reference)
    reach = compute_window_behavior_coverage(_reach_window(), HAND_NAMES, BIN, reference)

    assert static["moving_bin_fraction"] == pytest.approx(0.0)
    assert reach["moving_bin_fraction"] == pytest.approx(1.0)
    assert static["reference_peak_speed"] == pytest.approx(reference)
