from __future__ import annotations

import numpy as np
import pytest

from latentbrain.eval.split_audit import (
    BEHAVIOR_SPLIT_COLUMNS,
    COMPARISON_COLUMNS,
    NEURON_SPLIT_COLUMNS,
    SPLIT_COLUMNS,
    TRIAL_COLUMNS,
    compare_split_statistics,
    compute_behavior_split_statistics,
    compute_neuron_split_statistics,
    compute_split_statistics,
    compute_trial_statistics,
)

BEHAVIOR_NAMES = ["hand_pos_x", "hand_pos_y", "cursor_pos_x", "cursor_pos_y"]


def _spikes(trials: int = 6, time: int = 8, neurons: int = 4) -> np.ndarray:
    generator = np.random.default_rng(0)
    return generator.poisson(0.3, size=(trials, time, neurons)).astype(np.float64)


def _behavior(trials: int = 6, time: int = 8) -> np.ndarray:
    steps = np.linspace(0.0, 1.0, time)
    behavior = np.zeros((trials, time, 4))
    for trial in range(trials):
        behavior[trial, :, 0] = steps * (trial + 1)
        behavior[trial, :, 1] = steps * (trial + 1) * 0.5
        behavior[trial, :, 2] = steps
        behavior[trial, :, 3] = steps
    return behavior


def _labels(trials: int = 6) -> np.ndarray:
    return np.array(["train", "train", "validation", "validation", "test", "test"][:trials])


def test_trial_statistics_include_required_columns() -> None:
    stats = compute_trial_statistics(
        _spikes(),
        _behavior(),
        BEHAVIOR_NAMES,
        _labels(),
        20,
        np.array([0, 1]),
        np.array([2, 3]),
    )

    assert list(stats.columns) == TRIAL_COLUMNS
    assert len(stats) == 6
    assert bool(stats["behavior_available"].all())
    assert np.isfinite(stats["heldout_rate_hz"]).all()
    assert np.isfinite(stats["heldin_spikes"]).all()


def test_endpoint_direction_is_finite_when_behavior_exists() -> None:
    stats = compute_trial_statistics(
        _spikes(), _behavior(), BEHAVIOR_NAMES, _labels(), 20, np.array([0]), np.array([1])
    )

    for column in ("endpoint_dx", "endpoint_dy", "endpoint_angle_rad", "endpoint_distance"):
        assert np.isfinite(stats[column]).all()
    assert np.isfinite(stats["mean_speed"]).all()
    assert (stats["endpoint_distance"] > 0.0).all()
    assert (stats["endpoint_angle_rad"].abs() <= np.pi).all()


def test_missing_behavior_returns_nans_and_does_not_crash() -> None:
    stats = compute_trial_statistics(_spikes(), None, None, _labels(), 20)

    assert not bool(stats["behavior_available"].any())
    for column in ("endpoint_dx", "endpoint_dy", "endpoint_angle_rad", "endpoint_distance"):
        assert stats[column].isna().all()
    assert stats["heldin_spikes"].isna().all()
    assert np.isfinite(stats["population_rate_hz"]).all()


def test_missing_position_names_disable_behavior_columns() -> None:
    stats = compute_trial_statistics(_spikes(), _behavior(), ["a", "b", "c", "d"], _labels(), 20)

    assert not bool(stats["behavior_available"].any())
    assert stats["endpoint_distance"].isna().all()


def test_split_statistics_include_required_columns() -> None:
    trials = compute_trial_statistics(
        _spikes(), _behavior(), BEHAVIOR_NAMES, _labels(), 20, np.array([0, 1]), np.array([2, 3])
    )

    stats = compute_split_statistics(trials)

    assert list(stats.columns) == SPLIT_COLUMNS
    assert set(stats["split"]) == {"train", "validation", "test"}
    assert (stats["n_trials"] == 2).all()


def test_split_statistics_survive_missing_behavior() -> None:
    trials = compute_trial_statistics(_spikes(), None, None, _labels(), 20)

    stats = compute_split_statistics(trials)

    assert list(stats.columns) == SPLIT_COLUMNS
    assert stats["mean_endpoint_distance"].isna().all()
    assert np.isfinite(stats["mean_population_rate_hz"]).all()


def test_neuron_split_statistics_label_groups() -> None:
    stats = compute_neuron_split_statistics(
        _spikes(), _labels(), np.array([0, 1]), np.array([2, 3]), 20
    )

    assert list(stats.columns) == NEURON_SPLIT_COLUMNS
    assert set(stats["neuron_group"]) == {"heldin", "heldout"}
    assert len(stats) == 3 * 4


def test_behavior_split_statistics_include_required_columns() -> None:
    stats = compute_behavior_split_statistics(_behavior(), BEHAVIOR_NAMES, _labels())

    assert list(stats.columns) == BEHAVIOR_SPLIT_COLUMNS
    assert set(stats["behavior_name"]) == set(BEHAVIOR_NAMES)
    assert np.isfinite(stats["mean_absolute_change"]).all()


def test_split_comparison_computes_standardized_differences() -> None:
    trials = compute_trial_statistics(
        _spikes(), _behavior(), BEHAVIOR_NAMES, _labels(), 20, np.array([0, 1]), np.array([2, 3])
    )

    comparison = compare_split_statistics(trials, "validation", "test")

    assert list(comparison.columns) == COMPARISON_COLUMNS
    row = comparison[comparison["metric"] == "endpoint_distance"].iloc[0]
    assert row["difference"] == pytest.approx(row["split_a_mean"] - row["split_b_mean"])
    assert np.isfinite(row["standardized_difference"])


def test_split_comparison_returns_nan_for_missing_behavior_metrics() -> None:
    trials = compute_trial_statistics(_spikes(), None, None, _labels(), 20)

    comparison = compare_split_statistics(trials, "validation", "test")

    row = comparison[comparison["metric"] == "endpoint_distance"].iloc[0]
    assert np.isnan(row["standardized_difference"])
    spikes_row = comparison[comparison["metric"] == "total_spikes"].iloc[0]
    assert np.isfinite(spikes_row["difference"])


def test_trial_statistics_reject_mismatched_labels() -> None:
    with pytest.raises(ValueError, match="one entry per trial"):
        compute_trial_statistics(_spikes(), None, None, np.array(["train"]), 20)
