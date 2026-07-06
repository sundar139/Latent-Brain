from __future__ import annotations

import numpy as np
import pytest

from latentbrain.eval.behavior import derive_velocity_targets, select_behavior_targets


def test_behavior_target_selection_by_prefix_preserves_order() -> None:
    behavior = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    names = ["hand_pos_x", "hand_pos_y", "cursor_pos_x", "other"]

    selected, selected_names = select_behavior_targets(behavior, names, ["cursor_pos", "hand_pos"])

    assert selected.shape == (2, 3, 3)
    assert selected_names == ["hand_pos_x", "hand_pos_y", "cursor_pos_x"]


def test_missing_behavior_prefix_raises_clear_error() -> None:
    behavior = np.zeros((1, 2, 1), dtype=np.float64)

    with pytest.raises(ValueError, match="No behavior targets"):
        select_behavior_targets(behavior, ["hand_pos_x"], ["cursor_pos"])


def test_velocity_derivation_preserves_shape() -> None:
    positions = np.zeros((2, 5, 2), dtype=np.float64)

    velocity, names = derive_velocity_targets(positions, ["hand_pos_x", "hand_pos_y"], 5)

    assert velocity.shape == positions.shape
    assert names == ["hand_pos_velocity_x", "hand_pos_velocity_y"]


def test_constant_position_gives_zero_velocity() -> None:
    positions = np.ones((1, 5, 1), dtype=np.float64) * 3.0

    velocity, _ = derive_velocity_targets(positions, ["hand_pos_x"], 5)

    np.testing.assert_allclose(velocity, 0.0)


def test_linear_position_gives_constant_velocity() -> None:
    dt = 0.005
    positions = (np.arange(5, dtype=np.float64) * 2.0 * dt).reshape(1, 5, 1)

    velocity, _ = derive_velocity_targets(positions, ["hand_pos_x"], 5)

    np.testing.assert_allclose(velocity, 2.0)
