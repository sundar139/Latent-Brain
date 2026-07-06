from __future__ import annotations

import numpy as np


def _validate_behavior(behavior: np.ndarray, behavior_names: list[str]) -> np.ndarray:
    array = np.asarray(behavior, dtype=np.float64)
    if array.ndim != 3:
        msg = f"behavior must have rank 3; got shape {array.shape}"
        raise ValueError(msg)
    if array.shape[2] != len(behavior_names):
        msg = "behavior_names length must match behavior dimension"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = "behavior must be finite"
        raise ValueError(msg)
    return array


def select_behavior_targets(
    behavior: np.ndarray,
    behavior_names: list[str],
    prefixes: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Select behavior dimensions whose names start with any requested prefix."""
    if not prefixes:
        msg = "at least one behavior prefix is required"
        raise ValueError(msg)
    behavior_array = _validate_behavior(behavior, behavior_names)
    selected = [
        index
        for index, name in enumerate(behavior_names)
        if any(name.startswith(prefix) for prefix in prefixes)
    ]
    if not selected:
        msg = f"No behavior targets found for prefixes: {prefixes}"
        raise ValueError(msg)
    return behavior_array[:, :, selected], [behavior_names[index] for index in selected]


def _velocity_name(position_name: str) -> str:
    parts = position_name.split("_")
    axis = parts[-1] if len(parts) > 1 else "0"
    prefix = "_".join(parts[:-1]) if len(parts) > 1 else position_name
    return f"{prefix}_velocity_{axis}"


def derive_velocity_targets(
    positions: np.ndarray,
    position_names: list[str],
    bin_size_ms: int,
    method: str = "central_difference",
) -> tuple[np.ndarray, list[str]]:
    """Derive velocity targets in behavior units per second."""
    if bin_size_ms <= 0:
        msg = "bin_size_ms must be positive"
        raise ValueError(msg)
    if method != "central_difference":
        msg = "only central_difference velocity is supported"
        raise ValueError(msg)
    position_array = _validate_behavior(positions, position_names)
    velocity = np.gradient(position_array, bin_size_ms / 1000.0, axis=1)
    return np.asarray(velocity, dtype=np.float64), [_velocity_name(name) for name in position_names]
