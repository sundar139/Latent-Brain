from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

HAND_SOURCE = "hand_pos"
CURSOR_SOURCE = "cursor_pos"
BEHAVIOR_SOURCES = (HAND_SOURCE, CURSOR_SOURCE)

ENDPOINT_COLUMNS = [
    "endpoint_dx",
    "endpoint_dy",
    "endpoint_angle_rad",
    "endpoint_distance",
    "mean_speed",
    "peak_speed",
    "peak_speed_time_seconds",
    "movement_onset_time_seconds",
    "behavior_source",
]

COVERAGE_KEYS = (
    "behavior_source",
    "mean_speed",
    "mean_peak_speed",
    "mean_endpoint_distance",
    "endpoint_direction_entropy",
    "moving_bin_fraction",
    "trials_with_movement_fraction",
    "reference_peak_speed",
)

DIRECTION_BINS = 8

# A bin counts as "moving" when its speed clears this share of the reference peak speed.
MOVING_SPEED_FRACTION = 0.2


def resolve_behavior_source(behavior_names: list[str] | None) -> tuple[str, int, int] | None:
    """Prefer hand position; fall back to cursor position and report which was used."""
    if not behavior_names:
        return None
    for source in BEHAVIOR_SOURCES:
        try:
            return (
                source,
                behavior_names.index(f"{source}_x"),
                behavior_names.index(f"{source}_y"),
            )
        except ValueError:
            continue
    return None


def compute_hand_speed(
    behavior: np.ndarray,
    behavior_names: list[str],
    bin_size_seconds: float,
) -> np.ndarray:
    """Per-bin speed of the hand (or cursor) position, shape [trials, time]."""
    if bin_size_seconds <= 0.0:
        msg = "bin_size_seconds must be positive"
        raise ValueError(msg)
    values = np.asarray(behavior, dtype=np.float64)
    if values.ndim != 3:
        msg = "behavior must have shape [trials, time, variables]"
        raise ValueError(msg)
    resolved = resolve_behavior_source(behavior_names)
    if resolved is None:
        msg = "behavior must contain hand_pos or cursor_pos coordinates"
        raise ValueError(msg)
    _, x_index, y_index = resolved
    x = values[:, :, x_index]
    y = values[:, :, y_index]
    steps = np.hypot(np.diff(x, axis=1), np.diff(y, axis=1)) / bin_size_seconds
    # Repeat the first step so speed aligns bin-for-bin with the position samples.
    return np.asarray(np.concatenate([steps[:, :1], steps], axis=1))


def find_peak_speed_index(speed: np.ndarray) -> np.ndarray:
    values = np.asarray(speed, dtype=np.float64)
    if values.ndim != 2:
        msg = "speed must have shape [trials, time]"
        raise ValueError(msg)
    return np.asarray(np.argmax(values, axis=1))


def find_movement_onset_index(speed: np.ndarray, threshold_quantile: float) -> np.ndarray:
    """First bin whose speed reaches the trial's own speed quantile.

    On a trial that is static and then moves, the quantile can equal the minimum speed, which
    would place onset at the first bin. Fall back to a fraction of the trial's peak speed so
    onset always lands where movement actually begins.
    """
    if not 0.0 <= threshold_quantile <= 1.0:
        msg = "threshold_quantile must be in [0, 1]"
        raise ValueError(msg)
    values = np.asarray(speed, dtype=np.float64)
    if values.ndim != 2:
        msg = "speed must have shape [trials, time]"
        raise ValueError(msg)
    thresholds = np.quantile(values, threshold_quantile, axis=1, keepdims=True)
    minima = values.min(axis=1, keepdims=True)
    peaks = values.max(axis=1, keepdims=True)
    degenerate = thresholds <= minima
    thresholds = np.where(degenerate, minima + MOVING_SPEED_FRACTION * (peaks - minima), thresholds)
    reached = values >= thresholds
    # Every trial reaches its own threshold at the peak, so argmax is always defined.
    return np.asarray(np.argmax(reached, axis=1))


def compute_endpoint_features(
    behavior_window: np.ndarray,
    behavior_names: list[str],
    bin_size_seconds: float = 0.02,
    threshold_quantile: float = 0.7,
) -> pd.DataFrame:
    values = np.asarray(behavior_window, dtype=np.float64)
    resolved = resolve_behavior_source(behavior_names)
    if resolved is None:
        msg = "behavior must contain hand_pos or cursor_pos coordinates"
        raise ValueError(msg)
    source, x_index, y_index = resolved
    speed = compute_hand_speed(values, behavior_names, bin_size_seconds)
    peak_index = find_peak_speed_index(speed)
    onset_index = find_movement_onset_index(speed, threshold_quantile)
    x = values[:, :, x_index]
    y = values[:, :, y_index]
    dx = x[:, -1] - x[:, 0]
    dy = y[:, -1] - y[:, 0]
    return pd.DataFrame(
        {
            "endpoint_dx": dx,
            "endpoint_dy": dy,
            "endpoint_angle_rad": np.arctan2(dy, dx),
            "endpoint_distance": np.hypot(dx, dy),
            "mean_speed": speed.mean(axis=1),
            "peak_speed": speed.max(axis=1),
            "peak_speed_time_seconds": peak_index * bin_size_seconds,
            "movement_onset_time_seconds": onset_index * bin_size_seconds,
            "behavior_source": source,
        },
        columns=ENDPOINT_COLUMNS,
    )


def endpoint_direction_entropy(angles: np.ndarray, bins: int = DIRECTION_BINS) -> float:
    """Shannon entropy (nats) of reach directions binned into equal-width sectors."""
    finite = np.asarray(angles, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    shifted = (finite + np.pi) % (2.0 * np.pi)
    indices = np.clip((shifted / (2.0 * np.pi / bins)).astype(int), 0, bins - 1)
    counts = np.bincount(indices, minlength=bins).astype(np.float64)
    probabilities = counts[counts > 0.0] / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def global_peak_speed(
    behavior: np.ndarray, behavior_names: list[str], bin_size_seconds: float
) -> float:
    """Peak hand speed over the whole recording, used as a window-independent movement scale."""
    speed = compute_hand_speed(behavior, behavior_names, bin_size_seconds)
    return float(speed.max()) if speed.size else float("nan")


def compute_window_behavior_coverage(
    behavior_window: np.ndarray,
    behavior_names: list[str],
    bin_size_seconds: float,
    reference_peak_speed: float | None = None,
) -> dict[str, Any]:
    """How much actual movement a window contains, not just how long it is.

    Thresholding each window against its own peak speed is scale-free and therefore cannot
    compare windows: a window holding only pre-movement jitter looks just as "moving" as one
    holding a reach. Pass `reference_peak_speed` (the peak over the whole recording) to get a
    coverage number that is comparable across windows.
    """
    features = compute_endpoint_features(behavior_window, behavior_names, bin_size_seconds)
    speed = compute_hand_speed(behavior_window, behavior_names, bin_size_seconds)
    peaks = speed.max(axis=1, keepdims=True)
    scale = (
        np.full_like(peaks, float(reference_peak_speed))
        if reference_peak_speed is not None and np.isfinite(reference_peak_speed)
        else peaks
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        moving = speed >= (MOVING_SPEED_FRACTION * scale)
    trials_with_movement = np.asarray(peaks.reshape(-1) > 0.0)
    return {
        "behavior_source": str(features.iloc[0]["behavior_source"]),
        "mean_speed": float(features["mean_speed"].mean()),
        "mean_peak_speed": float(features["peak_speed"].mean()),
        "mean_endpoint_distance": float(features["endpoint_distance"].mean()),
        "endpoint_direction_entropy": endpoint_direction_entropy(
            features["endpoint_angle_rad"].to_numpy()
        ),
        "moving_bin_fraction": float(np.mean(moving)),
        "trials_with_movement_fraction": float(np.mean(trials_with_movement)),
        "reference_peak_speed": float(reference_peak_speed)
        if reference_peak_speed is not None
        else float("nan"),
    }
