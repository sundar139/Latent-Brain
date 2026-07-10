from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from latentbrain.data.schemas import NeuralDataset
from latentbrain.eval.fold_balance import (
    compare_fold_balance,
    compute_fold_balance_statistics,
    summarize_fold_balance,
)
from latentbrain.eval.movement_features import (
    compute_endpoint_features,
    compute_hand_speed,
    compute_window_behavior_coverage,
    endpoint_direction_entropy,
    find_movement_onset_index,
    find_peak_speed_index,
    global_peak_speed,
    resolve_behavior_source,
)
from latentbrain.eval.stratified_cv import (
    FACTOR_LATENT,
    SPLIT_MEAN_RATE_INVALID,
    build_repeated_stratified_folds,
    build_trial_features,
    score_folds,
    select_best_valid_method,
    summarize_methods,
)

FROM_START = "from_start"
PEAK_SPEED_CENTERED = "behavior_speed_peak_centered"
MOVEMENT_ONSET = "behavior_movement_onset"
CROP_POLICIES = (FROM_START, PEAK_SPEED_CENTERED, MOVEMENT_ONSET)
BEHAVIOR_ALIGNED_POLICIES = (PEAK_SPEED_CENTERED, MOVEMENT_ONSET)

WINDOW_SLICE_COLUMNS = [
    "trial_index",
    "start_bin",
    "end_bin",
    "clipped",
]

BEHAVIOR_STATISTICS_COLUMNS = [
    "window_name",
    "trial_index",
    "start_bin",
    "end_bin",
    "duration_seconds",
    "crop_policy",
    "endpoint_angle_rad",
    "endpoint_distance",
    "mean_speed",
    "peak_speed",
    "peak_speed_time_seconds",
    "movement_onset_time_seconds",
    "behavior_source",
    "clipped",
]

SCORE_COLUMNS = [
    "window_name",
    "report_label",
    "fold_repeat",
    "fold_index",
    "method_name",
    "method_type",
    "valid_model",
    "reportable_as_model_performance",
    "invalid_reason",
    "unified_bits_per_spike",
    "poisson_nll",
    "eval_spike_count",
    "eval_heldout_rate_hz",
    "train_trial_count",
    "eval_trial_count",
    "notes",
]

BALANCE_COLUMNS = [
    "window_name",
    "repeat_index",
    "fold_index",
    "n_trials",
    "mean_population_rate_hz",
    "mean_heldout_rate_hz",
    "mean_endpoint_distance",
    "mean_speed",
    "endpoint_direction_entropy",
    "fold_balance_warning",
]

# A window with less movement than this is treated as pre-movement rather than a reach window.
MIN_MOVING_BIN_FRACTION = 0.10


def _window_bins(duration_seconds: float, bin_size_seconds: float) -> int:
    if duration_seconds <= 0.0:
        msg = "window duration_seconds must be positive"
        raise ValueError(msg)
    return int(round(duration_seconds / bin_size_seconds))


def build_window_slices(
    spikes: np.ndarray,
    behavior: np.ndarray | None,
    behavior_names: list[str] | None,
    candidate: dict[str, Any],
    bin_size_seconds: float,
) -> pd.DataFrame:
    """Per-trial [start_bin, end_bin) slice for one window candidate."""
    counts = np.asarray(spikes)
    if counts.ndim != 3:
        msg = "spikes must have shape [trials, time, neurons]"
        raise ValueError(msg)
    trials, time_bins = counts.shape[0], counts.shape[1]
    policy = str(candidate["crop_policy"])
    if policy not in CROP_POLICIES:
        msg = f"crop_policy must be one of {CROP_POLICIES}"
        raise ValueError(msg)
    width = _window_bins(float(candidate["duration_seconds"]), bin_size_seconds)
    if width > time_bins:
        msg = (
            f"window {candidate['name']!r} requests {width} bins but trials have {time_bins}; "
            "shorten duration_seconds"
        )
        raise ValueError(msg)

    if policy == FROM_START:
        start_seconds = float(candidate.get("start_seconds", 0.0))
        offset = _window_bins(start_seconds, bin_size_seconds) if start_seconds > 0.0 else 0
        starts = np.full(trials, offset, dtype=np.int64)
    else:
        if behavior is None or not behavior_names:
            msg = f"window {candidate['name']!r} requires behavior data for policy {policy!r}"
            raise ValueError(msg)
        speed = compute_hand_speed(behavior, behavior_names, bin_size_seconds)
        if policy == PEAK_SPEED_CENTERED:
            centers = find_peak_speed_index(speed)
            starts = centers - width // 2
        else:
            onsets = find_movement_onset_index(
                speed, float(candidate.get("speed_threshold_quantile", 0.7))
            )
            pre_bins = _window_bins(
                float(candidate.get("pre_event_seconds", 0.0)), bin_size_seconds
            )
            starts = onsets - pre_bins
        starts = np.asarray(starts, dtype=np.int64)

    # Clipping keeps every window the same length; trials whose ideal window ran off either edge
    # are flagged so the report can say how many were shifted.
    clipped = (starts < 0) | (starts + width > time_bins)
    starts = np.clip(starts, 0, time_bins - width)
    return pd.DataFrame(
        {
            "trial_index": np.arange(trials, dtype=np.int64),
            "start_bin": starts,
            "end_bin": starts + width,
            "clipped": clipped,
        },
        columns=WINDOW_SLICE_COLUMNS,
    )


def apply_window_candidate(
    spikes: np.ndarray,
    behavior: np.ndarray | None,
    window_slices: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray | None]:
    counts = np.asarray(spikes)
    starts = window_slices["start_bin"].to_numpy(dtype=np.int64)
    ends = window_slices["end_bin"].to_numpy(dtype=np.int64)
    width = int(ends[0] - starts[0])
    offsets = np.arange(width, dtype=np.int64)
    indices = starts[:, None] + offsets[None, :]
    trials = np.arange(counts.shape[0], dtype=np.int64)[:, None]
    cropped_spikes = counts[trials, indices]
    cropped_behavior = None if behavior is None else np.asarray(behavior)[trials, indices]
    return np.asarray(cropped_spikes), cropped_behavior


def _windowed_dataset(
    dataset: NeuralDataset, spikes: np.ndarray, behavior: np.ndarray | None
) -> NeuralDataset:
    return NeuralDataset(
        spikes=spikes,
        rates=None,
        latents=None,
        trial_ids=dataset.trial_ids,
        time_ms=dataset.time_ms[: spikes.shape[1]],
        bin_size_ms=dataset.bin_size_ms,
        metadata=dict(dataset.metadata),
        behavior=behavior,
        behavior_names=dataset.behavior_names,
    )


def _fold_config(config: dict[str, Any]) -> dict[str, Any]:
    fold_config = copy.deepcopy(config)
    fold_config["cross_validation"] = dict(config["cross_validation"])
    stratification = dict(config["stratification"])
    fold_config["cross_validation"]["fallback_when_behavior_missing"] = str(
        stratification.pop("fallback_when_behavior_missing", "rate_only")
    )
    fold_config["cross_validation"]["stratification"] = stratification
    return fold_config


def evaluate_window_candidate(
    config: dict[str, Any],
    candidate: dict[str, Any],
    dataset: NeuralDataset,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Score one candidate window under behavior-stratified cross-validation."""
    from latentbrain.data.splits import create_neuron_mask  # noqa: PLC0415

    bin_size_seconds = float(config["binning"]["target_bin_size_ms"]) / 1000.0
    behavior_names = list(dataset.behavior_names) if dataset.behavior_names is not None else None
    slices = build_window_slices(
        dataset.spikes, dataset.behavior, behavior_names, candidate, bin_size_seconds
    )
    spikes, behavior = apply_window_candidate(dataset.spikes, dataset.behavior, slices)
    windowed = _windowed_dataset(dataset, spikes, behavior)

    window_name = str(candidate["name"])
    label = str(candidate.get("report_label", ""))

    behavior_statistics = pd.DataFrame(columns=BEHAVIOR_STATISTICS_COLUMNS)
    coverage: dict[str, Any] = {
        "behavior_source": "unavailable",
        "endpoint_direction_entropy": float("nan"),
        "moving_bin_fraction": float("nan"),
    }
    if behavior is not None and behavior_names and dataset.behavior is not None:
        features = compute_endpoint_features(behavior, behavior_names, bin_size_seconds)
        # Coverage is thresholded against the peak speed of the whole recording so that windows
        # can be compared; a per-window peak would make a pre-movement window look "moving".
        reference_peak = global_peak_speed(dataset.behavior, behavior_names, bin_size_seconds)
        coverage = compute_window_behavior_coverage(
            behavior, behavior_names, bin_size_seconds, reference_peak
        )
        behavior_statistics = pd.DataFrame(
            {
                "window_name": window_name,
                "trial_index": slices["trial_index"],
                "start_bin": slices["start_bin"],
                "end_bin": slices["end_bin"],
                "duration_seconds": float(candidate["duration_seconds"]),
                "crop_policy": str(candidate["crop_policy"]),
                "endpoint_angle_rad": features["endpoint_angle_rad"],
                "endpoint_distance": features["endpoint_distance"],
                "mean_speed": features["mean_speed"],
                "peak_speed": features["peak_speed"],
                "peak_speed_time_seconds": features["peak_speed_time_seconds"],
                "movement_onset_time_seconds": features["movement_onset_time_seconds"],
                "behavior_source": features["behavior_source"],
                "clipped": slices["clipped"],
            },
            columns=BEHAVIOR_STATISTICS_COLUMNS,
        )

    fold_config = _fold_config(config)
    reference_mask = create_neuron_mask(
        windowed.spikes.shape[2],
        float(fold_config["cross_validation"]["heldout_neuron_fraction"]),
        seed=int(fold_config["cross_validation"]["base_seed"]),
    )
    trial_features = build_trial_features(
        windowed.spikes,
        windowed.behavior,
        behavior_names,
        int(config["binning"]["target_bin_size_ms"]),
        np.flatnonzero(reference_mask.heldout),
    )
    fold_assignments = build_repeated_stratified_folds(trial_features, fold_config)
    balance = compute_fold_balance_statistics(fold_assignments)
    balance_summary = summarize_fold_balance(balance, compare_fold_balance(balance))
    balance = balance.copy()
    balance.insert(0, "window_name", window_name)
    balance["fold_balance_warning"] = balance_summary["fold_balance_warning"]

    scores = score_folds(windowed, fold_assignments, fold_config)
    scores = scores.rename(columns={"repeat_index": "fold_repeat"})
    scores.insert(0, "report_label", label)
    scores.insert(0, "window_name", window_name)

    diagnostics = {
        "window_name": window_name,
        "report_label": label,
        "crop_policy": str(candidate["crop_policy"]),
        "duration_seconds": float(candidate["duration_seconds"]),
        "clipped_trial_count": int(slices["clipped"].sum()),
        "fold_balance_warning": balance_summary["fold_balance_warning"],
        **coverage,
    }
    return (
        scores[SCORE_COLUMNS],
        behavior_statistics,
        balance[BALANCE_COLUMNS],
        diagnostics,
    )


def _method_mean(scores: pd.DataFrame, window_name: str, method_name: str) -> float:
    rows = scores[(scores["window_name"] == window_name) & (scores["method_name"] == method_name)]
    return float("nan") if rows.empty else float(rows["unified_bits_per_spike"].mean())


def window_method_summary(scores: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    statistics = dict(config["statistics"])
    frames: list[pd.DataFrame] = []
    for window_name in scores["window_name"].unique():
        subset = scores[scores["window_name"] == window_name].rename(
            columns={"fold_repeat": "repeat_index"}
        )
        summary = summarize_methods(
            subset,
            int(statistics["bootstrap_repeats"]),
            float(statistics["confidence_interval"]),
            int(statistics["bootstrap_seed"]),
        )
        summary.insert(0, "window_name", window_name)
        frames.append(summary)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_window_candidates(
    scores: pd.DataFrame,
    behavior_statistics: pd.DataFrame,
    balance_statistics: pd.DataFrame,
    references: dict[str, Any],
    diagnostics: list[dict[str, Any]],
    method_summary: pd.DataFrame,
    current_window_name: str,
) -> dict[str, Any]:
    """Recommend a window using valid-model performance and behavior coverage only."""
    by_window = {str(row["window_name"]): row for row in diagnostics}
    entropy_by_window = {
        name: float(row.get("endpoint_direction_entropy", float("nan")))
        for name, row in by_window.items()
    }

    def _summary_value(window_name: str, method_name: str, column: str) -> float:
        rows = method_summary[
            (method_summary["window_name"] == window_name)
            & (method_summary["method_name"] == method_name)
        ]
        return float("nan") if rows.empty else float(rows.iloc[0][column])

    current_mean = _summary_value(current_window_name, FACTOR_LATENT, "mean_unified_bits_per_spike")
    current_ci_low = float(
        references.get("current_from_start_factor_latent_ci95_low", float("nan"))
    )
    if not np.isfinite(current_ci_low):
        current_ci_low = _summary_value(current_window_name, FACTOR_LATENT, "ci95_low")
    current_entropy = entropy_by_window.get(current_window_name, float("nan"))

    eligible: list[str] = []
    for name, row in by_window.items():
        coverage_ok = (
            str(row.get("behavior_source", "unavailable")) != "unavailable"
            and float(row.get("moving_bin_fraction", 0.0)) >= MIN_MOVING_BIN_FRACTION
        )
        balance_ok = str(row.get("fold_balance_warning", "none")) == "none"
        factor_mean = _summary_value(name, FACTOR_LATENT, "mean_unified_bits_per_spike")
        # Rule 3: preserve or improve factor-latent, judged against the current CI lower bound
        # so that ordinary fold noise does not disqualify a window.
        preserves = np.isfinite(factor_mean) and factor_mean >= current_ci_low
        if coverage_ok and balance_ok and preserves:
            eligible.append(name)

    challengers = [
        name
        for name in eligible
        if name != current_window_name
        and np.isfinite(entropy_by_window.get(name, float("nan")))
        and np.isfinite(current_entropy)
        and entropy_by_window[name] > current_entropy
        and float(by_window[name].get("moving_bin_fraction", 0.0))
        > float(by_window.get(current_window_name, {}).get("moving_bin_fraction", 0.0))
    ]
    current_coverage = float(
        by_window.get(current_window_name, {}).get("moving_bin_fraction", float("nan"))
    )
    if challengers:
        recommended = max(
            challengers,
            key=lambda name: (
                entropy_by_window[name],
                _summary_value(name, FACTOR_LATENT, "mean_unified_bits_per_spike"),
            ),
        )
        current_supported = False
    else:
        recommended = current_window_name
        current_supported = True

    # The early-window label follows the current window's actual movement coverage, not merely
    # whether a challenger displaced it. A window with no moving bins is a pre-movement window
    # whichever way the recommendation lands.
    current_is_early_window = bool(
        recommended != current_window_name
        or (np.isfinite(current_coverage) and current_coverage < MIN_MOVING_BIN_FRACTION)
    )
    selection_note = (
        "Selection used valid-model performance and behavior coverage only; invalid controls "
        "were ignored."
    )
    if not current_supported:
        rationale = (
            f"{recommended} preserves factor-latent against the current window's CI lower bound "
            "while carrying more reach-direction diversity and more moving bins. "
            f"{current_window_name} must be labelled an early-window diagnostic. {selection_note}"
        )
    elif current_is_early_window:
        rationale = (
            "No candidate improved both behavior coverage and valid-model performance, so the "
            "current window is retained. Its movement coverage is below the floor, so it must "
            f"be labelled an early-window diagnostic. {selection_note}"
        )
    else:
        rationale = (
            "No candidate improved both behavior coverage and valid-model performance, so the "
            f"current window is retained on its own merits. {selection_note}"
        )

    coverage_warnings = [
        f"{name}: moving_bin_fraction={float(row.get('moving_bin_fraction', float('nan'))):.3f}"
        for name, row in by_window.items()
        if float(row.get("moving_bin_fraction", 0.0)) < MIN_MOVING_BIN_FRACTION
    ]
    best_mean = _summary_value(recommended, FACTOR_LATENT, "mean_unified_bits_per_spike")
    invalid_best = _summary_value(
        recommended, SPLIT_MEAN_RATE_INVALID, "mean_unified_bits_per_spike"
    )
    best_valid_method = select_best_valid_method(
        method_summary[method_summary["window_name"] == recommended]
    )
    return {
        "recommended_window_name": recommended,
        "recommended_reporting_mode": "stratified_cross_validation",
        "current_window_name": current_window_name,
        "current_window_still_supported": current_supported,
        "current_window_is_early_window_diagnostic": current_is_early_window,
        "current_window_moving_bin_fraction": current_coverage,
        "factor_latent_best_window_mean": best_mean,
        "factor_latent_current_window_mean": current_mean,
        "factor_latent_best_window_ci95_low": _summary_value(
            recommended, FACTOR_LATENT, "ci95_low"
        ),
        "factor_latent_best_window_ci95_high": _summary_value(
            recommended, FACTOR_LATENT, "ci95_high"
        ),
        "split_mean_invalid_best_window_mean": invalid_best,
        "invalid_control_gap_best_window": float(invalid_best - best_mean)
        if np.isfinite(invalid_best) and np.isfinite(best_mean)
        else float("nan"),
        "endpoint_direction_entropy_by_window": entropy_by_window,
        "endpoint_direction_entropy_current_window": current_entropy,
        "endpoint_direction_entropy_best_window": entropy_by_window.get(recommended, float("nan")),
        "moving_bin_fraction_by_window": {
            name: float(row.get("moving_bin_fraction", float("nan")))
            for name, row in by_window.items()
        },
        "behavior_coverage_warning": "; ".join(coverage_warnings) if coverage_warnings else "none",
        "eligible_windows": sorted(eligible),
        "window_selection_rationale": rationale,
        "best_valid_method": best_valid_method,
        "carried_forward_method": best_valid_method,
        "invalid_controls_excluded_from_window_selection": True,
        "invalid_controls_excluded_from_valid_model_selection": True,
        "single_split_results_reportable": False,
        "old_incompatible_mean_rate_values_used_as_targets": False,
        "official_benchmark_claim": False,
    }


def build_window_recommendations(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "recommended_window_name": summary["recommended_window_name"],
        "recommended_reporting_mode": summary["recommended_reporting_mode"],
        "current_window_still_supported": summary["current_window_still_supported"],
        "current_window_is_early_window_diagnostic": summary[
            "current_window_is_early_window_diagnostic"
        ],
        "window_selection_rationale": summary["window_selection_rationale"],
        "eligible_windows": summary["eligible_windows"],
        "behavior_coverage_warning": summary["behavior_coverage_warning"],
        "invalid_controls_excluded_from_window_selection": True,
        "carried_forward_method": summary["carried_forward_method"],
        "official_benchmark_claim": False,
    }


def speed_profiles(
    dataset: NeuralDataset, candidates: list[dict[str, Any]], bin_size_seconds: float
) -> dict[str, np.ndarray]:
    """Mean speed trace inside each candidate window, for the speed-profile figure."""
    if dataset.behavior is None or dataset.behavior_names is None:
        return {}
    behavior_names = list(dataset.behavior_names)
    if resolve_behavior_source(behavior_names) is None:
        return {}
    profiles: dict[str, np.ndarray] = {}
    for candidate in candidates:
        slices = build_window_slices(
            dataset.spikes, dataset.behavior, behavior_names, candidate, bin_size_seconds
        )
        _, behavior = apply_window_candidate(dataset.spikes, dataset.behavior, slices)
        if behavior is None:
            continue
        speed = compute_hand_speed(behavior, behavior_names, bin_size_seconds)
        profiles[str(candidate["name"])] = np.asarray(speed.mean(axis=0))
    return profiles


def window_entropy_table(diagnostics: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "window_name": row["window_name"],
                "report_label": row.get("report_label", ""),
                "crop_policy": row.get("crop_policy", ""),
                "duration_seconds": row.get("duration_seconds", float("nan")),
                "behavior_source": row.get("behavior_source", "unavailable"),
                "endpoint_direction_entropy": row.get("endpoint_direction_entropy", float("nan")),
                "moving_bin_fraction": row.get("moving_bin_fraction", float("nan")),
                "mean_endpoint_distance": row.get("mean_endpoint_distance", float("nan")),
                "mean_peak_speed": row.get("mean_peak_speed", float("nan")),
                "clipped_trial_count": row.get("clipped_trial_count", 0),
                "fold_balance_warning": row.get("fold_balance_warning", "none"),
            }
            for row in diagnostics
        ]
    )


__all__ = [
    "BEHAVIOR_ALIGNED_POLICIES",
    "CROP_POLICIES",
    "MIN_MOVING_BIN_FRACTION",
    "apply_window_candidate",
    "build_window_recommendations",
    "build_window_slices",
    "endpoint_direction_entropy",
    "evaluate_window_candidate",
    "speed_profiles",
    "summarize_window_candidates",
    "window_entropy_table",
    "window_method_summary",
]
