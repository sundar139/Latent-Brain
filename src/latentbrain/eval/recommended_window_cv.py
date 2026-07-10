from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from latentbrain.data.io import compute_dataset_hash, load_neural_dataset
from latentbrain.data.rebinning import rebin_neural_dataset, validate_rebin_factor
from latentbrain.data.schemas import NeuralDataset
from latentbrain.data.splits import create_neuron_mask
from latentbrain.data.validation import validate_neural_dataset
from latentbrain.eval.fold_balance import (
    compare_fold_balance,
    compute_fold_balance_statistics,
    summarize_fold_balance,
)
from latentbrain.eval.movement_features import (
    MOVING_SPEED_FRACTION,
    compute_endpoint_features,
    compute_hand_speed,
    endpoint_direction_entropy,
    global_peak_speed,
    resolve_behavior_source,
)
from latentbrain.eval.stratified_cv import (
    FACTOR_LATENT,
    SPLIT_MEAN_RATE_INVALID,
    build_repeated_stratified_folds,
    build_trial_features,
    score_folds,
    summarize_methods,
)
from latentbrain.eval.window_audit import apply_window_candidate, build_window_slices
from latentbrain.paths import get_repo_root, resolve_configured_path

RECOMMENDED_WINDOW_NAME = "behavior_speed_peak_centered_1p28s"
RECOMMENDED_REPORTING_MODE = "recommended_window_stratified_cross_validation"

BEHAVIOR_STATISTICS_COLUMNS = [
    "trial_index",
    "start_bin",
    "end_bin",
    "duration_seconds",
    "endpoint_angle_rad",
    "endpoint_distance",
    "mean_speed",
    "peak_speed",
    "moving_bin_fraction",
    "behavior_source",
    "clipped",
]
SCORE_COLUMNS = [
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
    "factor_analysis_random_state",
    "notes",
]
FOLD_BALANCE_COLUMNS = [
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
LEAKAGE_DIAGNOSTIC_COLUMNS = [
    "metric",
    "factor_latent_value",
    "split_mean_invalid_value",
    "difference_factor_minus_invalid",
    "interpretation",
]


def build_recommended_window_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Load, verify, rebin, and crop the configured peak-speed-centered dataset."""
    dataset_config = config["dataset"]
    processed_path = resolve_configured_path(str(dataset_config["processed_path"]), get_repo_root())
    if not processed_path.exists():
        msg = f"Processed dataset is missing: {processed_path}"
        raise FileNotFoundError(msg)
    dataset = load_neural_dataset(processed_path)
    validate_neural_dataset(dataset)
    dataset_hash = compute_dataset_hash(dataset)
    expected_hash = str(dataset_config.get("expected_hash", ""))
    if expected_hash and dataset_hash != expected_hash:
        msg = f"Dataset hash mismatch: expected {expected_hash}, got {dataset_hash}"
        raise ValueError(msg)

    target_bin_size_ms = int(config["binning"]["target_bin_size_ms"])
    validate_rebin_factor(int(dataset_config["original_bin_size_ms"]), target_bin_size_ms)
    rebinned = rebin_neural_dataset(dataset, target_bin_size_ms)
    behavior_names = list(rebinned.behavior_names) if rebinned.behavior_names is not None else None
    if rebinned.behavior is None or resolve_behavior_source(behavior_names) is None:
        msg = "recommended peak-speed-centered window requires hand_pos or cursor_pos behavior data"
        raise ValueError(msg)

    window = dict(config["window"])
    bin_size_seconds = target_bin_size_ms / 1000.0
    slices = build_window_slices(
        rebinned.spikes, rebinned.behavior, behavior_names, window, bin_size_seconds
    )
    spikes, behavior = apply_window_candidate(rebinned.spikes, rebinned.behavior, slices)
    assert behavior is not None and behavior_names is not None
    windowed = NeuralDataset(
        spikes=spikes,
        rates=None,
        latents=None,
        trial_ids=rebinned.trial_ids,
        time_ms=np.arange(spikes.shape[1], dtype=np.float64) * target_bin_size_ms,
        bin_size_ms=target_bin_size_ms,
        metadata=dict(rebinned.metadata),
        behavior=behavior,
        behavior_names=behavior_names,
    )

    features = compute_endpoint_features(behavior, behavior_names, bin_size_seconds)
    speed = compute_hand_speed(behavior, behavior_names, bin_size_seconds)
    reference_peak = global_peak_speed(rebinned.behavior, behavior_names, bin_size_seconds)
    moving_fraction = np.mean(speed >= MOVING_SPEED_FRACTION * reference_peak, axis=1)
    behavior_statistics = pd.DataFrame(
        {
            "trial_index": slices["trial_index"],
            "start_bin": slices["start_bin"],
            "end_bin": slices["end_bin"],
            "duration_seconds": float(window["duration_seconds"]),
            "endpoint_angle_rad": features["endpoint_angle_rad"],
            "endpoint_distance": features["endpoint_distance"],
            "mean_speed": features["mean_speed"],
            "peak_speed": features["peak_speed"],
            "moving_bin_fraction": moving_fraction,
            "behavior_source": features["behavior_source"],
            "clipped": slices["clipped"],
        },
        columns=BEHAVIOR_STATISTICS_COLUMNS,
    )
    return {
        "dataset": windowed,
        "dataset_hash": dataset_hash,
        "window_slices": slices,
        "behavior_statistics": behavior_statistics,
        "reference_peak_speed": reference_peak,
    }


def _fold_config(config: dict[str, Any]) -> dict[str, Any]:
    copied = dict(config)
    copied["cross_validation"] = dict(config["cross_validation"])
    copied["cross_validation"]["stratification"] = {
        key: value
        for key, value in config["stratification"].items()
        if key != "fallback_when_behavior_missing"
    }
    # Behavior is already required; rate_only is only the existing fold builder's spelling.
    copied["cross_validation"]["fallback_when_behavior_missing"] = "rate_only"
    return copied


def _method_value(method_summary: pd.DataFrame, method_name: str, column: str) -> float:
    rows = method_summary[method_summary["method_name"] == method_name]
    return float("nan") if rows.empty else float(rows.iloc[0][column])


def summarize_recommended_window_cv(
    scores: pd.DataFrame,
    behavior_statistics: pd.DataFrame,
    fold_balance: pd.DataFrame,
    references: dict[str, Any],
) -> dict[str, Any]:
    """Summarize the valid baseline and invalid leakage diagnostic separately."""
    method_summary = summarize_methods(
        scores,
        int(references.get("bootstrap_repeats", 10_000)),
        float(references.get("confidence_interval", 0.95)),
        int(references.get("bootstrap_seed", 1337)),
    )
    factor_mean = _method_value(method_summary, FACTOR_LATENT, "mean_unified_bits_per_spike")
    invalid_mean = _method_value(
        method_summary, SPLIT_MEAN_RATE_INVALID, "mean_unified_bits_per_spike"
    )
    difference = factor_mean - invalid_mean
    beats_invalid = bool(np.isfinite(difference) and difference > 0.0)
    warnings = sorted(
        {
            str(value)
            for value in fold_balance.get("fold_balance_warning", pd.Series(dtype=str))
            if str(value) != "none"
        }
    )
    return {
        "recommended_window_name": RECOMMENDED_WINDOW_NAME,
        "recommended_reporting_mode": RECOMMENDED_REPORTING_MODE,
        "factor_latent_mean": factor_mean,
        "factor_latent_std": _method_value(
            method_summary, FACTOR_LATENT, "std_unified_bits_per_spike"
        ),
        "factor_latent_ci95_low": _method_value(method_summary, FACTOR_LATENT, "ci95_low"),
        "factor_latent_ci95_high": _method_value(method_summary, FACTOR_LATENT, "ci95_high"),
        "factor_latent_positive_fraction": _method_value(
            method_summary, FACTOR_LATENT, "positive_fraction"
        ),
        "split_mean_invalid_mean": invalid_mean,
        "factor_latent_minus_split_mean_invalid": difference,
        "factor_latent_beats_invalid_control_mean": beats_invalid,
        "leakage_dominance_persists": not beats_invalid,
        "leakage_dominance_conclusion": (
            "The leakage dominance observed in the pre-movement window does not persist."
            if beats_invalid
            else "Target leakage remains dominant even in the recommended window."
        ),
        "moving_bin_fraction_mean": float(behavior_statistics["moving_bin_fraction"].mean()),
        "endpoint_direction_entropy_mean": endpoint_direction_entropy(
            behavior_statistics["endpoint_angle_rad"].to_numpy(dtype=np.float64)
        ),
        "fold_balance_warning": "; ".join(warnings) if warnings else "none",
        "invalid_controls_excluded_from_model_selection": True,
        "single_split_results_reportable": False,
        "old_mean_rate_values_used_as_targets": False,
        "official_leaderboard_claim": False,
        "protocol_frozen": True,
        "previous_from_start_factor_latent_mean": references.get(
            "previous_from_start_factor_latent_mean"
        ),
        "previous_from_start_split_mean_invalid_mean": references.get(
            "previous_from_start_split_mean_invalid_mean"
        ),
    }


def _leakage_diagnostics(summary: dict[str, Any]) -> pd.DataFrame:
    previous_factor = summary.get("previous_from_start_factor_latent_mean")
    previous_invalid = summary.get("previous_from_start_split_mean_invalid_mean")
    return pd.DataFrame(
        [
            {
                "metric": "recommended_window_cv_mean",
                "factor_latent_value": summary["factor_latent_mean"],
                "split_mean_invalid_value": summary["split_mean_invalid_mean"],
                "difference_factor_minus_invalid": summary[
                    "factor_latent_minus_split_mean_invalid"
                ],
                "interpretation": summary["leakage_dominance_conclusion"],
            },
            {
                "metric": "previous_from_start_diagnostic_mean",
                "factor_latent_value": previous_factor,
                "split_mean_invalid_value": previous_invalid,
                "difference_factor_minus_invalid": (
                    float(previous_factor) - float(previous_invalid)
                    if previous_factor is not None and previous_invalid is not None
                    else float("nan")
                ),
                "interpretation": "Early/pre-movement diagnostic; not a performance target.",
            },
        ],
        columns=LEAKAGE_DIAGNOSTIC_COLUMNS,
    )


def evaluate_recommended_window_cv(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    """Run CPU-only stratified CV on the recommended movement window."""
    built = build_recommended_window_dataset(config)
    dataset = built["dataset"]
    behavior_statistics = built["behavior_statistics"]
    assert isinstance(dataset, NeuralDataset)
    assert isinstance(behavior_statistics, pd.DataFrame)
    fold_config = _fold_config(config)
    cv = config["cross_validation"]
    reference_mask = create_neuron_mask(
        dataset.spikes.shape[2],
        float(cv["heldout_neuron_fraction"]),
        seed=int(cv["base_seed"]),
    )
    trial_features = build_trial_features(
        dataset.spikes,
        dataset.behavior,
        list(dataset.behavior_names) if dataset.behavior_names is not None else None,
        dataset.bin_size_ms,
        np.flatnonzero(reference_mask.heldout),
    )
    fold_assignments = build_repeated_stratified_folds(trial_features, fold_config)
    raw_balance = compute_fold_balance_statistics(fold_assignments)
    balance_summary = summarize_fold_balance(raw_balance, compare_fold_balance(raw_balance))
    fold_balance = raw_balance[FOLD_BALANCE_COLUMNS[:-1]].copy()
    fold_balance["fold_balance_warning"] = balance_summary["fold_balance_warning"]

    scores = score_folds(dataset, fold_assignments, fold_config).rename(
        columns={"repeat_index": "fold_repeat"}
    )[SCORE_COLUMNS]
    statistics = config["statistics"]
    method_summary = summarize_methods(
        scores.rename(columns={"fold_repeat": "repeat_index"}),
        int(statistics["bootstrap_repeats"]),
        float(statistics["confidence_interval"]),
        int(statistics["bootstrap_seed"]),
    )
    summary_references = {
        **dict(config.get("references", {})),
        "bootstrap_repeats": statistics["bootstrap_repeats"],
        "confidence_interval": statistics["confidence_interval"],
        "bootstrap_seed": statistics["bootstrap_seed"],
    }
    summary = summarize_recommended_window_cv(
        scores, behavior_statistics, fold_balance, summary_references
    )
    summary.update(
        {
            "dataset_name": config["dataset"]["name"],
            "dataset_hash": built["dataset_hash"],
            "bin_size_ms": dataset.bin_size_ms,
            "window_crop_policy": config["window"]["crop_policy"],
            "window_duration_seconds": float(config["window"]["duration_seconds"]),
            "reference_model": config["scoring"]["reference_model"],
            "fold_count": int(cv["fold_count"]),
            "repeats": int(cv["repeats"]),
            "total_folds": int(cv["fold_count"]) * int(cv["repeats"]),
            "assignment_method": cv["assignment_method"],
            **balance_summary,
        }
    )
    tables = {
        "method_summary": method_summary,
        "fold_assignments": fold_assignments,
        "behavior_statistics": behavior_statistics,
        "fold_balance": fold_balance[FOLD_BALANCE_COLUMNS],
        "leakage_diagnostics": _leakage_diagnostics(summary),
    }
    return scores, tables, summary


def build_recommended_window_protocol(
    config: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Serializable protocol contract for carried-forward MC_Maze Small reporting."""
    return {
        "dataset": dict(config["dataset"]),
        "binning": dict(config["binning"]),
        "window": dict(config["window"]),
        "scoring": dict(config["scoring"]),
        "cross_validation": dict(config["cross_validation"]),
        "stratification": dict(config["stratification"]),
        "methods": [dict(method) for method in config["methods"]],
        "statistics": dict(config["statistics"]),
        "recommended_reporting_mode": RECOMMENDED_REPORTING_MODE,
        "claim_safety": {
            "single_split_results_reportable": False,
            "official_leaderboard_claim": False,
            "old_mean_rate_values_used_as_targets": False,
            "invalid_controls_excluded_from_model_selection": True,
        },
        "leakage_dominance_persists": summary.get("leakage_dominance_persists"),
        "protocol_frozen": True,
    }
