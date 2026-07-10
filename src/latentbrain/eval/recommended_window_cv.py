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
    TRAIN_MEAN_RATE,
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


TRIAL_AWARE_SOURCE = "trial_aware_raw"
# A factor-analysis random-state spread wider than this is reported as a stability warning.
FACTOR_ANALYSIS_SENSITIVITY_TOLERANCE = 0.005

LARGE_SCORE_COLUMNS = [
    "repeat_index",
    "fold_index",
    "split_seed",
    "neuron_mask_seed",
    "method_name",
    "method_type",
    "valid_model",
    "reportable_as_model_performance",
    "invalid_reason",
    "factor_analysis_random_state",
    "train_trial_count",
    "eval_trial_count",
    "unified_bits_per_spike",
    "poisson_nll",
    "eval_spike_count",
    "eval_heldout_rate_hz",
    "notes",
]
LARGE_METHOD_SUMMARY_COLUMNS = [
    "method_name",
    "method_type",
    "valid_model",
    "reportable_as_model_performance",
    "n_scores",
    "mean_unified_bits_per_spike",
    "std_unified_bits_per_spike",
    "median_unified_bits_per_spike",
    "min_unified_bits_per_spike",
    "max_unified_bits_per_spike",
    "ci95_low",
    "ci95_high",
    "positive_fraction",
    "between_repeat_std",
    "within_repeat_std",
    "notes",
]
SENSITIVITY_COLUMNS = [
    "repeat_index",
    "fold_index",
    "split_seed",
    "factor_analysis_random_state",
    "unified_bits_per_spike",
    "poisson_nll",
    "difference_from_random_state_0",
    "notes",
]
FOLD_LEAKAGE_COLUMNS = [
    "repeat_index",
    "fold_index",
    "factor_latent_value",
    "split_mean_invalid_value",
    "factor_minus_invalid",
    "factor_beats_invalid",
    "interpretation",
]
COMPARISON_COLUMNS = [
    "dataset",
    "trial_count",
    "fold_count",
    "repeats",
    "eval_trials_per_fold",
    "factor_latent_mean",
    "factor_latent_std",
    "factor_latent_ci95_low",
    "factor_latent_ci95_high",
    "factor_latent_positive_fraction",
    "split_mean_invalid_mean",
    "factor_minus_invalid",
    "moving_bin_fraction",
    "endpoint_direction_entropy",
]

LARGE_BEHAVIOR_STATISTICS_COLUMNS = [
    *BEHAVIOR_STATISTICS_COLUMNS,
    "left_clipped",
    "right_clipped",
    "padded_bins",
    "peak_speed_in_window",
    "movement_onset_in_window",
]


def build_trial_aware_window_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Extract the frozen window from ragged raw trials, then rebin. Never from the global crop."""
    import json  # noqa: PLC0415

    from latentbrain.data.nlb import NLBConfig, load_trial_sequences  # noqa: PLC0415
    from latentbrain.data.validation import validate_trial_sequences  # noqa: PLC0415
    from latentbrain.eval.window_audit import (  # noqa: PLC0415
        evaluate_window_coverage,
        reference_peak_speed,
        trial_movement_table,
    )

    dataset_config = config["dataset"]
    repo_root = get_repo_root()
    processed_path = resolve_configured_path(str(dataset_config["processed_path"]), repo_root)
    if not processed_path.exists():
        msg = f"Processed dataset is missing: {processed_path}"
        raise FileNotFoundError(msg)
    reference_dataset = load_neural_dataset(processed_path)
    validate_neural_dataset(reference_dataset)
    dataset_hash = compute_dataset_hash(reference_dataset)
    expected_hash = str(dataset_config.get("expected_hash", ""))
    if expected_hash and dataset_hash != expected_hash:
        msg = f"Dataset hash mismatch: expected {expected_hash}, got {dataset_hash}"
        raise ValueError(msg)

    trial_source = config["trial_source"]
    if str(trial_source["type"]) != TRIAL_AWARE_SOURCE:
        msg = f"trial_source.type must be {TRIAL_AWARE_SOURCE}"
        raise ValueError(msg)
    if bool(trial_source["allow_global_crop_to_min"]):
        msg = (
            "trial_source.allow_global_crop_to_min must be false; the globally cropped array "
            "cannot source event-centered evaluation windows"
        )
        raise ValueError(msg)
    if not bool(config["binning"]["extract_before_rebin"]):
        msg = "binning.extract_before_rebin must be true"
        raise ValueError(msg)

    raw_dir = resolve_configured_path(str(dataset_config["raw_dir"]), repo_root)
    if not raw_dir.exists():
        msg = f"Raw dataset directory is missing: {raw_dir}"
        raise FileNotFoundError(msg)
    provenance_path = resolve_configured_path(str(dataset_config["provenance_path"]), repo_root)
    if not provenance_path.exists():
        msg = f"Provenance file is missing: {provenance_path}"
        raise FileNotFoundError(msg)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if str(provenance.get("processed_dataset_hash", dataset_hash)) != dataset_hash:
        msg = "provenance processed_dataset_hash does not match the processed artifact"
        raise ValueError(msg)
    nlb_config = NLBConfig.model_validate(provenance["config"])
    sequences = load_trial_sequences(raw_dir, nlb_config)
    validate_trial_sequences(sequences)
    if bool(trial_source["require_exact_trial_lengths"]) and int(
        sequences.trial_lengths.min()
    ) != int(reference_dataset.spikes.shape[1]):
        msg = "trial-aware minimum length does not match the processed artifact's global crop"
        raise ValueError(msg)

    target_bin_size_ms = int(config["binning"]["target_bin_size_ms"])
    validate_rebin_factor(int(dataset_config["original_bin_size_ms"]), target_bin_size_ms)
    window = dict(config["window"])
    movement = trial_movement_table(sequences)
    reference_peak = reference_peak_speed(sequences, target_bin_size_ms)
    coverage, _candidate_summary, extras = evaluate_window_coverage(
        sequences, movement, window, target_bin_size_ms, reference_peak
    )
    if bool(coverage["left_clipped"].any()) or bool(coverage["right_clipped"].any()):
        msg = "the frozen Large window must not clip any trial; the accepted audit reports zero"
        raise ValueError(msg)
    if int(coverage["padded_bins"].sum()) != 0:
        msg = "the frozen Large window must not pad any trial"
        raise ValueError(msg)

    spikes = extras["windowed_spikes"]
    behavior = extras["windowed_behavior"]
    expected_bins = int(round(float(window["duration_seconds"]) * 1000.0 / target_bin_size_ms))
    if spikes.shape[1] != expected_bins:
        msg = f"windowed dataset has {spikes.shape[1]} time bins, expected {expected_bins}"
        raise ValueError(msg)
    if spikes.shape[:2] != behavior.shape[:2]:
        msg = "windowed spikes and behavior are misaligned"
        raise ValueError(msg)
    behavior_names = list(sequences.behavior_names or [])
    windowed = NeuralDataset(
        spikes=spikes,
        rates=None,
        latents=None,
        trial_ids=np.asarray(sequences.trial_ids, dtype=np.int64),
        time_ms=np.arange(spikes.shape[1], dtype=np.float64) * target_bin_size_ms,
        bin_size_ms=target_bin_size_ms,
        metadata={
            "dataset_name": str(dataset_config["name"]),
            "source_dataset_hash": dataset_hash,
            "trial_source": TRIAL_AWARE_SOURCE,
            "window_name": str(window["name"]),
            "window_extraction_policy": "extract_at_source_bins_then_rebin",
            "global_crop_used_for_event_centered_windows": False,
        },
        behavior=behavior,
        behavior_names=behavior_names,
    )
    validate_neural_dataset(windowed)

    behavior_statistics = pd.DataFrame(
        {
            "trial_index": coverage["trial_index"],
            "start_bin": coverage["start_bin"],
            "end_bin": coverage["end_bin"],
            "duration_seconds": float(window["duration_seconds"]),
            "endpoint_angle_rad": coverage["endpoint_angle_rad"],
            "endpoint_distance": coverage["endpoint_distance"],
            "mean_speed": coverage["mean_speed"],
            "peak_speed": coverage["peak_speed"],
            "moving_bin_fraction": coverage["moving_bin_fraction"],
            "behavior_source": str(movement.iloc[0]["behavior_source"]),
            "clipped": coverage["left_clipped"] | coverage["right_clipped"],
            "left_clipped": coverage["left_clipped"],
            "right_clipped": coverage["right_clipped"],
            "padded_bins": coverage["padded_bins"],
            "peak_speed_in_window": coverage["peak_speed_in_window"],
            "movement_onset_in_window": coverage["movement_onset_in_window"],
        },
        columns=LARGE_BEHAVIOR_STATISTICS_COLUMNS,
    )
    return {
        "dataset": windowed,
        "dataset_hash": dataset_hash,
        "behavior_statistics": behavior_statistics,
        "reference_peak_speed": reference_peak,
        "trial_length_min": int(sequences.trial_lengths.min()),
        "trial_length_max": int(sequences.trial_lengths.max()),
        "trial_source_file": sequences.metadata.get("source_file"),
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


def _repeat_std(scores: pd.DataFrame, method_name: str) -> tuple[float, float]:
    """Between-repeat std of repeat means, and the mean within-repeat std."""
    rows = scores[scores["method_name"] == method_name]
    if rows.empty:
        return (float("nan"), float("nan"))
    grouped = rows.groupby("repeat_index")["unified_bits_per_spike"]
    means = grouped.mean().to_numpy(dtype=np.float64)
    stds = grouped.std(ddof=1).to_numpy(dtype=np.float64)
    between = float(np.std(means, ddof=1)) if means.size > 1 else 0.0
    within = float(np.nanmean(stds)) if stds.size else float("nan")
    return (between, within)


def build_large_method_summary(scores: pd.DataFrame, statistics: dict[str, Any]) -> pd.DataFrame:
    """Method summary with repeat-resolved stability columns."""
    summary = summarize_methods(
        scores,
        int(statistics["bootstrap_repeats"]),
        float(statistics["confidence_interval"]),
        int(statistics["bootstrap_seed"]),
    ).rename(columns={"n_folds": "n_scores"})
    between: list[float] = []
    within: list[float] = []
    for method_name in summary["method_name"]:
        method_between, method_within = _repeat_std(scores, str(method_name))
        between.append(method_between)
        within.append(method_within)
    summary["between_repeat_std"] = between
    summary["within_repeat_std"] = within
    return summary[LARGE_METHOD_SUMMARY_COLUMNS]


def factor_analysis_random_state_sensitivity(
    dataset: NeuralDataset,
    fold_assignments: pd.DataFrame,
    fold_config: dict[str, Any],
    baseline_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Rescore factor-latent across FactorAnalysis random states, independent of fold seeds."""
    sensitivity = fold_config.get("factor_analysis_sensitivity", {})
    states = [int(state) for state in sensitivity.get("random_states", [])]
    if len(set(states)) != len(states):
        msg = "factor_analysis_sensitivity.random_states must be unique"
        raise ValueError(msg)
    factor_method = next(
        method for method in fold_config["methods"] if str(method["name"]) == FACTOR_LATENT
    )
    baseline_state = int(factor_method.get("factor_analysis_random_state", 0))
    frames: list[pd.DataFrame] = []
    for state in states:
        if state == baseline_state:
            rows = baseline_scores[baseline_scores["method_name"] == FACTOR_LATENT].copy()
        else:
            state_config = dict(fold_config)
            state_config["methods"] = [{**factor_method, "factor_analysis_random_state": state}]
            rows = score_folds(dataset, fold_assignments, state_config)
        frames.append(
            pd.DataFrame(
                {
                    "repeat_index": rows["repeat_index"].to_numpy(),
                    "fold_index": rows["fold_index"].to_numpy(),
                    "split_seed": rows["split_seed"].to_numpy(),
                    "factor_analysis_random_state": state,
                    "unified_bits_per_spike": rows["unified_bits_per_spike"].to_numpy(),
                    "poisson_nll": rows["poisson_nll"].to_numpy(),
                }
            )
        )
    table = pd.concat(frames, ignore_index=True)
    reference = table[table["factor_analysis_random_state"] == 0].set_index(
        ["repeat_index", "fold_index"]
    )["unified_bits_per_spike"]
    keys = pd.MultiIndex.from_arrays([table["repeat_index"], table["fold_index"]])
    table["difference_from_random_state_0"] = (
        table["unified_bits_per_spike"].to_numpy() - reference.reindex(keys).to_numpy()
        if not reference.empty
        else float("nan")
    )
    table["notes"] = "FactorAnalysis random state is independent of the fold and repeat seeds."
    return table[SENSITIVITY_COLUMNS]


def summarize_factor_analysis_sensitivity(sensitivity: pd.DataFrame) -> dict[str, Any]:
    if sensitivity.empty:
        return {
            "factor_analysis_random_states": [],
            "factor_analysis_random_state_range": float("nan"),
            "factor_analysis_random_state_std": float("nan"),
            "factor_analysis_random_state_warning": "sensitivity was not evaluated",
        }
    means = sensitivity.groupby("factor_analysis_random_state")["unified_bits_per_spike"].mean()
    values = means.to_numpy(dtype=np.float64)
    spread = float(np.max(values) - np.min(values))
    deviation = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    warning = (
        f"factor-latent varies by {spread:.6f} bits/spike across FactorAnalysis random states, "
        f"above the {FACTOR_ANALYSIS_SENSITIVITY_TOLERANCE} tolerance"
        if spread > FACTOR_ANALYSIS_SENSITIVITY_TOLERANCE
        else "none"
    )
    return {
        "factor_analysis_random_states": [int(state) for state in means.index],
        "factor_analysis_random_state_range": spread,
        "factor_analysis_random_state_std": deviation,
        "factor_analysis_random_state_mean_by_state": {
            int(state): float(value) for state, value in means.items()
        },
        "factor_analysis_random_state_warning": warning,
    }


def build_fold_leakage_diagnostics(scores: pd.DataFrame) -> pd.DataFrame:
    """Per-fold factor-latent versus the invalid split-mean control."""
    pivot = scores.pivot_table(
        index=["repeat_index", "fold_index"],
        columns="method_name",
        values="unified_bits_per_spike",
    ).reset_index()
    factor = pivot[FACTOR_LATENT].to_numpy(dtype=np.float64)
    invalid = pivot[SPLIT_MEAN_RATE_INVALID].to_numpy(dtype=np.float64)
    difference = factor - invalid
    beats = difference > 0.0
    return pd.DataFrame(
        {
            "repeat_index": pivot["repeat_index"],
            "fold_index": pivot["fold_index"],
            "factor_latent_value": factor,
            "split_mean_invalid_value": invalid,
            "factor_minus_invalid": difference,
            "factor_beats_invalid": beats,
            "interpretation": [
                "factor-latent beats the invalid control on this fold"
                if value
                else "the invalid control still dominates on this fold"
                for value in beats
            ],
        },
        columns=FOLD_LEAKAGE_COLUMNS,
    )


def build_small_large_comparison(
    large_summary: dict[str, Any],
    small_summary: dict[str, Any] | None,
    references: dict[str, Any],
    large_rows: dict[str, Any],
) -> pd.DataFrame:
    """Protocol-stability comparison only. Never a cross-dataset performance claim."""
    source = small_summary or {}

    def _small(key: str, reference_key: str) -> Any:
        value = source.get(key)
        return references.get(reference_key) if value is None else value

    small_row = {
        "dataset": "mc_maze_small",
        "trial_count": source.get("trial_count"),
        "fold_count": source.get("fold_count"),
        "repeats": source.get("repeats"),
        "eval_trials_per_fold": source.get("eval_trials_per_fold"),
        "factor_latent_mean": _small("factor_latent_mean", "small_factor_latent_mean"),
        "factor_latent_std": _small("factor_latent_std", "small_factor_latent_std"),
        "factor_latent_ci95_low": _small("factor_latent_ci95_low", "small_factor_latent_ci95_low"),
        "factor_latent_ci95_high": _small(
            "factor_latent_ci95_high", "small_factor_latent_ci95_high"
        ),
        "factor_latent_positive_fraction": _small(
            "factor_latent_positive_fraction", "small_factor_latent_positive_fraction"
        ),
        "split_mean_invalid_mean": _small(
            "split_mean_invalid_mean", "small_split_mean_invalid_mean"
        ),
        "factor_minus_invalid": _small(
            "factor_latent_minus_split_mean_invalid", "small_factor_minus_invalid"
        ),
        "moving_bin_fraction": source.get("moving_bin_fraction_mean"),
        "endpoint_direction_entropy": source.get("endpoint_direction_entropy_mean"),
    }
    large_row = {
        "dataset": large_summary["dataset_name"],
        "trial_count": large_rows["trial_count"],
        "fold_count": large_summary["fold_count"],
        "repeats": large_summary["repeats"],
        "eval_trials_per_fold": large_rows["eval_trials_per_fold"],
        "factor_latent_mean": large_summary["factor_latent_mean"],
        "factor_latent_std": large_summary["factor_latent_std"],
        "factor_latent_ci95_low": large_summary["factor_latent_ci95_low"],
        "factor_latent_ci95_high": large_summary["factor_latent_ci95_high"],
        "factor_latent_positive_fraction": large_summary["factor_latent_positive_fraction"],
        "split_mean_invalid_mean": large_summary["split_mean_invalid_mean"],
        "factor_minus_invalid": large_summary["factor_latent_minus_split_mean_invalid"],
        "moving_bin_fraction": large_summary["moving_bin_fraction_mean"],
        "endpoint_direction_entropy": large_summary["endpoint_direction_entropy_mean"],
    }
    return pd.DataFrame([small_row, large_row], columns=COMPARISON_COLUMNS)


def summarize_small_large_comparison(
    comparison: pd.DataFrame,
    small_summary_available: bool,
) -> dict[str, Any]:
    """Stability-only conclusions. Cross-dataset performance comparison is forbidden."""
    small = comparison[comparison["dataset"] == "mc_maze_small"].iloc[0]
    large = comparison[comparison["dataset"] != "mc_maze_small"].iloc[0]
    conclusions: list[str] = []
    small_std = small["factor_latent_std"]
    large_std = large["factor_latent_std"]
    if pd.notna(small_std) and pd.notna(large_std):
        wider = "wider" if float(large_std) > float(small_std) else "narrower"
        conclusions.append(f"Large fold-to-fold variance is {wider} than Small under this protocol")
    small_positive = small["factor_latent_positive_fraction"]
    large_positive = large["factor_latent_positive_fraction"]
    if pd.notna(small_positive) and pd.notna(large_positive):
        conclusions.append(
            "Large positive-fold fraction is higher than Small"
            if float(large_positive) > float(small_positive)
            else "Large positive-fold fraction is lower than Small"
            if float(large_positive) < float(small_positive)
            else "Large and Small have the same positive-fold fraction"
        )
    for label, row in (("small", small), ("large", large)):
        difference = row["factor_minus_invalid"]
        if pd.notna(difference):
            persists = "persists" if float(difference) <= 0.0 else "does not persist"
            conclusions.append(f"leakage dominance {persists} on {label}")
    return {
        "small_summary_available": bool(small_summary_available),
        "small_large_comparison_conclusions": conclusions,
        "small_large_comparison_is_protocol_stability_only": True,
        "cross_dataset_performance_comparison_claimed": False,
    }


def evaluate_large_recommended_window_cv(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    """Trial-aware recommended-window CV. Behavior-frozen window; no neural model is trained."""
    import json  # noqa: PLC0415

    built = build_trial_aware_window_dataset(config)
    dataset = built["dataset"]
    behavior_statistics = built["behavior_statistics"]
    fold_config = _fold_config(config)
    cv = config["cross_validation"]
    if str(cv.get("heldout_mask_policy", "fixed_within_repeat")) != "fixed_within_repeat":
        msg = "cross_validation.heldout_mask_policy must be fixed_within_repeat"
        raise ValueError(msg)

    reference_mask = create_neuron_mask(
        dataset.spikes.shape[2], float(cv["heldout_neuron_fraction"]), seed=int(cv["base_seed"])
    )
    trial_features = build_trial_features(
        dataset.spikes,
        dataset.behavior,
        list(dataset.behavior_names or []),
        dataset.bin_size_ms,
        np.flatnonzero(reference_mask.heldout),
    )
    fold_assignments = build_repeated_stratified_folds(trial_features, fold_config)
    raw_balance = compute_fold_balance_statistics(fold_assignments)
    balance_summary = summarize_fold_balance(raw_balance, compare_fold_balance(raw_balance))
    fold_balance = raw_balance.copy()
    fold_balance["fold_balance_warning"] = balance_summary["fold_balance_warning"]

    scores = score_folds(dataset, fold_assignments, fold_config)[LARGE_SCORE_COLUMNS]
    train_mean_values = scores[scores["method_name"] == TRAIN_MEAN_RATE][
        "unified_bits_per_spike"
    ].to_numpy(dtype=np.float64)
    if train_mean_values.size and float(np.max(np.abs(train_mean_values))) > 1e-12:
        msg = "train mean-rate reference did not score exactly 0.0 bits/spike against itself"
        raise ValueError(msg)

    statistics = dict(config["statistics"])
    method_summary = build_large_method_summary(scores, statistics)
    sensitivity = (
        factor_analysis_random_state_sensitivity(dataset, fold_assignments, fold_config, scores)
        if bool(config.get("factor_analysis_sensitivity", {}).get("enabled", False))
        else pd.DataFrame(columns=SENSITIVITY_COLUMNS)
    )
    fold_leakage = build_fold_leakage_diagnostics(scores)

    summary_references = {
        **dict(config.get("references", {})),
        "bootstrap_repeats": statistics["bootstrap_repeats"],
        "confidence_interval": statistics["confidence_interval"],
        "bootstrap_seed": statistics["bootstrap_seed"],
    }
    summary = summarize_recommended_window_cv(
        scores.rename(columns={"repeat_index": "fold_repeat"}),
        behavior_statistics,
        fold_balance,
        summary_references,
    )
    between, within = _repeat_std(scores, FACTOR_LATENT)
    beats_fraction = float(fold_leakage["factor_beats_invalid"].mean())
    invalid_values = scores[scores["method_name"] == SPLIT_MEAN_RATE_INVALID][
        "unified_bits_per_spike"
    ].to_numpy(dtype=np.float64)
    trial_count = int(dataset.spikes.shape[0])
    eval_trials = int(round(trial_count / int(cv["fold_count"])))
    summary.update(
        {
            "dataset_name": str(config["dataset"]["name"]),
            "dataset_hash": built["dataset_hash"],
            "trial_source": TRIAL_AWARE_SOURCE,
            "trial_source_file": built["trial_source_file"],
            "trial_length_min": built["trial_length_min"],
            "trial_length_max": built["trial_length_max"],
            "global_crop_used_for_event_centered_windows": False,
            "window_name": str(config["window"]["name"]),
            "window_crop_policy": str(config["window"]["crop_policy"]),
            "window_duration_seconds": float(config["window"]["duration_seconds"]),
            "target_bin_size_ms": dataset.bin_size_ms,
            "bin_size_ms": dataset.bin_size_ms,
            "trial_count": trial_count,
            "time_bins": int(dataset.spikes.shape[1]),
            "neuron_count": int(dataset.spikes.shape[2]),
            "heldin_neuron_count": int(reference_mask.heldin.sum()),
            "heldout_neuron_count": int(reference_mask.heldout.sum()),
            "reference_model": str(config["scoring"]["reference_model"]),
            "fold_count": int(cv["fold_count"]),
            "repeats": int(cv["repeats"]),
            "total_folds": int(cv["fold_count"]) * int(cv["repeats"]),
            "train_trials_per_fold": trial_count - eval_trials,
            "eval_trials_per_fold": eval_trials,
            "heldout_mask_policy": str(cv["heldout_mask_policy"]),
            "assignment_method": str(cv["assignment_method"]),
            "train_mean_rate_mean": 0.0 if train_mean_values.size else float("nan"),
            "factor_latent_between_repeat_std": between,
            "factor_latent_within_repeat_std": within,
            "split_mean_invalid_std": float(np.std(invalid_values, ddof=1))
            if invalid_values.size > 1
            else 0.0,
            "factor_latent_beats_invalid_control_fraction": beats_fraction,
            **summarize_factor_analysis_sensitivity(sensitivity),
            **balance_summary,
        }
    )

    small_summary: dict[str, Any] | None = None
    small_path = config.get("inputs", {}).get("small_recommended_window_summary_path")
    if small_path:
        resolved = resolve_configured_path(str(small_path), get_repo_root())
        if resolved.exists():
            small_summary = json.loads(resolved.read_text(encoding="utf-8"))
    comparison = build_small_large_comparison(
        summary,
        small_summary,
        dict(config.get("references", {})),
        {"trial_count": trial_count, "eval_trials_per_fold": eval_trials},
    )
    summary.update(summarize_small_large_comparison(comparison, small_summary is not None))

    tables = {
        "method_summary": method_summary,
        "fold_assignments": fold_assignments,
        "behavior_statistics": behavior_statistics,
        "fold_balance": fold_balance,
        "leakage_diagnostics": fold_leakage,
        "factor_analysis_sensitivity": sensitivity,
        "small_large_comparison": comparison,
    }
    return scores, tables, summary


def build_recommended_window_protocol(
    config: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Serializable protocol contract for carried-forward recommended-window reporting."""
    # Annotated so the comprehension's keys widen to str; a Literal key type cannot be unpacked
    # into dict[str, Any].
    optional: dict[str, Any] = {
        key: dict(config[key])
        for key in ("trial_source", "factor_analysis_sensitivity")
        if key in config
    }
    return {
        **optional,
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
