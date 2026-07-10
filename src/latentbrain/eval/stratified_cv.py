from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from latentbrain.data.schemas import NeuralDataset, TrialSplit
from latentbrain.data.splits import create_neuron_mask
from latentbrain.eval.rate_controls import (
    compute_split_mean_rate_invalid_control,
    compute_train_mean_rate_control,
)
from latentbrain.eval.scoring import (
    ScoringConfig,
    score_heldout_prediction,
    train_heldout_mean_rate_reference,
)
from latentbrain.eval.seed_robustness import bootstrap_mean_ci
from latentbrain.eval.split_audit import factor_latent_heldout_rates

TRIAL_FEATURE_COLUMNS = [
    "trial_index",
    "total_spikes",
    "population_rate_hz",
    "heldout_rate_hz",
    "zero_fraction",
    "behavior_available",
    "endpoint_dx",
    "endpoint_dy",
    "endpoint_angle_rad",
    "endpoint_distance",
    "mean_speed",
    "endpoint_direction_bin",
    "endpoint_distance_bin",
    "mean_speed_bin",
    "population_rate_bin",
    "heldout_rate_bin",
]

FOLD_ASSIGNMENT_COLUMNS = [
    "repeat_index",
    "fold_index",
    "trial_index",
    "stratum",
    "total_spikes",
    "population_rate_hz",
    "heldout_rate_hz",
    "endpoint_angle_rad",
    "endpoint_distance",
    "mean_speed",
    "assignment_method",
    "seed",
]

SCORE_COLUMNS = [
    "repeat_index",
    "fold_index",
    "split_seed",
    "neuron_mask_seed",
    "method_name",
    "method_type",
    "valid_model",
    "reportable_as_model_performance",
    "invalid_reason",
    "train_trial_count",
    "eval_trial_count",
    "unified_bits_per_spike",
    "poisson_nll",
    "eval_spike_count",
    "eval_heldout_rate_hz",
    "factor_analysis_random_state",
    "notes",
]

METHOD_SUMMARY_COLUMNS = [
    "method_name",
    "method_type",
    "valid_model",
    "reportable_as_model_performance",
    "n_folds",
    "mean_unified_bits_per_spike",
    "std_unified_bits_per_spike",
    "median_unified_bits_per_spike",
    "min_unified_bits_per_spike",
    "max_unified_bits_per_spike",
    "ci95_low",
    "ci95_high",
    "positive_fraction",
    "notes",
]

ASSIGNMENT_METHODS = ("greedy_balanced",)
BEHAVIOR_FALLBACKS = ("rate_only",)
FACTOR_LATENT = "factor_latent"
TRAIN_MEAN_RATE = "train_mean_rate"
SPLIT_MEAN_RATE_INVALID = "split_mean_rate_invalid"

_POSITION_PREFIX = "hand_pos"

_BEHAVIOR_COLUMNS = (
    "endpoint_dx",
    "endpoint_dy",
    "endpoint_angle_rad",
    "endpoint_distance",
    "mean_speed",
)


def _position_columns(behavior_names: list[str] | None) -> tuple[int, int] | None:
    if not behavior_names:
        return None
    try:
        return (
            behavior_names.index(f"{_POSITION_PREFIX}_x"),
            behavior_names.index(f"{_POSITION_PREFIX}_y"),
        )
    except ValueError:
        return None


def build_trial_features(
    spikes: np.ndarray,
    behavior: np.ndarray | None,
    behavior_names: list[str] | None,
    bin_size_ms: int,
    heldout_indices: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-trial spike and behavior features used to stratify folds."""
    counts = np.asarray(spikes, dtype=np.float64)
    if counts.ndim != 3:
        msg = "spikes must have shape [trials, time, neurons]"
        raise ValueError(msg)
    seconds = counts.shape[1] * (bin_size_ms / 1000.0)
    seconds_per_bin = bin_size_ms / 1000.0
    columns = _position_columns(behavior_names)
    has_behavior = behavior is not None and columns is not None
    rows: list[dict[str, Any]] = []
    for index in range(counts.shape[0]):
        trial = counts[index]
        heldout = trial[:, heldout_indices] if heldout_indices is not None else None
        row: dict[str, Any] = {
            "trial_index": int(index),
            "total_spikes": float(trial.sum()),
            "population_rate_hz": float(trial.sum()) / (seconds * trial.shape[1]),
            "heldout_rate_hz": float(heldout.sum()) / (seconds * heldout.shape[1])
            if heldout is not None and heldout.shape[1] > 0
            else float("nan"),
            "zero_fraction": float(np.mean(trial == 0.0)),
            "behavior_available": bool(has_behavior),
        }
        for column in _BEHAVIOR_COLUMNS:
            row[column] = float("nan")
        if has_behavior:
            assert behavior is not None and columns is not None
            x = np.asarray(behavior[index][:, columns[0]], dtype=np.float64)
            y = np.asarray(behavior[index][:, columns[1]], dtype=np.float64)
            dx = float(x[-1] - x[0])
            dy = float(y[-1] - y[0])
            steps = np.hypot(np.diff(x), np.diff(y))
            row["endpoint_dx"] = dx
            row["endpoint_dy"] = dy
            row["endpoint_angle_rad"] = float(np.arctan2(dy, dx))
            row["endpoint_distance"] = float(np.hypot(dx, dy))
            row["mean_speed"] = (
                float(np.mean(steps) / seconds_per_bin) if steps.size else float("nan")
            )
        rows.append(row)
    features = pd.DataFrame(rows)
    for column in (
        "endpoint_direction_bin",
        "endpoint_distance_bin",
        "mean_speed_bin",
        "population_rate_bin",
        "heldout_rate_bin",
    ):
        features[column] = -1
    return features[TRIAL_FEATURE_COLUMNS]


def _direction_bin(angles: pd.Series, bins: int) -> pd.Series:
    # Wrap [-pi, pi) into equal-width sectors so opposite reaches never share a bin.
    shifted = (angles + np.pi) % (2.0 * np.pi)
    edges = np.floor(shifted / (2.0 * np.pi / bins)).astype("Int64")
    return edges.clip(0, bins - 1)


def _quantile_bin(values: pd.Series, bins: int) -> pd.Series:
    finite = values.dropna()
    if finite.empty or bins <= 1:
        return pd.Series([-1] * len(values), index=values.index, dtype="Int64")
    ranks = values.rank(method="first", na_option="keep")
    edges = np.ceil(ranks / (len(finite) / bins)) - 1
    return edges.clip(0, bins - 1).astype("Int64")


def build_strata_labels(trial_features: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    """Deterministic stratum label per trial. Behavior-derived terms drop out when absent."""
    settings = dict(config["cross_validation"]["stratification"])
    fallback = str(config["cross_validation"].get("fallback_when_behavior_missing", "rate_only"))
    if fallback not in BEHAVIOR_FALLBACKS:
        msg = f"fallback_when_behavior_missing must be one of {BEHAVIOR_FALLBACKS}"
        raise ValueError(msg)
    behavior_available = bool(trial_features["behavior_available"].all())
    features = trial_features.copy()
    parts: list[pd.Series] = []
    if behavior_available and bool(settings.get("use_endpoint_direction", False)):
        features["endpoint_direction_bin"] = _direction_bin(
            features["endpoint_angle_rad"], int(settings["endpoint_direction_bins"])
        )
        parts.append(features["endpoint_direction_bin"])
    if behavior_available and bool(settings.get("use_endpoint_distance", False)):
        features["endpoint_distance_bin"] = _quantile_bin(
            features["endpoint_distance"], int(settings["endpoint_distance_bins"])
        )
        parts.append(features["endpoint_distance_bin"])
    if behavior_available and bool(settings.get("use_mean_speed", False)):
        features["mean_speed_bin"] = _quantile_bin(
            features["mean_speed"], int(settings["mean_speed_bins"])
        )
        parts.append(features["mean_speed_bin"])
    if bool(settings.get("use_population_rate", False)):
        features["population_rate_bin"] = _quantile_bin(
            features["population_rate_hz"], int(settings["population_rate_bins"])
        )
        parts.append(features["population_rate_bin"])
    if bool(settings.get("use_heldout_rate", False)) and features["heldout_rate_hz"].notna().any():
        features["heldout_rate_bin"] = _quantile_bin(
            features["heldout_rate_hz"], int(settings["heldout_rate_bins"])
        )
        parts.append(features["heldout_rate_bin"])
    if not parts:
        msg = "at least one stratification variable must be usable"
        raise ValueError(msg)
    labels = parts[0].astype(str)
    for series in parts[1:]:
        labels = labels + "_" + series.astype(str)
    return labels.rename("stratum")


def _merge_small_strata(labels: pd.Series, min_trials_per_stratum: int) -> pd.Series:
    counts = labels.value_counts()
    small = set(counts[counts < max(int(min_trials_per_stratum), 1)].index)
    if not small:
        return labels
    # Pooling rare strata keeps fold sizes balanced; the alternative is a fold that never
    # sees a rare reach direction at all.
    return labels.where(~labels.isin(small), other="pooled_small_stratum")


def assign_stratified_folds(
    trial_features: pd.DataFrame,
    strata_labels: pd.Series,
    fold_count: int,
    seed: int,
    min_trials_per_stratum: int,
) -> pd.DataFrame:
    """Greedy balanced assignment: within each stratum, fill the currently smallest folds."""
    if fold_count < 3:
        msg = "fold_count must be at least 3"
        raise ValueError(msg)
    labels = _merge_small_strata(strata_labels, min_trials_per_stratum)
    generator = np.random.default_rng(seed)
    fold_sizes = np.zeros(fold_count, dtype=np.int64)
    assignment = pd.Series(-1, index=trial_features.index, dtype=np.int64)
    for stratum in sorted(labels.unique()):
        members = np.asarray(labels.index[labels == stratum])
        order = generator.permutation(len(members))
        for position in order:
            fold = int(np.argmin(fold_sizes))
            assignment.iloc[int(members[position])] = fold
            fold_sizes[fold] += 1
    frame = trial_features.copy()
    frame["fold_index"] = assignment.to_numpy()
    frame["stratum"] = labels.to_numpy()
    frame["seed"] = int(seed)
    frame["assignment_method"] = "greedy_balanced"
    return frame


def build_repeated_stratified_folds(
    trial_features: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    cross_validation = dict(config["cross_validation"])
    method = str(cross_validation.get("assignment_method", "greedy_balanced"))
    if method not in ASSIGNMENT_METHODS:
        msg = f"assignment_method must be one of {ASSIGNMENT_METHODS}"
        raise ValueError(msg)
    fold_count = int(cross_validation["fold_count"])
    repeats = int(cross_validation["repeats"])
    if repeats < 2:
        msg = "repeats must be at least 2"
        raise ValueError(msg)
    base_seed = int(cross_validation["base_seed"])
    minimum = int(cross_validation.get("min_trials_per_stratum", 2))
    labels = build_strata_labels(trial_features, config)
    frames: list[pd.DataFrame] = []
    for repeat_index in range(repeats):
        assigned = assign_stratified_folds(
            trial_features, labels, fold_count, base_seed + repeat_index, minimum
        )
        assigned.insert(0, "repeat_index", repeat_index)
        frames.append(assigned)
    folds = pd.concat(frames, ignore_index=True)
    return folds[FOLD_ASSIGNMENT_COLUMNS]


def build_random_folds(trial_features: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Matched unstratified folds: same fold count and repeat count, random assignment."""
    cross_validation = dict(config["cross_validation"])
    fold_count = int(cross_validation["fold_count"])
    total_folds = int(cross_validation.get("random_split_repeats", fold_count * 2))
    repeats = max(total_folds // fold_count, 1)
    base_seed = int(cross_validation["base_seed"]) + 10_000
    frames: list[pd.DataFrame] = []
    for repeat_index in range(repeats):
        generator = np.random.default_rng(base_seed + repeat_index)
        order = generator.permutation(len(trial_features))
        frame = trial_features.copy()
        frame["fold_index"] = np.zeros(len(trial_features), dtype=np.int64)
        for position, trial in enumerate(order):
            frame.iloc[int(trial), frame.columns.get_loc("fold_index")] = position % fold_count
        frame["stratum"] = "random"
        frame["seed"] = int(base_seed + repeat_index)
        frame["assignment_method"] = "random"
        frame.insert(0, "repeat_index", repeat_index)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)[FOLD_ASSIGNMENT_COLUMNS]


def _scoring_config(config: dict[str, Any]) -> ScoringConfig:
    scoring = config["scoring"]
    return ScoringConfig(
        bin_size_ms=int(config["binning"]["target_bin_size_ms"]),
        include_poisson_constant=bool(scoring["include_poisson_constant"]),
        min_rate_hz=float(scoring["min_rate_hz"]),
        max_rate_hz=float(scoring["max_rate_hz"]),
        reference_name=str(scoring["reference_model"]),
    )


def _factor_settings(method: dict[str, Any]) -> dict[str, float]:
    return {
        "latent_dim": float(method["latent_dim"]),
        "smoothing_sigma_ms": float(method["smoothing_sigma_ms"]),
        "heldout_decoder_alpha": float(method["heldout_decoder_alpha"]),
        "max_iter": 1000.0,
        "tol": 1.0e-4,
    }


def _counts(dataset: NeuralDataset, trial_ids: np.ndarray, neurons: np.ndarray) -> np.ndarray:
    return np.asarray(dataset.spikes[np.isin(dataset.trial_ids, trial_ids)][:, :, neurons])


def score_folds(
    dataset: NeuralDataset,
    fold_assignments: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Score each method on every held-out fold with the canonical unweighted scorer."""
    scoring = _scoring_config(config)
    methods = {str(method["name"]): dict(method) for method in config["methods"]}
    heldout_fraction = float(config["cross_validation"]["heldout_neuron_fraction"])
    base_seed = int(config["cross_validation"]["base_seed"])
    rows: list[dict[str, Any]] = []
    for repeat_index in sorted(fold_assignments["repeat_index"].unique()):
        repeat_rows = fold_assignments[fold_assignments["repeat_index"] == repeat_index]
        # The neuron mask is fixed within a repeat so folds differ only in trials.
        repeat_seed = base_seed + int(repeat_index)
        mask = create_neuron_mask(dataset.spikes.shape[2], heldout_fraction, seed=repeat_seed)
        heldin = np.flatnonzero(mask.heldin)
        heldout = np.flatnonzero(mask.heldout)
        for fold_index in sorted(repeat_rows["fold_index"].unique()):
            eval_trials = repeat_rows[repeat_rows["fold_index"] == fold_index][
                "trial_index"
            ].to_numpy()
            train_trials = repeat_rows[repeat_rows["fold_index"] != fold_index][
                "trial_index"
            ].to_numpy()
            train_counts = _counts(dataset, train_trials, heldout)
            eval_counts = _counts(dataset, eval_trials, heldout)
            reference = train_heldout_mean_rate_reference(train_counts, eval_counts.shape, scoring)
            predictions: dict[str, np.ndarray] = {}
            random_state = -1
            if TRAIN_MEAN_RATE in methods:
                predictions[TRAIN_MEAN_RATE] = compute_train_mean_rate_control(
                    train_counts, eval_counts.shape, scoring
                )["predicted_rates_hz"]
            if FACTOR_LATENT in methods:
                method = methods[FACTOR_LATENT]
                random_state = int(method.get("factor_analysis_random_state", 0))
                split = TrialSplit(train=train_trials, validation=eval_trials, test=eval_trials)
                predictions[FACTOR_LATENT] = factor_latent_heldout_rates(
                    dataset, split, heldin, heldout, scoring, _factor_settings(method), random_state
                )["validation"]
            if SPLIT_MEAN_RATE_INVALID in methods:
                predictions[SPLIT_MEAN_RATE_INVALID] = compute_split_mean_rate_invalid_control(
                    eval_counts, scoring
                )["predicted_rates_hz"]
            for method_name, predicted in predictions.items():
                method = methods[method_name]
                valid = bool(method.get("valid_model", False))
                scored = score_heldout_prediction(
                    eval_counts,
                    predicted,
                    reference,
                    scoring,
                    method_name,
                    "evaluation_fold",
                    "stratified_cv",
                    valid,
                )
                rows.append(
                    {
                        "repeat_index": int(repeat_index),
                        "fold_index": int(fold_index),
                        "split_seed": repeat_seed,
                        "neuron_mask_seed": repeat_seed,
                        "method_name": method_name,
                        "method_type": str(method.get("type", "")),
                        "valid_model": valid,
                        "reportable_as_model_performance": bool(
                            method.get("reportable_as_model_performance", False)
                        ),
                        "invalid_reason": str(method.get("invalid_reason", "")),
                        "train_trial_count": int(train_trials.size),
                        "eval_trial_count": int(eval_trials.size),
                        "unified_bits_per_spike": scored["bits_per_spike"],
                        "poisson_nll": scored["poisson_nll"],
                        "eval_spike_count": scored["spike_count"],
                        "eval_heldout_rate_hz": scored["observed_rate_hz"],
                        "factor_analysis_random_state": random_state
                        if method_name == FACTOR_LATENT
                        else -1,
                        "notes": str(method.get("notes", "")),
                    }
                )
    return pd.DataFrame(rows, columns=SCORE_COLUMNS)


def summarize_methods(
    scores: pd.DataFrame,
    repeats: int = 10000,
    confidence: float = 0.95,
    seed: int = 1337,
) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame(columns=METHOD_SUMMARY_COLUMNS)
    rows: list[dict[str, Any]] = []
    for method_name, group in scores.groupby("method_name", sort=True):
        values = group["unified_bits_per_spike"].to_numpy(dtype=np.float64)
        ci_low, ci_high = bootstrap_mean_ci(values, repeats, confidence, seed)
        rows.append(
            {
                "method_name": str(method_name),
                "method_type": str(group.iloc[0]["method_type"]),
                "valid_model": bool(group.iloc[0]["valid_model"]),
                "reportable_as_model_performance": bool(
                    group.iloc[0]["reportable_as_model_performance"]
                ),
                "n_folds": int(len(group)),
                "mean_unified_bits_per_spike": float(np.mean(values)),
                "std_unified_bits_per_spike": float(np.std(values, ddof=1))
                if values.size > 1
                else 0.0,
                "median_unified_bits_per_spike": float(np.median(values)),
                "min_unified_bits_per_spike": float(np.min(values)),
                "max_unified_bits_per_spike": float(np.max(values)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "positive_fraction": float(np.mean(values > 0.0)),
                "notes": str(group.iloc[0]["notes"]),
            }
        )
    return pd.DataFrame(rows, columns=METHOD_SUMMARY_COLUMNS)


def select_best_valid_method(method_summary: pd.DataFrame) -> str | None:
    """Best reportable valid model. Invalid controls and the reference can never win."""
    if method_summary.empty:
        return None
    eligible = method_summary[
        method_summary["valid_model"].astype(bool)
        & method_summary["reportable_as_model_performance"].astype(bool)
    ]
    if eligible.empty:
        return None
    ranked = eligible.sort_values("mean_unified_bits_per_spike", ascending=False, kind="mergesort")
    return str(ranked.iloc[0]["method_name"])


def _method_values(scores: pd.DataFrame, method_name: str) -> np.ndarray:
    rows = scores[scores["method_name"] == method_name]
    return np.asarray(rows["unified_bits_per_spike"].to_numpy(dtype=np.float64))


def compare_random_and_stratified(
    stratified_scores: pd.DataFrame, random_scores: pd.DataFrame
) -> dict[str, Any]:
    stratified = _method_values(stratified_scores, FACTOR_LATENT)
    random_values = _method_values(random_scores, FACTOR_LATENT)
    if stratified.size < 2 or random_values.size < 2:
        return {
            "stratified_factor_latent_mean": float("nan"),
            "stratified_factor_latent_std": float("nan"),
            "random_fold_factor_latent_mean": float("nan"),
            "random_fold_factor_latent_std": float("nan"),
            "stratification_reduces_variance": False,
            "variance_reduction_fraction": float("nan"),
        }
    stratified_variance = float(np.var(stratified, ddof=1))
    random_variance = float(np.var(random_values, ddof=1))
    reduction = (
        float(1.0 - stratified_variance / random_variance)
        if random_variance > 0.0
        else float("nan")
    )
    return {
        "stratified_factor_latent_mean": float(np.mean(stratified)),
        "stratified_factor_latent_std": float(np.std(stratified, ddof=1)),
        "random_fold_factor_latent_mean": float(np.mean(random_values)),
        "random_fold_factor_latent_std": float(np.std(random_values, ddof=1)),
        "stratified_factor_latent_variance": stratified_variance,
        "random_fold_factor_latent_variance": random_variance,
        "stratification_reduces_variance": bool(
            np.isfinite(reduction) and stratified_variance < random_variance
        ),
        "variance_reduction_fraction": reduction,
    }


def summarize_stratified_cv(
    scores: pd.DataFrame,
    method_summary: pd.DataFrame,
    fold_balance_summary: dict[str, Any],
    comparison: dict[str, Any],
    references: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    cross_validation = dict(config["cross_validation"])
    stratification = dict(cross_validation["stratification"])
    enabled = [key for key, value in stratification.items() if key.startswith("use_") and value]
    best_valid = select_best_valid_method(method_summary)

    def _summary_value(method_name: str, column: str) -> float:
        rows = method_summary[method_summary["method_name"] == method_name]
        return float("nan") if rows.empty else float(rows.iloc[0][column])

    return {
        "primary_metric": "unified_bits_per_spike",
        "reference_model": "train_heldout_mean_rate",
        "evaluation_metric_is_unweighted": True,
        "fold_count": int(cross_validation["fold_count"]),
        "repeats": int(cross_validation["repeats"]),
        "total_folds": int(cross_validation["fold_count"] * cross_validation["repeats"]),
        "assignment_method": str(cross_validation["assignment_method"]),
        "stratification_variables": enabled,
        "factor_latent_mean_unified_bits_per_spike": _summary_value(
            FACTOR_LATENT, "mean_unified_bits_per_spike"
        ),
        "factor_latent_std_unified_bits_per_spike": _summary_value(
            FACTOR_LATENT, "std_unified_bits_per_spike"
        ),
        "factor_latent_ci95_low": _summary_value(FACTOR_LATENT, "ci95_low"),
        "factor_latent_ci95_high": _summary_value(FACTOR_LATENT, "ci95_high"),
        "factor_latent_positive_fraction": _summary_value(FACTOR_LATENT, "positive_fraction"),
        "split_mean_rate_invalid_mean_unified_bits_per_spike": _summary_value(
            SPLIT_MEAN_RATE_INVALID, "mean_unified_bits_per_spike"
        ),
        "train_mean_rate_mean_unified_bits_per_spike": _summary_value(
            TRAIN_MEAN_RATE, "mean_unified_bits_per_spike"
        ),
        "best_valid_method": best_valid,
        "carried_forward_method": best_valid,
        "invalid_controls_excluded_from_valid_model_selection": True,
        "invalid_control_methods": sorted(
            str(name)
            for name in scores[~scores["valid_model"].astype(bool)]["method_name"].unique()
            if str(name) != TRAIN_MEAN_RATE
        ),
        "random_factor_latent_test_mean_reference": references.get(
            "repeated_split_factor_latent_test_mean"
        ),
        "random_factor_latent_test_positive_fraction_reference": references.get(
            "repeated_split_factor_latent_test_positive_fraction"
        ),
        **comparison,
        **fold_balance_summary,
        "single_split_results_reportable": False,
        "recommended_reporting_mode": "stratified_cross_validation",
        "old_incompatible_mean_rate_values_used_as_targets": False,
        "official_benchmark_claim": False,
    }
