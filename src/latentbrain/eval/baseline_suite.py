from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml

from latentbrain.data.schemas import NeuralDataset, TrialSplit
from latentbrain.data.splits import create_neuron_mask
from latentbrain.eval.cosmoothing import (
    fit_cosmoothing_ridge,
    fit_reduced_rank_cosmoothing,
    flatten_trial_time,
    predict_cosmoothing_rates,
)
from latentbrain.eval.rate_controls import (
    compute_split_mean_rate_invalid_control,
    compute_train_mean_rate_control,
)
from latentbrain.eval.recommended_window_cv import build_trial_aware_window_dataset
from latentbrain.eval.scoring import (
    ScoringConfig,
    score_heldout_prediction,
    train_heldout_mean_rate_reference,
)
from latentbrain.eval.seed_robustness import bootstrap_mean_ci
from latentbrain.eval.smoothing import smooth_spike_counts, spike_counts_to_rates_hz
from latentbrain.eval.split_audit import factor_latent_heldout_rates
from latentbrain.eval.sweeps import expand_grid
from latentbrain.paths import get_repo_root, resolve_configured_path

TRAIN_MEAN_RATE = "train_mean_rate"
SPLIT_MEAN_RATE_INVALID = "split_mean_rate_invalid"
FACTOR_LATENT_FIXED = "factor_latent_fixed"
REPRODUCTION_TOLERANCE = 1e-6

FORBIDDEN_OLD_PROTOCOLS = [
    "from_start_1p28s",
    "single_70_15_15_split",
    "seed_plus_run_index",
    "evaluation_target_calibration",
    "invalid_split_mean_as_model",
]
REQUIRED_CLAIM_SAFETY_RULES = [
    "single_split_results_reportable: false",
    "official_leaderboard_claim: false",
    "old_mean_rate_values_used_as_targets: false",
    "invalid_controls_excluded_from_model_selection: true",
]
REQUIRED_NEURAL_SEEDS = 5

INNER_SELECTION_COLUMNS = [
    "outer_repeat_index",
    "outer_fold_index",
    "method_name",
    "configuration_id",
    "inner_fold_index",
    "hyperparameters_json",
    "inner_unified_bits_per_spike",
    "inner_poisson_nll",
    "selection_eligible",
    "notes",
]
SELECTED_HYPERPARAMETER_COLUMNS = [
    "outer_repeat_index",
    "outer_fold_index",
    "method_name",
    "selected_configuration_id",
    "selected_hyperparameters_json",
    "mean_inner_unified_bits_per_spike",
    "std_inner_unified_bits_per_spike",
    "selection_rank",
    "tie_break_reason",
]
OUTER_SCORE_COLUMNS = [
    "repeat_index",
    "fold_index",
    "split_seed",
    "neuron_mask_seed",
    "method_name",
    "method_family",
    "valid_model",
    "reportable_as_model_performance",
    "invalid_reason",
    "selected_configuration_id",
    "selected_hyperparameters_json",
    "train_trial_count",
    "eval_trial_count",
    "unified_bits_per_spike",
    "poisson_nll",
    "eval_spike_count",
    "eval_heldout_rate_hz",
    "notes",
]
REPEAT_SCORE_COLUMNS = [
    "repeat_index",
    "neuron_mask_seed",
    "method_name",
    "mean_unified_bits_per_spike",
    "std_unified_bits_per_spike",
    "positive_fraction",
    "fold_count",
]
METHOD_SUMMARY_COLUMNS = [
    "method_name",
    "method_family",
    "valid_model",
    "reportable_as_model_performance",
    "n_outer_scores",
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
    "selected_as_baseline_to_beat",
    "selection_reason",
    "notes",
]
PAIRED_COMPARISON_COLUMNS = [
    "baseline_method",
    "comparison_method",
    "comparison_unit",
    "mean_paired_difference",
    "median_paired_difference",
    "ci95_low",
    "ci95_high",
    "positive_repeat_fraction",
    "superiority_supported",
    "comparison_interpretation",
]


@dataclass(frozen=True)
class OuterFold:
    """One accepted outer fold, with the neuron mask reused from the frozen protocol."""

    repeat_index: int
    fold_index: int
    split_seed: int
    neuron_mask_seed: int
    train_trials: np.ndarray
    eval_trials: np.ndarray
    heldin: np.ndarray
    heldout: np.ndarray


def _resolve(path_value: str) -> Path:
    return resolve_configured_path(str(path_value), get_repo_root())


def load_frozen_protocol(config: dict[str, Any]) -> dict[str, Any]:
    """Load the frozen recommended-window protocol; it is the only source of dataset paths."""
    path = _resolve(config["inputs"]["recommended_window_protocol_path"])
    if not path.exists():
        msg = f"frozen recommended-window protocol is missing: {path}"
        raise FileNotFoundError(msg)
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        msg = f"frozen protocol must contain a mapping: {path}"
        raise ValueError(msg)

    dataset = config["dataset"]
    if str(protocol["dataset"]["expected_hash"]) != str(dataset["expected_hash"]):
        msg = "baseline suite dataset hash does not match the frozen protocol"
        raise ValueError(msg)
    if str(protocol["window"]["name"]) != str(config["window"]["name"]):
        msg = "baseline suite window does not match the frozen protocol"
        raise ValueError(msg)
    if bool(protocol["trial_source"]["allow_global_crop_to_min"]):
        msg = "the frozen protocol must forbid the global crop_to_min source"
        raise ValueError(msg)
    if bool(config["trial_source"]["allow_global_crop_to_min"]):
        msg = (
            "trial_source.allow_global_crop_to_min must be false; the globally cropped array "
            "cannot source event-centered evaluation windows"
        )
        raise ValueError(msg)
    if not bool(protocol["binning"]["extract_before_rebin"]) or not bool(
        config["window"]["extract_before_rebin"]
    ):
        msg = "extraction must precede rebinning"
        raise ValueError(msg)
    if int(protocol["binning"]["target_bin_size_ms"]) != int(
        config["binning"]["target_bin_size_ms"]
    ):
        msg = "baseline suite target bin size does not match the frozen protocol"
        raise ValueError(msg)
    return protocol


def build_window_dataset(protocol: dict[str, Any]) -> tuple[NeuralDataset, str]:
    """Rebuild the exact frozen evaluation array from the trial-aware raw source."""
    built = build_trial_aware_window_dataset(
        {
            "dataset": protocol["dataset"],
            "trial_source": {**protocol["trial_source"], "require_exact_trial_lengths": True},
            "binning": protocol["binning"],
            "window": protocol["window"],
        }
    )
    return built["dataset"], str(built["dataset_hash"])


def load_outer_folds(
    config: dict[str, Any],
    protocol: dict[str, Any],
    n_neurons: int,
) -> list[OuterFold]:
    """Reuse the accepted outer fold assignments and per-repeat neuron masks verbatim."""
    outer = config["outer_cross_validation"]
    if not bool(outer["reuse_exact_assignments"]) or not bool(outer["reuse_exact_neuron_masks"]):
        msg = "outer_cross_validation must reuse the accepted assignments and neuron masks"
        raise ValueError(msg)
    path = _resolve(outer["source_assignments_path"])
    if not path.exists():
        msg = f"accepted outer fold assignments are missing: {path}"
        raise FileNotFoundError(msg)
    assignments = pd.read_csv(path)

    base_seed = int(outer["base_seed"])
    fold_count = int(outer["fold_count"])
    repeats = int(outer["repeats"])
    if sorted(assignments["repeat_index"].unique()) != list(range(repeats)):
        msg = "accepted assignments do not contain the configured repeats"
        raise ValueError(msg)
    if sorted(assignments["fold_index"].unique()) != list(range(fold_count)):
        msg = "accepted assignments do not contain the configured folds"
        raise ValueError(msg)
    heldout_fraction = float(protocol["cross_validation"]["heldout_neuron_fraction"])
    if int(protocol["cross_validation"]["base_seed"]) != base_seed:
        msg = "baseline suite base_seed does not match the frozen protocol"
        raise ValueError(msg)

    folds: list[OuterFold] = []
    for repeat_index in range(repeats):
        rows = assignments[assignments["repeat_index"] == repeat_index]
        seeds = rows["seed"].unique()
        if len(seeds) != 1 or int(seeds[0]) != base_seed + repeat_index:
            msg = f"repeat {repeat_index} split seed does not match the accepted assignments"
            raise ValueError(msg)
        repeat_seed = base_seed + repeat_index
        mask = create_neuron_mask(n_neurons, heldout_fraction, seed=repeat_seed)
        heldin = np.flatnonzero(mask.heldin)
        heldout = np.flatnonzero(mask.heldout)
        for fold_index in range(fold_count):
            eval_trials = rows[rows["fold_index"] == fold_index]["trial_index"].to_numpy()
            train_trials = rows[rows["fold_index"] != fold_index]["trial_index"].to_numpy()
            folds.append(
                OuterFold(
                    repeat_index=repeat_index,
                    fold_index=fold_index,
                    split_seed=repeat_seed,
                    neuron_mask_seed=repeat_seed,
                    train_trials=np.sort(train_trials),
                    eval_trials=np.sort(eval_trials),
                    heldin=heldin,
                    heldout=heldout,
                )
            )
    return folds


def _scoring(config: dict[str, Any]) -> ScoringConfig:
    scoring = config["scoring"]
    return ScoringConfig(
        bin_size_ms=int(config["binning"]["target_bin_size_ms"]),
        include_poisson_constant=bool(scoring["include_poisson_constant"]),
        min_rate_hz=float(scoring["min_rate_hz"]),
        max_rate_hz=float(scoring["max_rate_hz"]),
        reference_name=str(scoring["reference_model"]),
    )


class _SmoothingCache:
    """Smoothing has no learned statistics, so it is cached per sigma across every fold."""

    def __init__(self, spikes: np.ndarray, bin_size_ms: int) -> None:
        self._spikes = spikes
        self._bin_size_ms = bin_size_ms
        self._cache: dict[tuple[bytes, float], np.ndarray] = {}

    def rates(self, heldin: np.ndarray, sigma_ms: float) -> np.ndarray:
        # Key on the whole held-in index set. A digest of its size or first element collides
        # across repeats, and a collision would feed one repeat's held-out neurons into another
        # repeat's features: direct target leakage.
        key = (np.asarray(heldin, dtype=np.int64).tobytes(), float(sigma_ms))
        cached = self._cache.get(key)
        if cached is None:
            smoothed = smooth_spike_counts(
                self._spikes[:, :, heldin],
                self._bin_size_ms,
                method="gaussian",
                sigma_ms=float(sigma_ms),
                truncate=4.0,
            )
            cached = spike_counts_to_rates_hz(smoothed, self._bin_size_ms)
            self._cache[key] = cached
        return cached


def _counts(dataset: NeuralDataset, trials: np.ndarray, neurons: np.ndarray) -> np.ndarray:
    return np.asarray(dataset.spikes[np.isin(dataset.trial_ids, trials)][:, :, neurons])


def predict_ridge_family(
    dataset: NeuralDataset,
    cache: _SmoothingCache,
    train_trials: np.ndarray,
    eval_trials: np.ndarray,
    heldin: np.ndarray,
    heldout: np.ndarray,
    scoring: ScoringConfig,
    hyperparameters: dict[str, Any],
    reduced_rank: bool,
) -> np.ndarray:
    """Fit smoothing-ridge (optionally reduced rank) on training trials only, predict eval."""
    rates = cache.rates(heldin, float(hyperparameters["smoothing_sigma_ms"]))
    train_mask = np.isin(dataset.trial_ids, train_trials)
    eval_mask = np.isin(dataset.trial_ids, eval_trials)
    train_x = flatten_trial_time(rates[train_mask])
    train_y = flatten_trial_time(dataset.spikes[train_mask][:, :, heldout].astype(np.float64))
    bin_size_ms = scoring.bin_size_ms
    alpha = float(hyperparameters["alpha"])
    standardize = bool(hyperparameters["standardize_features"])
    intercept = bool(hyperparameters["fit_intercept"])
    model = (
        fit_reduced_rank_cosmoothing(
            train_x,
            train_y,
            bin_size_ms,
            alpha,
            int(hyperparameters["rank"]),
            scoring.min_rate_hz,
            scoring.max_rate_hz,
            standardize_features=standardize,
            fit_intercept=intercept,
        )
        if reduced_rank
        else fit_cosmoothing_ridge(
            train_x,
            train_y,
            bin_size_ms,
            alpha,
            scoring.min_rate_hz,
            scoring.max_rate_hz,
            standardize_features=standardize,
            fit_intercept=intercept,
        )
    )
    return predict_cosmoothing_rates(
        rates[eval_mask], model, scoring.min_rate_hz, scoring.max_rate_hz
    )


def predict_factor_latent(
    dataset: NeuralDataset,
    train_trials: np.ndarray,
    eval_trials: np.ndarray,
    heldin: np.ndarray,
    heldout: np.ndarray,
    scoring: ScoringConfig,
    hyperparameters: dict[str, Any],
) -> np.ndarray:
    split = TrialSplit(train=train_trials, validation=eval_trials, test=eval_trials)
    settings = {
        "latent_dim": float(hyperparameters["latent_dim"]),
        "smoothing_sigma_ms": float(hyperparameters["smoothing_sigma_ms"]),
        "heldout_decoder_alpha": float(hyperparameters["heldout_decoder_alpha"]),
        "max_iter": 1000.0,
        "tol": 1.0e-4,
    }
    predictions = factor_latent_heldout_rates(
        dataset,
        split,
        heldin,
        heldout,
        scoring,
        settings,
        int(hyperparameters["factor_analysis_random_state"]),
    )
    return np.asarray(predictions["validation"])


def predict_method(
    method: dict[str, Any],
    dataset: NeuralDataset,
    cache: _SmoothingCache,
    train_trials: np.ndarray,
    eval_trials: np.ndarray,
    heldin: np.ndarray,
    heldout: np.ndarray,
    scoring: ScoringConfig,
    hyperparameters: dict[str, Any],
) -> np.ndarray:
    family = str(method["family"])
    eval_counts = _counts(dataset, eval_trials, heldout)
    if family == "reference":
        train_counts = _counts(dataset, train_trials, heldout)
        rates = compute_train_mean_rate_control(train_counts, eval_counts.shape, scoring)
        return np.asarray(rates["predicted_rates_hz"])
    if family == "invalid_control":
        # The invalid control reads the evaluation fold's own targets. Diagnostic only.
        return np.asarray(
            compute_split_mean_rate_invalid_control(eval_counts, scoring)["predicted_rates_hz"]
        )
    if family == "factor_latent":
        return predict_factor_latent(
            dataset, train_trials, eval_trials, heldin, heldout, scoring, hyperparameters
        )
    if family in ("ridge", "reduced_rank"):
        return predict_ridge_family(
            dataset,
            cache,
            train_trials,
            eval_trials,
            heldin,
            heldout,
            scoring,
            hyperparameters,
            reduced_rank=family == "reduced_rank",
        )
    msg = f"unknown method family: {family}"
    raise ValueError(msg)


def _score(
    dataset: NeuralDataset,
    train_trials: np.ndarray,
    eval_trials: np.ndarray,
    heldout: np.ndarray,
    predicted: np.ndarray,
    scoring: ScoringConfig,
    method_name: str,
    valid_model: bool,
) -> dict[str, Any]:
    train_counts = _counts(dataset, train_trials, heldout)
    eval_counts = _counts(dataset, eval_trials, heldout)
    reference = train_heldout_mean_rate_reference(train_counts, eval_counts.shape, scoring)
    return score_heldout_prediction(
        eval_counts,
        predicted,
        reference,
        scoring,
        method_name,
        "evaluation_fold",
        "nested_baseline_suite",
        valid_model,
    )


def method_configurations(
    method: dict[str, Any], n_heldin: int, n_heldout: int
) -> list[dict[str, Any]]:
    """Finite configuration grid for one method, validated against the data dimensions."""
    if "hyperparameters" in method:
        return [dict(method["hyperparameters"])]
    search = method.get("search")
    if not search:
        return [{}]
    configurations = expand_grid(dict(search))
    max_rank = min(n_heldin, n_heldout)
    for configuration in configurations:
        rank = configuration.get("rank")
        if rank is not None and int(rank) > max_rank:
            msg = f"rank {rank} exceeds the maximum usable rank {max_rank}"
            raise ValueError(msg)
    return configurations


def inner_folds(train_trials: np.ndarray, fold_count: int, seed: int) -> list[np.ndarray]:
    """Deterministic inner folds built only from outer-training trials."""
    if fold_count < 2:
        msg = "inner_selection.fold_count must be at least 2"
        raise ValueError(msg)
    order = np.random.default_rng(seed).permutation(train_trials.size)
    return [np.sort(train_trials[order[index::fold_count]]) for index in range(fold_count)]


def _complexity(hyperparameters: dict[str, Any]) -> float:
    return float(hyperparameters.get("latent_dim", hyperparameters.get("rank", 0)))


def select_configuration(
    inner_scores: pd.DataFrame,
    configurations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Highest mean inner score, tie-broken by complexity, smoothing, then grid index."""
    grouped = inner_scores.groupby("configuration_id")["inner_unified_bits_per_spike"]
    means = grouped.mean()
    stds = grouped.std(ddof=1).fillna(0.0)
    indices = [int(index) for index in means.index]
    ranked = sorted(
        indices,
        key=lambda index: (
            -float(means[index]),
            _complexity(configurations[index]),
            float(configurations[index].get("smoothing_sigma_ms", 0.0)),
            index,
        ),
    )
    best = ranked[0]
    tied = [index for index in indices if float(means[index]) == float(means[best])]
    reasons = []
    if len(tied) > 1:
        reasons = ["lower_model_complexity", "lower_smoothing_sigma", "lower_search_index"]
    return {
        "selected_configuration_id": best,
        "selected_hyperparameters_json": json.dumps(configurations[best], sort_keys=True),
        "mean_inner_unified_bits_per_spike": float(means[best]),
        "std_inner_unified_bits_per_spike": float(stds[best]),
        "selection_rank": 1,
        "tie_break_reason": "; ".join(reasons) if reasons else "none",
    }


def run_baseline_suite(config: dict[str, Any]) -> dict[str, Any]:
    """Nested train-only selection over the frozen outer folds. No neural model is trained."""
    protocol = load_frozen_protocol(config)
    dataset, dataset_hash = build_window_dataset(protocol)
    folds = load_outer_folds(config, protocol, int(dataset.spikes.shape[2]))
    scoring = _scoring(config)
    cache = _SmoothingCache(dataset.spikes, scoring.bin_size_ms)
    inner = config["inner_selection"]
    inner_enabled = bool(inner["enabled"])
    inner_count = int(inner["fold_count"])
    inner_seed = int(inner["base_seed"])

    methods = [dict(method) for method in config["methods"]]
    configurations = {
        str(method["name"]): method_configurations(
            method, int(folds[0].heldin.size), int(folds[0].heldout.size)
        )
        for method in methods
    }

    inner_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []
    for fold in folds:
        inner_partitions = inner_folds(
            fold.train_trials,
            inner_count,
            inner_seed + fold.repeat_index * 100 + fold.fold_index,
        )
        for method in methods:
            name = str(method["name"])
            grid = configurations[name]
            valid = bool(method["valid_model"])
            needs_selection = inner_enabled and "search" in method and len(grid) > 1
            selection: dict[str, Any] = {
                "selected_configuration_id": 0,
                "selected_hyperparameters_json": json.dumps(grid[0], sort_keys=True),
                "mean_inner_unified_bits_per_spike": float("nan"),
                "std_inner_unified_bits_per_spike": float("nan"),
                "selection_rank": 1,
                "tie_break_reason": "fixed_configuration",
            }
            if needs_selection:
                scores: list[dict[str, Any]] = []
                for inner_index, inner_eval in enumerate(inner_partitions):
                    inner_train = np.setdiff1d(fold.train_trials, inner_eval)
                    for configuration_id, hyperparameters in enumerate(grid):
                        predicted = predict_method(
                            method,
                            dataset,
                            cache,
                            inner_train,
                            inner_eval,
                            fold.heldin,
                            fold.heldout,
                            scoring,
                            hyperparameters,
                        )
                        scored = _score(
                            dataset,
                            inner_train,
                            inner_eval,
                            fold.heldout,
                            predicted,
                            scoring,
                            name,
                            valid,
                        )
                        scores.append(
                            {
                                "outer_repeat_index": fold.repeat_index,
                                "outer_fold_index": fold.fold_index,
                                "method_name": name,
                                "configuration_id": configuration_id,
                                "inner_fold_index": inner_index,
                                "hyperparameters_json": json.dumps(hyperparameters, sort_keys=True),
                                "inner_unified_bits_per_spike": scored["bits_per_spike"],
                                "inner_poisson_nll": scored["poisson_nll"],
                                "selection_eligible": valid,
                                "notes": "inner folds are built only from outer-training trials",
                            }
                        )
                inner_frame = pd.DataFrame(scores)
                inner_rows.extend(scores)
                selection = select_configuration(inner_frame, grid)
            if needs_selection or "search" in method:
                selected_rows.append(
                    {
                        "outer_repeat_index": fold.repeat_index,
                        "outer_fold_index": fold.fold_index,
                        "method_name": name,
                        **selection,
                    }
                )

            hyperparameters = grid[int(selection["selected_configuration_id"])]
            predicted = predict_method(
                method,
                dataset,
                cache,
                fold.train_trials,
                fold.eval_trials,
                fold.heldin,
                fold.heldout,
                scoring,
                hyperparameters,
            )
            scored = _score(
                dataset,
                fold.train_trials,
                fold.eval_trials,
                fold.heldout,
                predicted,
                scoring,
                name,
                valid,
            )
            outer_rows.append(
                {
                    "repeat_index": fold.repeat_index,
                    "fold_index": fold.fold_index,
                    "split_seed": fold.split_seed,
                    "neuron_mask_seed": fold.neuron_mask_seed,
                    "method_name": name,
                    "method_family": str(method["family"]),
                    "valid_model": valid,
                    "reportable_as_model_performance": bool(
                        method["reportable_as_model_performance"]
                    ),
                    "invalid_reason": str(method.get("invalid_reason", "")),
                    "selected_configuration_id": int(selection["selected_configuration_id"]),
                    "selected_hyperparameters_json": json.dumps(hyperparameters, sort_keys=True),
                    "train_trial_count": int(fold.train_trials.size),
                    "eval_trial_count": int(fold.eval_trials.size),
                    "unified_bits_per_spike": scored["bits_per_spike"],
                    "poisson_nll": scored["poisson_nll"],
                    "eval_spike_count": scored["spike_count"],
                    "eval_heldout_rate_hz": scored["observed_rate_hz"],
                    "notes": str(method.get("notes", "")),
                }
            )

    return {
        "dataset": dataset,
        "dataset_hash": dataset_hash,
        "protocol": protocol,
        "configurations": configurations,
        "outer_scores": pd.DataFrame(outer_rows, columns=OUTER_SCORE_COLUMNS),
        "inner_selection": pd.DataFrame(inner_rows, columns=INNER_SELECTION_COLUMNS)
        if inner_rows
        else pd.DataFrame(columns=INNER_SELECTION_COLUMNS),
        "selected_hyperparameters": pd.DataFrame(
            selected_rows, columns=SELECTED_HYPERPARAMETER_COLUMNS
        )
        if selected_rows
        else pd.DataFrame(columns=SELECTED_HYPERPARAMETER_COLUMNS),
    }


def build_repeat_level_scores(outer_scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate folds to repeats; folds inside a repeat share a neuron mask."""
    rows: list[dict[str, Any]] = []
    for (repeat_index, method_name), group in outer_scores.groupby(
        ["repeat_index", "method_name"], sort=True
    ):
        values = group["unified_bits_per_spike"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "repeat_index": int(repeat_index),
                "neuron_mask_seed": int(group.iloc[0]["neuron_mask_seed"]),
                "method_name": str(method_name),
                "mean_unified_bits_per_spike": float(np.mean(values)),
                "std_unified_bits_per_spike": float(np.std(values, ddof=1))
                if values.size > 1
                else 0.0,
                "positive_fraction": float(np.mean(values > 0.0)),
                "fold_count": int(values.size),
            }
        )
    return pd.DataFrame(rows, columns=REPEAT_SCORE_COLUMNS)


def hierarchical_paired_bootstrap(
    paired: pd.DataFrame,
    repeats: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    """Resample repeats, then folds within each drawn repeat. Deterministic given `seed`.

    Folds inside a repeat share a neuron mask and overlapping training trials, so they are not
    independent samples. Resampling both levels keeps the interval honest.
    """
    groups = [
        group["difference"].to_numpy(dtype=np.float64)
        for _, group in paired.groupby("repeat_index", sort=True)
    ]
    if not groups:
        return (float("nan"), float("nan"))
    if len(groups) == 1:
        return bootstrap_mean_ci(groups[0], repeats, confidence, seed)
    generator = np.random.default_rng(seed)
    n_repeats = len(groups)
    means = np.empty(repeats, dtype=np.float64)
    for draw in range(repeats):
        chosen = generator.integers(0, n_repeats, size=n_repeats)
        repeat_means = np.empty(n_repeats, dtype=np.float64)
        for position, index in enumerate(chosen):
            values = groups[int(index)]
            folds = generator.integers(0, values.size, size=values.size)
            repeat_means[position] = values[folds].mean()
        means[draw] = repeat_means.mean()
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return (float(low), float(high))


def build_paired_comparisons(
    outer_scores: pd.DataFrame,
    repeat_scores: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Repeat-level paired comparisons against the incumbent baseline. Invalid controls excluded."""
    statistics = config["statistics"]
    selection = config["selection"]
    baseline = str(selection["baseline_to_beat"])
    minimum_fraction = float(selection["require_positive_fraction_minimum"])
    require_ci = bool(selection["require_positive_paired_ci"])

    valid_methods = sorted(
        str(name)
        for name in outer_scores[outer_scores["reportable_as_model_performance"].astype(bool)][
            "method_name"
        ].unique()
        if str(name) != baseline
    )
    baseline_folds = outer_scores[outer_scores["method_name"] == baseline]
    baseline_repeats = repeat_scores[repeat_scores["method_name"] == baseline].set_index(
        "repeat_index"
    )["mean_unified_bits_per_spike"]

    rows: list[dict[str, Any]] = []
    for method_name in valid_methods:
        folds = outer_scores[outer_scores["method_name"] == method_name]
        merged = folds.merge(
            baseline_folds,
            on=["repeat_index", "fold_index"],
            suffixes=("", "_baseline"),
        )
        paired = pd.DataFrame(
            {
                "repeat_index": merged["repeat_index"],
                "difference": merged["unified_bits_per_spike"]
                - merged["unified_bits_per_spike_baseline"],
            }
        )
        method_repeats = repeat_scores[repeat_scores["method_name"] == method_name].set_index(
            "repeat_index"
        )["mean_unified_bits_per_spike"]
        repeat_differences = (method_repeats - baseline_repeats).to_numpy(dtype=np.float64)
        ci_low, ci_high = hierarchical_paired_bootstrap(
            paired,
            int(statistics["bootstrap_repeats"]),
            float(statistics["confidence_interval"]),
            int(statistics["bootstrap_seed"]),
        )
        mean_difference = float(np.mean(repeat_differences))
        positive_fraction = float(np.mean(repeat_differences > 0.0))
        supported = bool(
            mean_difference > 0.0
            and (ci_low > 0.0 if require_ci else True)
            and positive_fraction >= minimum_fraction
        )
        interpretation = (
            f"{method_name} beats {baseline} on the paired repeat-level mean with a positive "
            "bootstrap interval; replacement is supported."
            if supported
            else f"{method_name} does not clear every replacement gate against {baseline}; "
            f"{baseline} is retained."
        )
        rows.append(
            {
                "baseline_method": baseline,
                "comparison_method": method_name,
                "comparison_unit": str(statistics["comparison_unit"]),
                "mean_paired_difference": mean_difference,
                "median_paired_difference": float(np.median(repeat_differences)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "positive_repeat_fraction": positive_fraction,
                "superiority_supported": supported,
                "comparison_interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows, columns=PAIRED_COMPARISON_COLUMNS)


def _repeat_std(values: pd.DataFrame) -> tuple[float, float]:
    grouped = values.groupby("repeat_index")["unified_bits_per_spike"]
    means = grouped.mean().to_numpy(dtype=np.float64)
    stds = grouped.std(ddof=1).to_numpy(dtype=np.float64)
    between = float(np.std(means, ddof=1)) if means.size > 1 else 0.0
    within = float(np.nanmean(stds)) if stds.size else float("nan")
    return (between, within)


def choose_baseline_to_beat(
    comparisons: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Replace the incumbent only when every declared gate passes. Otherwise retain it."""
    baseline = str(config["selection"]["baseline_to_beat"])
    if comparisons.empty:
        return {"baseline_to_beat": baseline, "baseline_replaced": False, "supported": False}
    winners = comparisons[comparisons["superiority_supported"].astype(bool)]
    if winners.empty:
        return {"baseline_to_beat": baseline, "baseline_replaced": False, "supported": False}
    best = winners.sort_values("mean_paired_difference", ascending=False, kind="mergesort").iloc[0]
    return {
        "baseline_to_beat": str(best["comparison_method"]),
        "baseline_replaced": True,
        "supported": True,
    }


def build_method_summary(
    outer_scores: pd.DataFrame,
    baseline_choice: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    statistics = config["statistics"]
    baseline = str(baseline_choice["baseline_to_beat"])
    rows: list[dict[str, Any]] = []
    for method_name, group in outer_scores.groupby("method_name", sort=True):
        values = group["unified_bits_per_spike"].to_numpy(dtype=np.float64)
        ci_low, ci_high = bootstrap_mean_ci(
            values,
            int(statistics["bootstrap_repeats"]),
            float(statistics["confidence_interval"]),
            int(statistics["bootstrap_seed"]),
        )
        between, within = _repeat_std(group)
        selected = str(method_name) == baseline
        if selected:
            reason = (
                "replaced the incumbent after clearing every paired gate"
                if baseline_choice["baseline_replaced"]
                else "retained: no valid method cleared every paired replacement gate"
            )
        elif not bool(group.iloc[0]["reportable_as_model_performance"]):
            reason = "not reportable as model performance; excluded from baseline selection"
        else:
            reason = "valid baseline that did not clear every paired replacement gate"
        rows.append(
            {
                "method_name": str(method_name),
                "method_family": str(group.iloc[0]["method_family"]),
                "valid_model": bool(group.iloc[0]["valid_model"]),
                "reportable_as_model_performance": bool(
                    group.iloc[0]["reportable_as_model_performance"]
                ),
                "n_outer_scores": int(values.size),
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
                "between_repeat_std": between,
                "within_repeat_std": within,
                "selected_as_baseline_to_beat": selected,
                "selection_reason": reason,
                "notes": str(group.iloc[0]["notes"]),
            }
        )
    return pd.DataFrame(rows, columns=METHOD_SUMMARY_COLUMNS)


def _accepted_factor_latent_mean(config: dict[str, Any]) -> float:
    path = _resolve(config["inputs"]["recommended_window_summary_path"])
    if not path.exists():
        msg = f"accepted recommended-window summary is missing: {path}"
        raise FileNotFoundError(msg)
    summary = json.loads(path.read_text(encoding="utf-8"))
    return float(summary["factor_latent_mean"])


def build_readiness(
    summary: dict[str, Any],
    config: dict[str, Any],
    method_summary: pd.DataFrame,
) -> dict[str, Any]:
    """Readiness plan for the later neural reevaluation. No neural model was trained here."""
    blockers: list[str] = []
    if not bool(summary["factor_latent_reproduced"]):
        blockers.append(
            "factor_latent_fixed did not reproduce the accepted recommended-window mean"
        )
    if not bool(summary["outer_assignments_reused"]) or not bool(summary["neuron_masks_reused"]):
        blockers.append("outer fold assignments or neuron masks were not reused verbatim")
    if bool(summary["official_leaderboard_claim"]) or bool(
        summary["single_split_results_reportable"]
    ):
        blockers.append("claim-safety flags are not satisfied")
    baseline = str(summary["baseline_to_beat"])
    rows = method_summary[method_summary["method_name"] == baseline]
    if rows.empty or not bool(rows.iloc[0]["reportable_as_model_performance"]):
        blockers.append("the baseline to beat is not a reportable valid model")
    return {
        "ready": not blockers,
        "dataset_hash": summary["dataset_hash"],
        "window_name": summary["window_name"],
        "target_bin_size_ms": summary["target_bin_size_ms"],
        "fold_assignment_source": str(config["outer_cross_validation"]["source_assignments_path"]),
        "neuron_mask_source": "recreated from the frozen protocol base_seed and repeat index",
        "baseline_to_beat": baseline,
        "baseline_mean": summary["best_valid_method_mean"]
        if summary["baseline_replaced"]
        else summary["factor_latent_fixed_mean"],
        "baseline_ci95_low": float(rows.iloc[0]["ci95_low"]) if not rows.empty else float("nan"),
        "baseline_ci95_high": float(rows.iloc[0]["ci95_high"]) if not rows.empty else float("nan"),
        "comparison_unit": str(config["statistics"]["comparison_unit"]),
        "required_neural_seeds": REQUIRED_NEURAL_SEEDS,
        "required_outer_repeats": int(config["outer_cross_validation"]["repeats"]),
        "required_claim_safety_rules": REQUIRED_CLAIM_SAFETY_RULES,
        "required_checkpoint_selection_metric": "unified_bits_per_spike on inner-training folds",
        "forbidden_old_protocols": FORBIDDEN_OLD_PROTOCOLS,
        "blockers": blockers,
        "neural_experiment_run_during_this_milestone": False,
        "note": "Readiness plan only. No neural model was trained, tuned, or scored here.",
    }


def summarize_baseline_suite(
    result: dict[str, Any],
    repeat_scores: pd.DataFrame,
    comparisons: pd.DataFrame,
    baseline_choice: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    outer = result["outer_scores"]
    fixed = outer[outer["method_name"] == FACTOR_LATENT_FIXED]["unified_bits_per_spike"]
    fixed_mean = float(fixed.mean()) if not fixed.empty else float("nan")
    accepted_mean = _accepted_factor_latent_mean(config)
    difference = float(fixed_mean - accepted_mean)
    reproduced = bool(np.isfinite(difference) and abs(difference) <= REPRODUCTION_TOLERANCE)

    reportable = outer[outer["reportable_as_model_performance"].astype(bool)]
    means = reportable.groupby("method_name")["unified_bits_per_spike"].mean()
    best_method = str(means.idxmax()) if not means.empty else ""
    baseline = str(baseline_choice["baseline_to_beat"])
    against_baseline = comparisons[comparisons["comparison_method"] == best_method]

    summary: dict[str, Any] = {
        "dataset_name": str(config["dataset"]["name"]),
        "dataset_hash": result["dataset_hash"],
        "window_name": str(config["window"]["name"]),
        "protocol_source": str(config["inputs"]["recommended_window_protocol_path"]),
        "outer_assignments_reused": True,
        "neuron_masks_reused": True,
        "inner_selection_enabled": bool(config["inner_selection"]["enabled"]),
        "inner_fold_count": int(config["inner_selection"]["fold_count"]),
        "outer_fold_count": int(config["outer_cross_validation"]["fold_count"]),
        "outer_repeats": int(config["outer_cross_validation"]["repeats"]),
        "total_outer_evaluations": int(
            len(outer[["repeat_index", "fold_index"]].drop_duplicates())
        ),
        "factor_latent_fixed_mean": fixed_mean,
        "factor_latent_accepted_mean": accepted_mean,
        "factor_latent_reproduced": reproduced,
        "factor_latent_reproduction_difference": difference,
        "valid_methods": sorted(reportable["method_name"].unique().tolist()),
        "best_valid_method": best_method,
        "best_valid_method_mean": float(means.max()) if not means.empty else float("nan"),
        "baseline_to_beat": baseline,
        "baseline_replaced": bool(baseline_choice["baseline_replaced"]),
        "baseline_replacement_supported": bool(baseline_choice["supported"]),
        "paired_ci_against_factor_latent": [
            float(against_baseline.iloc[0]["ci95_low"]),
            float(against_baseline.iloc[0]["ci95_high"]),
        ]
        if not against_baseline.empty
        else None,
        "paired_difference_against_factor_latent": float(
            against_baseline.iloc[0]["mean_paired_difference"]
        )
        if not against_baseline.empty
        else None,
        "positive_repeat_fraction_against_factor_latent": float(
            against_baseline.iloc[0]["positive_repeat_fraction"]
        )
        if not against_baseline.empty
        else None,
        "split_mean_invalid_mean": float(
            outer[outer["method_name"] == SPLIT_MEAN_RATE_INVALID]["unified_bits_per_spike"].mean()
        ),
        "train_mean_rate_mean": float(
            outer[outer["method_name"] == TRAIN_MEAN_RATE]["unified_bits_per_spike"].mean()
        ),
        "invalid_controls_excluded": True,
        "single_split_results_reportable": False,
        "official_leaderboard_claim": False,
        "old_mean_rate_values_used_as_targets": False,
        "cross_dataset_performance_comparison_claimed": False,
        "recommended_reporting_mode": "recommended_window_stratified_cross_validation",
        "target_bin_size_ms": int(config["binning"]["target_bin_size_ms"]),
        "comparison_unit": str(config["statistics"]["comparison_unit"]),
        "hierarchical_bootstrap": bool(config["statistics"]["hierarchical_bootstrap"]),
        "naive_independent_fold_test_used": False,
    }
    return summary


def build_baseline_protocol(config: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": dict(config["dataset"]),
        "trial_source": dict(config["trial_source"]),
        "window": dict(config["window"]),
        "binning": dict(config["binning"]),
        "outer_cross_validation": dict(config["outer_cross_validation"]),
        "inner_selection": dict(config["inner_selection"]),
        "scoring": dict(config["scoring"]),
        "statistics": dict(config["statistics"]),
        "selection": dict(config["selection"]),
        "baseline_to_beat": summary["baseline_to_beat"],
        "baseline_mean": summary["best_valid_method_mean"]
        if summary["baseline_replaced"]
        else summary["factor_latent_fixed_mean"],
        "claim_safety": {
            "single_split_results_reportable": False,
            "official_leaderboard_claim": False,
            "old_mean_rate_values_used_as_targets": False,
            "invalid_controls_excluded_from_model_selection": True,
        },
        "protocol_frozen": True,
    }
