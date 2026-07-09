from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from latentbrain.data.schemas import NeuralDataset, TrialSplit
from latentbrain.data.splits import create_neuron_mask, create_trial_split
from latentbrain.eval.rate_controls import (
    FACTOR_LATENT,
    ORACLE_SPLIT_SCALED_FACTOR_LATENT_INVALID,
    SPLIT_MEAN_RATE_INVALID,
    TRAIN_MEAN_RATE,
    TRAIN_PER_NEURON_MEAN_RATE,
    TRAIN_POPULATION_SCALED_MEAN_RATE,
    TRAIN_RATE_CALIBRATED_FACTOR_LATENT,
    apply_rate_calibration,
    compute_oracle_split_scaled_factor_latent_invalid_control,
    compute_split_mean_rate_invalid_control,
    compute_train_mean_rate_control,
    compute_train_per_neuron_mean_rate_control,
    compute_train_population_scaled_mean_rate_control,
    compute_train_rate_calibration,
    select_best_valid_method,
)
from latentbrain.eval.scoring import (
    ScoringConfig,
    score_heldout_prediction,
    train_heldout_mean_rate_reference,
)
from latentbrain.eval.seed_robustness import bootstrap_mean_ci
from latentbrain.eval.split_audit import factor_latent_heldout_rates

REPEATED_SPLIT_COLUMNS = [
    "split_seed",
    "factor_analysis_random_state",
    "method_name",
    "valid_model",
    "validation_unified_bits_per_spike",
    "test_unified_bits_per_spike",
    "validation_poisson_nll",
    "test_poisson_nll",
    "validation_heldout_rate_hz",
    "test_heldout_rate_hz",
    "validation_trial_count",
    "test_trial_count",
    "notes",
]

FA_SENSITIVITY_COLUMNS = [
    "split_seed",
    "factor_analysis_random_state",
    "validation_unified_bits_per_spike",
    "test_unified_bits_per_spike",
    "validation_poisson_nll",
    "test_poisson_nll",
    "difference_from_random_state_0_validation",
    "difference_from_random_state_0_test",
    "notes",
]

RATE_CONTROL_COLUMNS = [
    "split_seed",
    "split",
    "method_name",
    "valid_model",
    "invalid_reason",
    "unified_bits_per_spike",
    "poisson_nll",
    "heldout_rate_hz",
    "predicted_rate_hz",
    "rate_error_hz",
    "notes",
]

DECOMPOSITION_COLUMNS = [
    "split_seed",
    "split",
    "factor_latent_bits_per_spike",
    "train_rate_calibrated_factor_latent_bits_per_spike",
    "split_mean_rate_invalid_bits_per_spike",
    "oracle_split_scaled_factor_latent_invalid_bits_per_spike",
    "valid_calibration_gain",
    "invalid_oracle_gain",
    "split_mean_advantage_over_factor_latent",
    "rate_offset_explains_gap",
    "notes",
]

METHOD_SUMMARY_COLUMNS = [
    "method_name",
    "valid_model",
    "n_scores",
    "mean_validation_unified_bits_per_spike",
    "std_validation_unified_bits_per_spike",
    "mean_test_unified_bits_per_spike",
    "std_test_unified_bits_per_spike",
    "ci95_low_test",
    "ci95_high_test",
    "test_positive_fraction",
    "notes",
]

EVALUATION_SPLITS = ("validation", "test")

# A pure rescaling that recovers at least this share of the invalid split-mean advantage means
# the advantage is dominated by a split-level rate offset rather than by trial structure.
RATE_OFFSET_EXPLAINS_THRESHOLD = 0.5

# Test-mean gains below this are numerical noise, not improvements worth reporting.
NEGLIGIBLE_GAIN = 1.0e-4


def _factor_latent_settings(config: dict[str, Any]) -> dict[str, float]:
    settings = dict(config["factor_latent"])
    return {
        "latent_dim": float(settings["latent_dim"]),
        "smoothing_sigma_ms": float(settings["smoothing_sigma_ms"]),
        "heldout_decoder_alpha": float(settings["heldout_decoder_alpha"]),
        "max_iter": 1000.0,
        "tol": 1.0e-4,
    }


def _scoring_config(config: dict[str, Any]) -> ScoringConfig:
    scoring = config["scoring"]
    return ScoringConfig(
        bin_size_ms=int(config["binning"]["target_bin_size_ms"]),
        include_poisson_constant=bool(scoring["include_poisson_constant"]),
        min_rate_hz=float(scoring["min_rate_hz"]),
        max_rate_hz=float(scoring["max_rate_hz"]),
        reference_name=str(scoring["reference_model"]),
    )


def _split_and_indices(
    dataset: NeuralDataset, config: dict[str, Any], split_seed: int
) -> tuple[TrialSplit, np.ndarray, np.ndarray]:
    splits = config["splits"]
    split = create_trial_split(
        dataset.trial_ids,
        float(splits["train_fraction"]),
        float(splits["validation_fraction"]),
        float(splits["test_fraction"]),
        seed=int(split_seed),
    )
    mask = create_neuron_mask(
        dataset.spikes.shape[2], float(splits["heldout_neuron_fraction"]), seed=int(split_seed)
    )
    return split, np.flatnonzero(mask.heldin), np.flatnonzero(mask.heldout)


def _counts(dataset: NeuralDataset, trial_ids: np.ndarray, neurons: np.ndarray) -> np.ndarray:
    return np.asarray(dataset.spikes[np.isin(dataset.trial_ids, trial_ids)][:, :, neurons])


def _score(
    counts: np.ndarray,
    predicted: np.ndarray,
    reference: np.ndarray,
    scoring: ScoringConfig,
    method_name: str,
    split_name: str,
    valid_model: bool,
) -> dict[str, Any]:
    return score_heldout_prediction(
        counts, predicted, reference, scoring, method_name, split_name, "cv_rate_audit", valid_model
    )


def run_repeated_split_factor_latent(
    config: dict[str, Any], dataset: NeuralDataset
) -> pd.DataFrame:
    """Factor-latent scored over every split seed crossed with every FactorAnalysis seed."""
    scoring = _scoring_config(config)
    settings = _factor_latent_settings(config)
    rows: list[dict[str, Any]] = []
    for split_seed in [int(seed) for seed in config["splits"]["split_seeds"]]:
        split, heldin, heldout = _split_and_indices(dataset, config, split_seed)
        train_counts = _counts(dataset, split.train, heldout)
        for random_state in [
            int(state) for state in config["splits"]["factor_analysis_random_states"]
        ]:
            predictions = factor_latent_heldout_rates(
                dataset, split, heldin, heldout, scoring, settings, random_state
            )
            row: dict[str, Any] = {
                "split_seed": split_seed,
                "factor_analysis_random_state": random_state,
                "method_name": FACTOR_LATENT,
                "valid_model": True,
                "notes": "Train-only factor-analysis fit; canonical unweighted scoring.",
            }
            for split_name in EVALUATION_SPLITS:
                counts = _counts(dataset, getattr(split, split_name), heldout)
                reference = train_heldout_mean_rate_reference(train_counts, counts.shape, scoring)
                scored = _score(
                    counts,
                    predictions[split_name],
                    reference,
                    scoring,
                    FACTOR_LATENT,
                    split_name,
                    True,
                )
                row[f"{split_name}_unified_bits_per_spike"] = scored["bits_per_spike"]
                row[f"{split_name}_poisson_nll"] = scored["poisson_nll"]
                row[f"{split_name}_heldout_rate_hz"] = scored["observed_rate_hz"]
                row[f"{split_name}_trial_count"] = int(counts.shape[0])
            rows.append(row)
    return pd.DataFrame(rows, columns=REPEATED_SPLIT_COLUMNS)


def run_factor_analysis_random_state_sensitivity(
    repeated_scores: pd.DataFrame, split_seed: int
) -> pd.DataFrame:
    """Slice one split seed and express every FactorAnalysis seed relative to random_state 0."""
    if repeated_scores.empty:
        return pd.DataFrame(columns=FA_SENSITIVITY_COLUMNS)
    subset = repeated_scores[repeated_scores["split_seed"] == split_seed].copy()
    if subset.empty:
        return pd.DataFrame(columns=FA_SENSITIVITY_COLUMNS)
    baseline = subset[subset["factor_analysis_random_state"] == 0]
    validation_base = (
        float(baseline.iloc[0]["validation_unified_bits_per_spike"])
        if not baseline.empty
        else float("nan")
    )
    test_base = (
        float(baseline.iloc[0]["test_unified_bits_per_spike"])
        if not baseline.empty
        else float("nan")
    )
    subset["difference_from_random_state_0_validation"] = (
        subset["validation_unified_bits_per_spike"] - validation_base
    )
    subset["difference_from_random_state_0_test"] = (
        subset["test_unified_bits_per_spike"] - test_base
    )
    subset["notes"] = (
        "sklearn FactorAnalysis uses randomized SVD; only the random_state differs across rows."
    )
    return subset.sort_values("factor_analysis_random_state").reset_index(drop=True)[
        FA_SENSITIVITY_COLUMNS
    ]


def _control_predictions(
    dataset: NeuralDataset,
    config: dict[str, Any],
    scoring: ScoringConfig,
    split: TrialSplit,
    heldin: np.ndarray,
    heldout: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Predicted held-out rates per control per evaluation split."""
    enabled = dict(config["rate_controls"])
    settings = _factor_latent_settings(config)
    train_heldout = _counts(dataset, split.train, heldout)
    train_heldin = _counts(dataset, split.train, heldin)
    factor = factor_latent_heldout_rates(dataset, split, heldin, heldout, scoring, settings, 0)
    calibration = compute_train_rate_calibration(train_heldout, factor["train"], scoring)
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for split_name in EVALUATION_SPLITS:
        counts = _counts(dataset, getattr(split, split_name), heldout)
        split_heldin = _counts(dataset, getattr(split, split_name), heldin)
        per_split: dict[str, np.ndarray] = {}
        if bool(enabled.get("include_train_mean_rate", True)):
            per_split[TRAIN_MEAN_RATE] = compute_train_mean_rate_control(
                train_heldout, counts.shape, scoring
            )["predicted_rates_hz"]
        if bool(enabled.get("include_train_per_neuron_mean_rate", True)):
            per_split[TRAIN_PER_NEURON_MEAN_RATE] = compute_train_per_neuron_mean_rate_control(
                train_heldout, counts.shape, scoring
            )["predicted_rates_hz"]
        if bool(enabled.get("include_train_population_scaled_mean_rate", True)):
            per_split[TRAIN_POPULATION_SCALED_MEAN_RATE] = (
                compute_train_population_scaled_mean_rate_control(
                    train_heldout, train_heldin, split_heldin, counts.shape, scoring
                )["predicted_rates_hz"]
            )
        per_split[FACTOR_LATENT] = factor[split_name]
        if bool(enabled.get("include_train_split_rate_calibrated_factor_latent", True)):
            per_split[TRAIN_RATE_CALIBRATED_FACTOR_LATENT] = apply_rate_calibration(
                factor[split_name], calibration, scoring
            )
        if bool(enabled.get("include_split_mean_rate_invalid", True)):
            per_split[SPLIT_MEAN_RATE_INVALID] = compute_split_mean_rate_invalid_control(
                counts, scoring
            )["predicted_rates_hz"]
        if bool(enabled.get("include_oracle_split_scaled_factor_latent_invalid", True)):
            per_split[ORACLE_SPLIT_SCALED_FACTOR_LATENT_INVALID] = (
                compute_oracle_split_scaled_factor_latent_invalid_control(
                    factor[split_name], counts, scoring
                )["predicted_rates_hz"]
            )
        predictions[split_name] = per_split
    return predictions


def run_rate_control_audit(config: dict[str, Any], dataset: NeuralDataset) -> pd.DataFrame:
    from latentbrain.eval.rate_controls import (  # noqa: PLC0415
        CONTROL_NOTES,
        invalid_reason,
        is_valid_control,
    )

    scoring = _scoring_config(config)
    rows: list[dict[str, Any]] = []
    for split_seed in [int(seed) for seed in config["splits"]["split_seeds"]]:
        split, heldin, heldout = _split_and_indices(dataset, config, split_seed)
        train_heldout = _counts(dataset, split.train, heldout)
        predictions = _control_predictions(dataset, config, scoring, split, heldin, heldout)
        for split_name in EVALUATION_SPLITS:
            counts = _counts(dataset, getattr(split, split_name), heldout)
            reference = train_heldout_mean_rate_reference(train_heldout, counts.shape, scoring)
            for method_name, predicted in predictions[split_name].items():
                valid = is_valid_control(method_name)
                scored = _score(
                    counts, predicted, reference, scoring, method_name, split_name, valid
                )
                rows.append(
                    {
                        "split_seed": split_seed,
                        "split": split_name,
                        "method_name": method_name,
                        "valid_model": valid,
                        "invalid_reason": invalid_reason(method_name),
                        "unified_bits_per_spike": scored["bits_per_spike"],
                        "poisson_nll": scored["poisson_nll"],
                        "heldout_rate_hz": scored["observed_rate_hz"],
                        "predicted_rate_hz": scored["mean_predicted_rate_hz"],
                        "rate_error_hz": float(
                            scored["mean_predicted_rate_hz"] - scored["observed_rate_hz"]
                        ),
                        "notes": CONTROL_NOTES.get(method_name, invalid_reason(method_name)),
                    }
                )
    return pd.DataFrame(rows, columns=RATE_CONTROL_COLUMNS)


def decompose_rate_offset(rate_control_scores: pd.DataFrame) -> pd.DataFrame:
    if rate_control_scores.empty:
        return pd.DataFrame(columns=DECOMPOSITION_COLUMNS)
    pivot = rate_control_scores.pivot_table(
        index=["split_seed", "split"],
        columns="method_name",
        values="unified_bits_per_spike",
        aggfunc="first",
    )
    rows: list[dict[str, Any]] = []
    for (split_seed, split_name), values in pivot.iterrows():
        factor = float(values.get(FACTOR_LATENT, float("nan")))
        calibrated = float(values.get(TRAIN_RATE_CALIBRATED_FACTOR_LATENT, float("nan")))
        split_mean = float(values.get(SPLIT_MEAN_RATE_INVALID, float("nan")))
        oracle = float(values.get(ORACLE_SPLIT_SCALED_FACTOR_LATENT_INVALID, float("nan")))
        calibration_gain = calibrated - factor
        oracle_gain = oracle - factor
        split_mean_advantage = split_mean - factor
        explains = False
        if (
            np.isfinite(oracle_gain)
            and np.isfinite(split_mean_advantage)
            and split_mean_advantage > 0.0
        ):
            explains = bool(oracle_gain >= RATE_OFFSET_EXPLAINS_THRESHOLD * split_mean_advantage)
        rows.append(
            {
                "split_seed": int(split_seed),
                "split": str(split_name),
                "factor_latent_bits_per_spike": factor,
                "train_rate_calibrated_factor_latent_bits_per_spike": calibrated,
                "split_mean_rate_invalid_bits_per_spike": split_mean,
                "oracle_split_scaled_factor_latent_invalid_bits_per_spike": oracle,
                "valid_calibration_gain": calibration_gain,
                "invalid_oracle_gain": oracle_gain,
                "split_mean_advantage_over_factor_latent": split_mean_advantage,
                "rate_offset_explains_gap": explains,
                "notes": (
                    "Oracle and split-mean columns leak evaluation targets and are diagnostic only."
                ),
            }
        )
    return pd.DataFrame(rows, columns=DECOMPOSITION_COLUMNS)


def summarize_methods(
    rate_control_scores: pd.DataFrame,
    repeats: int = 10000,
    confidence: float = 0.95,
    seed: int = 1337,
) -> pd.DataFrame:
    if rate_control_scores.empty:
        return pd.DataFrame(columns=METHOD_SUMMARY_COLUMNS)
    rows: list[dict[str, Any]] = []
    for method_name, group in rate_control_scores.groupby("method_name", sort=True):
        validation = group[group["split"] == "validation"]["unified_bits_per_spike"].to_numpy(
            dtype=np.float64
        )
        test = group[group["split"] == "test"]["unified_bits_per_spike"].to_numpy(dtype=np.float64)
        ci_low, ci_high = bootstrap_mean_ci(test, repeats, confidence, seed)
        rows.append(
            {
                "method_name": str(method_name),
                "valid_model": bool(group.iloc[0]["valid_model"]),
                "n_scores": int(len(group)),
                "mean_validation_unified_bits_per_spike": float(np.mean(validation))
                if validation.size
                else float("nan"),
                "std_validation_unified_bits_per_spike": float(np.std(validation, ddof=1))
                if validation.size > 1
                else 0.0,
                "mean_test_unified_bits_per_spike": float(np.mean(test))
                if test.size
                else float("nan"),
                "std_test_unified_bits_per_spike": float(np.std(test, ddof=1))
                if test.size > 1
                else 0.0,
                "ci95_low_test": ci_low,
                "ci95_high_test": ci_high,
                "test_positive_fraction": float(np.mean(test > 0.0)) if test.size else float("nan"),
                "notes": str(group.iloc[0]["notes"]),
            }
        )
    return pd.DataFrame(rows, columns=METHOD_SUMMARY_COLUMNS)


def _range(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.max() - finite.min()) if finite.size else float("nan")


def summarize_cv_rate_audit(
    repeated_scores: pd.DataFrame,
    fa_sensitivity: pd.DataFrame,
    rate_controls: pd.DataFrame,
    decomposition: pd.DataFrame,
    method_summary: pd.DataFrame,
    references: dict[str, Any],
) -> dict[str, Any]:
    factor = repeated_scores[repeated_scores["method_name"] == FACTOR_LATENT]
    validation = factor["validation_unified_bits_per_spike"].to_numpy(dtype=np.float64)
    test = factor["test_unified_bits_per_spike"].to_numpy(dtype=np.float64)

    split_variance = float("nan")
    random_state_variance = float("nan")
    if not factor.empty:
        # Variance of per-split means isolates the trial-split effect; the mean of within-split
        # variances isolates the FactorAnalysis random_state effect. A split seed with a single
        # random_state contributes no within-split variance rather than a NaN.
        by_split = factor.groupby("split_seed")["test_unified_bits_per_spike"]
        split_means = by_split.mean().to_numpy(dtype=np.float64)
        if split_means.size > 1:
            split_variance = float(np.var(split_means, ddof=1))
        within = [
            float(np.var(group.to_numpy(dtype=np.float64), ddof=1))
            for _, group in by_split
            if group.size > 1
        ]
        if within:
            random_state_variance = float(np.mean(within))

    valid_summary = method_summary[method_summary["valid_model"].astype(bool)]
    non_trivial = valid_summary[valid_summary["method_name"] != TRAIN_MEAN_RATE]
    best_valid = select_best_valid_method(rate_controls)
    best_valid_test_mean = float("nan")
    if best_valid is not None:
        rows = method_summary[method_summary["method_name"] == best_valid]
        if not rows.empty:
            best_valid_test_mean = float(rows.iloc[0]["mean_test_unified_bits_per_spike"])

    def _method_test_mean(name: str) -> float:
        rows = method_summary[method_summary["method_name"] == name]
        return (
            float("nan") if rows.empty else float(rows.iloc[0]["mean_test_unified_bits_per_spike"])
        )

    factor_test_mean = _method_test_mean(FACTOR_LATENT)
    calibrated_test_mean = _method_test_mean(TRAIN_RATE_CALIBRATED_FACTOR_LATENT)
    population_scaled_test_mean = _method_test_mean(TRAIN_POPULATION_SCALED_MEAN_RATE)
    split_mean_test_mean = _method_test_mean(SPLIT_MEAN_RATE_INVALID)

    calibration_gain = (
        float(calibrated_test_mean - factor_test_mean)
        if np.isfinite(calibrated_test_mean) and np.isfinite(factor_test_mean)
        else float("nan")
    )
    calibration_helps = bool(np.isfinite(calibration_gain) and calibration_gain > 0.0)
    # A positive sign on a gain this small is numerical, not a real improvement. Report both so
    # `train_only_rate_calibration_helps: true` can never be read as a meaningful result.
    calibration_gain_negligible = bool(
        np.isfinite(calibration_gain) and abs(calibration_gain) < NEGLIGIBLE_GAIN
    )
    rate_offset_explains = (
        bool(decomposition["rate_offset_explains_gap"].mean() >= 0.5)
        if not decomposition.empty
        else False
    )
    invalid_dominates = bool(
        np.isfinite(split_mean_test_mean)
        and np.isfinite(best_valid_test_mean)
        and split_mean_test_mean > best_valid_test_mean
    )
    high_split_variance = bool(
        np.isfinite(split_variance)
        and np.isfinite(random_state_variance)
        and split_variance > random_state_variance
    )
    return {
        "primary_metric": "unified_bits_per_spike",
        "reference_model": "train_heldout_mean_rate",
        "evaluation_metric_is_unweighted": True,
        "factor_latent_repeated_split_validation_mean": float(np.mean(validation))
        if validation.size
        else float("nan"),
        "factor_latent_repeated_split_validation_std": float(np.std(validation, ddof=1))
        if validation.size > 1
        else 0.0,
        "factor_latent_repeated_split_test_mean": float(np.mean(test))
        if test.size
        else float("nan"),
        "factor_latent_repeated_split_test_std": float(np.std(test, ddof=1))
        if test.size > 1
        else 0.0,
        "factor_latent_test_positive_fraction": float(np.mean(test > 0.0))
        if test.size
        else float("nan"),
        "factor_analysis_random_state_validation_range": _range(
            fa_sensitivity["validation_unified_bits_per_spike"].to_numpy(dtype=np.float64)
        )
        if not fa_sensitivity.empty
        else float("nan"),
        "factor_analysis_random_state_test_range": _range(
            fa_sensitivity["test_unified_bits_per_spike"].to_numpy(dtype=np.float64)
        )
        if not fa_sensitivity.empty
        else float("nan"),
        "between_split_test_variance": split_variance,
        "within_split_random_state_test_variance": random_state_variance,
        "split_variance_exceeds_random_state_variance": high_split_variance,
        "best_valid_rate_control_method": best_valid,
        "best_valid_rate_control_test_mean": best_valid_test_mean,
        "best_non_trivial_valid_method": None
        if non_trivial.empty
        else str(
            non_trivial.sort_values("mean_test_unified_bits_per_spike", ascending=False).iloc[0][
                "method_name"
            ]
        ),
        "factor_latent_test_mean": factor_test_mean,
        "train_rate_calibrated_factor_latent_test_mean": calibrated_test_mean,
        "train_population_scaled_mean_rate_test_mean": population_scaled_test_mean,
        "split_mean_rate_invalid_test_mean": split_mean_test_mean,
        "invalid_split_mean_advantage_over_factor_latent": float(
            split_mean_test_mean - factor_test_mean
        )
        if np.isfinite(split_mean_test_mean) and np.isfinite(factor_test_mean)
        else float("nan"),
        "train_only_rate_calibration_helps": calibration_helps,
        "train_only_rate_calibration_test_gain": calibration_gain,
        "train_only_rate_calibration_gain_is_negligible": calibration_gain_negligible,
        "rate_offset_explains_split_mean_advantage": rate_offset_explains,
        "invalid_controls_dominate_valid_models": invalid_dominates,
        "invalid_controls_excluded_from_best_valid_model": True,
        "invalid_control_methods": list(
            method_summary[~method_summary["valid_model"].astype(bool)]["method_name"]
        ),
        "single_split_results_reportable": False,
        "recommended_reporting_mode": "repeated_split",
        "carried_forward_for_reporting": FACTOR_LATENT,
        "accepted_split_seed": references.get("accepted_split_seed"),
        "old_incompatible_mean_rate_values_used_as_targets": False,
        "official_benchmark_claim": False,
    }


def build_reporting_recommendations(summary: dict[str, Any]) -> dict[str, Any]:
    reasons = [
        "Repeated-split spread dominates any single-split number.",
        "FactorAnalysis random_state alone moves the metric materially.",
    ]
    if summary.get("invalid_controls_dominate_valid_models"):
        reasons.append(
            "An invalid control that reads evaluation targets beats every valid model, so an "
            "unmodeled split-level rate offset remains."
        )
    if summary.get("train_only_rate_calibration_gain_is_negligible"):
        reasons.append(
            "The train-only rate calibration changes the test mean by less than "
            f"{NEGLIGIBLE_GAIN}; its positive sign is numerical, not a real gain."
        )
    if summary.get("invalid_controls_dominate_valid_models") and not summary.get(
        "rate_offset_explains_split_mean_advantage"
    ):
        reasons.append(
            "Rescaling recovers little of the invalid advantage, so the leak is per-neuron "
            "evaluation-split mean information rather than a global rate offset; no valid "
            "rescaling can close it."
        )
    return {
        "single_split_results_reportable": False,
        "recommended_reporting_mode": "repeated_split",
        "report_as_diagnostics": [
            FACTOR_LATENT,
            TRAIN_POPULATION_SCALED_MEAN_RATE,
            TRAIN_RATE_CALIBRATED_FACTOR_LATENT,
        ],
        "must_label_invalid": list(summary.get("invalid_control_methods", [])),
        "carried_forward_for_reporting": FACTOR_LATENT,
        "neural_models_carried_forward": False,
        "rate_offset_warning": (
            "The invalid split-mean-rate control beats every valid model; treat all scores as "
            "validation-only diagnostics until the split-level rate offset is modeled."
            if summary.get("invalid_controls_dominate_valid_models")
            else "No invalid control dominates the valid models."
        ),
        "reasons": reasons,
        "official_benchmark_claim": False,
    }
