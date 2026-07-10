from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from latentbrain.eval.stratified_cv import (
    FACTOR_LATENT,
    FOLD_ASSIGNMENT_COLUMNS,
    METHOD_SUMMARY_COLUMNS,
    SPLIT_MEAN_RATE_INVALID,
    TRAIN_MEAN_RATE,
    TRIAL_FEATURE_COLUMNS,
    assign_stratified_folds,
    build_repeated_stratified_folds,
    build_strata_labels,
    build_trial_features,
    compare_random_and_stratified,
    select_best_valid_method,
    summarize_methods,
)

BEHAVIOR_NAMES = ["hand_pos_x", "hand_pos_y", "cursor_pos_x", "cursor_pos_y"]


def _spikes(trials: int = 40, time: int = 16, neurons: int = 8) -> np.ndarray:
    generator = np.random.default_rng(3)
    return generator.poisson(0.4, size=(trials, time, neurons)).astype(np.float64)


def _behavior(trials: int = 40, time: int = 16) -> np.ndarray:
    generator = np.random.default_rng(4)
    behavior = np.zeros((trials, time, 4))
    angles = np.linspace(-np.pi, np.pi, trials, endpoint=False)
    for trial in range(trials):
        radius = 1.0 + generator.random()
        behavior[trial, :, 0] = np.linspace(0.0, radius * np.cos(angles[trial]), time)
        behavior[trial, :, 1] = np.linspace(0.0, radius * np.sin(angles[trial]), time)
    return behavior


def _config(**overrides: Any) -> dict[str, Any]:
    config = {
        "cross_validation": {
            "fold_count": 5,
            "repeats": 3,
            "base_seed": 2027,
            "heldout_neuron_fraction": 0.25,
            "assignment_method": "greedy_balanced",
            "fallback_when_behavior_missing": "rate_only",
            "min_trials_per_stratum": 2,
            "random_split_repeats": 15,
            "stratification": {
                "use_endpoint_direction": True,
                "endpoint_direction_bins": 8,
                "use_endpoint_distance": True,
                "endpoint_distance_bins": 3,
                "use_mean_speed": True,
                "mean_speed_bins": 3,
                "use_population_rate": True,
                "population_rate_bins": 3,
                "use_heldout_rate": True,
                "heldout_rate_bins": 3,
            },
        }
    }
    config["cross_validation"].update(overrides)
    return config


def _features(with_behavior: bool = True) -> pd.DataFrame:
    return build_trial_features(
        _spikes(),
        _behavior() if with_behavior else None,
        BEHAVIOR_NAMES if with_behavior else None,
        20,
        np.array([6, 7]),
    )


def test_trial_features_include_required_columns() -> None:
    features = _features()

    assert list(features.columns) == TRIAL_FEATURE_COLUMNS
    assert len(features) == 40
    assert bool(features["behavior_available"].all())
    assert np.isfinite(features["endpoint_angle_rad"]).all()
    assert np.isfinite(features["heldout_rate_hz"]).all()
    assert (features["endpoint_angle_rad"].abs() <= np.pi).all()


def test_behavior_missing_fallback_produces_nan_columns() -> None:
    features = _features(with_behavior=False)

    assert not bool(features["behavior_available"].any())
    for column in ("endpoint_dx", "endpoint_angle_rad", "endpoint_distance", "mean_speed"):
        assert features[column].isna().all()
    assert np.isfinite(features["population_rate_hz"]).all()


def test_strata_labels_fall_back_to_rate_only_when_behavior_missing() -> None:
    labels = build_strata_labels(_features(with_behavior=False), _config())

    assert len(labels) == 40
    # Only the two rate bins remain, so labels look like "<rate>_<heldout>".
    assert labels.str.count("_").max() == 1


def test_strata_labels_are_deterministic() -> None:
    first = build_strata_labels(_features(), _config())
    second = build_strata_labels(_features(), _config())

    assert first.equals(second)
    assert first.nunique() > 1


def test_fold_assignment_is_deterministic_under_seed() -> None:
    features = _features()
    labels = build_strata_labels(features, _config())

    first = assign_stratified_folds(features, labels, 5, 2027, 2)
    second = assign_stratified_folds(features, labels, 5, 2027, 2)
    different = assign_stratified_folds(features, labels, 5, 7, 2)

    assert first["fold_index"].tolist() == second["fold_index"].tolist()
    assert first["fold_index"].tolist() != different["fold_index"].tolist()


def test_fold_assignment_rejects_small_fold_counts() -> None:
    features = _features()
    labels = build_strata_labels(features, _config())

    with pytest.raises(ValueError, match="fold_count must be at least 3"):
        assign_stratified_folds(features, labels, 2, 2027, 2)


def test_every_trial_appears_exactly_once_per_repeat() -> None:
    folds = build_repeated_stratified_folds(_features(), _config())

    assert list(folds.columns) == FOLD_ASSIGNMENT_COLUMNS
    for _, group in folds.groupby("repeat_index"):
        assert sorted(group["trial_index"]) == list(range(40))
        assert group["trial_index"].is_unique


def test_fold_counts_are_balanced_within_tolerance() -> None:
    folds = build_repeated_stratified_folds(_features(), _config())

    for _, group in folds.groupby("repeat_index"):
        sizes = group.groupby("fold_index").size().to_numpy()
        assert sizes.max() - sizes.min() <= 1


def test_repeats_below_two_are_rejected() -> None:
    with pytest.raises(ValueError, match="repeats must be at least 2"):
        build_repeated_stratified_folds(_features(), _config(repeats=1))


def test_unknown_assignment_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="assignment_method"):
        build_repeated_stratified_folds(_features(), _config(assignment_method="magic"))


def _scores(factor: list[float], invalid: list[float]) -> pd.DataFrame:
    rows = []
    for fold_index, (factor_bits, invalid_bits) in enumerate(zip(factor, invalid, strict=True)):
        for name, method_type, valid, reportable, bits in (
            (TRAIN_MEAN_RATE, "rate_control", False, False, 0.0),
            (FACTOR_LATENT, "factor_latent", True, True, factor_bits),
            (SPLIT_MEAN_RATE_INVALID, "invalid_control", False, False, invalid_bits),
        ):
            rows.append(
                {
                    "repeat_index": 0,
                    "fold_index": fold_index,
                    "method_name": name,
                    "method_type": method_type,
                    "valid_model": valid,
                    "reportable_as_model_performance": reportable,
                    "invalid_reason": "" if valid else "leaks",
                    "train_trial_count": 32,
                    "eval_trial_count": 8,
                    "unified_bits_per_spike": bits,
                    "poisson_nll": 100.0,
                    "eval_spike_count": 500.0,
                    "eval_heldout_rate_hz": 0.6,
                    "factor_analysis_random_state": 0 if name == FACTOR_LATENT else -1,
                    "notes": "",
                }
            )
    return pd.DataFrame(rows)


def test_method_summary_has_required_columns_and_is_deterministic() -> None:
    scores = _scores([0.01, 0.02, 0.03], [0.09, 0.10, 0.11])

    summary = summarize_methods(scores, 200, 0.95, 1337)
    again = summarize_methods(scores, 200, 0.95, 1337)

    assert list(summary.columns) == METHOD_SUMMARY_COLUMNS
    assert summary.equals(again)
    factor = summary[summary["method_name"] == FACTOR_LATENT].iloc[0]
    assert factor["mean_unified_bits_per_spike"] == pytest.approx(0.02)
    assert factor["positive_fraction"] == pytest.approx(1.0)


def test_invalid_controls_are_excluded_from_valid_model_selection() -> None:
    # The invalid control scores far higher, yet must never be selected.
    summary = summarize_methods(_scores([0.01, 0.02, 0.03], [0.09, 0.10, 0.11]), 200, 0.95, 1337)

    assert select_best_valid_method(summary) == FACTOR_LATENT


def test_reference_method_is_excluded_from_valid_model_selection() -> None:
    scores = _scores([-0.05, -0.04, -0.03], [0.09, 0.10, 0.11])
    summary = summarize_methods(scores, 200, 0.95, 1337)

    # train_mean_rate averages 0.0 and beats factor-latent here, but it is the reference.
    assert select_best_valid_method(summary) == FACTOR_LATENT


def test_best_valid_method_is_none_without_reportable_rows() -> None:
    summary = summarize_methods(_scores([0.01], [0.09]), 200, 0.95, 1337)
    summary["reportable_as_model_performance"] = False

    assert select_best_valid_method(summary) is None
    assert select_best_valid_method(pd.DataFrame()) is None


def test_variance_comparison_detects_reduction_and_reports_it_honestly() -> None:
    stratified = _scores([0.010, 0.011, 0.012], [0.09, 0.09, 0.09])
    random_wide = _scores([-0.02, 0.01, 0.05], [0.09, 0.09, 0.09])

    reduced = compare_random_and_stratified(stratified, random_wide)
    assert reduced["stratification_reduces_variance"] is True
    assert reduced["variance_reduction_fraction"] > 0.0

    # Swap the roles: stratification must be reported as not reducing variance.
    increased = compare_random_and_stratified(random_wide, stratified)
    assert increased["stratification_reduces_variance"] is False
    assert increased["variance_reduction_fraction"] < 0.0
