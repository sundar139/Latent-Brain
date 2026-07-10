from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

import latentbrain.data.nlb as nlb_module
from latentbrain.data.io import save_neural_dataset
from latentbrain.data.schemas import NeuralDataset, TrialSequences
from latentbrain.eval.baseline_suite import (
    FACTOR_LATENT_FIXED,
    FORBIDDEN_OLD_PROTOCOLS,
    REQUIRED_NEURAL_SEEDS,
    _SmoothingCache,
    build_method_summary,
    build_paired_comparisons,
    build_readiness,
    build_repeat_level_scores,
    choose_baseline_to_beat,
    hierarchical_paired_bootstrap,
    inner_folds,
    load_frozen_protocol,
    load_outer_folds,
    method_configurations,
    run_baseline_suite,
    select_configuration,
    summarize_baseline_suite,
)
from latentbrain.eval.cosmoothing import fit_reduced_rank_cosmoothing
from latentbrain.eval.seed_robustness import bootstrap_mean_ci

NAMES = ["hand_pos_x", "hand_pos_y", "cursor_pos_x", "cursor_pos_y"]
LENGTHS = (48, 56, 64, 48, 56, 64, 48, 56, 64, 48, 56, 64)
GLOBAL_BINS = min(LENGTHS)
WINDOW_SECONDS = 0.08  # 16 source bins at 5 ms -> 4 bins at 20 ms
FOLDS, REPEATS = 3, 2


def _trial(length: int, angle: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    spikes = generator.poisson(0.5, size=(length, 10)).astype(np.int64)
    times = np.arange(length, dtype=np.float64)
    position = np.cumsum(np.exp(-(((times - 0.4 * length) / (0.12 * length)) ** 2)))
    behavior = np.zeros((length, 4))
    behavior[:, 0] = position * np.cos(angle)
    behavior[:, 1] = position * np.sin(angle)
    behavior[:, 2] = behavior[:, 0]
    behavior[:, 3] = behavior[:, 1]
    return spikes, behavior


def _sequences() -> TrialSequences:
    angles = np.linspace(-np.pi, np.pi, len(LENGTHS), endpoint=False)
    spikes, behavior = [], []
    for index, length in enumerate(LENGTHS):
        trial_spikes, trial_behavior = _trial(length, angles[index], index)
        spikes.append(trial_spikes)
        behavior.append(trial_behavior)
    return TrialSequences(
        spikes=spikes,
        behavior=behavior,
        behavior_names=NAMES,
        trial_ids=np.arange(len(LENGTHS), dtype=np.int64),
        trial_lengths=np.asarray(LENGTHS, dtype=np.int64),
        bin_size_ms=5,
        metadata={"spikes_conserved": True, "source_file": "toy_train.nwb"},
    )


def _write_processed(path: Path, sequences: TrialSequences) -> str:
    spikes = np.stack([trial[:GLOBAL_BINS] for trial in sequences.spikes])
    behavior = np.stack([trial[:GLOBAL_BINS] for trial in sequences.behavior or []])
    dataset = NeuralDataset(
        spikes=spikes,
        rates=None,
        latents=None,
        trial_ids=sequences.trial_ids,
        time_ms=np.arange(GLOBAL_BINS, dtype=np.float64) * 5.0,
        bin_size_ms=5,
        metadata={"dataset_name": "toy_large"},
        behavior=behavior,
        behavior_names=NAMES,
    )
    save_neural_dataset(dataset, path)
    return str(dataset.metadata["dataset_hash"])


def _assignments(path: Path, base_seed: int = 2027) -> None:
    rows = []
    for repeat in range(REPEATS):
        order = np.random.default_rng(base_seed + repeat).permutation(len(LENGTHS))
        for position, trial in enumerate(order):
            rows.append(
                {
                    "repeat_index": repeat,
                    "fold_index": position % FOLDS,
                    "trial_index": int(trial),
                    "stratum": "toy",
                    "seed": base_seed + repeat,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _frozen_protocol(tmp_path: Path, processed: Path, provenance: Path, dataset_hash: str) -> Path:
    protocol = {
        "dataset": {
            "name": "toy_large",
            "processed_path": str(processed),
            "provenance_path": str(provenance),
            "raw_dir": str(tmp_path / "raw"),
            "expected_hash": dataset_hash,
            "original_bin_size_ms": 5,
        },
        "trial_source": {"type": "trial_aware_raw", "allow_global_crop_to_min": False},
        "binning": {"target_bin_size_ms": 20, "extract_before_rebin": True},
        "window": {
            "name": "behavior_speed_peak_centered_1p28s",
            "crop_policy": "behavior_speed_peak_centered",
            "duration_seconds": WINDOW_SECONDS,
        },
        "cross_validation": {
            "fold_count": FOLDS,
            "repeats": REPEATS,
            "base_seed": 2027,
            "heldout_neuron_fraction": 0.25,
        },
    }
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    return path


def _config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    sequences = _sequences()
    processed = tmp_path / "processed.npz"
    dataset_hash = _write_processed(processed, sequences)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(exist_ok=True)
    provenance = tmp_path / "provenance.json"
    nlb_config = yaml.safe_load(Path("configs/nlb_mc_maze_large.yaml").read_text(encoding="utf-8"))
    provenance.write_text(
        json.dumps({"config": nlb_config, "processed_dataset_hash": dataset_hash}), encoding="utf-8"
    )
    monkeypatch.setattr(nlb_module, "load_trial_sequences", lambda _root, _config: _sequences())

    protocol_path = _frozen_protocol(tmp_path, processed, provenance, dataset_hash)
    assignments_path = tmp_path / "assignments.csv"
    _assignments(assignments_path)
    cv_summary = tmp_path / "cv_summary.json"

    base = yaml.safe_load(
        Path("configs/mc_maze_large_baseline_suite.yaml").read_text(encoding="utf-8")
    )
    base["dataset"].update({"name": "toy_large", "expected_hash": dataset_hash})
    base["window"]["duration_seconds"] = WINDOW_SECONDS
    base["outer_cross_validation"].update(
        {
            "fold_count": FOLDS,
            "repeats": REPEATS,
            "source_assignments_path": str(assignments_path),
            "source_summary_path": str(cv_summary),
        }
    )
    base["inner_selection"]["fold_count"] = 2
    for method in base["methods"]:
        if method["name"] == "factor_latent_train_selected":
            method["search"] = {
                "latent_dim": [2, 4],
                "smoothing_sigma_ms": [120.0],
                "heldout_decoder_alpha": [10000.0],
                "standardize_features": [True],
                "fit_intercept": [True],
                "factor_analysis_random_state": [0],
            }
        if method["name"] == "smoothed_cosmoothing_ridge":
            method["search"] = {
                "smoothing_sigma_ms": [80.0, 160.0],
                "alpha": [10000.0],
                "standardize_features": [True],
                "fit_intercept": [True],
            }
        if method["name"] == "reduced_rank_cosmoothing":
            method["search"] = {
                "smoothing_sigma_ms": [80.0],
                "alpha": [10000.0],
                "rank": [1, 2],
                "standardize_features": [True],
                "fit_intercept": [True],
            }
        if method["name"] == FACTOR_LATENT_FIXED:
            method["hyperparameters"]["latent_dim"] = 2
    base["statistics"]["bootstrap_repeats"] = 200
    base["inputs"]["recommended_window_protocol_path"] = str(protocol_path)
    base["inputs"]["recommended_window_summary_path"] = str(cv_summary)
    base["reporting"]["output_dir"] = str(tmp_path / "out")
    return base


def _write_accepted_mean(config: dict[str, Any], mean: float) -> None:
    Path(config["inputs"]["recommended_window_summary_path"]).write_text(
        json.dumps({"factor_latent_mean": mean}), encoding="utf-8"
    )


def test_frozen_protocol_is_reused_and_global_crop_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)

    protocol = load_frozen_protocol(config)
    assert protocol["trial_source"]["type"] == "trial_aware_raw"

    config["trial_source"]["allow_global_crop_to_min"] = True
    with pytest.raises(ValueError, match="cannot source event-centered evaluation windows"):
        load_frozen_protocol(config)


def test_mismatched_protocol_hash_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    config["dataset"]["expected_hash"] = "0" * 64

    with pytest.raises(ValueError, match="dataset hash does not match the frozen protocol"):
        load_frozen_protocol(config)


def test_outer_assignments_and_neuron_masks_are_reused_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    protocol = load_frozen_protocol(config)

    folds = load_outer_folds(config, protocol, n_neurons=10)

    assert len(folds) == FOLDS * REPEATS
    for fold in folds:
        assert fold.neuron_mask_seed == 2027 + fold.repeat_index
        assert fold.split_seed == fold.neuron_mask_seed
        assert np.intersect1d(fold.train_trials, fold.eval_trials).size == 0
        assert fold.train_trials.size + fold.eval_trials.size == len(LENGTHS)
    for repeat in range(REPEATS):
        repeat_folds = [fold for fold in folds if fold.repeat_index == repeat]
        first = repeat_folds[0]
        for fold in repeat_folds[1:]:
            np.testing.assert_array_equal(fold.heldin, first.heldin)
        evaluated = np.concatenate([fold.eval_trials for fold in repeat_folds])
        assert sorted(evaluated) == list(range(len(LENGTHS)))


def test_mismatched_assignment_seed_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    protocol = load_frozen_protocol(config)
    path = Path(config["outer_cross_validation"]["source_assignments_path"])
    assignments = pd.read_csv(path)
    assignments["seed"] = 999
    assignments.to_csv(path, index=False)

    with pytest.raises(ValueError, match="split seed does not match the accepted assignments"):
        load_outer_folds(config, protocol, n_neurons=10)


def test_inner_folds_only_contain_outer_training_trials() -> None:
    train = np.array([0, 2, 4, 6, 8, 10], dtype=np.int64)

    partitions = inner_folds(train, fold_count=3, seed=4041)

    assert sorted(np.concatenate(partitions)) == sorted(train)
    for partition in partitions:
        assert np.isin(partition, train).all()
    repeated = inner_folds(train, fold_count=3, seed=4041)
    for first, second in zip(partitions, repeated, strict=True):
        np.testing.assert_array_equal(first, second)


def test_smoothing_cache_never_serves_another_masks_neurons() -> None:
    """Two masks of equal size sharing a first index must not share cached features.

    A collision would put one repeat's held-out neurons into another repeat's features.
    """
    generator = np.random.default_rng(0)
    spikes = generator.poisson(0.5, size=(4, 8, 6)).astype(np.int64)
    cache = _SmoothingCache(spikes, bin_size_ms=20)
    first = np.array([0, 1, 2], dtype=np.int64)
    second = np.array([0, 4, 5], dtype=np.int64)

    first_rates = cache.rates(first, 80.0).copy()
    second_rates = cache.rates(second, 80.0)

    assert first_rates.shape == second_rates.shape
    assert not np.allclose(first_rates, second_rates)
    np.testing.assert_allclose(cache.rates(first, 80.0), first_rates)


def test_method_configurations_reject_rank_above_dimensions() -> None:
    method = {"search": {"rank": [4, 64], "alpha": [1.0]}}

    with pytest.raises(ValueError, match="exceeds the maximum usable rank"):
        method_configurations(method, n_heldin=8, n_heldout=4)


def test_selection_prefers_highest_mean_then_lower_complexity() -> None:
    configurations = [
        {"latent_dim": 8, "smoothing_sigma_ms": 200.0},
        {"latent_dim": 2, "smoothing_sigma_ms": 200.0},
        {"latent_dim": 4, "smoothing_sigma_ms": 100.0},
    ]
    inner = pd.DataFrame(
        {
            "configuration_id": [0, 0, 1, 1, 2, 2],
            "inner_unified_bits_per_spike": [0.5, 0.5, 0.5, 0.5, 0.1, 0.1],
        }
    )

    selection = select_configuration(inner, configurations)

    assert selection["selected_configuration_id"] == 1
    assert "lower_model_complexity" in selection["tie_break_reason"]
    assert json.loads(selection["selected_hyperparameters_json"]) == configurations[1]


def test_reduced_rank_mapping_respects_selected_rank() -> None:
    generator = np.random.default_rng(0)
    features = generator.normal(size=(200, 6))
    targets = generator.poisson(1.0, size=(200, 5)).astype(np.float64)

    model = fit_reduced_rank_cosmoothing(
        features, targets, bin_size_ms=20, alpha=1.0, rank=2, min_rate_hz=1e-4, max_rate_hz=500.0
    )

    assert np.linalg.matrix_rank(model["coefficients"], tol=1e-8) <= 2
    with pytest.raises(ValueError, match="exceeds the maximum rank"):
        fit_reduced_rank_cosmoothing(
            features,
            targets,
            bin_size_ms=20,
            alpha=1.0,
            rank=9,
            min_rate_hz=1e-4,
            max_rate_hz=500.0,
        )


def _outer_scores(
    differences: dict[str, list[float]], repeats: int = 5, folds: int = 5
) -> pd.DataFrame:
    rows = []
    for method_name, per_repeat in differences.items():
        for repeat in range(repeats):
            for fold in range(folds):
                rows.append(
                    {
                        "repeat_index": repeat,
                        "fold_index": fold,
                        "split_seed": 2027 + repeat,
                        "neuron_mask_seed": 2027 + repeat,
                        "method_name": method_name,
                        "method_family": "ridge" if "ridge" in method_name else "factor_latent",
                        "valid_model": method_name != "split_mean_rate_invalid",
                        "reportable_as_model_performance": method_name
                        not in ("split_mean_rate_invalid", "train_mean_rate"),
                        "invalid_reason": "",
                        "unified_bits_per_spike": per_repeat[repeat] + 0.001 * fold,
                        "notes": "",
                    }
                )
    return pd.DataFrame(rows)


def _statistics_config(minimum: float = 0.80) -> dict[str, Any]:
    return {
        "statistics": {
            "confidence_interval": 0.95,
            "bootstrap_repeats": 500,
            "bootstrap_seed": 1337,
            "comparison_unit": "repeat",
            "hierarchical_bootstrap": True,
        },
        "selection": {
            "baseline_to_beat": FACTOR_LATENT_FIXED,
            "require_positive_paired_ci": True,
            "require_positive_fraction_minimum": minimum,
            "retain_existing_baseline_when_inconclusive": True,
        },
    }


def test_repeat_level_aggregation_is_correct() -> None:
    scores = _outer_scores({FACTOR_LATENT_FIXED: [0.10] * 5})

    repeats = build_repeat_level_scores(scores)

    assert len(repeats) == 5
    row = repeats.iloc[0]
    assert row["fold_count"] == 5
    assert row["mean_unified_bits_per_spike"] == pytest.approx(0.10 + 0.001 * 2)
    assert row["neuron_mask_seed"] == 2027


def test_hierarchical_bootstrap_is_deterministic() -> None:
    paired = pd.DataFrame({"repeat_index": [0, 0, 1, 1], "difference": [0.01, 0.02, 0.03, 0.04]})

    first = hierarchical_paired_bootstrap(paired, 500, 0.95, 1337)
    second = hierarchical_paired_bootstrap(paired, 500, 0.95, 1337)

    assert first == second
    assert first[0] < first[1]


def test_hierarchical_bootstrap_resamples_repeats_not_only_folds() -> None:
    """Repeats disagree strongly; a fold-only bootstrap would understate the interval."""
    paired = pd.DataFrame(
        {"repeat_index": [0, 0, 0, 1, 1, 1], "difference": [0.01] * 3 + [0.10] * 3}
    )

    low, high = hierarchical_paired_bootstrap(paired, 2000, 0.95, 1337)
    naive_low, naive_high = bootstrap_mean_ci(
        paired["difference"].to_numpy(dtype=float), 2000, 0.95, 1337
    )

    # Resampling repeats can draw two copies of either repeat, so the interval spans both means.
    assert low == pytest.approx(0.01, abs=1e-9)
    assert high == pytest.approx(0.10, abs=1e-9)
    assert (high - low) > (naive_high - naive_low)


def test_clear_winner_replaces_factor_latent() -> None:
    config = _statistics_config()
    scores = _outer_scores(
        {FACTOR_LATENT_FIXED: [0.10] * 5, "smoothed_cosmoothing_ridge": [0.20] * 5}
    )
    repeats = build_repeat_level_scores(scores)

    comparisons = build_paired_comparisons(scores, repeats, config)
    choice = choose_baseline_to_beat(comparisons, config)

    row = comparisons.iloc[0]
    assert row["comparison_unit"] == "repeat"
    assert row["mean_paired_difference"] == pytest.approx(0.10)
    assert row["positive_repeat_fraction"] == 1.0
    assert row["ci95_low"] > 0.0
    assert bool(row["superiority_supported"]) is True
    assert choice == {
        "baseline_to_beat": "smoothed_cosmoothing_ridge",
        "baseline_replaced": True,
        "supported": True,
    }


def test_inconclusive_comparison_retains_factor_latent() -> None:
    config = _statistics_config()
    scores = _outer_scores(
        {
            FACTOR_LATENT_FIXED: [0.10, 0.10, 0.10, 0.10, 0.10],
            "smoothed_cosmoothing_ridge": [0.12, 0.08, 0.13, 0.07, 0.11],
        }
    )
    repeats = build_repeat_level_scores(scores)

    comparisons = build_paired_comparisons(scores, repeats, config)
    choice = choose_baseline_to_beat(comparisons, config)

    assert bool(comparisons.iloc[0]["superiority_supported"]) is False
    assert choice["baseline_to_beat"] == FACTOR_LATENT_FIXED
    assert choice["baseline_replaced"] is False
    assert "retained" in comparisons.iloc[0]["comparison_interpretation"]


def test_positive_fraction_gate_blocks_replacement_without_consistency() -> None:
    config = _statistics_config()
    # Wins on the mean but only on three of five repeats.
    scores = _outer_scores(
        {
            FACTOR_LATENT_FIXED: [0.10] * 5,
            "smoothed_cosmoothing_ridge": [0.30, 0.30, 0.30, 0.05, 0.05],
        }
    )
    repeats = build_repeat_level_scores(scores)

    comparisons = build_paired_comparisons(scores, repeats, config)

    assert comparisons.iloc[0]["mean_paired_difference"] > 0.0
    assert comparisons.iloc[0]["positive_repeat_fraction"] == pytest.approx(0.6)
    assert bool(comparisons.iloc[0]["superiority_supported"]) is False


def test_invalid_control_never_enters_comparisons_or_baseline() -> None:
    config = _statistics_config()
    scores = _outer_scores({FACTOR_LATENT_FIXED: [0.10] * 5, "split_mean_rate_invalid": [0.90] * 5})
    repeats = build_repeat_level_scores(scores)

    comparisons = build_paired_comparisons(scores, repeats, config)
    choice = choose_baseline_to_beat(comparisons, config)
    summary = build_method_summary(scores, choice, config)

    assert comparisons.empty
    assert choice["baseline_to_beat"] == FACTOR_LATENT_FIXED
    invalid = summary[summary["method_name"] == "split_mean_rate_invalid"].iloc[0]
    assert bool(invalid["selected_as_baseline_to_beat"]) is False
    assert "excluded from baseline selection" in invalid["selection_reason"]


def _readiness_summary(**overrides: Any) -> dict[str, Any]:
    base = {
        "dataset_hash": "hash",
        "window_name": "behavior_speed_peak_centered_1p28s",
        "target_bin_size_ms": 20,
        "factor_latent_reproduced": True,
        "outer_assignments_reused": True,
        "neuron_masks_reused": True,
        "official_leaderboard_claim": False,
        "single_split_results_reportable": False,
        "baseline_to_beat": FACTOR_LATENT_FIXED,
        "baseline_replaced": False,
        "best_valid_method_mean": 0.2,
        "factor_latent_fixed_mean": 0.12,
    }
    return {**base, **overrides}


def _readiness_config() -> dict[str, Any]:
    return {
        "outer_cross_validation": {"source_assignments_path": "assignments.csv", "repeats": 5},
        "statistics": {"comparison_unit": "repeat"},
    }


def _readiness_methods(reportable: bool = True) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method_name": FACTOR_LATENT_FIXED,
                "reportable_as_model_performance": reportable,
                "ci95_low": 0.11,
                "ci95_high": 0.13,
            }
        ]
    )


def test_readiness_records_plan_and_confirms_no_neural_training() -> None:
    readiness = build_readiness(_readiness_summary(), _readiness_config(), _readiness_methods())

    assert readiness["ready"] is True
    assert readiness["blockers"] == []
    assert readiness["required_neural_seeds"] == REQUIRED_NEURAL_SEEDS >= 5
    assert readiness["forbidden_old_protocols"] == FORBIDDEN_OLD_PROTOCOLS
    assert readiness["neural_experiment_run_during_this_milestone"] is False
    assert readiness["baseline_mean"] == pytest.approx(0.12)


def test_readiness_fails_when_baseline_reproduction_fails() -> None:
    readiness = build_readiness(
        _readiness_summary(factor_latent_reproduced=False),
        _readiness_config(),
        _readiness_methods(),
    )

    assert readiness["ready"] is False
    assert any("did not reproduce" in blocker for blocker in readiness["blockers"])


def test_readiness_fails_on_protocol_mismatch() -> None:
    readiness = build_readiness(
        _readiness_summary(outer_assignments_reused=False),
        _readiness_config(),
        _readiness_methods(),
    )

    assert readiness["ready"] is False
    assert any("not reused verbatim" in blocker for blocker in readiness["blockers"])


def test_readiness_fails_when_baseline_is_not_reportable() -> None:
    readiness = build_readiness(
        _readiness_summary(), _readiness_config(), _readiness_methods(reportable=False)
    )

    assert readiness["ready"] is False
    assert any("not a reportable valid model" in blocker for blocker in readiness["blockers"])


def test_end_to_end_toy_suite_reproduces_its_own_factor_latent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    _write_accepted_mean(config, 0.0)

    result = run_baseline_suite(config)
    outer = result["outer_scores"]

    assert len(outer) == FOLDS * REPEATS * len(config["methods"])
    train_mean = outer[outer["method_name"] == "train_mean_rate"]["unified_bits_per_spike"]
    np.testing.assert_allclose(train_mean.to_numpy(), 0.0, atol=1e-12)

    invalid = outer[outer["method_name"] == "split_mean_rate_invalid"]
    assert not invalid["valid_model"].any()
    assert not invalid["reportable_as_model_performance"].any()

    selected = result["selected_hyperparameters"]
    for method_name in (
        "factor_latent_train_selected",
        "smoothed_cosmoothing_ridge",
        "reduced_rank_cosmoothing",
    ):
        rows = selected[selected["method_name"] == method_name]
        assert len(rows) == FOLDS * REPEATS
        grid = result["configurations"][method_name]
        for row in rows.itertuples():
            assert json.loads(row.selected_hyperparameters_json) in grid

    inner = result["inner_selection"]
    assert bool(inner["selection_eligible"].all())

    # The accepted mean was written as 0.0, so reproduction must fail and block readiness.
    fixed_mean = float(
        outer[outer["method_name"] == FACTOR_LATENT_FIXED]["unified_bits_per_spike"].mean()
    )
    _write_accepted_mean(config, fixed_mean)
    repeats = build_repeat_level_scores(outer)
    comparisons = build_paired_comparisons(outer, repeats, config)
    choice = choose_baseline_to_beat(comparisons, config)
    summary = summarize_baseline_suite(result, repeats, comparisons, choice, config)

    assert summary["factor_latent_reproduced"] is True
    assert summary["factor_latent_reproduction_difference"] == pytest.approx(0.0, abs=1e-12)
    assert summary["invalid_controls_excluded"] is True
    assert summary["official_leaderboard_claim"] is False
    assert summary["single_split_results_reportable"] is False
    assert summary["old_mean_rate_values_used_as_targets"] is False
    assert summary["naive_independent_fold_test_used"] is False
    assert "split_mean_rate_invalid" not in summary["valid_methods"]
    assert summary["total_outer_evaluations"] == FOLDS * REPEATS


def test_reproduction_mismatch_blocks_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    _write_accepted_mean(config, 99.0)

    result = run_baseline_suite(config)
    repeats = build_repeat_level_scores(result["outer_scores"])
    comparisons = build_paired_comparisons(result["outer_scores"], repeats, config)
    choice = choose_baseline_to_beat(comparisons, config)
    summary = summarize_baseline_suite(result, repeats, comparisons, choice, config)
    method_summary = build_method_summary(result["outer_scores"], choice, config)
    readiness = build_readiness(summary, config, method_summary)

    assert summary["factor_latent_reproduced"] is False
    assert readiness["ready"] is False
    assert any("did not reproduce" in blocker for blocker in readiness["blockers"])
