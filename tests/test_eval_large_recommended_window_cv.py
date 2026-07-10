from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

import latentbrain.data.nlb as nlb_module
import latentbrain.eval.recommended_window_cv as cv_module
from latentbrain.data.io import save_neural_dataset
from latentbrain.data.schemas import NeuralDataset, TrialSequences
from latentbrain.eval.recommended_window_cv import (
    FACTOR_ANALYSIS_SENSITIVITY_TOLERANCE,
    build_fold_leakage_diagnostics,
    build_large_method_summary,
    build_small_large_comparison,
    build_trial_aware_window_dataset,
    evaluate_large_recommended_window_cv,
    summarize_factor_analysis_sensitivity,
    summarize_small_large_comparison,
)
from latentbrain.eval.stratified_cv import (
    FACTOR_LATENT,
    SPLIT_MEAN_RATE_INVALID,
    TRAIN_MEAN_RATE,
    build_repeated_stratified_folds,
    build_trial_features,
    score_folds,
)

NAMES = ["hand_pos_x", "hand_pos_y", "cursor_pos_x", "cursor_pos_y"]
LENGTHS = (48, 56, 64, 48, 56, 64, 48, 56, 64, 48, 56, 64)
GLOBAL_BINS = min(LENGTHS)
WINDOW_SECONDS = 0.08  # 16 source bins at 5 ms -> 4 bins at 20 ms


def _trial(length: int, angle: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    spikes = generator.poisson(0.4, size=(length, 8)).astype(np.int64)
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
    base = yaml.safe_load(
        Path("configs/mc_maze_large_recommended_window_cv.yaml").read_text(encoding="utf-8")
    )
    base["dataset"].update(
        {
            "name": "toy_large",
            "processed_path": str(processed),
            "provenance_path": str(provenance),
            "raw_dir": str(raw_dir),
            "expected_hash": dataset_hash,
        }
    )
    base["window"]["duration_seconds"] = WINDOW_SECONDS
    base["cross_validation"].update({"fold_count": 3, "repeats": 2, "min_trials_per_stratum": 2})
    base["stratification"].update(
        {
            "use_endpoint_distance": False,
            "use_mean_speed": False,
            "use_population_rate": False,
            "use_heldout_rate": False,
            "endpoint_direction_bins": 4,
        }
    )
    base["methods"][1]["latent_dim"] = 2
    base["factor_analysis_sensitivity"]["random_states"] = [0, 2027]
    base["statistics"]["bootstrap_repeats"] = 200
    base["inputs"]["small_recommended_window_summary_path"] = str(tmp_path / "absent_small.json")
    base["reporting"]["output_dir"] = str(tmp_path / "out")
    return base


def test_trial_aware_window_extracts_before_rebin_with_expected_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)

    built = build_trial_aware_window_dataset(config)
    dataset = built["dataset"]

    assert dataset.spikes.shape == (len(LENGTHS), 4, 8)
    assert dataset.behavior is not None
    assert dataset.behavior.shape[:2] == dataset.spikes.shape[:2]
    assert dataset.bin_size_ms == 20
    assert dataset.metadata["trial_source"] == "trial_aware_raw"
    assert dataset.metadata["global_crop_used_for_event_centered_windows"] is False
    np.testing.assert_array_equal(dataset.trial_ids, np.arange(len(LENGTHS)))
    assert not built["behavior_statistics"]["clipped"].any()
    assert int(built["behavior_statistics"]["padded_bins"].sum()) == 0
    assert built["trial_length_min"] == GLOBAL_BINS
    assert built["trial_length_max"] == max(LENGTHS)


def test_global_crop_source_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path, monkeypatch)
    config["trial_source"]["allow_global_crop_to_min"] = True

    with pytest.raises(ValueError, match="cannot source event-centered evaluation windows"):
        build_trial_aware_window_dataset(config)


def test_extract_before_rebin_is_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path, monkeypatch)
    config["binning"]["extract_before_rebin"] = False

    with pytest.raises(ValueError, match="extract_before_rebin must be true"):
        build_trial_aware_window_dataset(config)


def test_hash_mismatch_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path, monkeypatch)
    config["dataset"]["expected_hash"] = "0" * 64

    with pytest.raises(ValueError, match="Dataset hash mismatch"):
        build_trial_aware_window_dataset(config)


def _folds_and_scores(config: dict[str, Any]) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    built = build_trial_aware_window_dataset(config)
    dataset = built["dataset"]
    fold_config = cv_module._fold_config(config)
    from latentbrain.data.splits import create_neuron_mask

    mask = create_neuron_mask(dataset.spikes.shape[2], 0.25, seed=2027)
    features = build_trial_features(
        dataset.spikes,
        dataset.behavior,
        NAMES,
        dataset.bin_size_ms,
        np.flatnonzero(mask.heldout),
    )
    folds = build_repeated_stratified_folds(features, fold_config)
    scores = score_folds(dataset, folds, fold_config)
    return dataset, folds, scores


def test_every_trial_evaluated_exactly_once_per_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dataset, folds, _scores = _folds_and_scores(_config(tmp_path, monkeypatch))

    for _repeat, group in folds.groupby("repeat_index"):
        assert sorted(group["trial_index"]) == list(range(len(LENGTHS)))
        counts = group.groupby("fold_index").size().to_numpy()
        assert counts.max() - counts.min() <= 1


def test_neuron_mask_is_fixed_within_repeat_and_varies_across_repeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dataset, _folds, scores = _folds_and_scores(_config(tmp_path, monkeypatch))

    for _repeat, group in scores.groupby("repeat_index"):
        assert group["neuron_mask_seed"].nunique() == 1
    assert scores.groupby("repeat_index")["neuron_mask_seed"].first().nunique() == 2


def test_train_mean_rate_scores_exactly_zero_and_invalid_stays_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dataset, _folds, scores = _folds_and_scores(_config(tmp_path, monkeypatch))

    train_mean = scores[scores["method_name"] == TRAIN_MEAN_RATE]
    np.testing.assert_allclose(train_mean["unified_bits_per_spike"].to_numpy(), 0.0, atol=1e-12)
    assert not train_mean["reportable_as_model_performance"].any()
    invalid = scores[scores["method_name"] == SPLIT_MEAN_RATE_INVALID]
    assert not invalid["valid_model"].any()
    assert not invalid["reportable_as_model_performance"].any()


def test_factor_analysis_random_state_is_independent_of_fold_and_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dataset, _folds, scores = _folds_and_scores(_config(tmp_path, monkeypatch))
    factor = scores[scores["method_name"] == FACTOR_LATENT]

    assert factor["factor_analysis_random_state"].nunique() == 1
    assert int(factor["factor_analysis_random_state"].iloc[0]) == 0
    assert factor["fold_index"].nunique() > 1
    assert factor["repeat_index"].nunique() > 1


def test_sensitivity_summary_computes_range_std_and_warning() -> None:
    table = pd.DataFrame(
        {
            "factor_analysis_random_state": [0, 0, 1, 1],
            "unified_bits_per_spike": [0.10, 0.12, 0.20, 0.22],
        }
    )

    summary = summarize_factor_analysis_sensitivity(table)

    assert summary["factor_analysis_random_states"] == [0, 1]
    assert summary["factor_analysis_random_state_range"] == pytest.approx(0.10)
    assert summary["factor_analysis_random_state_std"] == pytest.approx(
        float(np.std([0.11, 0.21], ddof=1))
    )
    assert "above the" in summary["factor_analysis_random_state_warning"]


def test_sensitivity_summary_is_silent_below_tolerance() -> None:
    table = pd.DataFrame(
        {
            "factor_analysis_random_state": [0, 1],
            "unified_bits_per_spike": [0.10, 0.10 + FACTOR_ANALYSIS_SENSITIVITY_TOLERANCE / 2.0],
        }
    )

    summary = summarize_factor_analysis_sensitivity(table)

    assert summary["factor_analysis_random_state_warning"] == "none"


def test_fold_leakage_diagnostics_compute_difference_and_dominance() -> None:
    scores = pd.DataFrame(
        {
            "repeat_index": [0, 0, 0, 0],
            "fold_index": [0, 0, 1, 1],
            "method_name": [FACTOR_LATENT, SPLIT_MEAN_RATE_INVALID] * 2,
            "unified_bits_per_spike": [0.10, 0.08, 0.05, 0.09],
        }
    )

    table = build_fold_leakage_diagnostics(scores)

    assert list(table["factor_minus_invalid"]) == pytest.approx([0.02, -0.04])
    assert list(table["factor_beats_invalid"]) == [True, False]
    assert "still dominates" in table.iloc[1]["interpretation"]
    assert float(table["factor_beats_invalid"].mean()) == 0.5


def test_method_summary_reports_repeat_resolved_stability() -> None:
    scores = pd.DataFrame(
        {
            "repeat_index": [0, 0, 1, 1],
            "fold_index": [0, 1, 0, 1],
            "method_name": [FACTOR_LATENT] * 4,
            "method_type": ["factor_latent"] * 4,
            "valid_model": [True] * 4,
            "reportable_as_model_performance": [True] * 4,
            "unified_bits_per_spike": [0.10, 0.12, 0.20, 0.22],
            "notes": [""] * 4,
        }
    )

    summary = build_large_method_summary(
        scores, {"bootstrap_repeats": 100, "confidence_interval": 0.95, "bootstrap_seed": 1337}
    )

    assert int(summary.iloc[0]["n_scores"]) == 4
    assert summary.iloc[0]["between_repeat_std"] == pytest.approx(
        float(np.std([0.11, 0.21], ddof=1))
    )
    assert summary.iloc[0]["within_repeat_std"] == pytest.approx(
        float(np.std([0.10, 0.12], ddof=1))
    )


def _large_summary() -> dict[str, Any]:
    return {
        "dataset_name": "mc_maze_large",
        "fold_count": 5,
        "repeats": 5,
        "factor_latent_mean": 0.05,
        "factor_latent_std": 0.01,
        "factor_latent_ci95_low": 0.04,
        "factor_latent_ci95_high": 0.06,
        "factor_latent_positive_fraction": 1.0,
        "split_mean_invalid_mean": 0.04,
        "factor_latent_minus_split_mean_invalid": 0.01,
        "moving_bin_fraction_mean": 0.85,
        "endpoint_direction_entropy_mean": 1.82,
    }


def test_comparison_falls_back_to_references_when_small_summary_missing() -> None:
    references = {
        "small_factor_latent_mean": 0.077,
        "small_factor_latent_std": None,
        "small_factor_latent_ci95_low": 0.071,
        "small_factor_latent_ci95_high": 0.082,
        "small_factor_latent_positive_fraction": 1.0,
        "small_split_mean_invalid_mean": 0.071,
        "small_factor_minus_invalid": 0.006,
    }

    comparison = build_small_large_comparison(
        _large_summary(), None, references, {"trial_count": 500, "eval_trials_per_fold": 100}
    )
    summary = summarize_small_large_comparison(comparison, small_summary_available=False)

    assert list(comparison["dataset"]) == ["mc_maze_small", "mc_maze_large"]
    assert comparison.iloc[0]["factor_latent_mean"] == pytest.approx(0.077)
    assert pd.isna(comparison.iloc[0]["trial_count"])
    assert summary["small_summary_available"] is False
    assert summary["cross_dataset_performance_comparison_claimed"] is False


def test_small_summary_availability_does_not_depend_on_optional_keys() -> None:
    """The Small summary has no trial_count; availability must not be inferred from it."""
    small_summary = {"factor_latent_mean": 0.077, "factor_latent_std": 0.0146}

    comparison = build_small_large_comparison(
        _large_summary(), small_summary, {}, {"trial_count": 500, "eval_trials_per_fold": 100}
    )
    summary = summarize_small_large_comparison(comparison, small_summary_available=True)

    assert pd.isna(comparison.iloc[0]["trial_count"])
    assert summary["small_summary_available"] is True


def test_equal_positive_fraction_conclusion_is_grammatical() -> None:
    comparison = build_small_large_comparison(
        _large_summary(),
        {"factor_latent_positive_fraction": 1.0, "factor_latent_std": 0.0146},
        {},
        {"trial_count": 500, "eval_trials_per_fold": 100},
    )

    summary = summarize_small_large_comparison(comparison, small_summary_available=True)

    conclusions = summary["small_large_comparison_conclusions"]
    assert "Large and Small have the same positive-fold fraction" in conclusions
    assert not any("equal than" in line for line in conclusions)


def test_comparison_conclusions_are_stability_only() -> None:
    references = {"small_factor_latent_mean": 0.077, "small_factor_minus_invalid": 0.006}
    small_summary = {
        "trial_count": 100,
        "fold_count": 5,
        "repeats": 5,
        "eval_trials_per_fold": 20,
        "factor_latent_mean": 0.077,
        "factor_latent_std": 0.02,
        "factor_latent_ci95_low": 0.071,
        "factor_latent_ci95_high": 0.082,
        "factor_latent_positive_fraction": 1.0,
        "split_mean_invalid_mean": 0.071,
        "factor_latent_minus_split_mean_invalid": 0.006,
        "moving_bin_fraction_mean": 0.577,
        "endpoint_direction_entropy_mean": 2.028,
    }

    comparison = build_small_large_comparison(
        _large_summary(),
        small_summary,
        references,
        {"trial_count": 500, "eval_trials_per_fold": 100},
    )
    summary = summarize_small_large_comparison(comparison, small_summary_available=True)

    text = " ".join(summary["small_large_comparison_conclusions"]).lower()
    assert "variance" in text
    assert "positive-fold fraction" in text
    assert "leakage dominance" in text
    for forbidden in ("better than", "worse than", "outperforms", "improvement over"):
        assert forbidden not in text
    assert summary["small_large_comparison_is_protocol_stability_only"] is True


def test_end_to_end_large_cv_on_toy_trials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path, monkeypatch)

    scores, tables, summary = evaluate_large_recommended_window_cv(config)

    assert summary["total_folds"] == 6
    assert summary["time_bins"] == 4
    assert summary["trial_source"] == "trial_aware_raw"
    assert summary["global_crop_used_for_event_centered_windows"] is False
    assert summary["heldout_mask_policy"] == "fixed_within_repeat"
    assert summary["single_split_results_reportable"] is False
    assert summary["official_leaderboard_claim"] is False
    assert summary["old_mean_rate_values_used_as_targets"] is False
    assert summary["recommended_reporting_mode"] == "recommended_window_stratified_cross_validation"
    assert summary["protocol_frozen"] is True
    assert summary["invalid_controls_excluded_from_model_selection"] is True
    assert summary["cross_dataset_performance_comparison_claimed"] is False

    assert len(scores) == 6 * 3
    assert set(tables["factor_analysis_sensitivity"]["factor_analysis_random_state"]) == {0, 2027}
    assert len(tables["leakage_diagnostics"]) == 6
    assert 0.0 <= summary["factor_latent_beats_invalid_control_fraction"] <= 1.0
    assert summary["leakage_dominance_persists"] in (True, False)
