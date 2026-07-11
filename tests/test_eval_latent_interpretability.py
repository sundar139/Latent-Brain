from __future__ import annotations

import inspect

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from latentbrain.data.schemas import NeuralDataset
from latentbrain.eval.latent_interpretability import (
    _aligned_similarity,
    _circular_shift,
    _curvature,
    _empirical_p_value,
    _participation_ratio,
    _path_length,
    build_claim_registry,
    build_final_recommendation,
    derive_behavior,
    direction_decoding,
    validate_config,
)


def _dataset() -> NeuralDataset:
    behavior = np.zeros((2, 4, 4), dtype=np.float64)
    behavior[:, :, 0] = np.arange(4) * 0.02
    behavior[:, :, 1] = np.arange(4) * 0.04
    behavior[:, :, 2:] = behavior[:, :, :2]
    return NeuralDataset(
        spikes=np.zeros((2, 4, 3), dtype=np.int64),
        rates=None,
        latents=None,
        trial_ids=np.arange(2),
        time_ms=np.arange(4) * 20.0,
        bin_size_ms=20,
        metadata={},
        behavior=behavior,
        behavior_names=["hand_pos_x", "hand_pos_y", "cursor_pos_x", "cursor_pos_y"],
    )


def test_behavior_derivation_respects_bin_size_and_is_finite() -> None:
    targets, direction, distance = derive_behavior(_dataset())
    assert np.allclose(targets[:, :, 2], 1.0)
    assert np.allclose(targets[:, :, 3], 2.0)
    assert np.isfinite(targets).all()
    assert np.array_equal(direction, derive_behavior(_dataset())[1])
    assert np.all(distance > 0.0)


def test_global_crop_is_rejected() -> None:
    config = {
        "dataset": {"name": "mc_maze_large"},
        "trial_source": {"type": "trial_aware_raw", "allow_global_crop_to_min": True},
    }
    with pytest.raises(ValueError, match="global crop"):
        validate_config(config)


def test_geometry_and_alignment_known_cases() -> None:
    flat = np.column_stack([np.arange(20), np.zeros(20), np.zeros(20)])
    effective, _ = _participation_ratio(flat)
    assert effective == pytest.approx(1.0)
    trajectory = np.column_stack([np.arange(5), np.zeros(5)])
    assert _path_length(trajectory) == pytest.approx(4.0)
    assert _curvature(trajectory) == pytest.approx(0.0)
    source = np.eye(4, 2)
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    similarity = _aligned_similarity(source @ rotation, source @ rotation, source, source)
    assert similarity["aligned_centroid_correlation"] > 0.99


def test_controls_nonzero_deterministic_and_empirical_formula() -> None:
    values = np.arange(24).reshape(2, 4, 3)
    first = _circular_shift(values, np.random.default_rng(7))
    second = _circular_shift(values, np.random.default_rng(7))
    assert np.array_equal(first, second)
    assert not np.array_equal(first, values)
    assert _empirical_p_value(2.0, np.array([1.0, 2.0, 3.0])) == pytest.approx(0.75)


def test_claim_registry_blocks_unsafe_claims_and_readiness() -> None:
    controls = pd.DataFrame(
        [{"statistic": "continuous_mean_r2_across_trial_permutation", "empirical_p_value": 0.01}]
    )
    registry = build_claim_registry(
        {"continuous_mean_r2": 0.4, "direction_balanced_accuracy": 0.5}, controls
    )
    causal = registry[registry["claim_id"] == "causal_generation"].iloc[0]
    assert causal["claim_status"] == "unsupported"
    assert "causal" in causal["forbidden_wording"].lower()
    recommendation = build_final_recommendation(
        registry,
        {
            "all_25_outer_folds_complete": False,
            "baseline_scores_reproduced": True,
            "behavior_decoding_complete": True,
            "direction_decoding_complete": True,
            "shuffle_controls_complete": True,
            "representation_stability_complete": True,
        },
    )
    assert recommendation["ready_for_final_report"] is False
    assert recommendation["final_report_blockers"]


def test_direction_decoder_is_deterministic_native_eight_class() -> None:
    rng = np.random.default_rng(12)
    labels = np.tile(np.arange(8), 10)
    latents = np.eye(8)[labels][:, None, :] + rng.normal(0.0, 0.01, (80, 4, 8))
    record = {
        "repeat_index": 0,
        "fold_index": 0,
        "train_trials": np.arange(64),
        "eval_trials": np.arange(64, 80),
        "train_latents": latents[:64],
        "eval_latents": latents[64:],
    }
    config = {"decoding": {"inner_folds": 3}}
    first, first_confusion = direction_decoding([record], labels, config)
    second, second_confusion = direction_decoding([record], labels, config)
    assert "multi_class" not in inspect.getsource(direction_decoding)
    assert first.iloc[0]["selected_C"] in {0.01, 0.1, 1.0, 10.0}
    metrics = first.iloc[0][["balanced_accuracy", "macro_f1"]].astype(float)
    assert np.isfinite(metrics).all()
    assert first_confusion.shape == (8, 8)
    assert np.array_equal(first_confusion, second_confusion)
    assert first.equals(second)
