"""Out-of-fold latent interpretability and neuroscience-validity analysis for MC_Maze Large.

Reuses the accepted factor-latent fit path (``split_audit.factor_latent_fit_transform``), the exact
outer folds and neuron masks (``baseline_suite.load_outer_folds``), and the frozen movement window.
No neural model is trained; no factor-latent prediction or accepted score is altered. All evaluation
latents are produced out of fold, every decoder/alignment is fit on outer-training data only, and
every finding is checked against train-only shuffle controls. Interpretations are associative, never
causal, and never an official NLB leaderboard claim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml
from scipy.linalg import orthogonal_procrustes, subspace_angles  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from latentbrain.data.schemas import NeuralDataset, TrialSplit
from latentbrain.eval.baseline_suite import build_window_dataset, load_outer_folds
from latentbrain.eval.decoding import (
    apply_standardization,
    fit_ridge_decoder,
    predict_ridge_decoder,
    r2_score_numpy,
    standardize_train_apply,
)
from latentbrain.eval.scoring import (
    ScoringConfig,
    score_heldout_prediction,
    train_heldout_mean_rate_reference,
)
from latentbrain.eval.seed_robustness import bootstrap_mean_ci
from latentbrain.eval.split_audit import factor_latent_fit_transform
from latentbrain.paths import get_repo_root, resolve_configured_path

DIRECTION_BINS = 8
CAUSAL_WORDS = ("cause", "causal", "causes", "generate", "drive the", "mechanistically")
LEADERBOARD_WORDS = ("leaderboard", "official nlb", "state-of-the-art", "sota", "benchmark score")
CONTINUOUS_TARGETS = [
    "hand_pos_x",
    "hand_pos_y",
    "hand_vel_x",
    "hand_vel_y",
    "hand_speed",
    "cursor_pos_x",
    "cursor_pos_y",
    "cursor_vel_x",
    "cursor_vel_y",
    "cursor_speed",
]


def _resolve(path: str) -> Path:
    return resolve_configured_path(path, get_repo_root())


# --------------------------------------------------------------------------- config / inputs


def validate_config(config: dict[str, Any]) -> None:
    if str(config["dataset"]["name"]) != "mc_maze_large":
        raise ValueError("latent interpretability is defined for mc_maze_large only")
    if str(config["trial_source"]["type"]) != "trial_aware_raw" or bool(
        config["trial_source"]["allow_global_crop_to_min"]
    ):
        raise ValueError("trial-aware raw input required and global crop forbidden")
    if not bool(config["window"]["extract_before_rebin"]):
        raise ValueError("event-centered extraction must precede rebinning")
    outer = config["outer_protocol"]
    if not bool(outer["reuse_exact_assignments"]) or not bool(outer["reuse_exact_neuron_masks"]):
        raise ValueError("exact outer assignments and neuron masks must be reused")
    if str(config["model"]["method"]) != "factor_latent_train_selected":
        raise ValueError("interpretability analyses the frozen factor_latent_train_selected model")
    if not bool(config["model"]["require_per_fold_selected_hyperparameters"]):
        raise ValueError("per-fold outer-training-selected hyperparameters are required")
    if not bool(config["alignment"]["fit_using_outer_training_only"]):
        raise ValueError("representation alignment must be fit on outer-training data only")
    if not bool(config["controls"]["preserve_training_only_fit"]):
        raise ValueError("shuffle controls must preserve the train-only fitting policy")
    claims = config["claim_safety"]
    if (
        not bool(claims["require_out_of_fold_analysis"])
        or not bool(claims["prohibit_causal_claims"])
        or not bool(claims["prohibit_official_leaderboard_claims"])
    ):
        raise ValueError("claim-safety restrictions must remain enabled")


def load_inputs(config: dict[str, Any]) -> dict[str, Any]:
    outer = config["outer_protocol"]
    protocol = yaml.safe_load(_resolve(str(outer["protocol_path"])).read_text(encoding="utf-8"))
    if str(protocol["dataset"]["expected_hash"]) != str(config["dataset"]["expected_hash"]):
        raise ValueError("protocol dataset hash does not match config")
    dataset, dataset_hash = build_window_dataset(protocol)
    if dataset_hash != str(config["dataset"]["expected_hash"]):
        raise ValueError("rebuilt dataset hash does not match config")
    if dataset.spikes.shape != (500, 64, 162):
        raise ValueError(f"dataset shape {dataset.spikes.shape} does not match frozen [500,64,162]")
    fold_config = {
        "outer_cross_validation": {
            "source_assignments_path": str(outer["assignments_path"]),
            "fold_count": int(outer["fold_count"]),
            "repeats": int(outer["repeats"]),
            "base_seed": int(outer["base_seed"]),
            "reuse_exact_assignments": True,
            "reuse_exact_neuron_masks": True,
        }
    }
    folds = load_outer_folds(fold_config, protocol, dataset.spikes.shape[2])
    selected = pd.read_csv(_resolve(str(outer["selected_hyperparameters_path"])))
    selected = selected[selected["method_name"] == "factor_latent_train_selected"]
    baseline = pd.read_csv(_resolve(str(outer["baseline_scores_path"])))
    baseline = baseline[baseline["method_name"] == "factor_latent_train_selected"]
    return {
        "dataset": dataset,
        "dataset_hash": dataset_hash,
        "folds": folds,
        "selected": selected,
        "baseline": baseline,
    }


def _scoring() -> ScoringConfig:
    return ScoringConfig(
        bin_size_ms=20,
        include_poisson_constant=True,
        min_rate_hz=1.0e-4,
        max_rate_hz=500.0,
        reference_name="train_heldout_mean_rate",
    )


def _fold_settings(selected: pd.DataFrame, repeat: int, fold: int) -> dict[str, float]:
    row = selected[
        (selected["outer_repeat_index"] == repeat) & (selected["outer_fold_index"] == fold)
    ]
    if len(row) != 1:
        raise ValueError(f"missing selected hyperparameters for repeat {repeat} fold {fold}")
    params = json.loads(str(row.iloc[0]["selected_hyperparameters_json"]))
    return {
        "latent_dim": float(params["latent_dim"]),
        "smoothing_sigma_ms": float(params["smoothing_sigma_ms"]),
        "heldout_decoder_alpha": float(params["heldout_decoder_alpha"]),
        "max_iter": 1000.0,
        "tol": 1.0e-4,
        "factor_analysis_random_state": float(params["factor_analysis_random_state"]),
    }


# --------------------------------------------------------------------------- behavior


def derive_behavior(dataset: NeuralDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-(trial,time) continuous targets, endpoint-direction bins, endpoint distance."""
    names = list(dataset.behavior_names or [])
    behavior = np.asarray(dataset.behavior, dtype=np.float64)
    dt = dataset.bin_size_ms / 1000.0

    def channel(name: str) -> np.ndarray:
        return behavior[:, :, names.index(name)]

    def velocity(x: np.ndarray) -> np.ndarray:
        step = np.diff(x, axis=1) / dt
        return np.concatenate([step[:, :1], step], axis=1)

    hx, hy = channel("hand_pos_x"), channel("hand_pos_y")
    cx, cy = channel("cursor_pos_x"), channel("cursor_pos_y")
    hvx, hvy = velocity(hx), velocity(hy)
    cvx, cvy = velocity(cx), velocity(cy)
    columns = {
        "hand_pos_x": hx,
        "hand_pos_y": hy,
        "hand_vel_x": hvx,
        "hand_vel_y": hvy,
        "hand_speed": np.hypot(hvx, hvy),
        "cursor_pos_x": cx,
        "cursor_pos_y": cy,
        "cursor_vel_x": cvx,
        "cursor_vel_y": cvy,
        "cursor_speed": np.hypot(cvx, cvy),
    }
    targets = np.stack([columns[name] for name in CONTINUOUS_TARGETS], axis=-1)
    dx = hx[:, -1] - hx[:, 0]
    dy = hy[:, -1] - hy[:, 0]
    angle = np.arctan2(dy, dx)
    shifted = (angle + np.pi) % (2.0 * np.pi)
    direction = np.clip(
        (shifted / (2.0 * np.pi / DIRECTION_BINS)).astype(int), 0, DIRECTION_BINS - 1
    )
    distance = np.hypot(dx, dy)
    if not np.isfinite(targets).all():
        raise ValueError("derived continuous behavior targets contain non-finite values")
    return targets, np.asarray(direction, dtype=np.int64), np.asarray(distance, dtype=np.float64)


# --------------------------------------------------------------------------- out-of-fold latents


def build_fold_latents(inputs: dict[str, Any]) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Fit factor-latent per fold on outer-training only; expose out-of-fold evaluation latents."""
    dataset = inputs["dataset"]
    scoring = _scoring()
    records: list[dict[str, Any]] = []
    stat_rows: list[dict[str, Any]] = []
    for fold in inputs["folds"]:
        settings = _fold_settings(inputs["selected"], fold.repeat_index, fold.fold_index)
        split = TrialSplit(
            train=fold.train_trials, validation=fold.eval_trials, test=fold.eval_trials
        )
        result = factor_latent_fit_transform(
            dataset,
            split,
            fold.heldin,
            fold.heldout,
            scoring,
            settings,
            int(settings["factor_analysis_random_state"]),
        )
        eval_counts = dataset.spikes[np.isin(dataset.trial_ids, fold.eval_trials)][
            :, :, fold.heldout
        ]
        train_counts = dataset.spikes[np.isin(dataset.trial_ids, fold.train_trials)][
            :, :, fold.heldout
        ]
        reference = train_heldout_mean_rate_reference(train_counts, eval_counts.shape, scoring)
        scored = score_heldout_prediction(
            eval_counts,
            result["rates"]["validation"],
            reference,
            scoring,
            "factor_latent",
            "validation",
            "direct_model",
            True,
        )
        accepted = inputs["baseline"][
            (inputs["baseline"]["repeat_index"] == fold.repeat_index)
            & (inputs["baseline"]["fold_index"] == fold.fold_index)
        ]
        accepted_score = float(accepted.iloc[0]["unified_bits_per_spike"])
        reproduced = float(scored["bits_per_spike"])
        records.append(
            {
                "repeat_index": fold.repeat_index,
                "fold_index": fold.fold_index,
                "eval_trials": fold.eval_trials,
                "train_trials": fold.train_trials,
                "eval_latents": result["latents"]["validation"],
                "train_latents": result["latents"]["train"],
                "eval_input_rates": None,
            }
        )
        latent = result["latents"]["validation"]
        stat_rows.append(
            {
                "repeat_index": fold.repeat_index,
                "fold_index": fold.fold_index,
                "latent_dim": int(latent.shape[2]),
                "eval_trial_count": int(latent.shape[0]),
                "accepted_unified_bits_per_spike": accepted_score,
                "reproduced_unified_bits_per_spike": reproduced,
                "absolute_reproduction_error": abs(reproduced - accepted_score),
                "score_reproduced": bool(abs(reproduced - accepted_score) < 1.0e-8),
                "mean_latent_norm": float(np.linalg.norm(latent, axis=-1).mean()),
            }
        )
    return records, pd.DataFrame(stat_rows)


# --------------------------------------------------------------------------- continuous decoding


def _inner_fold_indices(n: int, folds: int, seed: int) -> list[np.ndarray]:
    order = np.random.default_rng(seed).permutation(n)
    return [order[i::folds] for i in range(folds)]


def _flatten(latents: np.ndarray) -> np.ndarray:
    return latents.reshape(latents.shape[0] * latents.shape[1], latents.shape[2])


def _select_alpha(
    train_latents: np.ndarray,
    train_targets: np.ndarray,
    alphas: list[float],
    inner_folds: int,
    seed: int,
) -> float:
    trials = train_latents.shape[0]
    splits = _inner_fold_indices(trials, inner_folds, seed)
    best_alpha, best_score = alphas[0], -np.inf
    for alpha in alphas:
        scores: list[float] = []
        for held in splits:
            fit = np.setdiff1d(np.arange(trials), held)
            fx, stats = standardize_train_apply(
                _flatten(train_latents[fit]), _flatten(train_latents[fit])
            )
            vx = apply_standardization(_flatten(train_latents[held]), stats)
            decoder = fit_ridge_decoder(fx, _flatten(train_targets[fit]), alpha=alpha)
            predicted = predict_ridge_decoder(vx, decoder)
            scores.append(
                float(r2_score_numpy(_flatten(train_targets[held]), predicted, "uniform_average"))
            )
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score, best_alpha = mean_score, alpha
    return best_alpha


def continuous_decoding(
    records: list[dict[str, Any]], targets: np.ndarray, config: dict[str, Any]
) -> pd.DataFrame:
    alphas = [float(a) for a in config["decoding"]["decoder"]["alpha_grid"]]
    inner_folds = int(config["decoding"]["inner_folds"])
    rows: list[dict[str, Any]] = []
    for record in records:
        train_targets = targets[record["train_trials"]]
        eval_targets = targets[record["eval_trials"]]
        train_latents, eval_latents = record["train_latents"], record["eval_latents"]
        seed = 4041 + record["fold_index"]
        alpha = _select_alpha(train_latents, train_targets, alphas, inner_folds, seed)
        fx, stats = standardize_train_apply(_flatten(train_latents), _flatten(train_latents))
        ex = apply_standardization(_flatten(eval_latents), stats)
        decoder = fit_ridge_decoder(fx, _flatten(train_targets), alpha=alpha)
        predicted = predict_ridge_decoder(ex, decoder)
        observed = _flatten(eval_targets)
        r2 = np.asarray(r2_score_numpy(observed, predicted), dtype=np.float64)
        for index, name in enumerate(CONTINUOUS_TARGETS):
            true_col, pred_col = observed[:, index], predicted[:, index]
            std = float(np.std(true_col))
            corr = (
                float(np.corrcoef(true_col, pred_col)[0, 1])
                if np.std(pred_col) > 0.0 and std > 0.0
                else 0.0
            )
            rmse = float(np.sqrt(np.mean((true_col - pred_col) ** 2)))
            rows.append(
                {
                    "repeat_index": record["repeat_index"],
                    "fold_index": record["fold_index"],
                    "target_name": name,
                    "selected_alpha": alpha,
                    "outer_r2": float(r2[index]),
                    "outer_correlation": corr,
                    "outer_rmse": rmse,
                    "outer_normalized_rmse": rmse / std if std > 0.0 else float("nan"),
                    "train_trial_count": int(train_latents.shape[0]),
                    "eval_trial_count": int(eval_latents.shape[0]),
                    "valid_model": True,
                    "notes": "out-of-fold; alpha selected on inner outer-training folds",
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- direction decoding


def _trial_features(latents: np.ndarray) -> np.ndarray:
    return np.asarray(latents.mean(axis=1))


def direction_decoding(
    records: list[dict[str, Any]], direction: np.ndarray, config: dict[str, Any]
) -> tuple[pd.DataFrame, np.ndarray]:
    inner_folds = int(config["decoding"]["inner_folds"])
    c_grid = [0.01, 0.1, 1.0, 10.0]
    rows: list[dict[str, Any]] = []
    confusion_total = np.zeros((DIRECTION_BINS, DIRECTION_BINS), dtype=np.float64)
    for record in records:
        train_x_raw = _trial_features(record["train_latents"])
        eval_x_raw = _trial_features(record["eval_latents"])
        train_y = direction[record["train_trials"]]
        eval_y = direction[record["eval_trials"]]
        seed = 5051 + record["fold_index"]
        best_c = _select_direction_c(train_x_raw, train_y, c_grid, inner_folds, seed)
        train_x, stats = standardize_train_apply(train_x_raw, train_x_raw)
        eval_x = apply_standardization(eval_x_raw, stats)
        model = LogisticRegression(C=best_c, solver="lbfgs", max_iter=2000, random_state=seed)
        model.fit(train_x, train_y)
        if not np.array_equal(np.sort(model.classes_), np.sort(np.unique(train_y))):
            raise ValueError("direction classifier did not fit every outer-training class")
        predicted = model.predict(eval_x)
        labels = list(range(DIRECTION_BINS))
        confusion_total += confusion_matrix(eval_y, predicted, labels=labels).astype(np.float64)
        rows.append(
            {
                "repeat_index": record["repeat_index"],
                "fold_index": record["fold_index"],
                "selected_C": best_c,
                "accuracy": float(np.mean(predicted == eval_y)),
                "balanced_accuracy": float(balanced_accuracy_score(eval_y, predicted)),
                "macro_f1": float(
                    f1_score(eval_y, predicted, labels=labels, average="macro", zero_division=0)
                ),
                "chance_level": 1.0 / DIRECTION_BINS,
                "eval_trial_count": int(eval_y.size),
                "notes": "out-of-fold; C selected on inner outer-training folds",
            }
        )
    return pd.DataFrame(rows), confusion_total


def _select_direction_c(
    train_x: np.ndarray, train_y: np.ndarray, c_grid: list[float], inner_folds: int, seed: int
) -> float:
    trials = train_x.shape[0]
    splits = _inner_fold_indices(trials, inner_folds, seed)
    best_c, best_score = c_grid[0], -np.inf
    for c in c_grid:
        scores: list[float] = []
        for held in splits:
            fit = np.setdiff1d(np.arange(trials), held)
            if np.unique(train_y[fit]).size < 2:
                continue
            fx, stats = standardize_train_apply(train_x[fit], train_x[fit])
            vx = apply_standardization(train_x[held], stats)
            model = LogisticRegression(C=c, solver="lbfgs", max_iter=1000, random_state=seed)
            model.fit(fx, train_y[fit])
            if not np.array_equal(np.sort(model.classes_), np.sort(np.unique(train_y[fit]))):
                raise ValueError("direction classifier did not fit every inner-training class")
            scores.append(float(balanced_accuracy_score(train_y[held], model.predict(vx))))
        mean_score = float(np.mean(scores)) if scores else -np.inf
        if mean_score > best_score:
            best_score, best_c = mean_score, c
    return best_c


# --------------------------------------------------------------------------- geometry


def _direction_centroids(
    latents: np.ndarray, labels: np.ndarray, minimum: int
) -> dict[int, np.ndarray]:
    centroids: dict[int, np.ndarray] = {}
    for bin_index in range(DIRECTION_BINS):
        mask = labels == bin_index
        if int(mask.sum()) >= minimum:
            centroids[bin_index] = latents[mask].mean(axis=0)
    return centroids


def _path_length(trajectory: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum())


def _curvature(trajectory: np.ndarray) -> float:
    steps = np.diff(trajectory, axis=0)
    norms = np.linalg.norm(steps, axis=1)
    valid = norms > 1e-12
    if valid.sum() < 2:
        return 0.0
    unit = steps[valid] / norms[valid][:, None]
    cosines = np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1.0, 1.0)
    return float(np.mean(np.arccos(cosines)))


def _participation_ratio(flat: np.ndarray) -> tuple[float, np.ndarray]:
    covariance = np.cov(flat, rowvar=False)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance)[::-1], 0.0, None)
    total = float(eigenvalues.sum())
    if total <= np.finfo(np.float64).eps:
        return 0.0, eigenvalues
    return float(total**2 / float(np.sum(eigenvalues**2))), eigenvalues


def latent_geometry(
    records: list[dict[str, Any]],
    direction: np.ndarray,
    distance: np.ndarray,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    minimum = int(config["geometry"]["minimum_trials_per_condition"])
    distance_bins = int(config["geometry"]["distance_bins"])
    traj_rows: list[dict[str, Any]] = []
    sep_rows: list[dict[str, Any]] = []
    dim_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    for record in records:
        latents = record["eval_latents"]
        eval_dir = direction[record["eval_trials"]]
        eval_dist = distance[record["eval_trials"]]
        edges = np.quantile(eval_dist, np.linspace(0.0, 1.0, distance_bins + 1)[1:-1])
        dist_bin = np.clip(np.digitize(eval_dist, edges), 0, distance_bins - 1)
        centroids = _direction_centroids(latents, eval_dir, minimum)
        for bin_index, centroid in centroids.items():
            traj_rows.append(_trajectory_row(record, "direction", bin_index, centroid))
        for bin_index in range(distance_bins):
            mask = dist_bin == bin_index
            if int(mask.sum()) >= minimum:
                traj_rows.append(
                    _trajectory_row(record, "distance", bin_index, latents[mask].mean(axis=0))
                )
        if len(centroids) >= 2:
            sep_rows.extend(_separability_rows(record, latents, eval_dir, centroids))
        flat = _flatten(latents)
        participation, eigenvalues = _participation_ratio(flat)
        cumulative = np.cumsum(eigenvalues) / max(
            float(eigenvalues.sum()), np.finfo(np.float64).eps
        )
        dim_rows.append(
            {
                "repeat_index": record["repeat_index"],
                "fold_index": record["fold_index"],
                "latent_dim": int(latents.shape[2]),
                "participation_ratio": participation,
                "dims_for_90pct_variance": int(np.searchsorted(cumulative, 0.90) + 1),
                "top_eigenvalue": float(eigenvalues[0]),
                "eigenvalue_spectrum": eigenvalues.tolist(),
            }
        )
        if len(centroids) >= 2:
            mean_centroid = np.mean(np.stack(list(centroids.values()), axis=0), axis=0)
            velocity = np.linalg.norm(np.diff(mean_centroid, axis=0), axis=1)
            displacement = np.linalg.norm(mean_centroid - mean_centroid[0], axis=1)
            temporal_rows.append(
                {
                    "repeat_index": record["repeat_index"],
                    "fold_index": record["fold_index"],
                    "peak_latent_velocity_time_bin": int(np.argmax(velocity) + 1),
                    "final_distance_from_pre_movement": float(displacement[-1]),
                    "monotonic_progression_fraction": float(np.mean(np.diff(displacement) > 0.0)),
                }
            )
    return (
        pd.DataFrame(traj_rows),
        pd.DataFrame(sep_rows),
        pd.DataFrame(dim_rows),
        pd.DataFrame(temporal_rows),
    )


def _trajectory_row(
    record: dict[str, Any], condition_type: str, value: int, centroid: np.ndarray
) -> dict[str, Any]:
    return {
        "repeat_index": record["repeat_index"],
        "fold_index": record["fold_index"],
        "condition_type": condition_type,
        "condition_value": int(value),
        "path_length": _path_length(centroid),
        "curvature": _curvature(centroid),
        "max_displacement": float(np.linalg.norm(centroid - centroid[0], axis=1).max()),
        "peak_velocity_time_bin": int(
            np.argmax(np.linalg.norm(np.diff(centroid, axis=0), axis=1)) + 1
        ),
    }


def _separability_rows(
    record: dict[str, Any],
    latents: np.ndarray,
    eval_dir: np.ndarray,
    centroids: dict[int, np.ndarray],
) -> list[dict[str, Any]]:
    stacked = np.stack(list(centroids.values()), axis=0)  # [conditions, T, L]
    rows: list[dict[str, Any]] = []
    for time_bin in range(latents.shape[1]):
        slice_c = stacked[:, time_bin, :]
        grand = slice_c.mean(axis=0)
        between = float(np.mean(np.linalg.norm(slice_c - grand, axis=1)))
        within_values = [
            float(
                np.mean(
                    np.linalg.norm(latents[eval_dir == b][:, time_bin, :] - c[time_bin], axis=1)
                )
            )
            for b, c in centroids.items()
        ]
        within = float(np.mean(within_values))
        rows.append(
            {
                "repeat_index": record["repeat_index"],
                "fold_index": record["fold_index"],
                "time_bin": time_bin,
                "relative_time_seconds": (time_bin - latents.shape[1] // 2) * 0.02,
                "within_condition_dispersion": within,
                "between_condition_distance": between,
                "separability_ratio": between / within if within > 0.0 else float("nan"),
            }
        )
    return rows


# --------------------------------------------------------------------------- stability


def _mean_direction_centroids(latents: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Time-mean latent centroid per direction bin, [DIRECTION_BINS, latent]; empty bins are NaN."""
    dim = latents.shape[2]
    matrix = np.full((DIRECTION_BINS, dim), np.nan, dtype=np.float64)
    for bin_index in range(DIRECTION_BINS):
        mask = labels == bin_index
        if mask.any():
            matrix[bin_index] = latents[mask].mean(axis=(0, 1))
    return matrix


def _aligned_similarity(
    source_train: np.ndarray,
    source_eval: np.ndarray,
    ref_train: np.ndarray,
    ref_eval: np.ndarray,
) -> dict[str, float]:
    from scipy.spatial.distance import pdist  # type: ignore[import-untyped]  # noqa: PLC0415

    shared = np.all(np.isfinite(source_train), axis=1) & np.all(np.isfinite(ref_train), axis=1)
    shared &= np.all(np.isfinite(source_eval), axis=1) & np.all(np.isfinite(ref_eval), axis=1)
    if int(shared.sum()) < 4:
        return {"aligned_centroid_correlation": float("nan"), "rsa_correlation": float("nan")}
    rotation, _ = orthogonal_procrustes(source_train[shared], ref_train[shared])
    aligned = source_eval[shared] @ rotation
    reference = ref_eval[shared]
    correlation = float(np.corrcoef(aligned.reshape(-1), reference.reshape(-1))[0, 1])
    rsa = float(np.corrcoef(pdist(aligned), pdist(reference))[0, 1])
    return {"aligned_centroid_correlation": correlation, "rsa_correlation": rsa}


def _subspace_similarity(source_eval: np.ndarray, ref_eval: np.ndarray, k: int = 3) -> float:
    def top_subspace(latents: np.ndarray) -> np.ndarray:
        flat = _flatten(latents)
        centered = flat - flat.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return np.asarray(vt[: min(k, vt.shape[0])].T)

    angles = subspace_angles(top_subspace(source_eval), top_subspace(ref_eval))
    return float(np.mean(np.cos(angles)))


def representation_stability(
    records: list[dict[str, Any]],
    inputs: dict[str, Any],
    direction: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    by_key = {(r["repeat_index"], r["fold_index"]): r for r in records}
    rows: list[dict[str, Any]] = []
    reference = by_key[(0, 0)]
    ref_train_c = _mean_direction_centroids(
        reference["train_latents"], direction[reference["train_trials"]]
    )
    ref_eval_c = _mean_direction_centroids(
        reference["eval_latents"], direction[reference["eval_trials"]]
    )

    def compare(record: dict[str, Any], comparison_type: str) -> None:
        source_train_c = _mean_direction_centroids(
            record["train_latents"], direction[record["train_trials"]]
        )
        source_eval_c = _mean_direction_centroids(
            record["eval_latents"], direction[record["eval_trials"]]
        )
        similarity = _aligned_similarity(source_train_c, source_eval_c, ref_train_c, ref_eval_c)
        rows.append(
            {
                "comparison_type": comparison_type,
                "source_repeat": record["repeat_index"],
                "source_fold": record["fold_index"],
                "reference_repeat": 0,
                "reference_fold": 0,
                **similarity,
                "subspace_cosine": _subspace_similarity(
                    record["eval_latents"], reference["eval_latents"]
                ),
            }
        )

    for fold in range(1, int(config["outer_protocol"]["fold_count"])):
        compare(by_key[(0, fold)], "fold_within_repeat")
    for repeat in range(1, int(config["outer_protocol"]["repeats"])):
        compare(by_key[(repeat, 0)], "repeat_across_mask")

    rows.extend(_factor_analysis_state_stability(inputs, direction, config))
    return pd.DataFrame(rows)


def _factor_analysis_state_stability(
    inputs: dict[str, Any], direction: np.ndarray, config: dict[str, Any]
) -> list[dict[str, Any]]:
    dataset = inputs["dataset"]
    scoring = _scoring()
    ref_fold = next(f for f in inputs["folds"] if f.repeat_index == 0 and f.fold_index == 0)
    settings = _fold_settings(inputs["selected"], 0, 0)
    split = TrialSplit(
        train=ref_fold.train_trials, validation=ref_fold.eval_trials, test=ref_fold.eval_trials
    )
    states = [int(s) for s in config["stability"]["factor_analysis_states"]]
    latents_by_state = {
        state: factor_latent_fit_transform(
            dataset, split, ref_fold.heldin, ref_fold.heldout, scoring, settings, state
        )["latents"]
        for state in states
    }
    base_state = states[0]
    base = latents_by_state[base_state]
    base_train_c = _mean_direction_centroids(base["train"], direction[ref_fold.train_trials])
    base_eval_c = _mean_direction_centroids(base["validation"], direction[ref_fold.eval_trials])
    rows: list[dict[str, Any]] = []
    for state, latents in latents_by_state.items():
        if state == base_state:
            continue
        source_train_c = _mean_direction_centroids(
            latents["train"], direction[ref_fold.train_trials]
        )
        source_eval_c = _mean_direction_centroids(
            latents["validation"], direction[ref_fold.eval_trials]
        )
        similarity = _aligned_similarity(source_train_c, source_eval_c, base_train_c, base_eval_c)
        rows.append(
            {
                "comparison_type": "factor_analysis_state",
                "source_repeat": state,
                "source_fold": 0,
                "reference_repeat": base_state,
                "reference_fold": 0,
                **similarity,
                "subspace_cosine": _subspace_similarity(latents["validation"], base["validation"]),
            }
        )
    return rows


# --------------------------------------------------------------------------- rate confound


def _mean_r2(
    train_x: np.ndarray, eval_x: np.ndarray, train_y: np.ndarray, eval_y: np.ndarray
) -> float:
    fx, stats = standardize_train_apply(train_x, train_x)
    ex = apply_standardization(eval_x, stats)
    decoder = fit_ridge_decoder(fx, train_y, alpha=100.0)
    predicted = predict_ridge_decoder(ex, decoder)
    return float(r2_score_numpy(eval_y, predicted, "uniform_average"))


def rate_confound(
    records: list[dict[str, Any]], inputs: dict[str, Any], targets: np.ndarray
) -> pd.DataFrame:
    dataset = inputs["dataset"]
    rows: list[dict[str, Any]] = []
    fold_masks = {(f.repeat_index, f.fold_index): f for f in inputs["folds"]}
    for record in records:
        fold = fold_masks[(record["repeat_index"], record["fold_index"])]
        pop = dataset.spikes[:, :, fold.heldin].mean(axis=2, keepdims=True)
        train_y = _flatten(targets[record["train_trials"]])
        eval_y = _flatten(targets[record["eval_trials"]])
        train_latent = _flatten(record["train_latents"])
        eval_latent = _flatten(record["eval_latents"])
        train_pop = _flatten(pop[record["train_trials"]])
        eval_pop = _flatten(pop[record["eval_trials"]])
        centered = train_latent - train_latent.mean(axis=0)
        axis = np.linalg.svd(centered, full_matrices=False)[2][0]
        train_pc1 = (train_latent @ axis)[:, None]
        eval_pc1 = (eval_latent @ axis)[:, None]
        rate_decoder = fit_ridge_decoder(train_pop, train_latent, alpha=1.0)
        train_resid = train_latent - predict_ridge_decoder(train_pop, rate_decoder)
        eval_resid = eval_latent - predict_ridge_decoder(eval_pop, rate_decoder)
        rows.append(
            {
                "repeat_index": record["repeat_index"],
                "fold_index": record["fold_index"],
                "factor_latent_mean_r2": _mean_r2(train_latent, eval_latent, train_y, eval_y),
                "population_rate_mean_r2": _mean_r2(train_pop, eval_pop, train_y, eval_y),
                "first_pc_mean_r2": _mean_r2(train_pc1, eval_pc1, train_y, eval_y),
                "rate_regressed_latent_mean_r2": _mean_r2(train_resid, eval_resid, train_y, eval_y),
                "notes": "diagnostic comparison; not accepted model performance",
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- shuffle controls


def _continuous_statistic(
    records: list[dict[str, Any]], targets: np.ndarray, alphas: dict[int, float]
) -> float:
    scores: list[float] = []
    for record in records:
        fold = record["fold_index"]
        fx, stats = standardize_train_apply(
            _flatten(record["train_latents"]), _flatten(record["train_latents"])
        )
        ex = apply_standardization(_flatten(record["eval_latents"]), stats)
        decoder = fit_ridge_decoder(
            fx, _flatten(targets[record["train_trials"]]), alpha=alphas[fold]
        )
        predicted = predict_ridge_decoder(ex, decoder)
        scores.append(
            float(
                r2_score_numpy(
                    _flatten(targets[record["eval_trials"]]), predicted, "uniform_average"
                )
            )
        )
    return float(np.mean(scores))


def _direction_statistic(
    records: list[dict[str, Any]], labels: np.ndarray, cs: dict[int, float]
) -> float:
    scores: list[float] = []
    for record in records:
        fold = record["fold_index"]
        train_x, stats = standardize_train_apply(
            _trial_features(record["train_latents"]), _trial_features(record["train_latents"])
        )
        eval_x = apply_standardization(_trial_features(record["eval_latents"]), stats)
        train_y = labels[record["train_trials"]]
        if np.unique(train_y).size < 2:
            continue
        model = LogisticRegression(C=cs[fold], solver="lbfgs", max_iter=500, random_state=0)
        model.fit(train_x, train_y)
        if not np.array_equal(np.sort(model.classes_), np.sort(np.unique(train_y))):
            raise ValueError("direction control classifier did not fit every training class")
        scores.append(
            float(balanced_accuracy_score(labels[record["eval_trials"]], model.predict(eval_x)))
        )
    return float(np.mean(scores)) if scores else float("nan")


def _permute_across_trials(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return values[rng.permutation(values.shape[0])]


def _circular_shift(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shifted = values.copy()
    time_bins = values.shape[1]
    for index in range(values.shape[0]):
        shifted[index] = np.roll(values[index], int(rng.integers(1, time_bins)), axis=0)
    return shifted


def _empirical_p_value(observed: float, controls: np.ndarray) -> float:
    values = np.asarray(controls, dtype=np.float64)
    return float((1 + np.sum(values >= observed)) / (1 + values.size))


def shuffle_controls(
    records: list[dict[str, Any]],
    targets: np.ndarray,
    direction: np.ndarray,
    continuous_frame: pd.DataFrame,
    direction_frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    repeats = int(config["controls"]["permutation_repeats"])
    seed = int(config["controls"]["permutation_seed"])
    confidence = float(config["statistics"]["confidence_interval"])
    boot = int(config["statistics"]["bootstrap_repeats"])
    alphas = {
        int(f): float(a)
        for f, a in continuous_frame.groupby("fold_index")["selected_alpha"].first().items()
    }
    cs = {
        int(f): float(c) for f, c in direction_frame.set_index("fold_index")["selected_C"].items()
    }

    observed_continuous = _continuous_statistic(records, targets, alphas)
    observed_direction = _direction_statistic(records, direction, cs)
    rng = np.random.default_rng(seed)
    across_continuous: list[float] = []
    circular_continuous: list[float] = []
    across_direction: list[float] = []
    label_direction: list[float] = []
    for _ in range(repeats):
        across_continuous.append(
            _continuous_statistic(records, _permute_across_trials(targets, rng), alphas)
        )
        circular_continuous.append(
            _continuous_statistic(records, _circular_shift(targets, rng), alphas)
        )
        across_direction.append(
            _direction_statistic(records, _permute_across_trials(direction, rng), cs)
        )
        label_direction.append(
            _direction_statistic(records, direction[rng.permutation(direction.shape[0])], cs)
        )

    def summarize(name: str, observed: float, control: list[float]) -> dict[str, Any]:
        values = np.asarray(control, dtype=np.float64)
        low, high = bootstrap_mean_ci(values, boot, confidence, seed)
        p_value = _empirical_p_value(observed, values)
        return {
            "statistic": name,
            "observed_value": observed,
            "control_mean": float(np.nanmean(values)),
            "control_ci95_low": low,
            "control_ci95_high": high,
            "empirical_p_value": p_value,
            "permutation_repeats": repeats,
            "exceeds_control": bool(observed > float(np.nanmean(values))),
        }

    return pd.DataFrame(
        [
            summarize(
                "continuous_mean_r2_across_trial_permutation",
                observed_continuous,
                across_continuous,
            ),
            summarize(
                "continuous_mean_r2_circular_shift", observed_continuous, circular_continuous
            ),
            summarize(
                "direction_balanced_accuracy_across_trial_permutation",
                observed_direction,
                across_direction,
            ),
            summarize(
                "direction_balanced_accuracy_label_permutation", observed_direction, label_direction
            ),
        ]
    )


CLAIM_COLUMNS = [
    "claim_id",
    "candidate_claim",
    "evidence_type",
    "primary_metric",
    "observed_value",
    "control_value",
    "confidence_interval",
    "repeat_consistency",
    "claim_status",
    "allowed_wording",
    "forbidden_wording",
    "limitations",
]


def build_claim_registry(findings: dict[str, Any], controls: pd.DataFrame) -> pd.DataFrame:
    """Claim-safe registry. Support requires observed evidence and completed controls."""
    control_complete = not controls.empty
    templates = [
        ("hand_position", "factor latents predict hand position", "continuous_mean_r2"),
        ("hand_velocity", "factor latents predict hand velocity", "continuous_mean_r2"),
        ("movement_speed", "factor latents predict movement speed", "continuous_mean_r2"),
        (
            "endpoint_direction",
            "factor latents encode endpoint direction",
            "direction_balanced_accuracy",
        ),
        (
            "direction_geometry",
            "reach directions occupy separable latent trajectories",
            "direction_separability",
        ),
        (
            "distance_geometry",
            "distance modulates latent trajectory magnitude",
            "distance_modulation",
        ),
        ("fold_stability", "latent geometry is stable across folds", "fold_stability"),
        ("mask_stability", "latent geometry is stable across neuron masks", "mask_stability"),
        ("beyond_rate", "factor latents contain information beyond population rate", "beyond_rate"),
        (
            "low_dimension",
            "latent dimensionality is substantially lower than neuron count",
            "effective_dimension",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for claim_id, claim, metric in templates:
        value = float(findings.get(metric, float("nan")))
        supported = bool(control_complete and np.isfinite(value) and value > 0.0)
        status = (
            "supported"
            if supported
            else "descriptive_only"
            if np.isfinite(value)
            else "unsupported"
        )
        rows.append(
            {
                "claim_id": claim_id,
                "candidate_claim": claim,
                "evidence_type": "out_of_fold_association",
                "primary_metric": metric,
                "observed_value": value,
                "control_value": float("nan"),
                "confidence_interval": "",
                "repeat_consistency": "",
                "claim_status": status,
                "allowed_wording": claim if supported else f"Descriptively, {claim}.",
                "forbidden_wording": "causal; mechanistic; official leaderboard; state-of-the-art",
                "limitations": "Associative out-of-fold analysis; observational motor task.",
            }
        )
    rows.append(
        {
            "claim_id": "causal_generation",
            "candidate_claim": (
                "neural dynamics are causally generated by the inferred latent space"
            ),
            "evidence_type": "none",
            "primary_metric": "none",
            "observed_value": float("nan"),
            "control_value": float("nan"),
            "confidence_interval": "",
            "repeat_consistency": "",
            "claim_status": "unsupported",
            "allowed_wording": "No causal inference is supported.",
            "forbidden_wording": "causal; causes; generates; drives",
            "limitations": "Observational decoding and geometry cannot establish causality.",
        }
    )
    return pd.DataFrame(rows, columns=CLAIM_COLUMNS)


def build_final_recommendation(registry: pd.DataFrame, checks: dict[str, bool]) -> dict[str, Any]:
    required = [
        "all_25_outer_folds_complete",
        "baseline_scores_reproduced",
        "behavior_decoding_complete",
        "direction_decoding_complete",
        "shuffle_controls_complete",
        "representation_stability_complete",
    ]
    blockers = [key for key in required if not bool(checks.get(key, False))]
    supported = registry[registry["claim_status"] == "supported"]["candidate_claim"].tolist()
    descriptive = registry[registry["claim_status"] == "descriptive_only"][
        "candidate_claim"
    ].tolist()
    unsupported = registry[registry["claim_status"] == "unsupported"]["candidate_claim"].tolist()
    return {
        "analysis_complete": not blockers,
        "integrity_checks_passed": not blockers,
        "out_of_fold_latents_used": True,
        "shuffle_controls_complete": bool(checks.get("shuffle_controls_complete", False)),
        "supported_claims": supported,
        "descriptive_only_claims": descriptive,
        "unsupported_claims": unsupported,
        "primary_neuroscience_finding": supported[0]
        if supported
        else "No claim cleared support criteria.",
        "secondary_findings": supported[1:],
        "major_limitations": [
            "Observational analysis supports association and prediction, not causality.",
            "Latent axes are rotationally non-identifiable.",
            "Results are local and are not official NLB leaderboard measurements.",
        ],
        "ready_for_final_report": not blockers,
        "final_report_blockers": blockers,
        "official_leaderboard_claim": False,
        "causal_claim_allowed": False,
    }
