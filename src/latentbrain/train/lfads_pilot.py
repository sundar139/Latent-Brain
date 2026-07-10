from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
import yaml
from torch.utils.data import DataLoader

from latentbrain.data.schemas import NeuralDataset, NeuronMask, TrialSplit
from latentbrain.eval.baseline_suite import build_window_dataset, load_outer_folds
from latentbrain.eval.behavior import derive_velocity_targets
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
from latentbrain.models.lfads_gru import LFADSGRU, LFADSGRUConfig
from latentbrain.paths import get_repo_root, resolve_configured_path
from latentbrain.randomness import seed_everything
from latentbrain.torch.checkpoints import load_checkpoint
from latentbrain.torch.datasets import NeuralTrialDataset, create_dataloaders, create_torch_datasets
from latentbrain.train.lfads_trainer import TrainingState, train_lfads_gru

EXPECTED_REPEAT = 0
EXPECTED_FOLDS = [0, 1, 2, 3, 4]
EXPECTED_SEEDS = [2027, 2028, 2029, 2030, 2031]
EXPECTED_SHAPE = (500, 64, 162)
RUN_COLUMNS = [
    "repeat_index",
    "fold_index",
    "split_seed",
    "neuron_mask_seed",
    "initialization_seed",
    "status",
    "best_epoch",
    "checkpoint_source",
    "inner_validation_unified_bits_per_spike",
    "outer_unified_bits_per_spike",
    "outer_poisson_nll",
    "outer_behavior_mean_r2",
    "baseline_outer_unified_bits_per_spike",
    "paired_difference_vs_baseline",
    "training_seconds",
    "peak_cuda_memory_mb",
    "final_train_loss",
    "final_inner_validation_loss",
    "notes",
]
CHECKPOINT_COLUMNS = [
    "repeat_index",
    "fold_index",
    "initialization_seed",
    "epoch",
    "checkpoint_type",
    "selection_split",
    "selection_metric",
    "selection_metric_value",
    "checkpoint_path",
    "checkpoint_sha256",
]


@dataclass(frozen=True, slots=True)
class PilotRun:
    repeat_index: int
    fold_index: int
    initialization_seed: int

    @property
    def run_id(self) -> str:
        return (
            f"repeat_{self.repeat_index:03d}/fold_{self.fold_index:03d}/"
            f"seed_{self.initialization_seed}"
        )


def _resolve(path: str) -> Path:
    return resolve_configured_path(path, get_repo_root())


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        msg = f"{label} is missing: {path}"
        raise FileNotFoundError(msg)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"malformed {label}: {path}"
        raise ValueError(msg) from exc
    if not isinstance(loaded, dict):
        msg = f"{label} must contain a JSON object: {path}"
        raise ValueError(msg)
    return loaded


def validate_lfads_pilot_config(config: dict[str, Any]) -> None:
    outer = config["outer_protocol"]
    initialization = config["initialization"]
    if int(outer["repeat_index"]) != EXPECTED_REPEAT:
        msg = "outer_protocol.repeat_index must be 0 for the feasibility pilot"
        raise ValueError(msg)
    if [int(value) for value in outer["fold_indices"]] != EXPECTED_FOLDS:
        msg = "outer_protocol.fold_indices must be exactly [0, 1, 2, 3, 4]"
        raise ValueError(msg)
    if [int(value) for value in initialization["seeds"]] != EXPECTED_SEEDS:
        msg = "initialization.seeds must be exactly [2027, 2028, 2029, 2030, 2031]"
        raise ValueError(msg)
    if str(initialization["seed_policy"]) != "exact_declared_seed":
        msg = (
            "initialization.seed_policy must be exact_declared_seed; seed + run_index is forbidden"
        )
        raise ValueError(msg)
    if str(config["trial_source"]["type"]) != "trial_aware_raw" or bool(
        config["trial_source"]["allow_global_crop_to_min"]
    ):
        msg = "trial-aware raw input is required and global crop is forbidden"
        raise ValueError(msg)
    if not bool(config["window"]["extract_before_rebin"]):
        msg = "event-centered extraction must occur before rebinning"
        raise ValueError(msg)
    if not bool(outer["reuse_exact_assignments"]) or not bool(outer["reuse_exact_neuron_mask"]):
        msg = "exact outer assignments and neuron mask must be reused"
        raise ValueError(msg)
    inner = config["inner_checkpoint_selection"]
    if not bool(inner["enabled"]) or not bool(inner["stratified"]):
        msg = "stratified inner checkpoint selection must be enabled"
        raise ValueError(msg)
    if bool(inner["use_outer_evaluation_for_selection"]):
        msg = "outer evaluation cannot be used for checkpoint selection"
        raise ValueError(msg)
    if str(config["training"]["device"]) != "cuda":
        msg = "training.device must be cuda"
        raise ValueError(msg)
    if str(config["training"]["optimizer"]) != "adamw":
        msg = "training.optimizer must be adamw"
        raise ValueError(msg)
    if str(config["training"]["scheduler"]) != "cosine":
        msg = "training.scheduler must be cosine"
        raise ValueError(msg)
    if (
        str(config["training"]["checkpoint_metric"]) != ("inner_validation_unified_bits_per_spike")
        or str(config["training"]["checkpoint_mode"]) != "max"
    ):
        msg = "checkpoints must maximize inner_validation_unified_bits_per_spike"
        raise ValueError(msg)
    if bool(config["model"]["controller_enabled"]):
        msg = "controller variants are outside this pilot"
        raise ValueError(msg)
    if str(config["baseline"]["method"]) != "factor_latent_train_selected":
        msg = "baseline.method must be factor_latent_train_selected"
        raise ValueError(msg)


def build_pilot_run_schedule(config: dict[str, Any]) -> list[PilotRun]:
    validate_lfads_pilot_config(config)
    repeat = int(config["outer_protocol"]["repeat_index"])
    return [
        PilotRun(repeat, int(fold), int(seed))
        for fold in config["outer_protocol"]["fold_indices"]
        for seed in config["initialization"]["seeds"]
    ]


def build_inner_split(
    outer_train_trials: np.ndarray,
    assignments: pd.DataFrame,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact-size deterministic stratified split over outer-training trials only."""
    trials = np.asarray(outer_train_trials, dtype=np.int64)
    if trials.ndim != 1 or trials.size < 2:
        msg = "outer_train_trials must be a one-dimensional array with at least two trials"
        raise ValueError(msg)
    if validation_fraction <= 0.0 or validation_fraction >= 1.0:
        msg = "validation_fraction must be in (0, 1)"
        raise ValueError(msg)
    rows = assignments[assignments["trial_index"].isin(trials)].copy()
    if set(rows["trial_index"].astype(int)) != set(trials.tolist()):
        msg = "inner split assignments do not cover every outer-training trial"
        raise ValueError(msg)
    counts = rows["stratum"].value_counts()
    rows.loc[rows["stratum"].isin(counts[counts < 2].index), "stratum"] = (
        "pooled_inner_small_stratum"
    )
    generator = np.random.default_rng(seed)
    groups: list[tuple[str, np.ndarray, float]] = []
    desired = int(round(trials.size * validation_fraction))
    for label, group in rows.groupby("stratum", sort=True):
        values = group["trial_index"].to_numpy(dtype=np.int64)
        values = values[generator.permutation(values.size)]
        exact = values.size * validation_fraction
        groups.append((str(label), values, exact - math.floor(exact)))
    quotas = [
        min(math.floor(values.size * validation_fraction), values.size - 1)
        for _, values, _ in groups
    ]
    while sum(quotas) < desired:
        candidates = [
            index for index, (_, values, _) in enumerate(groups) if quotas[index] < values.size - 1
        ]
        if not candidates:
            break
        candidates.sort(key=lambda index: (-groups[index][2], groups[index][0]))
        quotas[candidates[(sum(quotas) - desired) % len(candidates)]] += 1
    validation = np.sort(
        np.concatenate(
            [values[:quota] for (_, values, _), quota in zip(groups, quotas, strict=True)]
        )
    )
    if validation.size != desired:
        msg = f"inner validation split has {validation.size} trials, expected {desired}"
        raise RuntimeError(msg)
    train = np.setdiff1d(trials, validation)
    return train, validation


def validate_input_target_separation(
    heldin: np.ndarray,
    heldout: np.ndarray,
    input_dim: int,
    output_dim: int,
) -> None:
    heldin_indices = np.asarray(heldin, dtype=np.int64)
    heldout_indices = np.asarray(heldout, dtype=np.int64)
    if np.intersect1d(heldin_indices, heldout_indices).size:
        msg = "held-in inputs and held-out targets overlap"
        raise ValueError(msg)
    if heldin_indices.size != input_dim:
        msg = "input_dim does not equal the held-in neuron count"
        raise ValueError(msg)
    if np.union1d(heldin_indices, heldout_indices).size != output_dim:
        msg = "held-in and held-out neurons must cover output_dim"
        raise ValueError(msg)


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint_record(record: dict[str, Any]) -> None:
    if str(record.get("selection_split")) != "inner_validation":
        msg = "selected checkpoint must come from inner_validation"
        raise ValueError(msg)
    if str(record.get("selection_metric")) != "inner_validation_unified_bits_per_spike":
        msg = "selected checkpoint must use inner-validation unified bits/spike"
        raise ValueError(msg)
    if str(record.get("checkpoint_type")) != "best":
        msg = "selected checkpoint must be the best inner-validation checkpoint"
        raise ValueError(msg)


def _finite(frame: pd.DataFrame, columns: list[str]) -> bool:
    if frame.empty or any(column not in frame for column in columns):
        return False
    return bool(np.isfinite(frame[columns].to_numpy(dtype=np.float64)).all())


def build_full_evaluation_recommendation(
    runs: pd.DataFrame,
    config: dict[str, Any],
    leakage_checks_passed: bool,
) -> dict[str, Any]:
    gates = config["pilot_gates"]
    expected = len(EXPECTED_FOLDS) * len(EXPECTED_SEEDS)
    completed = runs[runs["status"] == "completed"] if "status" in runs else runs.iloc[:0]
    failed_fraction = 1.0 - len(completed) / expected
    scores_finite = _finite(completed, ["outer_unified_bits_per_spike"])
    losses_finite = _finite(completed, ["final_train_loss", "final_inner_validation_loss"])
    checkpoint_valid = bool(
        len(completed) == expected
        and (completed["checkpoint_source"].astype(str) == "inner_validation").all()
    )
    mean_score = (
        float(completed["outer_unified_bits_per_spike"].mean()) if scores_finite else float("nan")
    )
    score_std = (
        float(completed["outer_unified_bits_per_spike"].std(ddof=1))
        if scores_finite
        else float("nan")
    )
    seed_means = (
        completed.groupby("initialization_seed")["outer_unified_bits_per_spike"].mean()
        if scores_finite
        else pd.Series(dtype=float)
    )
    seed_std = float(seed_means.std(ddof=1)) if len(seed_means) > 1 else float("nan")
    positive_seed_fraction = float((seed_means > 0.0).mean()) if len(seed_means) else 0.0
    positive_run_fraction = (
        float((completed["outer_unified_bits_per_spike"] > 0.0).mean()) if scores_finite else 0.0
    )
    mean_difference = (
        float(completed["paired_difference_vs_baseline"].mean())
        if _finite(completed, ["paired_difference_vs_baseline"])
        else float("nan")
    )
    total_seconds = (
        float(completed["training_seconds"].sum())
        if _finite(completed, ["training_seconds"])
        else float("nan")
    )
    peak_memory = (
        float(completed["peak_cuda_memory_mb"].max())
        if _finite(completed, ["peak_cuda_memory_mb"])
        else float("nan")
    )
    checks = {
        "all 25 runs completed": len(completed) == expected,
        "failed run fraction is allowed": failed_fraction
        <= float(gates["maximum_failed_run_fraction"]),
        "all scores are finite": scores_finite,
        "all losses are finite": losses_finite,
        "checkpoint selection uses inner validation": checkpoint_valid,
        "leakage checks passed": bool(leakage_checks_passed),
        "mean score is non-negative": bool(
            np.isfinite(mean_score)
            and mean_score >= float(gates["minimum_mean_unified_bits_per_spike"])
        ),
        "positive seed fraction clears gate": positive_seed_fraction
        >= float(gates["minimum_positive_seed_fraction"]),
        "seed-level standard deviation clears gate": bool(
            np.isfinite(seed_std) and seed_std <= float(gates["maximum_seed_std"])
        ),
        f"mean paired difference is at least {gates['full_evaluation_margin_over_baseline']}": bool(
            np.isfinite(mean_difference)
            and mean_difference >= float(gates["full_evaluation_margin_over_baseline"])
        ),
    }
    reasons = [f"failed: {name}" for name, passed in checks.items() if not passed] or [
        "all gates passed"
    ]
    return {
        "proceed": bool(all(checks.values())),
        "all_runs_completed": len(completed) == expected,
        "failed_run_fraction": failed_fraction,
        "mean_unified_bits_per_spike": mean_score,
        "score_std": score_std,
        "seed_level_std": seed_std,
        "positive_run_fraction": positive_run_fraction,
        "positive_seed_fraction": positive_seed_fraction,
        "mean_paired_difference_vs_baseline": mean_difference,
        "checkpoint_selection_valid": checkpoint_valid,
        "leakage_checks_passed": bool(leakage_checks_passed),
        "runtime_estimate_full_evaluation_hours": total_seconds * 5.0 / 3600.0,
        "estimated_peak_cuda_memory_mb": peak_memory,
        "reasons": reasons,
        "pilot_final_claim_allowed": False,
    }


def _scoring(config: dict[str, Any]) -> ScoringConfig:
    settings = config["scoring"]
    return ScoringConfig(
        bin_size_ms=int(config["binning"]["target_bin_size_ms"]),
        include_poisson_constant=bool(settings["include_poisson_constant"]),
        min_rate_hz=float(settings["min_rate_hz"]),
        max_rate_hz=float(settings["max_rate_hz"]),
        reference_name=str(settings["reference_model"]),
    )


def _counts(
    dataset: NeuralDataset, trial_ids: np.ndarray, neuron_indices: np.ndarray
) -> np.ndarray:
    return np.asarray(dataset.spikes[np.isin(dataset.trial_ids, trial_ids)][:, :, neuron_indices])


def _mask_from_fold(heldin: np.ndarray, heldout: np.ndarray, n_neurons: int) -> NeuronMask:
    heldin_mask = np.zeros(n_neurons, dtype=np.bool_)
    heldout_mask = np.zeros(n_neurons, dtype=np.bool_)
    heldin_mask[heldin] = True
    heldout_mask[heldout] = True
    return NeuronMask(heldin=heldin_mask, heldout=heldout_mask)


def _predict(
    model: LFADSGRU,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    rates: list[np.ndarray] = []
    factors: list[np.ndarray] = []
    trial_ids: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch["heldin_spikes"].to(device))
            rates.append(output["rates_hz"].detach().cpu().numpy())
            factors.append(output["factors"].detach().cpu().numpy())
            trial_ids.append(batch["trial_id"].detach().cpu().numpy())
    return {
        "rates": np.concatenate(rates),
        "factors": np.concatenate(factors),
        "trial_ids": np.concatenate(trial_ids),
    }


def _loader(
    dataset: NeuralDataset,
    trial_ids: np.ndarray,
    mask: NeuronMask,
    batch_size: int,
    split_name: str,
) -> DataLoader[dict[str, torch.Tensor]]:
    return DataLoader(
        NeuralTrialDataset(dataset, trial_ids, mask, dataset.spikes.shape[1], split_name),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def _inner_scorer(
    dataset: NeuralDataset,
    train_trials: np.ndarray,
    validation_trials: np.ndarray,
    heldout: np.ndarray,
    validation_loader: DataLoader[dict[str, torch.Tensor]],
    scoring: ScoringConfig,
    device: torch.device,
) -> Any:
    train_counts = _counts(dataset, train_trials, heldout)
    validation_counts = _counts(dataset, validation_trials, heldout)
    reference = train_heldout_mean_rate_reference(train_counts, validation_counts.shape, scoring)

    def score(model: LFADSGRU) -> float:
        predicted = _predict(model, validation_loader, device)["rates"][:, :, heldout]
        row = score_heldout_prediction(
            validation_counts,
            predicted,
            reference,
            scoring,
            "lfads",
            "inner_validation",
            "direct_model",
            True,
        )
        return float(row["bits_per_spike"])

    return score


def _behavior_mean_r2(
    dataset: NeuralDataset,
    train_prediction: dict[str, np.ndarray],
    eval_prediction: dict[str, np.ndarray],
) -> float:
    if dataset.behavior is None or dataset.behavior_names is None:
        return float("nan")
    velocities, _ = derive_velocity_targets(
        dataset.behavior,
        list(dataset.behavior_names),
        dataset.bin_size_ms,
    )
    trial_to_index = {int(trial): index for index, trial in enumerate(dataset.trial_ids)}

    def targets(trials: np.ndarray) -> np.ndarray:
        indices = [trial_to_index[int(trial)] for trial in trials]
        selected = velocities[indices]
        return selected.reshape(-1, selected.shape[-1])

    train_factors_raw = train_prediction["factors"].reshape(
        -1, train_prediction["factors"].shape[-1]
    )
    eval_factors_raw = eval_prediction["factors"].reshape(-1, eval_prediction["factors"].shape[-1])
    train_factors, factor_stats = standardize_train_apply(train_factors_raw, train_factors_raw)
    eval_factors = apply_standardization(eval_factors_raw, factor_stats)
    train_targets_raw = targets(train_prediction["trial_ids"])
    eval_targets_raw = targets(eval_prediction["trial_ids"])
    train_targets, target_stats = standardize_train_apply(train_targets_raw, train_targets_raw)
    eval_targets = apply_standardization(eval_targets_raw, target_stats)
    decoder = fit_ridge_decoder(train_factors, train_targets, alpha=100.0, fit_intercept=True)
    predicted = predict_ridge_decoder(eval_factors, decoder)
    return float(r2_score_numpy(eval_targets, predicted, multioutput="uniform_average"))


def _run_config(config: dict[str, Any], seed: int, output_dir: Path) -> dict[str, Any]:
    model = config["model"]
    training = copy.deepcopy(config["training"])
    training["seed"] = seed
    return {
        "dataset": {
            "name": config["dataset"]["name"],
            "expected_hash": config["dataset"]["expected_hash"],
            "bin_size_ms": int(config["binning"]["target_bin_size_ms"]),
        },
        "model": {
            "name": "lfads_gru",
            "input_dim": None,
            "output_dim": "all",
            "encoder_hidden_dim": int(model["encoder_hidden_dim"]),
            "generator_hidden_dim": int(model["generator_hidden_dim"]),
            "latent_dim": int(model["latent_dim"]),
            "factor_dim": int(model["factor_dim"]),
            "dropout": float(model["dropout_rate"]),
            "min_rate_hz": math.exp(float(model["log_rate_min"])),
            "max_rate_hz": math.exp(float(model["log_rate_max"])),
        },
        "training": training,
        "evaluation": {"evaluate_splits": ["train", "validation"]},
        "reporting": {"output_dir": str(output_dir)},
    }


def build_pilot_model(
    config: dict[str, Any],
    input_dim: int,
    output_dim: int,
    initialization_seed: int,
) -> LFADSGRU:
    """Seed before construction so initialization equals the declared seed exactly."""
    seed_everything(
        initialization_seed,
        deterministic=bool(config["initialization"]["deterministic_algorithms"]),
    )
    settings = config["model"]
    return LFADSGRU(
        LFADSGRUConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            encoder_hidden_dim=int(settings["encoder_hidden_dim"]),
            generator_hidden_dim=int(settings["generator_hidden_dim"]),
            latent_dim=int(settings["latent_dim"]),
            factor_dim=int(settings["factor_dim"]),
            dropout=float(settings["dropout_rate"]),
            min_rate_hz=math.exp(float(settings["log_rate_min"])),
            max_rate_hz=math.exp(float(settings["log_rate_max"])),
        )
    )


def _baseline_by_fold(config: dict[str, Any]) -> dict[int, float]:
    path = _resolve(str(config["outer_protocol"]["baseline_scores_path"]))
    if not path.exists():
        msg = f"baseline outer-fold scores are missing: {path}"
        raise FileNotFoundError(msg)
    scores = pd.read_csv(path)
    rows = scores[
        (scores["repeat_index"] == EXPECTED_REPEAT)
        & (scores["method_name"] == str(config["baseline"]["method"]))
    ]
    if sorted(rows["fold_index"].astype(int).tolist()) != EXPECTED_FOLDS:
        msg = "pilot-repeat baseline scores do not contain exactly five folds"
        raise ValueError(msg)
    return {
        int(row.fold_index): float(row.unified_bits_per_spike)
        for row in rows.itertuples(index=False)
    }


def _load_protocol_inputs(config: dict[str, Any]) -> dict[str, Any]:
    outer = config["outer_protocol"]
    readiness_path = _resolve(str(outer["readiness_path"]))
    readiness = _load_json(readiness_path, "neural reevaluation readiness artifact")
    if not bool(readiness.get("ready")) or readiness.get("blockers"):
        msg = "neural reevaluation readiness artifact does not permit training"
        raise ValueError(msg)
    if str(readiness.get("dataset_hash")) != str(config["dataset"]["expected_hash"]):
        msg = "readiness dataset hash does not match pilot config"
        raise ValueError(msg)
    if str(readiness.get("window_name")) != str(config["window"]["name"]):
        msg = "readiness window does not match pilot config"
        raise ValueError(msg)
    if str(readiness.get("baseline_to_beat")) != str(config["baseline"]["method"]):
        msg = "readiness baseline does not match pilot config"
        raise ValueError(msg)
    expected_mean = float(config["baseline"]["expected_overall_mean"])
    if not math.isclose(
        float(readiness["baseline_mean"]), expected_mean, rel_tol=0.0, abs_tol=1e-12
    ):
        msg = "readiness baseline mean does not match pilot config"
        raise ValueError(msg)

    summary = _load_json(_resolve(str(outer["baseline_summary_path"])), "baseline suite summary")
    if str(summary.get("baseline_to_beat")) != str(config["baseline"]["method"]):
        msg = "baseline suite summary does not match pilot baseline"
        raise ValueError(msg)
    assignments_path = _resolve(str(outer["assignments_path"]))
    protocol_path = assignments_path.parent / "recommended_window_protocol.yaml"
    if not protocol_path.exists():
        msg = f"frozen baseline protocol is missing: {protocol_path}"
        raise FileNotFoundError(msg)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        msg = "frozen baseline protocol must contain a mapping"
        raise ValueError(msg)
    dataset, dataset_hash = build_window_dataset(protocol)
    if dataset.spikes.shape != EXPECTED_SHAPE:
        msg = f"frozen pilot spikes have shape {dataset.spikes.shape}, expected {EXPECTED_SHAPE}"
        raise ValueError(msg)
    if dataset.behavior is None or dataset.behavior.shape != (500, 64, 4):
        msg = "frozen pilot behavior must have shape (500, 64, 4)"
        raise ValueError(msg)
    if dataset_hash != str(config["dataset"]["expected_hash"]):
        msg = "rebuilt pilot dataset hash does not match config"
        raise ValueError(msg)
    if not np.array_equal(dataset.trial_ids, np.arange(EXPECTED_SHAPE[0])):
        msg = "frozen assignment trial_index no longer maps exactly to dataset trial_ids"
        raise ValueError(msg)

    fold_config = {
        "outer_cross_validation": {
            "source_assignments_path": str(outer["assignments_path"]),
            "fold_count": int(protocol["cross_validation"]["fold_count"]),
            "repeats": int(protocol["cross_validation"]["repeats"]),
            "base_seed": int(protocol["cross_validation"]["base_seed"]),
            "reuse_exact_assignments": True,
            "reuse_exact_neuron_masks": True,
        }
    }
    folds = [
        fold
        for fold in load_outer_folds(fold_config, protocol, EXPECTED_SHAPE[2])
        if fold.repeat_index == EXPECTED_REPEAT
    ]
    if [fold.fold_index for fold in folds] != EXPECTED_FOLDS:
        msg = "exact repeat-0 outer folds were not recovered"
        raise ValueError(msg)
    assignments = pd.read_csv(_resolve(str(outer["assignments_path"])))
    assignments = assignments[assignments["repeat_index"] == EXPECTED_REPEAT]
    if len(assignments) != EXPECTED_SHAPE[0]:
        msg = "repeat-0 assignments must contain exactly 500 trials"
        raise ValueError(msg)
    return {
        "readiness": readiness,
        "protocol": protocol,
        "dataset": dataset,
        "folds": folds,
        "assignments": assignments,
        "baseline_by_fold": _baseline_by_fold(config),
    }


def _train_one(
    run: PilotRun,
    fold: Any,
    dataset: NeuralDataset,
    assignments: pd.DataFrame,
    baseline_score: float,
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    inner_seed = int(config["inner_checkpoint_selection"]["split_seed_base"]) + run.fold_index
    inner_train, inner_validation = build_inner_split(
        fold.train_trials,
        assignments,
        float(config["inner_checkpoint_selection"]["validation_fraction"]),
        inner_seed,
    )
    if (
        np.intersect1d(inner_train, fold.eval_trials).size
        or np.intersect1d(inner_validation, fold.eval_trials).size
    ):
        msg = "outer-evaluation trials entered the inner checkpoint split"
        raise RuntimeError(msg)
    validate_input_target_separation(
        fold.heldin, fold.heldout, int(fold.heldin.size), EXPECTED_SHAPE[2]
    )
    mask = _mask_from_fold(fold.heldin, fold.heldout, EXPECTED_SHAPE[2])
    split = TrialSplit(train=inner_train, validation=inner_validation, test=fold.eval_trials)
    loaders = create_dataloaders(
        create_torch_datasets(dataset, split, mask, EXPECTED_SHAPE[1]),
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"]["num_workers"]),
        drop_last=False,
        seed=run.initialization_seed,
    )
    run_dir = output_dir / "runs" / run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    concrete = _run_config(config, run.initialization_seed, run_dir)
    concrete["model"]["input_dim"] = int(fold.heldin.size)
    concrete["model"]["resolved_output_dim"] = EXPECTED_SHAPE[2]
    concrete["pilot"] = {
        "repeat_index": run.repeat_index,
        "fold_index": run.fold_index,
        "initialization_seed": run.initialization_seed,
        "inner_split_seed": inner_seed,
        "selection_split": "inner_validation",
        "outer_evaluation_used_for_selection": False,
        "normalization_fit_trials": inner_train.tolist(),
    }
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(concrete, sort_keys=False), encoding="utf-8"
    )
    model = build_pilot_model(
        config,
        input_dim=int(fold.heldin.size),
        output_dim=EXPECTED_SHAPE[2],
        initialization_seed=run.initialization_seed,
    )
    scoring = _scoring(config)
    scorer = _inner_scorer(
        dataset,
        inner_train,
        inner_validation,
        fold.heldout,
        loaders["validation"],
        scoring,
        device,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    state: TrainingState = train_lfads_gru(
        model,
        loaders,
        concrete,
        run_dir,
        device,
        checkpoint_scorer=scorer,
    )
    training_seconds = time.perf_counter() - started
    peak_memory = (
        float(torch.cuda.max_memory_allocated(device) / (1024.0**2))
        if device.type == "cuda"
        else 0.0
    )
    checkpoint = run_dir / "checkpoints" / "best_validation.pt"
    load_checkpoint(checkpoint, model, map_location=device)
    selected = {
        "repeat_index": run.repeat_index,
        "fold_index": run.fold_index,
        "initialization_seed": run.initialization_seed,
        "epoch": state.best_epoch,
        "checkpoint_type": "best",
        "selection_split": "inner_validation",
        "selection_metric": "inner_validation_unified_bits_per_spike",
        "selection_metric_value": state.best_metric,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
    }
    validate_checkpoint_record(selected)

    outer_train_loader = _loader(
        dataset, fold.train_trials, mask, int(config["training"]["batch_size"]), "outer_train"
    )
    outer_eval_loader = _loader(
        dataset, fold.eval_trials, mask, int(config["training"]["batch_size"]), "outer_evaluation"
    )
    outer_train_prediction = _predict(model, outer_train_loader, device)
    outer_prediction = _predict(model, outer_eval_loader, device)
    all_rates = outer_prediction["rates"]
    expected_output_shape = (fold.eval_trials.size, EXPECTED_SHAPE[1], EXPECTED_SHAPE[2])
    if all_rates.shape != expected_output_shape:
        msg = f"outer model output has shape {all_rates.shape}, expected {expected_output_shape}"
        raise RuntimeError(msg)
    if not np.isfinite(all_rates).all() or np.any(all_rates <= 0.0):
        msg = "outer model rates must be finite and strictly positive"
        raise RuntimeError(msg)
    outer_counts = _counts(dataset, fold.eval_trials, fold.heldout)
    train_counts = _counts(dataset, fold.train_trials, fold.heldout)
    reference = train_heldout_mean_rate_reference(train_counts, outer_counts.shape, scoring)
    scored = score_heldout_prediction(
        outer_counts,
        all_rates[:, :, fold.heldout],
        reference,
        scoring,
        "lfads",
        "outer_evaluation",
        "direct_model",
        True,
    )
    behavior_r2 = _behavior_mean_r2(dataset, outer_train_prediction, outer_prediction)
    final = state.history[-1]
    row = {
        "repeat_index": run.repeat_index,
        "fold_index": run.fold_index,
        "split_seed": int(fold.split_seed),
        "neuron_mask_seed": int(fold.neuron_mask_seed),
        "initialization_seed": run.initialization_seed,
        "status": "completed",
        "best_epoch": state.best_epoch,
        "checkpoint_source": "inner_validation",
        "inner_validation_unified_bits_per_spike": state.best_metric,
        "outer_unified_bits_per_spike": float(scored["bits_per_spike"]),
        "outer_poisson_nll": float(scored["poisson_nll"]),
        "outer_behavior_mean_r2": behavior_r2,
        "baseline_outer_unified_bits_per_spike": baseline_score,
        "paired_difference_vs_baseline": float(scored["bits_per_spike"]) - baseline_score,
        "training_seconds": training_seconds,
        "peak_cuda_memory_mb": peak_memory,
        "final_train_loss": float(final["train_total_loss"]),
        "final_inner_validation_loss": float(final["validation_total_loss"]),
        "notes": "outer evaluation performed once after inner-validation checkpoint selection",
    }
    resource = {
        "repeat_index": run.repeat_index,
        "fold_index": run.fold_index,
        "initialization_seed": run.initialization_seed,
        "training_seconds": training_seconds,
        "best_epoch": state.best_epoch,
        "epochs_completed": len(state.history),
        "peak_cuda_memory_mb": peak_memory,
        "batch_size": int(config["training"]["batch_size"]),
        "mixed_precision_enabled": bool(config["training"]["mixed_precision"]),
        "early_stopping_triggered": state.early_stopping_triggered,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
    }
    return row, selected, resource


def _aggregate(
    runs: pd.DataFrame,
    checkpoints: pd.DataFrame,
    resources: pd.DataFrame,
    config: dict[str, Any],
    inputs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
    completed = runs[runs["status"] == "completed"].copy()
    fold_seed = completed[
        [
            "fold_index",
            "initialization_seed",
            "outer_unified_bits_per_spike",
            "baseline_outer_unified_bits_per_spike",
            "paired_difference_vs_baseline",
        ]
    ].copy()
    fold_seed["positive_score"] = fold_seed["outer_unified_bits_per_spike"] > 0.0
    fold_seed["beats_baseline"] = fold_seed["paired_difference_vs_baseline"] > 0.0

    seed_rows = []
    for seed, group in fold_seed.groupby("initialization_seed", sort=True):
        resource = resources[resources["initialization_seed"] == seed]
        seed_rows.append(
            {
                "initialization_seed": int(seed),
                "completed_folds": len(group),
                "mean_outer_unified_bits_per_spike": float(
                    group["outer_unified_bits_per_spike"].mean()
                ),
                "std_outer_unified_bits_per_spike": float(
                    group["outer_unified_bits_per_spike"].std(ddof=1)
                ),
                "mean_paired_difference_vs_baseline": float(
                    group["paired_difference_vs_baseline"].mean()
                ),
                "positive_fold_fraction": float(group["positive_score"].mean()),
                "beats_baseline_fold_fraction": float(group["beats_baseline"].mean()),
                "mean_training_seconds": float(resource["training_seconds"].mean()),
                "maximum_peak_cuda_memory_mb": float(resource["peak_cuda_memory_mb"].max()),
            }
        )
    seed_summary = pd.DataFrame(seed_rows)

    fold_rows = []
    statistics = config["statistics"]
    for fold, group in fold_seed.groupby("fold_index", sort=True):
        values = group["outer_unified_bits_per_spike"].to_numpy(dtype=np.float64)
        low, high = bootstrap_mean_ci(
            values,
            int(statistics["bootstrap_repeats"]),
            float(statistics["confidence_interval"]),
            int(statistics["bootstrap_seed"]) + int(fold),
        )
        fold_rows.append(
            {
                "fold_index": int(fold),
                "completed_seeds": len(group),
                "mean_outer_unified_bits_per_spike": float(np.mean(values)),
                "std_across_seeds": float(np.std(values, ddof=1)),
                "ci95_low": low,
                "ci95_high": high,
                "baseline_outer_unified_bits_per_spike": float(
                    group["baseline_outer_unified_bits_per_spike"].iloc[0]
                ),
                "mean_paired_difference_vs_baseline": float(
                    group["paired_difference_vs_baseline"].mean()
                ),
                "positive_seed_fraction": float(group["positive_score"].mean()),
                "beats_baseline_seed_fraction": float(group["beats_baseline"].mean()),
            }
        )
    fold_summary = pd.DataFrame(fold_rows)
    paired = fold_seed[
        ["fold_index", "initialization_seed", "paired_difference_vs_baseline"]
    ].copy()
    recommendation = build_full_evaluation_recommendation(
        runs,
        config,
        leakage_checks_passed=bool(
            len(completed) == 25 and (completed["checkpoint_source"] == "inner_validation").all()
        ),
    )
    summary = {
        "dataset_name": config["dataset"]["name"],
        "dataset_hash": config["dataset"]["expected_hash"],
        "data_shape": list(EXPECTED_SHAPE),
        "behavior_shape": [500, 64, 4],
        "input_neuron_count": 122,
        "output_neuron_count": 162,
        "heldout_neuron_count": 40,
        "repeat_index": EXPECTED_REPEAT,
        "fold_indices": EXPECTED_FOLDS,
        "initialization_seeds": EXPECTED_SEEDS,
        "scheduled_runs": 25,
        "completed_runs": len(completed),
        "failed_runs": int((runs["status"] == "failed").sum()),
        "mean_unified_bits_per_spike": recommendation["mean_unified_bits_per_spike"],
        "score_std": recommendation["score_std"],
        "seed_level_std": recommendation["seed_level_std"],
        "positive_run_fraction": recommendation["positive_run_fraction"],
        "positive_seed_fraction": recommendation["positive_seed_fraction"],
        "pilot_repeat_baseline_mean": float(np.mean(list(inputs["baseline_by_fold"].values()))),
        "mean_paired_difference_vs_baseline": recommendation["mean_paired_difference_vs_baseline"],
        "checkpoint_selection_split": "inner_validation",
        "checkpoint_selection_valid": recommendation["checkpoint_selection_valid"],
        "leakage_checks_passed": recommendation["leakage_checks_passed"],
        "baseline_to_beat": config["baseline"]["method"],
        "full_evaluation_recommended": recommendation["proceed"],
        "single_split_results_reportable": False,
        "official_leaderboard_claim": False,
        "pilot_final_claim_allowed": False,
        "statement": (
            "The pilot evaluates feasibility and seed stability on one held-out-neuron mask. "
            "It is not a final multi-repeat model comparison."
        ),
    }
    protocol = {
        "dataset": config["dataset"],
        "trial_source": config["trial_source"],
        "window": config["window"],
        "binning": config["binning"],
        "outer_protocol": config["outer_protocol"],
        "initialization": config["initialization"],
        "model": config["model"],
        "training": config["training"],
        "inner_checkpoint_selection": config["inner_checkpoint_selection"],
        "scoring": config["scoring"],
        "baseline": config["baseline"],
        "protocol_frozen": True,
        "outer_evaluation_used_for_selection": False,
        "pilot_final_claim_allowed": False,
    }
    tables = {
        "lfads_pilot_runs": runs[RUN_COLUMNS],
        "fold_seed_scores": fold_seed,
        "seed_summary": seed_summary,
        "fold_summary": fold_summary,
        "paired_baseline_comparison": paired,
        "checkpoint_manifest": checkpoints.reindex(columns=CHECKPOINT_COLUMNS),
        "training_resource_summary": resources,
    }
    return summary, tables, protocol, recommendation


def run_lfads_pilot(config: dict[str, Any]) -> dict[str, Any]:
    """Run or resume the fixed 25-run repeat-0 LFADS feasibility pilot."""
    validate_lfads_pilot_config(config)
    inputs = _load_protocol_inputs(config)
    if not torch.cuda.is_available():
        msg = "CUDA was requested, but torch.cuda.is_available() is False."
        raise RuntimeError(msg)
    device = torch.device("cuda")
    output_dir = _resolve(str(config["reporting"]["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / "lfads_pilot_runs.csv"
    checkpoint_path = output_dir / "checkpoint_manifest.csv"
    resource_path = output_dir / "training_resource_summary.csv"
    runs = pd.read_csv(runs_path) if runs_path.exists() else pd.DataFrame(columns=RUN_COLUMNS)
    checkpoints = (
        pd.read_csv(checkpoint_path)
        if checkpoint_path.exists()
        else pd.DataFrame(columns=CHECKPOINT_COLUMNS)
    )
    resources = pd.read_csv(resource_path) if resource_path.exists() else pd.DataFrame()
    completed: set[tuple[int, int]] = set()
    for row in runs[runs["status"] == "completed"].itertuples(index=False):
        key = (int(row.fold_index), int(row.initialization_seed))
        manifest = checkpoints[
            (checkpoints["fold_index"] == key[0]) & (checkpoints["initialization_seed"] == key[1])
        ]
        if len(manifest) != 1:
            continue
        checkpoint_file = Path(str(manifest.iloc[0]["checkpoint_path"]))
        if checkpoint_file.exists() and checkpoint_sha256(checkpoint_file) == str(
            manifest.iloc[0]["checkpoint_sha256"]
        ):
            completed.add(key)
    folds = {fold.fold_index: fold for fold in inputs["folds"]}
    for run in build_pilot_run_schedule(config):
        if (run.fold_index, run.initialization_seed) in completed:
            continue
        same_run = (runs["fold_index"] == run.fold_index) & (
            runs["initialization_seed"] == run.initialization_seed
        )
        runs = runs[~same_run]
        if not checkpoints.empty:
            checkpoints = checkpoints[
                ~(
                    (checkpoints["fold_index"] == run.fold_index)
                    & (checkpoints["initialization_seed"] == run.initialization_seed)
                )
            ]
        if not resources.empty:
            resources = resources[
                ~(
                    (resources["fold_index"] == run.fold_index)
                    & (resources["initialization_seed"] == run.initialization_seed)
                )
            ]
        try:
            row, checkpoint, resource = _train_one(
                run,
                folds[run.fold_index],
                inputs["dataset"],
                inputs["assignments"],
                float(inputs["baseline_by_fold"][run.fold_index]),
                config,
                output_dir,
                device,
            )
        except Exception as exc:
            failed = dict.fromkeys(RUN_COLUMNS)
            failed.update(
                {
                    "repeat_index": run.repeat_index,
                    "fold_index": run.fold_index,
                    "initialization_seed": run.initialization_seed,
                    "status": "failed",
                    "notes": str(exc),
                }
            )
            runs = pd.concat([runs, pd.DataFrame([failed])], ignore_index=True)
            runs.to_csv(runs_path, index=False)
            raise
        runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
        checkpoints = pd.concat([checkpoints, pd.DataFrame([checkpoint])], ignore_index=True)
        resources = pd.concat([resources, pd.DataFrame([resource])], ignore_index=True)
        runs.to_csv(runs_path, index=False)
        checkpoints.to_csv(checkpoint_path, index=False)
        resources.to_csv(resource_path, index=False)

    summary, tables, protocol, recommendation = _aggregate(
        runs, checkpoints, resources, config, inputs
    )
    from latentbrain.eval.reporting import write_lfads_pilot_outputs  # noqa: PLC0415

    write_lfads_pilot_outputs(output_dir, summary, tables, protocol, recommendation)
    return {
        "summary": summary,
        "tables": tables,
        "protocol": protocol,
        "recommendation": recommendation,
        "output_dir": output_dir,
    }
