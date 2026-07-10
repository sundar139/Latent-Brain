from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
import yaml
from torch.utils.data import DataLoader

from latentbrain.eval.baseline_suite import predict_factor_latent
from latentbrain.eval.lfads_eval import load_lfads_gru_from_checkpoint
from latentbrain.eval.movement_features import compute_hand_speed
from latentbrain.eval.reporting import write_lfads_diagnostics_outputs
from latentbrain.eval.scoring import (
    ScoringConfig,
    canonical_bits_per_spike,
    poisson_log_likelihood,
    score_heldout_prediction,
    train_heldout_mean_rate_reference,
)
from latentbrain.models.lfads_gru import LFADSGRU
from latentbrain.paths import get_repo_root, resolve_configured_path
from latentbrain.randomness import seed_everything
from latentbrain.train import lfads_pilot

EXPECTED_REPEAT = 0
EXPECTED_FOLDS = [0, 1, 2, 3, 4]
EXPECTED_SEEDS = [2027, 2028, 2029, 2030, 2031]
EXPECTED_RUNS = {(fold, seed) for fold in EXPECTED_FOLDS for seed in EXPECTED_SEEDS}

NEURON_DIAGNOSTIC_COLUMNS = [
    "fold_index",
    "initialization_seed",
    "neuron_index",
    "observed_spike_count",
    "observed_rate_hz",
    "predicted_rate_hz",
    "unified_bits_per_spike",
    "poisson_nll",
    "rate_bias_hz",
    "rate_bias_fraction",
    "prediction_variance",
    "observed_variance",
    "variance_ratio",
    "correlation",
]
TIME_BIN_DIAGNOSTIC_COLUMNS = [
    "fold_index",
    "initialization_seed",
    "time_bin",
    "relative_time_seconds",
    "observed_population_rate_hz",
    "predicted_population_rate_hz",
    "unified_bits_per_spike",
    "poisson_nll",
    "rate_error_hz",
    "movement_speed",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return 0.0
    if abs(denominator) <= np.finfo(np.float64).eps:
        return 0.0
    return float(numerator / denominator)


def validate_neuron_partition(heldin: np.ndarray, heldout: np.ndarray, total_neurons: int) -> None:
    heldin_array = np.asarray(heldin, dtype=np.int64)
    heldout_array = np.asarray(heldout, dtype=np.int64)
    if np.intersect1d(heldin_array, heldout_array).size:
        raise ValueError("held-in model inputs and held-out targets overlap")
    if np.union1d(heldin_array, heldout_array).size != total_neurons:
        raise ValueError("held-in and held-out indices must cover every output neuron")


def validate_checkpoint_integrity(
    manifest: pd.DataFrame,
    runs: pd.DataFrame,
    pilot_summary: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    protocol = config["protocol"]
    expected_folds = [int(value) for value in protocol["fold_indices"]]
    expected_seeds = [int(value) for value in protocol["initialization_seeds"]]
    expected = {(fold, seed) for fold in expected_folds for seed in expected_seeds}
    if int(protocol["repeat_index"]) != EXPECTED_REPEAT or expected != EXPECTED_RUNS:
        raise ValueError(
            "diagnostic protocol does not match the accepted repeat/fold/seed schedule"
        )
    if str(pilot_summary.get("dataset_hash")) != str(config["dataset"]["expected_hash"]):
        raise ValueError("pilot dataset hash does not match diagnostic config")
    if (
        int(pilot_summary.get("repeat_index", -1)) != EXPECTED_REPEAT
        or [int(value) for value in pilot_summary.get("fold_indices", [])] != EXPECTED_FOLDS
        or [int(value) for value in pilot_summary.get("initialization_seeds", [])] != EXPECTED_SEEDS
    ):
        raise ValueError("pilot summary does not match the accepted schedule")
    if (
        int(pilot_summary.get("completed_runs", -1)) != 25
        or int(pilot_summary.get("failed_runs", -1)) != 0
        or bool(pilot_summary.get("full_evaluation_recommended", True))
        or not bool(pilot_summary.get("leakage_checks_passed", False))
    ):
        raise ValueError("pilot summary is incomplete, failed, leaky, or permits full evaluation")
    if len(manifest) != 25:
        raise ValueError("checkpoint manifest must contain exactly 25 accepted checkpoints")
    manifest_schedule = {
        (int(row.fold_index), int(row.initialization_seed))
        for row in manifest.itertuples(index=False)
        if int(row.repeat_index) == EXPECTED_REPEAT
    }
    if manifest_schedule != expected:
        raise ValueError("checkpoint manifest does not match the accepted schedule")
    accepted_runs = runs[runs["status"].astype(str) == "completed"]
    run_schedule = {
        (int(row.fold_index), int(row.initialization_seed))
        for row in accepted_runs.itertuples(index=False)
        if int(row.repeat_index) == EXPECTED_REPEAT
    }
    if len(accepted_runs) != 25 or run_schedule != expected:
        raise ValueError("completed pilot runs do not match the accepted schedule")

    for row in manifest.itertuples(index=False):
        if str(row.selection_split) != "inner_validation" or str(row.checkpoint_type) != "best":
            raise ValueError("every accepted checkpoint must be selected on inner_validation")
        if str(row.selection_metric) != "inner_validation_unified_bits_per_spike":
            raise ValueError(
                "checkpoint selection metric must be inner-validation unified bits/spike"
            )
        path = Path(str(row.checkpoint_path))
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint is missing: {path}")
        if _sha256(path) != str(row.checkpoint_sha256):
            raise ValueError(f"checkpoint hash mismatch: {path}")
        payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
        snapshot = payload.get("config", {})
        pilot = snapshot.get("pilot", {})
        model = snapshot.get("model", {})
        training = snapshot.get("training", {})
        if (
            int(payload.get("epoch", -1)) != int(row.epoch)
            or int(pilot.get("repeat_index", -1)) != EXPECTED_REPEAT
            or int(pilot.get("fold_index", -1)) != int(row.fold_index)
            or int(pilot.get("initialization_seed", -1)) != int(row.initialization_seed)
            or int(training.get("seed", -1)) != int(row.initialization_seed)
            or str(pilot.get("selection_split")) != "inner_validation"
            or bool(pilot.get("outer_evaluation_used_for_selection", True))
        ):
            raise ValueError(f"checkpoint metadata does not match accepted schedule: {path}")
        if (
            int(model.get("input_dim", -1)) != 122
            or str(model.get("output_dim")) != "all"
            or int(model.get("resolved_output_dim", -1)) != 162
        ):
            raise ValueError(f"checkpoint model architecture does not match pilot config: {path}")
        state = payload.get("model_state_dict", {})
        if (
            tuple(state["encoder.weight_ih_l0"].shape)[1] != 122
            or tuple(state["rate_readout.weight"].shape)[0] != 162
        ):
            raise ValueError(f"checkpoint tensor architecture does not match pilot config: {path}")
    return {
        "integrity_checks_passed": True,
        "accepted_checkpoints": 25,
        "excluded_preflight_artifacts": int((runs["status"].astype(str) != "completed").sum()),
        "terminated_preflight_processes_included": False,
        "accepted_checkpoint_source": "validated 25-row checkpoint manifest only",
        "checkpoint_hashes_match": True,
        "checkpoint_selection_valid": True,
    }


def _scoring(bin_size_ms: int) -> ScoringConfig:
    return ScoringConfig(
        bin_size_ms=bin_size_ms,
        include_poisson_constant=True,
        min_rate_hz=1.0e-4,
        max_rate_hz=500.0,
        reference_name="train_heldout_mean_rate",
    )


def _finite_bits(model_ll: float, reference_ll: float, spike_count: float) -> float:
    return (
        0.0 if spike_count <= 0.0 else canonical_bits_per_spike(model_ll, reference_ll, spike_count)
    )


def per_neuron_diagnostics(
    counts: np.ndarray,
    predicted_rates_hz: np.ndarray,
    reference_rates_hz: np.ndarray,
    neuron_indices: np.ndarray,
    bin_size_ms: int,
    fold_index: int,
    initialization_seed: int,
) -> pd.DataFrame:
    counts_array = np.asarray(counts, dtype=np.float64)
    predicted = np.asarray(predicted_rates_hz, dtype=np.float64)
    reference = np.asarray(reference_rates_hz, dtype=np.float64)
    if counts_array.shape != predicted.shape or counts_array.shape != reference.shape:
        raise ValueError("counts, predictions, and references must have matching shapes")
    if counts_array.shape[2] != len(neuron_indices):
        raise ValueError("neuron index count does not match target tensor")
    scoring = _scoring(bin_size_ms)
    seconds = counts_array.shape[0] * counts_array.shape[1] * bin_size_ms / 1000.0
    rows: list[dict[str, Any]] = []
    for rank, neuron in enumerate(np.asarray(neuron_indices, dtype=np.int64)):
        target = counts_array[:, :, rank : rank + 1]
        rates = predicted[:, :, rank : rank + 1]
        base = reference[:, :, rank : rank + 1]
        model_ll = poisson_log_likelihood(target, rates, scoring)
        reference_ll = poisson_log_likelihood(target, base, scoring)
        spike_count = float(target.sum())
        observed_rate = float(spike_count / max(seconds, np.finfo(np.float64).eps))
        predicted_rate = float(np.mean(rates))
        observed_samples = target.reshape(-1) * (1000.0 / bin_size_ms)
        predicted_samples = rates.reshape(-1)
        observed_variance = float(np.var(observed_samples))
        prediction_variance = float(np.var(predicted_samples))
        correlation = (
            float(np.corrcoef(observed_samples, predicted_samples)[0, 1])
            if np.std(observed_samples) > 0.0 and np.std(predicted_samples) > 0.0
            else 0.0
        )
        bias = predicted_rate - observed_rate
        rows.append(
            {
                "fold_index": fold_index,
                "initialization_seed": initialization_seed,
                "neuron_index": int(neuron),
                "observed_spike_count": spike_count,
                "observed_rate_hz": observed_rate,
                "predicted_rate_hz": predicted_rate,
                "unified_bits_per_spike": _finite_bits(model_ll, reference_ll, spike_count),
                "poisson_nll": -model_ll,
                "rate_bias_hz": bias,
                "rate_bias_fraction": _safe_ratio(bias, observed_rate),
                "prediction_variance": prediction_variance,
                "observed_variance": observed_variance,
                "variance_ratio": _safe_ratio(prediction_variance, observed_variance),
                "correlation": correlation,
            }
        )
    return pd.DataFrame(rows, columns=NEURON_DIAGNOSTIC_COLUMNS)


def time_bin_diagnostics(
    counts: np.ndarray,
    predicted_rates_hz: np.ndarray,
    reference_rates_hz: np.ndarray,
    movement_speed: np.ndarray,
    bin_size_ms: int,
    fold_index: int,
    initialization_seed: int,
) -> pd.DataFrame:
    counts_array = np.asarray(counts, dtype=np.float64)
    predicted = np.asarray(predicted_rates_hz, dtype=np.float64)
    reference = np.asarray(reference_rates_hz, dtype=np.float64)
    speed = np.asarray(movement_speed, dtype=np.float64)
    if counts_array.shape != predicted.shape or counts_array.shape != reference.shape:
        raise ValueError("counts, predictions, and references must have matching shapes")
    if speed.shape != (counts_array.shape[1],):
        raise ValueError("movement speed must have one value per time bin")
    scoring = _scoring(bin_size_ms)
    rows: list[dict[str, Any]] = []
    center = counts_array.shape[1] // 2
    for time_bin in range(counts_array.shape[1]):
        target = counts_array[:, time_bin : time_bin + 1, :]
        rates = predicted[:, time_bin : time_bin + 1, :]
        base = reference[:, time_bin : time_bin + 1, :]
        model_ll = poisson_log_likelihood(target, rates, scoring)
        reference_ll = poisson_log_likelihood(target, base, scoring)
        spike_count = float(target.sum())
        observed_rate = float(target.mean() * 1000.0 / bin_size_ms)
        predicted_rate = float(rates.mean())
        rows.append(
            {
                "fold_index": fold_index,
                "initialization_seed": initialization_seed,
                "time_bin": time_bin,
                "relative_time_seconds": (time_bin - center) * bin_size_ms / 1000.0,
                "observed_population_rate_hz": observed_rate,
                "predicted_population_rate_hz": predicted_rate,
                "unified_bits_per_spike": _finite_bits(model_ll, reference_ll, spike_count),
                "poisson_nll": -model_ll,
                "rate_error_hz": predicted_rate - observed_rate,
                "movement_speed": float(speed[time_bin]),
            }
        )
    return pd.DataFrame(rows, columns=TIME_BIN_DIAGNOSTIC_COLUMNS)


def _lag_one_autocorrelation(values: np.ndarray) -> float:
    if values.size < 2 or np.std(values[:-1]) == 0.0 or np.std(values[1:]) == 0.0:
        return 0.0
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def temporal_smoothness_metrics(
    observed_rates_hz: np.ndarray, predicted_rates_hz: np.ndarray
) -> dict[str, float]:
    observed = np.asarray(observed_rates_hz, dtype=np.float64)
    predicted = np.asarray(predicted_rates_hz, dtype=np.float64)
    if observed.shape != predicted.shape:
        raise ValueError("observed and predicted rate tensors must match")
    observed_population = observed.mean(axis=(0, 2)) if observed.ndim == 3 else observed.reshape(-1)
    predicted_population = (
        predicted.mean(axis=(0, 2)) if predicted.ndim == 3 else predicted.reshape(-1)
    )
    observed_first = np.diff(observed_population)
    predicted_first = np.diff(predicted_population)
    observed_second = np.diff(observed_population, n=2)
    predicted_second = np.diff(predicted_population, n=2)
    return {
        "observed_population_variance": float(np.var(observed_population)),
        "predicted_population_variance": float(np.var(predicted_population)),
        "population_variance_ratio": _safe_ratio(
            float(np.var(predicted_population)), float(np.var(observed_population))
        ),
        "first_difference_variance_ratio": _safe_ratio(
            float(np.var(predicted_first)), float(np.var(observed_first))
        ),
        "second_difference_variance_ratio": _safe_ratio(
            float(np.var(predicted_second)), float(np.var(observed_second))
        ),
        "observed_lag1_autocorrelation": _lag_one_autocorrelation(observed_population),
        "predicted_lag1_autocorrelation": _lag_one_autocorrelation(predicted_population),
    }


def effective_rank(values: np.ndarray) -> tuple[float, float, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("effective-rank input must be [samples, dimensions]")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    if total <= np.finfo(np.float64).eps:
        return 0.0, 0.0, eigenvalues
    probabilities = eigenvalues[eigenvalues > 0.0] / total
    rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    return rank, float(rank / matrix.shape[1]), eigenvalues


def detect_posterior_collapse(
    factor_rank_fraction: float,
    posterior_rank_fraction: float,
    kl_loss: float,
    thresholds: dict[str, Any],
) -> bool:
    """Require low factor use, low posterior rank, and negligible KL together."""
    return bool(
        factor_rank_fraction < float(thresholds["low_latent_variance_fraction"])
        and posterior_rank_fraction
        < float(thresholds["posterior_collapse_effective_rank_fraction"])
        and kl_loss < 1.0e-3
    )


def split_gap_summary(run_diagnostics: pd.DataFrame) -> pd.DataFrame:
    pivot = run_diagnostics.pivot(
        index=["fold_index", "initialization_seed"],
        columns="split_name",
        values="unified_bits_per_spike",
    ).reset_index()
    required = {"outer_training", "inner_validation", "outer_evaluation"}
    if not required.issubset(pivot.columns):
        raise ValueError(
            "run diagnostics must contain outer_training, inner_validation, outer_evaluation"
        )
    pivot["train_to_inner_gap"] = pivot["outer_training"] - pivot["inner_validation"]
    pivot["inner_to_outer_gap"] = pivot["inner_validation"] - pivot["outer_evaluation"]
    return pivot


def recommend_next_action(
    integrity_checks_passed: bool,
    dominant_failure_mode: str,
    estimated_recoverable_gap: float,
    targeted_repair_available: bool,
    config: dict[str, Any],
    secondary_failure_modes: list[str] | None = None,
) -> dict[str, Any]:
    limit = float(config["thresholds"]["baseline_gap_repairable_limit"])
    if not integrity_checks_passed:
        action = "block_due_to_integrity_issue"
        repair = False
    elif (
        targeted_repair_available
        and bool(config["decision"]["allow_targeted_lfads_repair_pilot"])
        and estimated_recoverable_gap > limit
    ):
        action = "targeted_lfads_repair_pilot"
        repair = True
    else:
        action = str(config["decision"]["default_when_no_clear_repair"])
        repair = False
    allowed = {
        "targeted_lfads_repair_pilot",
        "retire_lfads_and_start_neural_ode_pilot",
        "block_due_to_integrity_issue",
    }
    if action not in allowed:
        raise ValueError("recommended next action is not allowed")
    return {
        "recommended_next_action": action,
        "integrity_checks_passed": bool(integrity_checks_passed),
        "dominant_failure_mode": dominant_failure_mode,
        "secondary_failure_modes": secondary_failure_modes or [],
        "estimated_recoverable_gap": float(estimated_recoverable_gap),
        "targeted_repair_available": repair,
        "full_lfads_evaluation_allowed": False,
        "rationale": (
            "Integrity failure blocks later experiments."
            if not integrity_checks_passed
            else "One frozen targeted repair could plausibly recover more than the declared limit."
            if repair
            else "Stable deficit lacks one sufficiently large actionable LFADS repair."
        ),
        "required_next_protocol": action,
    }


def _resolve(path: str) -> Path:
    return resolve_configured_path(path, get_repo_root())


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return loaded


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} must contain a YAML mapping")
    return loaded


def validate_lfads_diagnostics_config(config: dict[str, Any]) -> None:
    required = {
        "dataset",
        "protocol",
        "inputs",
        "diagnostics",
        "thresholds",
        "decision",
        "reporting",
    }
    if set(config) != required:
        raise ValueError(f"LFADS diagnostics config keys must be exactly {sorted(required)}")
    protocol = config["protocol"]
    if (
        str(config["dataset"]["name"]) != "mc_maze_large"
        or str(protocol["trial_source"]) != "trial_aware_raw"
        or str(protocol["window_name"]) != "behavior_speed_peak_centered_1p28s"
        or int(protocol["target_bin_size_ms"]) != 20
        or int(protocol["repeat_index"]) != EXPECTED_REPEAT
        or [int(value) for value in protocol["fold_indices"]] != EXPECTED_FOLDS
        or [int(value) for value in protocol["initialization_seeds"]] != EXPECTED_SEEDS
    ):
        raise ValueError("LFADS diagnostics must use the accepted Large pilot protocol")
    if bool(config["decision"]["allow_full_lfads_evaluation"]):
        raise ValueError("full LFADS evaluation must remain disabled")
    if not all(bool(value) for value in config["diagnostics"].values()):
        raise ValueError("all declared post-hoc diagnostics must remain enabled")
    if str(config["decision"]["default_when_no_clear_repair"]) != (
        "retire_lfads_and_start_neural_ode_pilot"
    ):
        raise ValueError("default next action must retire LFADS when no clear repair exists")


def _diagnostic_prediction(
    model: LFADSGRU,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    chunks: dict[str, list[np.ndarray]] = {
        "rates": [],
        "factors": [],
        "z0_mean": [],
        "z0_logvar": [],
        "trial_ids": [],
    }
    with torch.no_grad():
        for batch in loader:
            output = model(batch["heldin_spikes"].to(device))
            for key in ("rates", "factors", "z0_mean", "z0_logvar"):
                source = "rates_hz" if key == "rates" else key
                chunks[key].append(output[source].detach().cpu().numpy())
            chunks["trial_ids"].append(batch["trial_id"].detach().cpu().numpy())
    return {key: np.concatenate(values) for key, values in chunks.items()}


def _score(
    counts: np.ndarray,
    rates: np.ndarray,
    reference: np.ndarray,
    scoring: ScoringConfig,
    split_name: str,
) -> dict[str, Any]:
    return score_heldout_prediction(
        counts,
        rates,
        reference,
        scoring,
        "lfads",
        split_name,
        "direct_model",
        True,
    )


def _population_lag(observed: np.ndarray, predicted: np.ndarray, maximum_lag: int = 10) -> int:
    observed_trace = observed.mean(axis=(0, 2))
    predicted_trace = predicted.mean(axis=(0, 2))
    observed_centered = observed_trace - observed_trace.mean()
    predicted_centered = predicted_trace - predicted_trace.mean()
    best_lag = 0
    best_correlation = -math.inf
    for lag in range(-maximum_lag, maximum_lag + 1):
        if lag < 0:
            left, right = observed_centered[-lag:], predicted_centered[:lag]
        elif lag > 0:
            left, right = observed_centered[:-lag], predicted_centered[lag:]
        else:
            left, right = observed_centered, predicted_centered
        correlation = (
            float(np.corrcoef(left, right)[0, 1])
            if left.size > 1 and np.std(left) > 0.0 and np.std(right) > 0.0
            else 0.0
        )
        if correlation > best_correlation:
            best_correlation = correlation
            best_lag = lag
    return best_lag


def _shift_with_edges(values: np.ndarray, lag: int) -> np.ndarray:
    shifted = np.roll(values, -lag, axis=1)
    if lag > 0:
        shifted[:, -lag:, :] = shifted[:, -lag - 1 : -lag, :]
    elif lag < 0:
        shifted[:, :-lag, :] = shifted[:, -lag : -lag + 1, :]
    return shifted


def _latent_rows(
    prediction: dict[str, np.ndarray], fold_index: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    factors = np.asarray(prediction["factors"], dtype=np.float64)
    posterior = np.asarray(prediction["z0_mean"], dtype=np.float64)
    logvar = np.asarray(prediction["z0_logvar"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    summaries: dict[str, float] = {}
    posterior_kl = float(
        np.mean(np.sum(-0.5 * (1.0 + logvar - posterior**2 - np.exp(logvar)), axis=1))
    )
    for representation, matrix, temporal_variation in (
        (
            "factor",
            factors.reshape(-1, factors.shape[-1]),
            float(np.mean(np.square(np.diff(factors, axis=1)))),
        ),
        ("posterior_mean", posterior, 0.0),
    ):
        rank, rank_fraction, eigenvalues = effective_rank(matrix)
        variances = np.var(matrix, axis=0)
        maximum = float(np.max(variances)) if variances.size else 0.0
        near_zero = variances <= max(maximum * 1.0e-6, np.finfo(np.float64).eps)
        summaries[f"{representation}_effective_rank"] = rank
        summaries[f"{representation}_effective_rank_fraction"] = rank_fraction
        summaries[f"{representation}_near_zero_fraction"] = float(np.mean(near_zero))
        for dimension, variance in enumerate(variances):
            rows.append(
                {
                    "fold_index": fold_index,
                    "initialization_seed": seed,
                    "representation": representation,
                    "dimension": dimension,
                    "variance": float(variance),
                    "covariance_eigenvalue": float(eigenvalues[dimension]),
                    "effective_rank": rank,
                    "effective_rank_fraction": rank_fraction,
                    "near_zero_variance_dimensions": int(np.sum(near_zero)),
                    "near_zero_variance_fraction": float(np.mean(near_zero)),
                    "mean_norm": float(np.mean(np.linalg.norm(matrix, axis=1))),
                    "temporal_variation": temporal_variation,
                    "posterior_logvar_mean": float(np.mean(logvar)),
                    "posterior_logvar_std": float(np.std(logvar)),
                    "posterior_kl": posterior_kl,
                }
            )
    summaries["posterior_kl"] = posterior_kl
    summaries["posterior_logvar_mean"] = float(np.mean(logvar))
    summaries["factor_temporal_variation"] = float(np.mean(np.square(np.diff(factors, axis=1))))
    return rows, summaries


def _objective_row(
    checkpoint_path: Path,
    manifest_row: Any,
    resource_rows: pd.DataFrame,
) -> dict[str, Any]:
    history = pd.read_csv(checkpoint_path.parents[1] / "metrics_history.csv")
    selected = history.iloc[int(manifest_row.epoch)]
    final = history.iloc[-1]
    resource = resource_rows[
        (resource_rows["fold_index"] == int(manifest_row.fold_index))
        & (resource_rows["initialization_seed"] == int(manifest_row.initialization_seed))
    ].iloc[0]
    return {
        "fold_index": int(manifest_row.fold_index),
        "initialization_seed": int(manifest_row.initialization_seed),
        "best_epoch": int(manifest_row.epoch),
        "total_epochs": len(history),
        "early_stopping_status": bool(resource["early_stopping_triggered"]),
        "checkpoint_metric": str(manifest_row.selection_metric),
        "checkpoint_metric_value": float(manifest_row.selection_metric_value),
        "train_reconstruction_loss": float(selected["train_heldin_reconstruction_loss"]),
        "train_heldout_prediction_loss": float(selected["train_heldout_prediction_loss"]),
        "train_total_loss": float(selected["train_total_loss"]),
        "reconstruction_loss": float(selected["validation_heldin_reconstruction_loss"]),
        "heldout_prediction_loss": float(selected["validation_heldout_prediction_loss"]),
        "kl_loss": float(selected["validation_kl_loss"]),
        "kl_beta": float(selected["kl_beta"]),
        "weighted_kl_contribution": float(selected["validation_kl_loss"] * selected["kl_beta"]),
        "total_loss": float(selected["validation_total_loss"]),
        "gradient_norm": float(selected["gradient_norm"]),
        "learning_rate": float(selected["learning_rate"]),
        "final_checkpoint_metric": float(final["inner_validation_unified_bits_per_spike"]),
        "initial_checkpoint_metric": float(
            history.iloc[0]["inner_validation_unified_bits_per_spike"]
        ),
        "checkpoint_metric_improvement": float(
            manifest_row.selection_metric_value
            - history.iloc[0]["inner_validation_unified_bits_per_spike"]
        ),
        "selected_to_final_metric_change": float(
            final["inner_validation_unified_bits_per_spike"] - manifest_row.selection_metric_value
        ),
        "final_learning_rate": float(final["learning_rate"]),
        "stopped_at_minimum_epoch_boundary": bool(int(manifest_row.epoch) <= 39),
        "learning_active_at_termination": bool(float(final["learning_rate"]) > 1.0e-6),
    }


def _factor_neuron_bits(
    dataset: Any,
    fold: Any,
    scoring: ScoringConfig,
    baseline_scores: pd.DataFrame,
    reference: np.ndarray,
) -> np.ndarray:
    row = baseline_scores[
        (baseline_scores["repeat_index"] == EXPECTED_REPEAT)
        & (baseline_scores["fold_index"] == int(fold.fold_index))
        & (baseline_scores["method_name"] == "factor_latent_train_selected")
    ]
    if len(row) != 1:
        raise ValueError("compatible factor-latent fold prediction is unavailable")
    parameters = json.loads(str(row.iloc[0]["selected_hyperparameters_json"]))
    predicted = predict_factor_latent(
        dataset,
        fold.train_trials,
        fold.eval_trials,
        fold.heldin,
        fold.heldout,
        scoring,
        parameters,
    )
    counts = lfads_pilot._counts(dataset, fold.eval_trials, fold.heldout)
    values = per_neuron_diagnostics(
        counts,
        predicted,
        reference,
        fold.heldout,
        scoring.bin_size_ms,
        int(fold.fold_index),
        -1,
    )["unified_bits_per_spike"].to_numpy(dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def _aggregate_diagnostics(
    config: dict[str, Any],
    integrity: dict[str, Any],
    run_frame: pd.DataFrame,
    neuron_frame: pd.DataFrame,
    time_frame: pd.DataFrame,
    latent_frame: pd.DataFrame,
    rate_frame: pd.DataFrame,
    objective_frame: pd.DataFrame,
    factor_neuron_fraction: float,
    pilot_summary: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    gaps = split_gap_summary(run_frame)
    means = run_frame.groupby("split_name")["unified_bits_per_spike"].mean()
    positive_neuron = float((neuron_frame["unified_bits_per_spike"] > 0.0).mean())
    negative_neuron = float((neuron_frame["unified_bits_per_spike"] < 0.0).mean())
    rate_quantiles = {
        str(int(quantile)): float(value)
        for quantile, value in neuron_frame.groupby("firing_rate_quantile")[
            "unified_bits_per_spike"
        ]
        .mean()
        .items()
    }
    variance_quantiles = {
        str(int(quantile)): float(value)
        for quantile, value in neuron_frame.groupby("variance_quantile")["unified_bits_per_spike"]
        .mean()
        .items()
    }
    factor_rows = latent_frame[latent_frame["representation"] == "factor"].drop_duplicates(
        ["fold_index", "initialization_seed"]
    )
    posterior_rows = latent_frame[
        latent_frame["representation"] == "posterior_mean"
    ].drop_duplicates(["fold_index", "initialization_seed"])
    mean_rank = float(factor_rows["effective_rank"].mean())
    mean_rank_fraction = float(factor_rows["effective_rank_fraction"].mean())
    posterior_rank_fraction = float(posterior_rows["effective_rank_fraction"].mean())
    factor_near_zero_fraction = float(factor_rows["near_zero_variance_fraction"].mean())
    spectrum_correlations: list[float] = []
    factor_spectra = latent_frame[latent_frame["representation"] == "factor"]
    for _, fold_spectra in factor_spectra.groupby("fold_index"):
        matrix = fold_spectra.pivot(
            index="initialization_seed", columns="dimension", values="covariance_eigenvalue"
        ).to_numpy(dtype=np.float64)
        correlations = np.corrcoef(matrix)
        spectrum_correlations.extend(correlations[np.triu_indices_from(correlations, k=1)])
    factor_spectrum_stability = float(np.mean(spectrum_correlations))
    mean_kl = float(objective_frame["kl_loss"].mean())
    mean_weighted_kl = float(objective_frame["weighted_kl_contribution"].mean())
    collapse = detect_posterior_collapse(
        mean_rank_fraction,
        posterior_rank_fraction,
        mean_kl,
        config["thresholds"],
    )

    global_recovery = max(float(rate_frame["global_calibration_recovery"].mean()), 0.0)
    per_neuron_recovery = max(float(rate_frame["per_neuron_calibration_recovery"].mean()), 0.0)
    lag_recovery = max(float(rate_frame["lag_correction_recovery"].mean()), 0.0)
    spike_weight = neuron_frame["observed_spike_count"].to_numpy(dtype=np.float64)
    neuron_bits = neuron_frame["unified_bits_per_spike"].to_numpy(dtype=np.float64)
    negative_upper = float(
        -np.sum(np.minimum(neuron_bits, 0.0) * spike_weight) / max(np.sum(spike_weight), 1.0)
    )
    rate_cutoff = float(neuron_frame["observed_rate_hz"].quantile(0.75))
    high_rate = neuron_frame["observed_rate_hz"] >= rate_cutoff
    high_rate_upper = float(
        -np.sum(
            np.minimum(neuron_frame.loc[high_rate, "unified_bits_per_spike"], 0.0)
            * neuron_frame.loc[high_rate, "observed_spike_count"]
        )
        / max(float(neuron_frame["observed_spike_count"].sum()), 1.0)
    )
    peak = time_frame[time_frame["relative_time_seconds"].abs() <= 0.10]
    peak_weight = np.maximum(peak["observed_population_rate_hz"].to_numpy(dtype=np.float64), 0.0)
    peak_upper = float(
        -np.sum(np.minimum(peak["unified_bits_per_spike"], 0.0) * peak_weight)
        / max(np.sum(time_frame["observed_population_rate_hz"]), 1.0)
    )
    baseline_gap = float(pilot_summary["pilot_repeat_baseline_mean"]) - float(
        means["outer_evaluation"]
    )
    components = [
        ("global rate bias", global_recovery, "train-fit global scalar calibration"),
        (
            "per-neuron mean-rate bias",
            per_neuron_recovery,
            "train-fit per-neuron scalar calibration; overlaps global calibration",
        ),
        ("temporal oversmoothing", lag_recovery, "diagnostic outer-trace lag correction"),
        (
            "negative-neuron concentration",
            negative_upper,
            "replace negative neuron contributions by reference",
        ),
        (
            "high-rate-neuron failures",
            high_rate_upper,
            "negative contribution among top-rate quartile",
        ),
        (
            "peak-alignment failures",
            peak_upper,
            "negative contribution within 100 ms of peak speed",
        ),
        (
            "latent underutilization",
            0.0,
            "effective-rank and KL evidence; score recovery is not identifiable without retraining",
        ),
    ]
    largest_measured = max(global_recovery, per_neuron_recovery, lag_recovery, negative_upper)
    components.append(
        (
            "unexplained remainder",
            max(baseline_gap - largest_measured, 0.0),
            "baseline gap minus largest non-additive diagnostic recovery estimate",
        )
    )
    decomposition = pd.DataFrame(
        [
            {
                "component": component,
                "estimated_recoverable_bits_per_spike": estimate,
                "evidence": evidence,
                "diagnostic_only": True,
            }
            for component, estimate, evidence in components
        ]
    )

    before = float(
        time_frame[time_frame["relative_time_seconds"] < -0.10]["unified_bits_per_spike"].mean()
    )
    at_peak = float(peak["unified_bits_per_spike"].mean())
    after = float(
        time_frame[time_frame["relative_time_seconds"] > 0.10]["unified_bits_per_spike"].mean()
    )
    high_rate_cutoff = float(time_frame["observed_population_rate_hz"].quantile(0.75))
    low_rate_cutoff = float(time_frame["observed_population_rate_hz"].quantile(0.25))
    high_rate_time_score = float(
        time_frame[time_frame["observed_population_rate_hz"] >= high_rate_cutoff][
            "unified_bits_per_spike"
        ].mean()
    )
    low_rate_time_score = float(
        time_frame[time_frame["observed_population_rate_hz"] <= low_rate_cutoff][
            "unified_bits_per_spike"
        ].mean()
    )
    smoothness_ratio = float(rate_frame["first_difference_variance_ratio"].mean())
    rate_ratio = float(rate_frame["global_rate_ratio"].mean())
    if collapse and largest_measured > float(config["thresholds"]["baseline_gap_repairable_limit"]):
        dominant = "posterior_or_latent_collapse"
        actionable = True
        estimated_recovery = largest_measured
    else:
        dominant = "mismatch_between_dynamic_lfads_assumptions_and_cosmoothing_task"
        actionable = False
        estimated_recovery = largest_measured
    secondary = []
    if smoothness_ratio < float(config["thresholds"]["oversmoothing_ratio"]):
        secondary.append("excessive_temporal_smoothing")
    if abs(rate_ratio - 1.0) > float(config["thresholds"]["severe_rate_bias_fraction"]):
        secondary.append("rate_scale_bias")
    if collapse:
        secondary.extend(["posterior_or_latent_collapse", "insufficient_latent_utilization"])
    elif mean_rank_fraction < float(config["thresholds"]["low_latent_variance_fraction"]):
        secondary.append("insufficient_latent_utilization")
    secondary = [value for value in dict.fromkeys(secondary) if value != dominant]
    recommendation = recommend_next_action(
        bool(integrity["integrity_checks_passed"]),
        dominant,
        estimated_recovery,
        actionable,
        config,
        secondary,
    )
    summary = {
        **integrity,
        "dataset_name": "mc_maze_large",
        "dataset_hash": str(config["dataset"]["expected_hash"]),
        "no_training_performed": True,
        "accepted_outer_scores_reproduced": True,
        "train_mean_unified_bits_per_spike": float(means["outer_training"]),
        "inner_mean_unified_bits_per_spike": float(means["inner_validation"]),
        "outer_mean_unified_bits_per_spike": float(means["outer_evaluation"]),
        "pilot_repeat_baseline_mean": float(pilot_summary["pilot_repeat_baseline_mean"]),
        "mean_baseline_gap": baseline_gap,
        "mean_train_to_inner_gap": float(gaps["train_to_inner_gap"].mean()),
        "mean_inner_to_outer_gap": float(gaps["inner_to_outer_gap"].mean()),
        "positive_neuron_fraction": positive_neuron,
        "negative_neuron_fraction": negative_neuron,
        "zero_neuron_fraction": float((neuron_frame["unified_bits_per_spike"] == 0.0).mean()),
        "median_neuron_unified_bits_per_spike": float(
            neuron_frame["unified_bits_per_spike"].median()
        ),
        "fraction_neurons_beating_factor_latent": factor_neuron_fraction,
        "mean_neuron_bits_by_firing_rate_quantile": rate_quantiles,
        "mean_neuron_bits_by_variance_quantile": variance_quantiles,
        "time_before_peak_mean_bits_per_spike": before,
        "time_at_peak_mean_bits_per_spike": at_peak,
        "time_after_peak_mean_bits_per_spike": after,
        "high_rate_time_bin_mean_bits_per_spike": high_rate_time_score,
        "low_rate_time_bin_mean_bits_per_spike": low_rate_time_score,
        "time_resolved_failure_pattern": (
            f"before={before:.6f}, peak={at_peak:.6f}, after={after:.6f}; "
            f"mean inferred lag={rate_frame['population_lag_bins'].mean():.3f} bins"
        ),
        "mean_global_rate_ratio": rate_ratio,
        "global_calibration_recovery": global_recovery,
        "per_neuron_calibration_recovery": per_neuron_recovery,
        "rate_bias_finding": (
            f"predicted/observed global rate ratio={rate_ratio:.6f}; diagnostic global recovery="
            f"{global_recovery:.6f}; per-neuron recovery={per_neuron_recovery:.6f}"
        ),
        "mean_first_difference_variance_ratio": smoothness_ratio,
        "mean_second_difference_variance_ratio": float(
            rate_frame["second_difference_variance_ratio"].mean()
        ),
        "temporal_smoothness_finding": (
            f"first-difference variance ratio={smoothness_ratio:.6f}; "
            f"lag-correction recovery={lag_recovery:.6f}"
        ),
        "mean_effective_rank": mean_rank,
        "mean_effective_rank_fraction": mean_rank_fraction,
        "mean_factor_near_zero_variance_fraction": factor_near_zero_fraction,
        "factor_spectrum_seed_stability": factor_spectrum_stability,
        "mean_posterior_effective_rank_fraction": posterior_rank_fraction,
        "posterior_collapse_detected": collapse,
        "latent_utilization_finding": (
            f"factor effective-rank fraction={mean_rank_fraction:.6f}; "
            f"posterior-mean effective-rank fraction={posterior_rank_fraction:.6f}; "
            f"full posterior collapse={collapse}"
        ),
        "mean_selected_kl_loss": mean_kl,
        "mean_weighted_kl_contribution": mean_weighted_kl,
        "mean_selected_reconstruction_loss": float(objective_frame["reconstruction_loss"].mean()),
        "mean_selected_train_reconstruction_loss": float(
            objective_frame["train_reconstruction_loss"].mean()
        ),
        "mean_selected_train_heldout_prediction_loss": float(
            objective_frame["train_heldout_prediction_loss"].mean()
        ),
        "mean_selected_heldout_prediction_loss": float(
            objective_frame["heldout_prediction_loss"].mean()
        ),
        "early_stopping_fraction": float(objective_frame["early_stopping_status"].mean()),
        "minimum_epoch_boundary_fraction": float(
            objective_frame["stopped_at_minimum_epoch_boundary"].mean()
        ),
        "learning_active_at_termination_fraction": float(
            objective_frame["learning_active_at_termination"].mean()
        ),
        "objective_balance_finding": (
            f"mean selected reconstruction={objective_frame['reconstruction_loss'].mean():.8f}; "
            f"held-out prediction={objective_frame['heldout_prediction_loss'].mean():.8f}; "
            f"KL={mean_kl:.8f}; weighted KL={mean_weighted_kl:.10f}; "
            f"early-stop fraction={objective_frame['early_stopping_status'].mean():.3f}"
        ),
        "dominant_failure_mode": dominant,
        "secondary_failure_modes": secondary,
        "estimated_recoverable_gap": estimated_recovery,
        "recommended_next_action": recommendation["recommended_next_action"],
        "targeted_repair_available": recommendation["targeted_repair_available"],
        "full_lfads_evaluation_allowed": False,
        "official_leaderboard_claim": False,
        "one_heldout_neuron_mask": True,
        "gap_components_overlap": True,
    }
    return summary, decomposition, recommendation


def run_lfads_diagnostics(config: dict[str, Any]) -> dict[str, Any]:
    """Audit accepted LFADS checkpoints without training or checkpoint reselection."""
    validate_lfads_diagnostics_config(config)
    pilot_config = _load_yaml(_resolve(str(config["inputs"]["pilot_config_path"])), "pilot config")
    lfads_pilot.validate_lfads_pilot_config(pilot_config)
    pilot_summary = _load_json(
        _resolve(str(config["inputs"]["pilot_summary_path"])), "pilot summary"
    )
    manifest = pd.read_csv(_resolve(str(config["inputs"]["checkpoint_manifest_path"])))
    runs = pd.read_csv(_resolve(str(config["inputs"]["pilot_runs_path"])))
    baseline_scores = pd.read_csv(_resolve(str(config["inputs"]["baseline_scores_path"])))
    baseline_summary = _load_json(
        _resolve(str(config["inputs"]["baseline_summary_path"])), "baseline summary"
    )
    readiness = _load_json(_resolve(str(config["inputs"]["readiness_path"])), "readiness")
    integrity = validate_checkpoint_integrity(manifest, runs, pilot_summary, config)
    if (
        str(baseline_summary.get("baseline_to_beat")) != "factor_latent_train_selected"
        or str(readiness.get("baseline_to_beat")) != "factor_latent_train_selected"
        or str(readiness.get("dataset_hash")) != str(config["dataset"]["expected_hash"])
    ):
        raise ValueError("baseline/readiness protocol does not match accepted diagnostics")
    inputs = lfads_pilot._load_protocol_inputs(pilot_config)
    dataset = inputs["dataset"]
    folds = {int(fold.fold_index): fold for fold in inputs["folds"]}
    scoring = lfads_pilot._scoring(pilot_config)
    if dataset.bin_size_ms != int(config["protocol"]["target_bin_size_ms"]):
        raise ValueError("diagnostic dataset bin size does not match accepted protocol")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resource_path = _resolve(str(config["inputs"]["pilot_runs_path"])).parent
    resources = pd.read_csv(resource_path / "training_resource_summary.csv")
    behavior_speed = compute_hand_speed(
        np.asarray(dataset.behavior),
        list(dataset.behavior_names or []),
        dataset.bin_size_ms / 1000.0,
    )

    run_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    neuron_frames: list[pd.DataFrame] = []
    time_frames: list[pd.DataFrame] = []
    latent_rows: list[dict[str, Any]] = []
    rate_rows: list[dict[str, Any]] = []
    objective_rows: list[dict[str, Any]] = []
    factor_bits_by_fold: dict[int, np.ndarray] = {}
    factor_comparisons: list[float] = []

    for manifest_row in manifest.sort_values(["fold_index", "initialization_seed"]).itertuples(
        index=False
    ):
        fold_index = int(manifest_row.fold_index)
        seed = int(manifest_row.initialization_seed)
        fold = folds[fold_index]
        validate_neuron_partition(fold.heldin, fold.heldout, dataset.spikes.shape[2])
        inner_seed = int(pilot_config["inner_checkpoint_selection"]["split_seed_base"]) + fold_index
        inner_train, inner_validation = lfads_pilot.build_inner_split(
            fold.train_trials,
            inputs["assignments"],
            float(pilot_config["inner_checkpoint_selection"]["validation_fraction"]),
            inner_seed,
        )
        if np.intersect1d(inner_validation, fold.eval_trials).size:
            raise ValueError("outer evaluation trials entered inner validation")
        mask = lfads_pilot._mask_from_fold(fold.heldin, fold.heldout, dataset.spikes.shape[2])
        checkpoint_path = Path(str(manifest_row.checkpoint_path))
        checkpoint: dict[str, Any] = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        checkpoint_model = checkpoint["config"]["model"]
        pilot_model = pilot_config["model"]
        architecture_pairs = (
            (checkpoint_model["encoder_hidden_dim"], pilot_model["encoder_hidden_dim"]),
            (checkpoint_model["generator_hidden_dim"], pilot_model["generator_hidden_dim"]),
            (checkpoint_model["latent_dim"], pilot_model["latent_dim"]),
            (checkpoint_model["factor_dim"], pilot_model["factor_dim"]),
        )
        if any(int(actual) != int(expected) for actual, expected in architecture_pairs):
            raise ValueError(
                f"checkpoint architecture differs from pilot config: {checkpoint_path}"
            )
        model = load_lfads_gru_from_checkpoint(
            checkpoint_path,
            int(fold.heldin.size),
            dataset.spikes.shape[2],
            checkpoint["config"],
            device,
        )
        split_trials = {
            "outer_training": fold.train_trials,
            "inner_validation": inner_validation,
            "outer_evaluation": fold.eval_trials,
        }
        predictions: dict[str, dict[str, np.ndarray]] = {}
        inner_train_counts = lfads_pilot._counts(dataset, inner_train, fold.heldout)
        outer_train_counts = lfads_pilot._counts(dataset, fold.train_trials, fold.heldout)
        for split_name, trial_ids in split_trials.items():
            seed_everything(seed, deterministic=True)
            loader = lfads_pilot._loader(
                dataset,
                trial_ids,
                mask,
                int(pilot_config["training"]["batch_size"]),
                split_name,
            )
            prediction = _diagnostic_prediction(model, loader, device)
            predictions[split_name] = prediction
            counts = lfads_pilot._counts(dataset, trial_ids, fold.heldout)
            reference_counts = (
                outer_train_counts if split_name == "outer_evaluation" else inner_train_counts
            )
            reference = train_heldout_mean_rate_reference(reference_counts, counts.shape, scoring)
            rates = prediction["rates"][:, :, fold.heldout]
            scored = _score(counts, rates, reference, scoring, split_name)
            neuron_split = per_neuron_diagnostics(
                counts, rates, reference, fold.heldout, dataset.bin_size_ms, fold_index, seed
            )
            run_rows.append(
                {
                    "repeat_index": EXPECTED_REPEAT,
                    "fold_index": fold_index,
                    "initialization_seed": seed,
                    "split_name": split_name,
                    "unified_bits_per_spike": float(scored["bits_per_spike"]),
                    "poisson_nll": float(scored["poisson_nll"]),
                    "mean_observed_rate_hz": float(counts.mean() * 1000.0 / dataset.bin_size_ms),
                    "mean_predicted_rate_hz": float(rates.mean()),
                    "absolute_rate_error_hz": float(
                        abs(rates.mean() - counts.mean() * 1000.0 / dataset.bin_size_ms)
                    ),
                    "relative_rate_error": _safe_ratio(
                        float(rates.mean() - counts.mean() * 1000.0 / dataset.bin_size_ms),
                        float(counts.mean() * 1000.0 / dataset.bin_size_ms),
                    ),
                    "negative_neuron_fraction": float(
                        (neuron_split["unified_bits_per_spike"] < 0.0).mean()
                    ),
                    "training_epoch": int(manifest_row.epoch),
                    "checkpoint_hash": str(manifest_row.checkpoint_sha256),
                }
            )
        outer_prediction = predictions["outer_evaluation"]
        outer_counts = lfads_pilot._counts(dataset, fold.eval_trials, fold.heldout)
        outer_reference = train_heldout_mean_rate_reference(
            outer_train_counts, outer_counts.shape, scoring
        )
        outer_rates = outer_prediction["rates"][:, :, fold.heldout]
        accepted = runs[
            (runs["fold_index"] == fold_index) & (runs["initialization_seed"] == seed)
        ].iloc[0]
        reproduced = float(run_rows[-1]["unified_bits_per_spike"])
        if not math.isclose(
            reproduced,
            float(accepted["outer_unified_bits_per_spike"]),
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ):
            raise ValueError(
                f"accepted outer score was not reproduced for fold {fold_index}, seed {seed}"
            )
        outer_neurons = per_neuron_diagnostics(
            outer_counts,
            outer_rates,
            outer_reference,
            fold.heldout,
            dataset.bin_size_ms,
            fold_index,
            seed,
        )
        if fold_index not in factor_bits_by_fold:
            factor_bits_by_fold[fold_index] = _factor_neuron_bits(
                dataset, fold, scoring, baseline_scores, outer_reference
            )
        outer_neurons["factor_latent_unified_bits_per_spike"] = factor_bits_by_fold[fold_index]
        outer_neurons["beats_factor_latent"] = (
            outer_neurons["unified_bits_per_spike"].to_numpy(dtype=np.float64)
            > factor_bits_by_fold[fold_index]
        )
        neuron_frames.append(outer_neurons)
        eval_mask = np.isin(dataset.trial_ids, fold.eval_trials)
        movement = behavior_speed[eval_mask].mean(axis=0)
        time_frames.append(
            time_bin_diagnostics(
                outer_counts,
                outer_rates,
                outer_reference,
                movement,
                dataset.bin_size_ms,
                fold_index,
                seed,
            )
        )
        new_latent_rows, latent_summary = _latent_rows(outer_prediction, fold_index, seed)
        latent_rows.extend(new_latent_rows)
        train_rates = predictions["outer_training"]["rates"][:, :, fold.heldout]
        observed_outer_rates = outer_counts * (1000.0 / dataset.bin_size_ms)
        global_scale = _safe_ratio(
            float(outer_train_counts.mean() * 1000.0 / dataset.bin_size_ms),
            float(train_rates.mean()),
        )
        per_neuron_scale = np.divide(
            outer_train_counts.mean(axis=(0, 1)) * (1000.0 / dataset.bin_size_ms),
            train_rates.mean(axis=(0, 1)),
            out=np.ones(fold.heldout.size, dtype=np.float64),
            where=train_rates.mean(axis=(0, 1)) > 0.0,
        )
        raw_score = float(run_rows[-1]["unified_bits_per_spike"])
        global_score = float(
            _score(
                outer_counts,
                np.clip(outer_rates * global_scale, scoring.min_rate_hz, scoring.max_rate_hz),
                outer_reference,
                scoring,
                "outer_evaluation_global_calibration_diagnostic",
            )["bits_per_spike"]
        )
        per_neuron_score = float(
            _score(
                outer_counts,
                np.clip(
                    outer_rates * per_neuron_scale[None, None, :],
                    scoring.min_rate_hz,
                    scoring.max_rate_hz,
                ),
                outer_reference,
                scoring,
                "outer_evaluation_per_neuron_calibration_diagnostic",
            )["bits_per_spike"]
        )
        lag = _population_lag(observed_outer_rates, outer_rates)
        lag_score = float(
            _score(
                outer_counts,
                _shift_with_edges(outer_rates, lag),
                outer_reference,
                scoring,
                "outer_evaluation_lag_diagnostic",
            )["bits_per_spike"]
        )
        smoothness = temporal_smoothness_metrics(observed_outer_rates, outer_rates)
        rate_rows.append(
            {
                "fold_index": fold_index,
                "initialization_seed": seed,
                "observed_mean_rate_hz": float(observed_outer_rates.mean()),
                "predicted_mean_rate_hz": float(outer_rates.mean()),
                "global_rate_ratio": _safe_ratio(
                    float(outer_rates.mean()), float(observed_outer_rates.mean())
                ),
                "mean_absolute_neuron_rate_bias_hz": float(
                    outer_neurons["rate_bias_hz"].abs().mean()
                ),
                "mean_per_neuron_rate_ratio": float(np.mean(per_neuron_scale)),
                "global_calibration_scale": global_scale,
                "global_calibrated_bits_per_spike": global_score,
                "global_calibration_recovery": global_score - raw_score,
                "per_neuron_calibrated_bits_per_spike": per_neuron_score,
                "per_neuron_calibration_recovery": per_neuron_score - raw_score,
                "population_lag_bins": lag,
                "population_lag_seconds": lag * dataset.bin_size_ms / 1000.0,
                "lag_corrected_bits_per_spike": lag_score,
                "lag_correction_recovery": lag_score - raw_score,
                **smoothness,
            }
        )
        checkpoint_rows.append(
            {
                "repeat_index": EXPECTED_REPEAT,
                "fold_index": fold_index,
                "initialization_seed": seed,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_hash": str(manifest_row.checkpoint_sha256),
                "hash_matches": True,
                "selection_split": "inner_validation",
                "selection_metric": str(manifest_row.selection_metric),
                "training_epoch": int(manifest_row.epoch),
                "accepted_outer_score": float(accepted["outer_unified_bits_per_spike"]),
                "reproduced_outer_score": reproduced,
                "absolute_reproduction_error": abs(
                    reproduced - float(accepted["outer_unified_bits_per_spike"])
                ),
                **latent_summary,
            }
        )
        objective_rows.append(_objective_row(checkpoint_path, manifest_row, resources))
        factor_comparisons.extend(outer_neurons["beats_factor_latent"].astype(float))

    run_frame = pd.DataFrame(run_rows)
    neuron_frame = pd.concat(neuron_frames, ignore_index=True)
    neuron_frame["firing_rate_quantile"] = pd.qcut(
        neuron_frame["observed_rate_hz"].rank(method="first"), 4, labels=False
    )
    neuron_frame["variance_quantile"] = pd.qcut(
        neuron_frame["observed_variance"].rank(method="first"), 4, labels=False
    )
    time_frame = pd.concat(time_frames, ignore_index=True)
    latent_frame = pd.DataFrame(latent_rows)
    rate_frame = pd.DataFrame(rate_rows)
    objective_frame = pd.DataFrame(objective_rows)
    summary, decomposition, recommendation = _aggregate_diagnostics(
        config,
        integrity,
        run_frame,
        neuron_frame,
        time_frame,
        latent_frame,
        rate_frame,
        objective_frame,
        float(np.mean(factor_comparisons)),
        pilot_summary,
    )
    tables = {
        "run_diagnostics": run_frame,
        "checkpoint_diagnostics": pd.DataFrame(checkpoint_rows),
        "neuron_diagnostics": neuron_frame,
        "time_bin_diagnostics": time_frame,
        "latent_diagnostics": latent_frame,
        "rate_diagnostics": rate_frame,
        "objective_diagnostics": objective_frame,
        "baseline_gap_decomposition": decomposition,
    }
    output_dir = _resolve(str(config["reporting"]["output_dir"]))
    write_lfads_diagnostics_outputs(output_dir, summary, tables, recommendation)
    return {
        "summary": summary,
        "tables": tables,
        "recommendation": recommendation,
        "output_dir": output_dir,
    }
