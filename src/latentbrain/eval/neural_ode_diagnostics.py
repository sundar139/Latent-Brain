"""Post-hoc targeted diagnostic for the accepted MC_Maze Large neural-ODE pilot checkpoints.

Loads the 25 already-selected checkpoints from the deterministic neural-ODE pilot and audits
them without training or reselecting anything. Reuses model-agnostic scoring/diagnostic
utilities from lfads_diagnostics.py and orchestration helpers from train/neural_ode_pilot.py
and train/lfads_pilot.py. Counterfactual readouts/calibrations are diagnostic-only and never
alter the accepted checkpoint or its outer score.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
import yaml
from torch.utils.data import DataLoader

from latentbrain.data.schemas import NeuralDataset
from latentbrain.eval.baseline_suite import predict_factor_latent
from latentbrain.eval.decoding import (
    apply_standardization,
    fit_ridge_decoder,
    predict_ridge_decoder,
    standardize_train_apply,
)
from latentbrain.eval.lfads_diagnostics import (
    _population_lag,
    _safe_ratio,
    _shift_with_edges,
    effective_rank,
    per_neuron_diagnostics,
    split_gap_summary,
    temporal_smoothness_metrics,
    time_bin_diagnostics,
)
from latentbrain.eval.metrics import safe_clip_rates
from latentbrain.eval.movement_features import compute_hand_speed
from latentbrain.eval.reporting import write_neural_ode_diagnostics_outputs
from latentbrain.eval.scoring import (
    ScoringConfig,
    score_heldout_prediction,
    train_heldout_mean_rate_reference,
)
from latentbrain.models.neural_sde import NeuralSDE
from latentbrain.paths import get_repo_root, resolve_configured_path
from latentbrain.randomness import seed_everything
from latentbrain.torch.checkpoints import load_checkpoint
from latentbrain.train import lfads_pilot, neural_ode_pilot
from latentbrain.train.neural_sde_tuning import _loss_for_batch

EXPECTED_REPEAT = 0
EXPECTED_FOLDS = [0, 1, 2, 3, 4]
EXPECTED_SEEDS = [2027, 2028, 2029, 2030, 2031]
EXPECTED_RUNS = {(fold, seed) for fold in EXPECTED_FOLDS for seed in EXPECTED_SEEDS}
FROZEN_ALPHA_GRID = (10.0, 100.0, 1000.0, 10000.0)
RUN_DIAGNOSTIC_COLUMNS = [
    "repeat_index",
    "fold_index",
    "initialization_seed",
    "split_name",
    "unified_bits_per_spike",
    "poisson_nll",
    "mean_observed_rate_hz",
    "mean_predicted_rate_hz",
    "absolute_rate_error_hz",
    "relative_rate_error",
    "checkpoint_epoch",
    "checkpoint_hash",
]
COUNTERFACTUAL_COLUMNS = [
    "fold_index",
    "initialization_seed",
    "method",
    "outer_unified_bits_per_spike",
    "accepted_outer_unified_bits_per_spike",
    "recovery_vs_accepted",
    "fit_policy",
    "diagnostic_only",
]


def _resolve(path: str) -> Path:
    return resolve_configured_path(path, get_repo_root())


def validate_neural_ode_diagnostics_config(config: dict[str, Any]) -> None:
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
        msg = f"neural-ODE diagnostics config keys must be exactly {sorted(required)}"
        raise ValueError(msg)
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
        msg = "neural-ODE diagnostics must use the accepted Large pilot protocol"
        raise ValueError(msg)
    if bool(config["decision"]["full_evaluation_currently_allowed"]):
        msg = "full evaluation must remain disallowed by default in this diagnostic"
        raise ValueError(msg)
    if not bool(config["decision"]["prohibit_broad_sweep"]):
        msg = "broad sweeps must remain prohibited"
        raise ValueError(msg)
    if not all(bool(value) for value in config["diagnostics"].values()):
        msg = "all declared post-hoc diagnostics must remain enabled"
        raise ValueError(msg)
    if str(config["decision"]["default_when_no_clear_repair"]) != (
        "retire_neural_ode_and_close_neural_model_search"
    ):
        msg = "default next action must retire the neural-ODE search when no clear repair exists"
        raise ValueError(msg)


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
        msg = "diagnostic protocol does not match the accepted repeat/fold/seed schedule"
        raise ValueError(msg)
    if str(pilot_summary.get("dataset_hash")) != str(config["dataset"]["expected_hash"]):
        msg = "pilot dataset hash does not match diagnostic config"
        raise ValueError(msg)
    if (
        int(pilot_summary.get("repeat_index", -1)) != EXPECTED_REPEAT
        or [int(value) for value in pilot_summary.get("fold_indices", [])] != EXPECTED_FOLDS
        or [int(value) for value in pilot_summary.get("initialization_seeds", [])] != EXPECTED_SEEDS
    ):
        msg = "pilot summary does not match the accepted schedule"
        raise ValueError(msg)
    if (
        int(pilot_summary.get("completed_runs", -1)) != 25
        or int(pilot_summary.get("failed_runs", -1)) != 0
        or bool(pilot_summary.get("diffusion_enabled", True))
        or not bool(pilot_summary.get("leakage_checks_passed", False))
        or not bool(pilot_summary.get("checkpoint_selection_valid", False))
    ):
        msg = (
            "pilot summary is incomplete, failed, leaky, uses diffusion, or fails checkpoint "
            "validity"
        )
        raise ValueError(msg)
    if len(manifest) != 25:
        msg = "checkpoint manifest must contain exactly 25 accepted checkpoints"
        raise ValueError(msg)
    manifest_schedule = {
        (int(row.fold_index), int(row.initialization_seed))
        for row in manifest.itertuples(index=False)
        if int(row.repeat_index) == EXPECTED_REPEAT
    }
    if manifest_schedule != expected:
        msg = "checkpoint manifest does not match the accepted schedule"
        raise ValueError(msg)
    accepted_runs = runs[runs["status"].astype(str) == "completed"]
    run_schedule = {
        (int(row.fold_index), int(row.initialization_seed))
        for row in accepted_runs.itertuples(index=False)
        if int(row.repeat_index) == EXPECTED_REPEAT
    }
    if len(accepted_runs) != 25 or run_schedule != expected:
        msg = "completed pilot runs do not match the accepted schedule"
        raise ValueError(msg)
    digests = set(manifest["model_config_digest"]) | set()
    solver_digests = set(manifest["solver_config_digest"])
    if len(digests) != 1 or len(solver_digests) != 1:
        msg = "accepted checkpoints do not share one frozen model/solver configuration"
        raise ValueError(msg)

    for row in manifest.itertuples(index=False):
        if str(row.selection_split) != "inner_validation" or str(row.checkpoint_type) != "best":
            msg = "every accepted checkpoint must be selected on inner_validation"
            raise ValueError(msg)
        if str(row.selection_metric) != "inner_validation_unified_bits_per_spike":
            msg = "checkpoint selection metric must be inner-validation unified bits/spike"
            raise ValueError(msg)
        path = Path(str(row.checkpoint_path))
        if not path.is_file():
            msg = f"checkpoint is missing: {path}"
            raise FileNotFoundError(msg)
        if lfads_pilot.checkpoint_sha256(path) != str(row.checkpoint_sha256):
            msg = f"checkpoint hash mismatch: {path}"
            raise ValueError(msg)
        payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
        snapshot = payload.get("config", {})
        pilot = snapshot.get("pilot", {})
        model = snapshot.get("model", {})
        if (
            int(payload.get("epoch", -1)) != int(row.epoch)
            or int(pilot.get("repeat_index", -1)) != EXPECTED_REPEAT
            or int(pilot.get("fold_index", -1)) != int(row.fold_index)
            or int(pilot.get("initialization_seed", -1)) != int(row.initialization_seed)
            or str(pilot.get("selection_split")) != "inner_validation"
            or bool(pilot.get("outer_evaluation_used_for_selection", True))
        ):
            msg = f"checkpoint metadata does not match accepted schedule: {path}"
            raise ValueError(msg)
        if (
            int(model.get("input_dim", -1)) != neural_ode_pilot.INPUT_NEURON_COUNT
            or int(model.get("resolved_output_dim", -1)) != lfads_pilot.EXPECTED_SHAPE[2]
            or float(model.get("diffusion_scale", -1.0)) != 0.0
        ):
            msg = f"checkpoint model architecture does not match pilot config: {path}"
            raise ValueError(msg)
    return {
        "integrity_checks_passed": True,
        "accepted_checkpoints": 25,
        "excluded_preflight_artifacts": int((runs["status"].astype(str) != "completed").sum()),
        "terminated_preflight_processes_included": False,
        "accepted_checkpoint_source": "validated 25-row checkpoint manifest only",
        "checkpoint_hashes_match": True,
        "checkpoint_selection_valid": True,
        "diffusion_disabled_confirmed": True,
    }


def verify_score_reproduction(
    reproduced: float, accepted: float, fold_index: int, seed: int
) -> None:
    """Recomputed outer score must match the accepted pilot score bit-for-bit (within 1e-10)."""
    if not math.isclose(reproduced, accepted, rel_tol=0.0, abs_tol=1.0e-10):
        msg = f"accepted outer score was not reproduced for fold {fold_index}, seed {seed}"
        raise ValueError(msg)


def _scoring(bin_size_ms: int) -> ScoringConfig:
    return ScoringConfig(
        bin_size_ms=bin_size_ms,
        include_poisson_constant=True,
        min_rate_hz=1.0e-4,
        max_rate_hz=500.0,
        reference_name="train_heldout_mean_rate",
    )


def decoder_spectrum_diagnostics(model: NeuralSDE) -> dict[str, Any]:
    """Singular-value/condition/saturation diagnostics of the trained rate readout."""
    weight = model.rate_readout.weight.detach().cpu().numpy().astype(np.float64)
    singular_values = np.linalg.svd(weight, compute_uv=False)
    rank, rank_fraction, _ = effective_rank(weight.T)
    condition = float(singular_values.max() / max(singular_values.min(), np.finfo(np.float64).eps))
    row_norms = np.linalg.norm(weight, axis=1)
    return {
        "decoder_singular_values": singular_values.tolist(),
        "decoder_effective_rank": rank,
        "decoder_effective_rank_fraction": rank_fraction,
        "decoder_condition_number": condition,
        "decoder_mean_output_weight_norm": float(row_norms.mean()),
        "decoder_heldout_neuron_weight_norm": row_norms,
        "decoder_bias_mean": float(model.rate_readout.bias.detach().cpu().numpy().mean()),
    }


def _diagnostic_prediction(
    model: NeuralSDE, loader: DataLoader[dict[str, torch.Tensor]], device: torch.device
) -> dict[str, np.ndarray]:
    model.eval()
    chunks: dict[str, list[np.ndarray]] = {
        "rates": [],
        "factors": [],
        "latents": [],
        "drift": [],
        "z0_mean": [],
        "z0_logvar": [],
        "trial_ids": [],
    }
    with torch.no_grad():
        for batch in loader:
            output = model(batch["heldin_spikes"].to(device), deterministic=True)
            for key in ("rates_hz", "factors", "latents", "drift", "z0_mean", "z0_logvar"):
                target = "rates" if key == "rates_hz" else key
                chunks[target].append(output[key].detach().cpu().numpy())
            chunks["trial_ids"].append(batch["trial_id"].detach().cpu().numpy())
    return {key: np.concatenate(value) for key, value in chunks.items()}


def static_state_rates(model: NeuralSDE, z0_mean: np.ndarray, time_bins: int) -> np.ndarray:
    """Decode the encoded initial state with no learned temporal evolution (frozen weights)."""
    model.eval()
    with torch.no_grad():
        z0 = torch.as_tensor(z0_mean, dtype=torch.float32, device=model.rate_readout.weight.device)
        static_factors = model.factor_readout(z0).unsqueeze(1).expand(-1, time_bins, -1)
        rates = torch.nn.functional.softplus(model.rate_readout(static_factors))
        clamped = torch.clamp(rates, min=model.config.min_rate_hz, max=model.config.max_rate_hz)
    return np.asarray(clamped.detach().cpu().numpy())


def frozen_latent_linear_readout(
    train_factors: np.ndarray,
    train_counts: np.ndarray,
    inner_train_factors: np.ndarray,
    inner_train_counts: np.ndarray,
    inner_validation_factors: np.ndarray,
    inner_validation_counts: np.ndarray,
    outer_factors: np.ndarray,
    outer_counts: np.ndarray,
    outer_reference: np.ndarray,
    scoring: ScoringConfig,
) -> dict[str, Any]:
    """Ridge readout: alpha selected on inner train/validation only, fit on outer train only."""

    def _flat(array: np.ndarray) -> np.ndarray:
        return array.reshape(-1, array.shape[-1])

    def _targets(counts: np.ndarray) -> np.ndarray:
        return safe_clip_rates(
            counts.reshape(-1, counts.shape[-1]) * (1000.0 / scoring.bin_size_ms),
            scoring.min_rate_hz,
            scoring.max_rate_hz,
        )

    inner_train_flat, inner_train_stats = standardize_train_apply(
        _flat(inner_train_factors), _flat(inner_train_factors)
    )
    inner_validation_flat = apply_standardization(
        _flat(inner_validation_factors), inner_train_stats
    )
    inner_reference = train_heldout_mean_rate_reference(
        inner_train_counts, inner_validation_counts.shape, scoring
    )
    best_alpha, best_metric = FROZEN_ALPHA_GRID[0], -math.inf
    for alpha in FROZEN_ALPHA_GRID:
        decoder = fit_ridge_decoder(
            inner_train_flat, _targets(inner_train_counts), alpha=alpha, fit_intercept=True
        )
        predicted = safe_clip_rates(
            predict_ridge_decoder(inner_validation_flat, decoder),
            scoring.min_rate_hz,
            scoring.max_rate_hz,
        ).reshape(inner_validation_counts.shape)
        scored = score_heldout_prediction(
            inner_validation_counts,
            predicted,
            inner_reference,
            scoring,
            "neural_ode_frozen_readout_selection",
            "inner_validation",
            "diagnostic_counterfactual",
            True,
        )
        if float(scored["bits_per_spike"]) > best_metric:
            best_metric = float(scored["bits_per_spike"])
            best_alpha = alpha

    train_flat, train_stats = standardize_train_apply(_flat(train_factors), _flat(train_factors))
    outer_flat = apply_standardization(_flat(outer_factors), train_stats)
    decoder = fit_ridge_decoder(
        train_flat, _targets(train_counts), alpha=best_alpha, fit_intercept=True
    )
    predicted_outer = safe_clip_rates(
        predict_ridge_decoder(outer_flat, decoder), scoring.min_rate_hz, scoring.max_rate_hz
    ).reshape(outer_counts.shape)
    scored_outer = score_heldout_prediction(
        outer_counts,
        predicted_outer,
        outer_reference,
        scoring,
        "neural_ode_frozen_readout",
        "outer_evaluation",
        "diagnostic_counterfactual",
        True,
    )
    return {
        "selected_alpha": best_alpha,
        "inner_validation_selection_bits_per_spike": best_metric,
        "outer_unified_bits_per_spike": float(scored_outer["bits_per_spike"]),
    }


def scalar_rate_calibration(
    train_counts: np.ndarray,
    train_rates: np.ndarray,
    outer_counts: np.ndarray,
    outer_rates: np.ndarray,
    outer_reference: np.ndarray,
    scoring: ScoringConfig,
) -> dict[str, Any]:
    """Global scalar rescaling fit on outer-training data only; evaluated once on outer data."""
    observed_train_mean = float(train_counts.mean() * 1000.0 / scoring.bin_size_ms)
    predicted_train_mean = float(train_rates.mean())
    scale = _safe_ratio(observed_train_mean, predicted_train_mean) or 1.0
    calibrated = safe_clip_rates(outer_rates * scale, scoring.min_rate_hz, scoring.max_rate_hz)
    scored = score_heldout_prediction(
        outer_counts,
        calibrated,
        outer_reference,
        scoring,
        "neural_ode_scalar_calibration",
        "outer_evaluation",
        "diagnostic_counterfactual",
        True,
    )
    return {"scale": scale, "outer_unified_bits_per_spike": float(scored["bits_per_spike"])}


def _drift_jacobian_stats(
    model: NeuralSDE, latents: np.ndarray, sample_count: int = 32, seed: int = 1337
) -> dict[str, float]:
    """Local drift Jacobian norm/trace at a subsample of realized states (not a global proof)."""
    device = model.rate_readout.weight.device
    flat = np.asarray(latents, dtype=np.float32).reshape(-1, latents.shape[-1])
    generator = np.random.default_rng(seed)
    indices = generator.choice(flat.shape[0], size=min(sample_count, flat.shape[0]), replace=False)
    norms: list[float] = []
    traces: list[float] = []
    time_feature = torch.zeros(1, dtype=torch.float32, device=device)

    def drift_fn(state: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = model.drift_net(torch.cat([state, time_feature], dim=-1))
        return result

    for index in indices:
        z = torch.as_tensor(flat[index], dtype=torch.float32, device=device)
        z.requires_grad_(True)
        jacobian = torch.autograd.functional.jacobian(drift_fn, z)  # type: ignore[no-untyped-call]
        norms.append(float(jacobian.norm().detach().cpu()))
        traces.append(float(torch.diagonal(jacobian).sum().detach().cpu()))
    return {
        "mean_drift_jacobian_norm": float(np.mean(norms)),
        "mean_drift_jacobian_trace": float(np.mean(traces)),
        "contraction_fraction": float(np.mean(np.asarray(traces) < 0.0)),
    }


def _objective_row(
    model: NeuralSDE,
    checkpoint_path: Path,
    manifest_row: Any,
    inner_validation_loader: DataLoader[dict[str, torch.Tensor]],
    resource_rows: pd.DataFrame,
    bin_size_ms: int,
    training_config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    history = pd.read_csv(checkpoint_path.parents[1] / "metrics_history.csv")
    selected = history.iloc[int(manifest_row.epoch)]
    final = history.iloc[-1]
    resource = resource_rows[
        (resource_rows["fold_index"] == int(manifest_row.fold_index))
        & (resource_rows["initialization_seed"] == int(manifest_row.initialization_seed))
    ].iloc[0]
    kl_beta = float(selected["kl_beta"])
    model.eval()
    losses: dict[str, list[float]] = {
        "heldin_reconstruction_loss": [],
        "heldout_prediction_loss": [],
        "z0_kl_loss": [],
        "drift_regularization_loss": [],
    }
    with torch.no_grad():
        for batch in inner_validation_loader:
            loss, _, _ = _loss_for_batch(
                model, batch, device, bin_size_ms, kl_beta, training_config
            )
            for key in losses:
                losses[key].append(float(loss[key].detach().cpu()))
    reconstruction = float(np.mean(losses["heldin_reconstruction_loss"]))
    heldout_prediction = float(np.mean(losses["heldout_prediction_loss"]))
    kl_loss = float(np.mean(losses["z0_kl_loss"]))
    drift_regularization = float(np.mean(losses["drift_regularization_loss"]))
    return {
        "fold_index": int(manifest_row.fold_index),
        "initialization_seed": int(manifest_row.initialization_seed),
        "best_epoch": int(manifest_row.epoch),
        "total_epochs": len(history),
        "early_stopping_status": bool(resource["early_stopping_triggered"]),
        "checkpoint_metric_value": float(manifest_row.selection_metric_value),
        "train_total_loss": float(selected["train_total_loss"]),
        "validation_total_loss": float(selected["validation_total_loss"]),
        "inner_validation_reconstruction_loss": reconstruction,
        "inner_validation_heldout_prediction_loss": heldout_prediction,
        "inner_validation_z0_kl_loss": kl_loss,
        "weighted_kl_contribution": kl_loss * kl_beta,
        "inner_validation_drift_regularization_loss": drift_regularization,
        "configured_heldout_loss_weight": float(training_config.get("heldout_loss_weight", 1.0)),
        "configured_heldin_loss_weight": float(training_config.get("heldin_loss_weight", 1.0)),
        "configured_drift_regularization_scale": float(
            training_config.get("drift_regularization_scale", 0.0)
        ),
        "gradient_norm": float(selected["gradient_norm"]),
        "learning_rate": float(selected["learning_rate"]),
        "final_learning_rate": float(final["learning_rate"]),
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
        "learning_active_at_termination": bool(float(final["learning_rate"]) > 1.0e-7),
    }


def _factor_neuron_bits(
    dataset: NeuralDataset,
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
        msg = "compatible factor-latent fold prediction is unavailable"
        raise ValueError(msg)
    parameters = json.loads(str(row.iloc[0]["selected_hyperparameters_json"]))
    predicted = predict_factor_latent(
        dataset, fold.train_trials, fold.eval_trials, fold.heldin, fold.heldout, scoring, parameters
    )
    counts = lfads_pilot._counts(dataset, fold.eval_trials, fold.heldout)
    values = per_neuron_diagnostics(
        counts, predicted, reference, fold.heldout, scoring.bin_size_ms, int(fold.fold_index), -1
    )["unified_bits_per_spike"].to_numpy(dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


REPAIR_MAP = {
    "trained decoder limitation": "replace_or_retrain_only_the_heldout_readout",
    "excessive drift regularization": "reduce_one_verified_excessive_regularization_term",
    "checkpoint-selection mismatch": "correct_one_verified_checkpoint_objective_mismatch",
    "latent dimension bottleneck": (
        "increase_one_verified_bottleneck_dimension_to_a_predeclared_value"
    ),
    "late-window temporal failure": "correct_one_verified_temporal_discretization_problem",
}


def build_next_action_recommendation(
    integrity_checks_passed: bool,
    exact_required_recovery: float,
    decomposition: pd.DataFrame,
    positive_and_stable: bool,
    config: dict[str, Any],
) -> dict[str, Any]:
    actionable = decomposition[decomposition["component"] != "unexplained remainder"]
    dominant_row = actionable.loc[actionable["estimated_recoverable_bits_per_spike"].idxmax()]
    dominant_component = str(dominant_row["component"])
    dominant_estimate = float(dominant_row["estimated_recoverable_bits_per_spike"])
    repair_available = bool(
        integrity_checks_passed
        and positive_and_stable
        and dominant_estimate >= exact_required_recovery
        and dominant_component in REPAIR_MAP
    )
    if not integrity_checks_passed:
        action = "block_due_to_integrity_issue"
        rationale = "checkpoint integrity, reproduction, or leakage checks failed"
        proposed_repair = None
    elif repair_available:
        action = "run_targeted_neural_ode_repair_pilot"
        proposed_repair = REPAIR_MAP[dominant_component]
        rationale = (
            f"one dominant actionable limitation ({dominant_component}) has an estimated "
            f"recovery of {dominant_estimate:.6f}, at or above the required "
            f"{exact_required_recovery:.6f} bits/spike"
        )
    else:
        action = str(config["decision"]["default_when_no_clear_repair"])
        proposed_repair = None
        rationale = (
            "integrity is sound and the pilot is stable, but no single frozen correction can "
            "plausibly recover the required gap; broad tuning would be required"
        )
    return {
        "recommended_next_action": action,
        "integrity_checks_passed": bool(integrity_checks_passed),
        "dominant_failure_mode": dominant_component,
        "secondary_failure_modes": [
            str(name)
            for name in actionable.loc[
                actionable["estimated_recoverable_bits_per_spike"] > 0.0, "component"
            ]
            if name != dominant_component
        ],
        "exact_required_recovery": exact_required_recovery,
        "estimated_recoverable_gap": dominant_estimate,
        "targeted_repair_available": repair_available,
        "proposed_single_repair": proposed_repair,
        "full_evaluation_allowed": False,
        "broad_sweep_allowed": False,
        "rationale": rationale,
        "required_next_protocol": action,
    }


def run_neural_ode_diagnostics(config: dict[str, Any]) -> dict[str, Any]:
    """Audit accepted neural-ODE checkpoints without training or checkpoint reselection."""
    validate_neural_ode_diagnostics_config(config)
    pilot_config_path = _resolve(str(config["inputs"]["pilot_config_path"]))
    pilot_config = yaml.safe_load(pilot_config_path.read_text(encoding="utf-8"))
    neural_ode_pilot.validate_neural_ode_pilot_config(pilot_config)
    pilot_summary = json.loads(
        _resolve(str(config["inputs"]["pilot_summary_path"])).read_text(encoding="utf-8")
    )
    manifest = pd.read_csv(_resolve(str(config["inputs"]["checkpoint_manifest_path"])))
    runs = pd.read_csv(_resolve(str(config["inputs"]["pilot_runs_path"])))
    baseline_scores = pd.read_csv(_resolve(str(config["inputs"]["baseline_scores_path"])))
    baseline_summary = json.loads(
        _resolve(str(config["inputs"]["baseline_summary_path"])).read_text(encoding="utf-8")
    )
    if str(baseline_summary.get("baseline_to_beat")) != "factor_latent_train_selected":
        msg = "baseline summary does not match the accepted diagnostics baseline"
        raise ValueError(msg)
    integrity = validate_checkpoint_integrity(manifest, runs, pilot_summary, config)

    inputs = lfads_pilot._load_protocol_inputs(pilot_config)
    dataset = inputs["dataset"]
    folds = {int(fold.fold_index): fold for fold in inputs["folds"]}
    scoring = _scoring(int(config["protocol"]["target_bin_size_ms"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resource_path = _resolve(str(config["inputs"]["pilot_runs_path"])).parent
    resources = pd.read_csv(resource_path / "training_resource_summary.csv")
    behavior_speed = compute_hand_speed(
        np.asarray(dataset.behavior),
        list(dataset.behavior_names or []),
        dataset.bin_size_ms / 1000.0,
    )
    training_config = neural_ode_pilot._concrete_training(pilot_config, dataset.bin_size_ms)[
        "training"
    ]

    run_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    neuron_frames: list[pd.DataFrame] = []
    time_frames: list[pd.DataFrame] = []
    latent_rows: list[dict[str, Any]] = []
    decoder_rows: list[dict[str, Any]] = []
    dynamics_rows: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []
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
        lfads_pilot.validate_input_target_separation(
            fold.heldin, fold.heldout, int(fold.heldin.size), lfads_pilot.EXPECTED_SHAPE[2]
        )
        inner_seed = int(pilot_config["inner_checkpoint_selection"]["split_seed_base"]) + fold_index
        inner_train, inner_validation = lfads_pilot.build_inner_split(
            fold.train_trials,
            inputs["assignments"],
            float(pilot_config["inner_checkpoint_selection"]["validation_fraction"]),
            inner_seed,
        )
        if np.intersect1d(inner_validation, fold.eval_trials).size:
            msg = "outer evaluation trials entered inner validation"
            raise ValueError(msg)
        mask = lfads_pilot._mask_from_fold(fold.heldin, fold.heldout, dataset.spikes.shape[2])
        checkpoint_path = Path(str(manifest_row.checkpoint_path))

        model = neural_ode_pilot.build_pilot_model(
            pilot_config, int(fold.heldin.size), dataset.spikes.shape[2], seed
        )
        load_checkpoint(checkpoint_path, model, map_location=device)
        model.to(device)
        model.eval()

        split_trials = {
            "outer_training": fold.train_trials,
            "inner_train": inner_train,
            "inner_validation": inner_validation,
            "outer_evaluation": fold.eval_trials,
        }
        predictions: dict[str, dict[str, np.ndarray]] = {}
        loaders: dict[str, DataLoader[dict[str, torch.Tensor]]] = {}
        inner_train_counts = lfads_pilot._counts(dataset, inner_train, fold.heldout)
        outer_train_counts = lfads_pilot._counts(dataset, fold.train_trials, fold.heldout)
        for split_name, trial_ids in split_trials.items():
            seed_everything(seed, deterministic=True)
            loader = neural_ode_pilot._loader(
                dataset, trial_ids, mask, int(pilot_config["training"]["batch_size"]), split_name
            )
            loaders[split_name] = loader
            prediction = _diagnostic_prediction(model, loader, device)
            predictions[split_name] = prediction
            if split_name == "inner_train":
                continue
            counts = lfads_pilot._counts(dataset, trial_ids, fold.heldout)
            reference_counts = (
                outer_train_counts if split_name == "outer_evaluation" else inner_train_counts
            )
            reference = train_heldout_mean_rate_reference(reference_counts, counts.shape, scoring)
            rates = prediction["rates"][:, :, fold.heldout]
            scored = score_heldout_prediction(
                counts, rates, reference, scoring, "neural_ode", split_name, "direct_model", True
            )
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
                    "checkpoint_epoch": int(manifest_row.epoch),
                    "checkpoint_hash": str(manifest_row.checkpoint_sha256),
                }
            )
            if split_name == "outer_training":
                neuron_frames.append(neuron_split.assign(split_name="outer_training"))

        outer_prediction = predictions["outer_evaluation"]
        outer_counts = lfads_pilot._counts(dataset, fold.eval_trials, fold.heldout)
        outer_reference = train_heldout_mean_rate_reference(
            outer_train_counts, outer_counts.shape, scoring
        )
        outer_rates = outer_prediction["rates"][:, :, fold.heldout]
        accepted = runs[
            (runs["fold_index"] == fold_index) & (runs["initialization_seed"] == seed)
        ].iloc[0]
        reproduced = next(
            row["unified_bits_per_spike"]
            for row in run_rows[::-1]
            if row["fold_index"] == fold_index
            and row["initialization_seed"] == seed
            and row["split_name"] == "outer_evaluation"
        )
        verify_score_reproduction(
            reproduced, float(accepted["outer_unified_bits_per_spike"]), fold_index, seed
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
        outer_neurons["split_name"] = "outer_evaluation"
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

        for representation, tensor in (
            ("factor", outer_prediction["factors"]),
            ("latent", outer_prediction["latents"]),
            ("z0", outer_prediction["z0_mean"]),
        ):
            matrix = np.asarray(tensor, dtype=np.float64)
            flat = matrix.reshape(-1, matrix.shape[-1])
            rank, rank_fraction, eigenvalues = effective_rank(flat)
            variances = np.var(flat, axis=0)
            maximum = float(np.max(variances)) if variances.size else 0.0
            near_zero = variances <= max(maximum * 1.0e-6, np.finfo(np.float64).eps)
            for dimension, variance in enumerate(variances):
                latent_rows.append(
                    {
                        "fold_index": fold_index,
                        "initialization_seed": seed,
                        "representation": representation,
                        "dimension": int(dimension),
                        "variance": float(variance),
                        "covariance_eigenvalue": float(eigenvalues[dimension]),
                        "effective_rank": rank,
                        "effective_rank_fraction": rank_fraction,
                        "near_zero_variance_dimensions": int(np.sum(near_zero)),
                        "near_zero_variance_fraction": float(np.mean(near_zero)),
                    }
                )

        drift_stats = _drift_jacobian_stats(model, outer_prediction["latents"])
        latent_norms = np.linalg.norm(outer_prediction["latents"], axis=-1)
        drift_norms = np.linalg.norm(outer_prediction["drift"], axis=-1)
        dynamics_rows.append(
            {
                "fold_index": fold_index,
                "initialization_seed": seed,
                "mean_state_norm": float(latent_norms.mean()),
                "terminal_state_norm": float(latent_norms[:, -1].mean()),
                "mean_drift_norm": float(drift_norms.mean()),
                "first_difference_variance": float(
                    np.mean(np.square(np.diff(outer_prediction["latents"], axis=1)))
                ),
                "second_difference_variance": float(
                    np.mean(np.square(np.diff(outer_prediction["latents"], n=2, axis=1)))
                ),
                **drift_stats,
            }
        )

        decoder_diagnostics = decoder_spectrum_diagnostics(model)
        decoder_rows.append(
            {
                "fold_index": fold_index,
                "initialization_seed": seed,
                "decoder_effective_rank": decoder_diagnostics["decoder_effective_rank"],
                "decoder_effective_rank_fraction": decoder_diagnostics[
                    "decoder_effective_rank_fraction"
                ],
                "decoder_condition_number": decoder_diagnostics["decoder_condition_number"],
                "decoder_mean_output_weight_norm": decoder_diagnostics[
                    "decoder_mean_output_weight_norm"
                ],
                "decoder_bias_mean": decoder_diagnostics["decoder_bias_mean"],
                "fraction_rates_clipped_at_min": float(
                    np.mean(outer_rates <= scoring.min_rate_hz * 1.000001)
                ),
                "fraction_rates_clipped_at_max": float(
                    np.mean(outer_rates >= scoring.max_rate_hz * 0.999999)
                ),
            }
        )

        raw_score = reproduced
        static_rates_full = static_state_rates(
            model, outer_prediction["z0_mean"], lfads_pilot.EXPECTED_SHAPE[1]
        )
        static_scored = score_heldout_prediction(
            outer_counts,
            static_rates_full[:, :, fold.heldout],
            outer_reference,
            scoring,
            "neural_ode_static_state",
            "outer_evaluation",
            "diagnostic_counterfactual",
            True,
        )
        for method, outer_bits, fit_policy in (
            (
                "static_encoder_only_no_dynamics",
                float(static_scored["bits_per_spike"]),
                "encoded initial state only; no train-only fitting, same frozen weights",
            ),
        ):
            counterfactual_rows.append(
                {
                    "fold_index": fold_index,
                    "initialization_seed": seed,
                    "method": method,
                    "outer_unified_bits_per_spike": outer_bits,
                    "accepted_outer_unified_bits_per_spike": raw_score,
                    "recovery_vs_accepted": outer_bits - raw_score,
                    "fit_policy": fit_policy,
                    "diagnostic_only": True,
                }
            )

        readout = frozen_latent_linear_readout(
            predictions["outer_training"]["factors"],
            outer_train_counts,
            predictions["inner_train"]["factors"],
            inner_train_counts,
            predictions["inner_validation"]["factors"],
            lfads_pilot._counts(dataset, inner_validation, fold.heldout),
            outer_prediction["factors"],
            outer_counts,
            outer_reference,
            scoring,
        )
        counterfactual_rows.append(
            {
                "fold_index": fold_index,
                "initialization_seed": seed,
                "method": "frozen_latent_linear_readout",
                "outer_unified_bits_per_spike": readout["outer_unified_bits_per_spike"],
                "accepted_outer_unified_bits_per_spike": raw_score,
                "recovery_vs_accepted": readout["outer_unified_bits_per_spike"] - raw_score,
                "fit_policy": (
                    f"alpha selected on inner train/validation "
                    f"(chosen={readout['selected_alpha']}); fit on outer-training trials only; "
                    "single outer-evaluation score"
                ),
                "diagnostic_only": True,
            }
        )

        calibration = scalar_rate_calibration(
            outer_train_counts,
            predictions["outer_training"]["rates"][:, :, fold.heldout],
            outer_counts,
            outer_rates,
            outer_reference,
            scoring,
        )
        counterfactual_rows.append(
            {
                "fold_index": fold_index,
                "initialization_seed": seed,
                "method": "scalar_rate_calibration",
                "outer_unified_bits_per_spike": calibration["outer_unified_bits_per_spike"],
                "accepted_outer_unified_bits_per_spike": raw_score,
                "recovery_vs_accepted": calibration["outer_unified_bits_per_spike"] - raw_score,
                "fit_policy": (
                    f"global scale={calibration['scale']:.6f} fit on outer-training trials only"
                ),
                "diagnostic_only": True,
            }
        )

        observed_outer_rates = outer_counts * (1000.0 / dataset.bin_size_ms)
        smoothness = temporal_smoothness_metrics(observed_outer_rates, outer_rates)
        lag = _population_lag(observed_outer_rates, outer_rates)
        lag_score = float(
            score_heldout_prediction(
                outer_counts,
                _shift_with_edges(outer_rates, lag),
                outer_reference,
                scoring,
                "neural_ode_lag_diagnostic",
                "outer_evaluation",
                "diagnostic_counterfactual",
                True,
            )["bits_per_spike"]
        )
        rate_rows.append(
            {
                "fold_index": fold_index,
                "initialization_seed": seed,
                "observed_mean_rate_hz": float(observed_outer_rates.mean()),
                "predicted_mean_rate_hz": float(outer_rates.mean()),
                "global_rate_ratio": _safe_ratio(
                    float(outer_rates.mean()), float(observed_outer_rates.mean())
                ),
                "population_lag_bins": lag,
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
            }
        )
        objective_rows.append(
            _objective_row(
                model,
                checkpoint_path,
                manifest_row,
                loaders["inner_validation"],
                resources,
                dataset.bin_size_ms,
                training_config,
                device,
            )
        )
        factor_comparisons.extend(outer_neurons["beats_factor_latent"].astype(float))

    run_frame = pd.DataFrame(run_rows, columns=RUN_DIAGNOSTIC_COLUMNS)
    neuron_frame = pd.concat(neuron_frames, ignore_index=True)
    neuron_frame["firing_rate_quantile"] = pd.qcut(
        neuron_frame["observed_rate_hz"].rank(method="first"), 4, labels=False
    )
    neuron_frame["variance_quantile"] = pd.qcut(
        neuron_frame["observed_variance"].rank(method="first"), 4, labels=False
    )
    time_frame = pd.concat(time_frames, ignore_index=True)
    latent_frame = pd.DataFrame(latent_rows)
    decoder_frame = pd.DataFrame(decoder_rows)
    dynamics_frame = pd.DataFrame(dynamics_rows)
    counterfactual_frame = pd.DataFrame(counterfactual_rows, columns=COUNTERFACTUAL_COLUMNS)
    rate_frame = pd.DataFrame(rate_rows)
    objective_frame = pd.DataFrame(objective_rows)

    summary, decomposition, recommendation = _aggregate_diagnostics(
        config,
        integrity,
        run_frame,
        neuron_frame,
        time_frame,
        latent_frame,
        decoder_frame,
        dynamics_frame,
        counterfactual_frame,
        rate_frame,
        objective_frame,
        float(np.mean(factor_comparisons)),
        pilot_summary,
        pilot_config,
    )
    tables = {
        "split_diagnostics": run_frame,
        "checkpoint_integrity": pd.DataFrame(checkpoint_rows),
        "neuron_diagnostics": neuron_frame,
        "time_bin_diagnostics": time_frame,
        "latent_diagnostics": latent_frame,
        "decoder_diagnostics": decoder_frame,
        "dynamics_diagnostics": dynamics_frame,
        "counterfactual_diagnostics": counterfactual_frame,
        "objective_diagnostics": objective_frame,
        "baseline_gap_decomposition": decomposition,
    }
    output_dir = _resolve(str(config["reporting"]["output_dir"]))
    write_neural_ode_diagnostics_outputs(output_dir, summary, tables, recommendation)
    return {
        "summary": summary,
        "tables": tables,
        "recommendation": recommendation,
        "output_dir": output_dir,
    }


def _aggregate_diagnostics(
    config: dict[str, Any],
    integrity: dict[str, Any],
    run_frame: pd.DataFrame,
    neuron_frame: pd.DataFrame,
    time_frame: pd.DataFrame,
    latent_frame: pd.DataFrame,
    decoder_frame: pd.DataFrame,
    dynamics_frame: pd.DataFrame,
    counterfactual_frame: pd.DataFrame,
    rate_frame: pd.DataFrame,
    objective_frame: pd.DataFrame,
    factor_neuron_fraction: float,
    pilot_summary: dict[str, Any],
    pilot_config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    gaps = split_gap_summary(run_frame[run_frame["split_name"] != "inner_train"])
    means = run_frame.groupby("split_name")["unified_bits_per_spike"].mean()
    outer_neurons = neuron_frame[neuron_frame["split_name"] == "outer_evaluation"]
    positive_neuron = float((outer_neurons["unified_bits_per_spike"] > 0.0).mean())
    negative_neuron = float((outer_neurons["unified_bits_per_spike"] < 0.0).mean())
    factor_rows = latent_frame[latent_frame["representation"] == "factor"].drop_duplicates(
        ["fold_index", "initialization_seed"]
    )
    z0_rows = latent_frame[latent_frame["representation"] == "z0"].drop_duplicates(
        ["fold_index", "initialization_seed"]
    )
    mean_rank = float(factor_rows["effective_rank"].mean())
    mean_rank_fraction = float(factor_rows["effective_rank_fraction"].mean())
    z0_rank_fraction = float(z0_rows["effective_rank_fraction"].mean())

    margin = float(pilot_config["pilot_gates"]["full_evaluation_margin_over_baseline"])
    pilot_diff = float(pilot_summary["mean_paired_difference_vs_baseline"])
    exact_required_recovery = margin - pilot_diff
    positive_and_stable = bool(
        float(pilot_summary["mean_unified_bits_per_spike"]) >= 0.0
        and float(pilot_summary["positive_seed_fraction"]) >= 0.60
        and float(pilot_summary["seed_mean_std"]) <= 0.05
    )

    def _method_recovery(method: str) -> float:
        rows = counterfactual_frame[counterfactual_frame["method"] == method]
        return float(rows["recovery_vs_accepted"].mean()) if len(rows) else float("nan")

    readout_recovery = max(_method_recovery("frozen_latent_linear_readout"), 0.0)
    static_recovery = max(_method_recovery("static_encoder_only_no_dynamics"), 0.0)
    calibration_recovery = max(_method_recovery("scalar_rate_calibration"), 0.0)
    lag_recovery = max(float(rate_frame["lag_correction_recovery"].mean()), 0.0)
    spike_weight = outer_neurons["observed_spike_count"].to_numpy(dtype=np.float64)
    neuron_bits = outer_neurons["unified_bits_per_spike"].to_numpy(dtype=np.float64)
    negative_upper = float(
        -np.sum(np.minimum(neuron_bits, 0.0) * spike_weight) / max(np.sum(spike_weight), 1.0)
    )
    late_window = time_frame[time_frame["relative_time_seconds"] > 0.0]
    late_weight = np.maximum(
        late_window["observed_population_rate_hz"].to_numpy(dtype=np.float64), 0.0
    )
    late_upper = float(
        -np.sum(np.minimum(late_window["unified_bits_per_spike"], 0.0) * late_weight)
        / max(np.sum(time_frame["observed_population_rate_hz"]), 1.0)
    )
    baseline_gap = float(pilot_summary["pilot_repeat_baseline_mean"]) - float(
        means["outer_evaluation"]
    )

    condition_warning = float(config["thresholds"]["decoder_condition_warning"])
    low_rank_threshold = float(config["thresholds"]["low_effective_rank_fraction"])
    decoder_ill_conditioned = bool(
        (decoder_frame["decoder_condition_number"] > condition_warning).mean() > 0.5
    )
    decoder_low_rank = bool(
        decoder_frame["decoder_effective_rank_fraction"].mean() < low_rank_threshold
    )
    smoothing_ratio = float(rate_frame["first_difference_variance_ratio"].mean())
    oversmoothed = smoothing_ratio < float(config["thresholds"]["severe_temporal_smoothing_ratio"])
    drift_regularization_scale = float(
        objective_frame["configured_drift_regularization_scale"].mean()
    )

    components = [
        (
            "trained decoder limitation",
            readout_recovery if (decoder_ill_conditioned or decoder_low_rank) else 0.0,
            (
                f"frozen train-only linear readout recovery={readout_recovery:.6f}; "
                f"decoder condition ill-conditioned={decoder_ill_conditioned}; "
                f"decoder low-rank={decoder_low_rank}"
            ),
        ),
        (
            "learned dynamics limitation",
            static_recovery,
            (
                f"static encoder-only (no evolved dynamics) recovery vs accepted="
                f"{static_recovery:.6f}; positive means dynamics currently hurt relative to the "
                "encoded initial state alone"
            ),
        ),
        (
            "global rate calibration",
            calibration_recovery,
            f"train-fit global scalar calibration recovery={calibration_recovery:.6f}",
        ),
        (
            "excessive drift regularization",
            static_recovery if (oversmoothed and drift_regularization_scale > 0.0) else 0.0,
            (
                f"first-difference variance ratio={smoothing_ratio:.6f}; "
                f"configured drift_regularization_scale={drift_regularization_scale:.8f}"
            ),
        ),
        (
            "checkpoint-selection mismatch",
            max(float(objective_frame["selected_to_final_metric_change"].mean()), 0.0),
            "mean (final epoch inner-validation metric - selected checkpoint metric)",
        ),
        (
            "latent dimension bottleneck",
            readout_recovery if mean_rank_fraction < low_rank_threshold else 0.0,
            f"factor effective-rank fraction={mean_rank_fraction:.6f}",
        ),
        (
            "late-window temporal failure",
            late_upper,
            "negative contribution weighted by observed rate in the second half of the window",
        ),
        (
            "negative-neuron concentration",
            negative_upper,
            "replace negative neuron contributions by reference",
        ),
        (
            "temporal lag misalignment",
            lag_recovery,
            "diagnostic outer-trace lag correction; overlaps late-window temporal failure",
        ),
    ]
    largest_measured = max(estimate for _, estimate, _ in components)
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
                "overlaps_other_components": component != "unexplained remainder",
            }
            for component, estimate, evidence in components
        ]
    )

    recommendation = build_next_action_recommendation(
        bool(integrity["integrity_checks_passed"]),
        exact_required_recovery,
        decomposition,
        positive_and_stable,
        config,
    )
    summary = {
        **integrity,
        "dataset_name": "mc_maze_large",
        "dataset_hash": str(config["dataset"]["expected_hash"]),
        "no_replacement_model_trained": True,
        "accepted_outer_scores_reproduced": True,
        "outer_training_mean_unified_bits_per_spike": float(means["outer_training"]),
        "inner_validation_mean_unified_bits_per_spike": float(means["inner_validation"]),
        "outer_evaluation_mean_unified_bits_per_spike": float(means["outer_evaluation"]),
        "pilot_repeat_baseline_mean": float(pilot_summary["pilot_repeat_baseline_mean"]),
        "mean_baseline_gap": baseline_gap,
        "mean_train_to_inner_gap": float(gaps["train_to_inner_gap"].mean()),
        "mean_inner_to_outer_gap": float(gaps["inner_to_outer_gap"].mean()),
        "positive_neuron_fraction": positive_neuron,
        "negative_neuron_fraction": negative_neuron,
        "median_neuron_unified_bits_per_spike": float(
            outer_neurons["unified_bits_per_spike"].median()
        ),
        "fraction_neurons_beating_factor_latent": factor_neuron_fraction,
        "mean_effective_rank": mean_rank,
        "mean_effective_rank_fraction": mean_rank_fraction,
        "mean_z0_effective_rank_fraction": z0_rank_fraction,
        "mean_decoder_condition_number": float(decoder_frame["decoder_condition_number"].mean()),
        "decoder_ill_conditioned": decoder_ill_conditioned,
        "decoder_low_rank": decoder_low_rank,
        "static_state_recovery_vs_accepted": static_recovery,
        "frozen_readout_recovery_vs_accepted": readout_recovery,
        "scalar_calibration_recovery_vs_accepted": calibration_recovery,
        "mean_first_difference_variance_ratio": smoothing_ratio,
        "temporal_oversmoothing_detected": oversmoothed,
        "mean_drift_jacobian_norm": float(dynamics_frame["mean_drift_jacobian_norm"].mean()),
        "solver_discretization_directly_testable": False,
        "solver_discretization_note": (
            "solver step size cannot be varied without retraining; not directly testable from "
            "frozen checkpoints in this diagnostic"
        ),
        "exact_required_recovery": exact_required_recovery,
        "estimated_recoverable_gap": recommendation["estimated_recoverable_gap"],
        "dominant_failure_mode": recommendation["dominant_failure_mode"],
        "secondary_failure_modes": recommendation["secondary_failure_modes"],
        "targeted_repair_available": recommendation["targeted_repair_available"],
        "proposed_single_repair": recommendation["proposed_single_repair"],
        "recommended_next_action": recommendation["recommended_next_action"],
        "full_evaluation_allowed": False,
        "broad_sweep_allowed": False,
        "lfads_remains_retired": True,
        "one_heldout_neuron_mask": True,
        "gap_components_overlap": True,
        "official_leaderboard_claim": False,
    }
    return summary, decomposition, recommendation
