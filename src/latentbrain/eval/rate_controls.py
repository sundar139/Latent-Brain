from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from latentbrain.eval.metrics import safe_clip_rates
from latentbrain.eval.scoring import ScoringConfig, train_heldout_mean_rate_reference

TRAIN_MEAN_RATE = "train_mean_rate"
TRAIN_PER_NEURON_MEAN_RATE = "train_per_neuron_mean_rate"
TRAIN_POPULATION_SCALED_MEAN_RATE = "train_population_scaled_mean_rate"
FACTOR_LATENT = "factor_latent"
TRAIN_RATE_CALIBRATED_FACTOR_LATENT = "train_rate_calibrated_factor_latent"
SPLIT_MEAN_RATE_INVALID = "split_mean_rate_invalid"
ORACLE_SPLIT_SCALED_FACTOR_LATENT_INVALID = "oracle_split_scaled_factor_latent_invalid"

VALID_CONTROLS = (
    TRAIN_MEAN_RATE,
    TRAIN_PER_NEURON_MEAN_RATE,
    TRAIN_POPULATION_SCALED_MEAN_RATE,
    FACTOR_LATENT,
    TRAIN_RATE_CALIBRATED_FACTOR_LATENT,
)

INVALID_CONTROLS = (
    SPLIT_MEAN_RATE_INVALID,
    ORACLE_SPLIT_SCALED_FACTOR_LATENT_INVALID,
)

KNOWN_CONTROLS = (*VALID_CONTROLS, *INVALID_CONTROLS)

INVALID_REASONS: dict[str, str] = {
    SPLIT_MEAN_RATE_INVALID: (
        "Fit on the evaluation split's own held-out target counts; leaks evaluation targets."
    ),
    ORACLE_SPLIT_SCALED_FACTOR_LATENT_INVALID: (
        "Rescaled using the evaluation split's own held-out target mean; leaks evaluation targets."
    ),
}

CONTROL_NOTES: dict[str, str] = {
    TRAIN_MEAN_RATE: "Canonical train-only held-out mean rate; scores 0.0 against itself.",
    TRAIN_PER_NEURON_MEAN_RATE: (
        "Per-neuron train-only mean rate. Identical to the canonical reference by construction, "
        "so it also scores 0.0; retained to make that degeneracy explicit."
    ),
    TRAIN_POPULATION_SCALED_MEAN_RATE: (
        "Train-only held-out profile rescaled by a population factor read from held-in spikes, "
        "which are legal model inputs. No held-out target counts are used."
    ),
    FACTOR_LATENT: "Train-only factor-analysis latents decoded to held-out rates.",
    TRAIN_RATE_CALIBRATED_FACTOR_LATENT: (
        "Factor-latent rescaled by a calibration estimated on train held-out counts only."
    ),
}

CALIBRATION_METHODS = ("multiplicative", "log_bias")


def is_valid_control(method_name: str) -> bool:
    return method_name in VALID_CONTROLS


def invalid_reason(method_name: str) -> str:
    return INVALID_REASONS.get(method_name, "")


def _population_rate_hz(counts: np.ndarray, bin_size_ms: int) -> float:
    seconds = counts.shape[0] * counts.shape[1] * (bin_size_ms / 1000.0)
    if seconds <= 0.0 or counts.shape[2] == 0:
        return float("nan")
    return float(counts.sum() / (seconds * counts.shape[2]))


def compute_train_mean_rate_control(
    train_heldout_counts: np.ndarray,
    target_shape: tuple[int, int, int],
    scoring: ScoringConfig,
) -> dict[str, Any]:
    """Canonical train-only held-out mean rate. Equals the reference, so it scores 0.0."""
    rates = train_heldout_mean_rate_reference(train_heldout_counts, target_shape, scoring)
    return {
        "method_name": TRAIN_MEAN_RATE,
        "valid_model": True,
        "invalid_reason": "",
        "predicted_rates_hz": rates,
        "notes": CONTROL_NOTES[TRAIN_MEAN_RATE],
    }


def compute_train_per_neuron_mean_rate_control(
    train_heldout_counts: np.ndarray,
    target_shape: tuple[int, int, int],
    scoring: ScoringConfig,
) -> dict[str, Any]:
    """Per-neuron train mean rate. The canonical reference already is this, so the score is 0.0."""
    rates = train_heldout_mean_rate_reference(train_heldout_counts, target_shape, scoring)
    return {
        "method_name": TRAIN_PER_NEURON_MEAN_RATE,
        "valid_model": True,
        "invalid_reason": "",
        "predicted_rates_hz": rates,
        "notes": CONTROL_NOTES[TRAIN_PER_NEURON_MEAN_RATE],
    }


def compute_train_population_scaled_mean_rate_control(
    train_heldout_counts: np.ndarray,
    train_heldin_counts: np.ndarray,
    split_heldin_counts: np.ndarray,
    target_shape: tuple[int, int, int],
    scoring: ScoringConfig,
) -> dict[str, Any]:
    """Train held-out profile scaled by a held-in population factor.

    Held-in spikes are model inputs, not held-out targets, so this stays a valid control. It
    tests whether the split-level rate offset is recoverable without touching evaluation targets.
    """
    profile = train_heldout_mean_rate_reference(train_heldout_counts, target_shape, scoring)
    train_population = _population_rate_hz(train_heldin_counts, scoring.bin_size_ms)
    split_population = _population_rate_hz(split_heldin_counts, scoring.bin_size_ms)
    scale = 1.0
    if np.isfinite(train_population) and train_population > 0.0 and np.isfinite(split_population):
        scale = float(split_population / train_population)
    rates = safe_clip_rates(profile * scale, scoring.min_rate_hz, scoring.max_rate_hz)
    return {
        "method_name": TRAIN_POPULATION_SCALED_MEAN_RATE,
        "valid_model": True,
        "invalid_reason": "",
        "predicted_rates_hz": rates,
        "population_scale": scale,
        "notes": CONTROL_NOTES[TRAIN_POPULATION_SCALED_MEAN_RATE],
    }


def compute_split_mean_rate_invalid_control(
    split_heldout_counts: np.ndarray,
    scoring: ScoringConfig,
) -> dict[str, Any]:
    """Invalid: each held-out neuron predicted by its mean rate on the evaluation split itself."""
    rates = train_heldout_mean_rate_reference(
        split_heldout_counts, split_heldout_counts.shape, scoring
    )
    return {
        "method_name": SPLIT_MEAN_RATE_INVALID,
        "valid_model": False,
        "invalid_reason": INVALID_REASONS[SPLIT_MEAN_RATE_INVALID],
        "predicted_rates_hz": rates,
        "notes": INVALID_REASONS[SPLIT_MEAN_RATE_INVALID],
    }


def compute_train_rate_calibration(
    train_heldout_counts: np.ndarray,
    train_predicted_rates_hz: np.ndarray,
    scoring: ScoringConfig,
    method: str = "multiplicative",
) -> dict[str, Any]:
    """Estimate one scalar calibration from train held-out counts and train predictions only."""
    if method not in CALIBRATION_METHODS:
        msg = f"calibration method must be one of {CALIBRATION_METHODS}"
        raise ValueError(msg)
    counts = np.asarray(train_heldout_counts, dtype=np.float64)
    predicted = np.asarray(train_predicted_rates_hz, dtype=np.float64)
    if counts.shape != predicted.shape:
        msg = "train counts and train predicted rates must have the same shape"
        raise ValueError(msg)
    seconds_per_bin = scoring.bin_size_ms / 1000.0
    observed_rate = float(counts.sum()) / (counts.size * seconds_per_bin)
    predicted_rate = float(predicted.mean())
    scale = 1.0
    if predicted_rate > 0.0 and np.isfinite(predicted_rate):
        scale = observed_rate / predicted_rate
    return {
        "method": method,
        "scale": float(scale),
        "log_bias": float(np.log(scale)) if scale > 0.0 else 0.0,
        "train_observed_rate_hz": observed_rate,
        "train_predicted_rate_hz": predicted_rate,
        "fit_on": "train_heldout_counts_only",
    }


def apply_rate_calibration(
    predicted_rates_hz: np.ndarray,
    calibration: dict[str, Any],
    scoring: ScoringConfig,
) -> np.ndarray:
    """Apply a fixed train-derived calibration.

    Multiplicative and log-bias parameterizations agree by construction.
    """
    rates = np.asarray(predicted_rates_hz, dtype=np.float64)
    method = str(calibration["method"])
    if method == "multiplicative":
        calibrated = rates * float(calibration["scale"])
    elif method == "log_bias":
        floor = np.clip(rates, a_min=1e-300, a_max=None)
        calibrated = np.exp(np.log(floor) + float(calibration["log_bias"]))
    else:
        msg = f"calibration method must be one of {CALIBRATION_METHODS}"
        raise ValueError(msg)
    return safe_clip_rates(calibrated, scoring.min_rate_hz, scoring.max_rate_hz)


def compute_oracle_split_scaled_factor_latent_invalid_control(
    predicted_rates_hz: np.ndarray,
    split_heldout_counts: np.ndarray,
    scoring: ScoringConfig,
) -> dict[str, Any]:
    """Invalid: factor-latent rescaled to match the evaluation split's own observed mean rate."""
    counts = np.asarray(split_heldout_counts, dtype=np.float64)
    predicted = np.asarray(predicted_rates_hz, dtype=np.float64)
    seconds_per_bin = scoring.bin_size_ms / 1000.0
    observed_rate = float(counts.sum()) / (counts.size * seconds_per_bin)
    predicted_rate = float(predicted.mean())
    scale = 1.0
    if predicted_rate > 0.0 and np.isfinite(predicted_rate):
        scale = observed_rate / predicted_rate
    rates = safe_clip_rates(predicted * scale, scoring.min_rate_hz, scoring.max_rate_hz)
    return {
        "method_name": ORACLE_SPLIT_SCALED_FACTOR_LATENT_INVALID,
        "valid_model": False,
        "invalid_reason": INVALID_REASONS[ORACLE_SPLIT_SCALED_FACTOR_LATENT_INVALID],
        "predicted_rates_hz": rates,
        "oracle_scale": scale,
        "notes": INVALID_REASONS[ORACLE_SPLIT_SCALED_FACTOR_LATENT_INVALID],
    }


def select_best_valid_method(
    scores: pd.DataFrame,
    metric: str = "unified_bits_per_spike",
) -> str | None:
    """Best method among valid models only. Invalid controls can never win."""
    if scores.empty:
        return None
    valid = scores[scores["valid_model"].astype(bool)]
    if valid.empty:
        return None
    ranked = valid.groupby("method_name", sort=True)[metric].mean().sort_values(ascending=False)
    return str(ranked.index[0])
