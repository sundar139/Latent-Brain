from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from latentbrain.data.schemas import NeuralDataset, TrialSplit
from latentbrain.data.splits import create_neuron_mask, create_trial_split
from latentbrain.eval.decoding import (
    fit_ridge_decoder,
    predict_ridge_decoder,
    standardize_train_apply,
)
from latentbrain.eval.metrics import safe_clip_rates
from latentbrain.eval.scoring import (
    ScoringConfig,
    score_heldout_prediction,
    train_heldout_mean_rate_reference,
)
from latentbrain.eval.smoothing import smooth_spike_counts, spike_counts_to_rates_hz
from latentbrain.models.factor_latent import FactorLatentModel

TRIAL_COLUMNS = [
    "trial_index",
    "split",
    "total_spikes",
    "heldin_spikes",
    "heldout_spikes",
    "population_rate_hz",
    "heldout_rate_hz",
    "zero_fraction",
    "behavior_available",
    "endpoint_dx",
    "endpoint_dy",
    "endpoint_angle_rad",
    "endpoint_distance",
    "mean_speed",
]

SPLIT_COLUMNS = [
    "split",
    "n_trials",
    "mean_total_spikes",
    "std_total_spikes",
    "mean_population_rate_hz",
    "std_population_rate_hz",
    "mean_heldout_rate_hz",
    "std_heldout_rate_hz",
    "mean_zero_fraction",
    "std_zero_fraction",
    "mean_endpoint_distance",
    "std_endpoint_distance",
    "mean_speed",
    "std_mean_speed",
]

NEURON_SPLIT_COLUMNS = [
    "split",
    "neuron_index",
    "neuron_group",
    "mean_rate_hz",
    "std_rate_hz",
    "total_spikes",
    "zero_fraction",
]

BEHAVIOR_SPLIT_COLUMNS = [
    "split",
    "behavior_name",
    "mean",
    "std",
    "min",
    "max",
    "mean_absolute_change",
]

COMPARISON_COLUMNS = [
    "metric",
    "split_a",
    "split_b",
    "split_a_mean",
    "split_b_mean",
    "difference",
    "standardized_difference",
]

COMPARISON_METRICS = (
    "total_spikes",
    "population_rate_hz",
    "heldout_rate_hz",
    "zero_fraction",
    "endpoint_distance",
    "mean_speed",
)

_POSITION_PREFIX = "hand_pos"


def _seconds_per_trial(time_bins: int, bin_size_ms: int) -> float:
    return float(time_bins) * (bin_size_ms / 1000.0)


def _position_columns(behavior_names: list[str] | None) -> tuple[int, int] | None:
    if not behavior_names:
        return None
    try:
        return (
            behavior_names.index(f"{_POSITION_PREFIX}_x"),
            behavior_names.index(f"{_POSITION_PREFIX}_y"),
        )
    except ValueError:
        return None


def _trial_behavior_summary(
    trial_behavior: np.ndarray, columns: tuple[int, int], bin_size_ms: int
) -> dict[str, float]:
    x = np.asarray(trial_behavior[:, columns[0]], dtype=np.float64)
    y = np.asarray(trial_behavior[:, columns[1]], dtype=np.float64)
    dx = float(x[-1] - x[0])
    dy = float(y[-1] - y[0])
    steps = np.hypot(np.diff(x), np.diff(y))
    seconds_per_bin = bin_size_ms / 1000.0
    return {
        "endpoint_dx": dx,
        "endpoint_dy": dy,
        "endpoint_angle_rad": float(np.arctan2(dy, dx)),
        "endpoint_distance": float(np.hypot(dx, dy)),
        "mean_speed": float(np.mean(steps) / seconds_per_bin) if steps.size else float("nan"),
    }


def compute_trial_statistics(
    spikes: np.ndarray,
    behavior: np.ndarray | None,
    behavior_names: list[str] | None,
    split_labels: np.ndarray,
    bin_size_ms: int,
    heldin_indices: np.ndarray | None = None,
    heldout_indices: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-trial spike and behavior statistics. Behavior columns are NaN when unavailable."""
    counts = np.asarray(spikes, dtype=np.float64)
    if counts.ndim != 3:
        msg = "spikes must have shape [trials, time, neurons]"
        raise ValueError(msg)
    if len(split_labels) != counts.shape[0]:
        msg = "split_labels must have one entry per trial"
        raise ValueError(msg)
    seconds = _seconds_per_trial(counts.shape[1], bin_size_ms)
    position_columns = _position_columns(behavior_names)
    has_behavior = behavior is not None and position_columns is not None
    rows: list[dict[str, object]] = []
    for index in range(counts.shape[0]):
        trial = counts[index]
        total = float(trial.sum())
        heldin = (
            float(trial[:, heldin_indices].sum()) if heldin_indices is not None else float("nan")
        )
        heldout_counts = trial[:, heldout_indices] if heldout_indices is not None else None
        row: dict[str, object] = {
            "trial_index": int(index),
            "split": str(split_labels[index]),
            "total_spikes": total,
            "heldin_spikes": heldin,
            "heldout_spikes": float(heldout_counts.sum())
            if heldout_counts is not None
            else float("nan"),
            "population_rate_hz": total / (seconds * trial.shape[1]),
            "heldout_rate_hz": float(heldout_counts.sum()) / (seconds * heldout_counts.shape[1])
            if heldout_counts is not None and heldout_counts.shape[1] > 0
            else float("nan"),
            "zero_fraction": float(np.mean(trial == 0.0)),
            "behavior_available": bool(has_behavior),
            "endpoint_dx": float("nan"),
            "endpoint_dy": float("nan"),
            "endpoint_angle_rad": float("nan"),
            "endpoint_distance": float("nan"),
            "mean_speed": float("nan"),
        }
        if has_behavior:
            assert behavior is not None and position_columns is not None
            row.update(_trial_behavior_summary(behavior[index], position_columns, bin_size_ms))
        rows.append(row)
    return pd.DataFrame(rows, columns=TRIAL_COLUMNS)


def _mean_std(values: pd.Series) -> tuple[float, float]:
    finite = values.dropna()
    if finite.empty:
        return (float("nan"), float("nan"))
    std = float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
    return (float(finite.mean()), std)


def compute_split_statistics(trial_statistics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, group in trial_statistics.groupby("split", sort=True):
        row: dict[str, object] = {"split": str(split_name), "n_trials": int(len(group))}
        for source, target in (
            ("total_spikes", "total_spikes"),
            ("population_rate_hz", "population_rate_hz"),
            ("heldout_rate_hz", "heldout_rate_hz"),
            ("zero_fraction", "zero_fraction"),
            ("endpoint_distance", "endpoint_distance"),
        ):
            mean, std = _mean_std(group[source])
            row[f"mean_{target}"] = mean
            row[f"std_{target}"] = std
        speed_mean, speed_std = _mean_std(group["mean_speed"])
        row["mean_speed"] = speed_mean
        row["std_mean_speed"] = speed_std
        rows.append(row)
    return pd.DataFrame(rows, columns=SPLIT_COLUMNS)


def compute_neuron_split_statistics(
    spikes: np.ndarray,
    split_labels: np.ndarray,
    heldin_indices: np.ndarray,
    heldout_indices: np.ndarray,
    bin_size_ms: int,
) -> pd.DataFrame:
    counts = np.asarray(spikes, dtype=np.float64)
    seconds_per_bin = bin_size_ms / 1000.0
    group_by_neuron: dict[int, str] = {int(index): "heldin" for index in heldin_indices}
    group_by_neuron.update({int(index): "heldout" for index in heldout_indices})
    rows: list[dict[str, object]] = []
    for split_name in sorted({str(label) for label in split_labels}):
        mask = np.asarray([str(label) == split_name for label in split_labels])
        subset = counts[mask]
        if subset.shape[0] == 0:
            continue
        seconds = subset.shape[0] * subset.shape[1] * seconds_per_bin
        for neuron_index in sorted(group_by_neuron):
            neuron = subset[:, :, neuron_index]
            rows.append(
                {
                    "split": split_name,
                    "neuron_index": int(neuron_index),
                    "neuron_group": group_by_neuron[neuron_index],
                    "mean_rate_hz": float(neuron.sum()) / seconds,
                    "std_rate_hz": float(np.std(neuron / seconds_per_bin)),
                    "total_spikes": float(neuron.sum()),
                    "zero_fraction": float(np.mean(neuron == 0.0)),
                }
            )
    return pd.DataFrame(rows, columns=NEURON_SPLIT_COLUMNS)


def compute_behavior_split_statistics(
    behavior: np.ndarray,
    behavior_names: list[str],
    split_labels: np.ndarray,
) -> pd.DataFrame:
    values = np.asarray(behavior, dtype=np.float64)
    if values.ndim != 3:
        msg = "behavior must have shape [trials, time, variables]"
        raise ValueError(msg)
    rows: list[dict[str, object]] = []
    for split_name in sorted({str(label) for label in split_labels}):
        mask = np.asarray([str(label) == split_name for label in split_labels])
        subset = values[mask]
        if subset.shape[0] == 0:
            continue
        for column, name in enumerate(behavior_names):
            series = subset[:, :, column]
            rows.append(
                {
                    "split": split_name,
                    "behavior_name": str(name),
                    "mean": float(np.mean(series)),
                    "std": float(np.std(series)),
                    "min": float(np.min(series)),
                    "max": float(np.max(series)),
                    "mean_absolute_change": float(np.mean(np.abs(np.diff(series, axis=1)))),
                }
            )
    return pd.DataFrame(rows, columns=BEHAVIOR_SPLIT_COLUMNS)


def compare_split_statistics(
    trial_statistics: pd.DataFrame,
    split_a: str,
    split_b: str,
) -> pd.DataFrame:
    """Per-metric mean difference between two splits, standardized by pooled trial spread."""
    a = trial_statistics[trial_statistics["split"] == split_a]
    b = trial_statistics[trial_statistics["split"] == split_b]
    rows: list[dict[str, object]] = []
    for metric in COMPARISON_METRICS:
        a_values = a[metric].dropna()
        b_values = b[metric].dropna()
        if a_values.empty or b_values.empty:
            rows.append(
                {
                    "metric": metric,
                    "split_a": split_a,
                    "split_b": split_b,
                    "split_a_mean": float("nan"),
                    "split_b_mean": float("nan"),
                    "difference": float("nan"),
                    "standardized_difference": float("nan"),
                }
            )
            continue
        a_mean = float(a_values.mean())
        b_mean = float(b_values.mean())
        pooled = float(np.sqrt((np.var(a_values, ddof=1) + np.var(b_values, ddof=1)) / 2.0))
        standardized = float("nan") if pooled == 0.0 else (a_mean - b_mean) / pooled
        rows.append(
            {
                "metric": metric,
                "split_a": split_a,
                "split_b": split_b,
                "split_a_mean": a_mean,
                "split_b_mean": b_mean,
                "difference": a_mean - b_mean,
                "standardized_difference": standardized,
            }
        )
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


REPEATED_SPLIT_COLUMNS = [
    "split_seed",
    "method_name",
    "validation_unified_bits_per_spike",
    "test_unified_bits_per_spike",
    "validation_poisson_nll",
    "test_poisson_nll",
    "validation_spike_count",
    "test_spike_count",
    "validation_heldout_rate_hz",
    "test_heldout_rate_hz",
    "notes",
]

FACTOR_LATENT_DEFAULTS: dict[str, float] = {
    "latent_dim": 8,
    "smoothing_sigma_ms": 200.0,
    "heldout_decoder_alpha": 10000.0,
    "max_iter": 1000,
    "tol": 1.0e-4,
}


def factor_latent_heldout_rates(
    dataset: NeuralDataset,
    split: TrialSplit,
    heldin_indices: np.ndarray,
    heldout_indices: np.ndarray,
    scoring: ScoringConfig,
    settings: dict[str, float],
    random_state: int = 0,
) -> dict[str, np.ndarray]:
    """Train-only factor-analysis latents decoded to held-out rates, per split."""
    bin_size_ms = scoring.bin_size_ms
    spikes = dataset.spikes
    smoothed = smooth_spike_counts(
        spikes[:, :, heldin_indices],
        bin_size_ms,
        method="gaussian",
        sigma_ms=float(settings["smoothing_sigma_ms"]),
        truncate=4.0,
    )
    input_rates = spike_counts_to_rates_hz(smoothed, bin_size_ms)

    def _flat(values: np.ndarray) -> np.ndarray:
        return np.asarray(values.reshape(values.shape[0] * values.shape[1], values.shape[2]))

    split_masks = {
        name: np.isin(dataset.trial_ids, getattr(split, name))
        for name in ("train", "validation", "test")
    }
    train_features_raw = _flat(input_rates[split_masks["train"]])
    train_features, feature_stats = standardize_train_apply(train_features_raw, train_features_raw)
    model = FactorLatentModel(
        latent_dim=int(settings["latent_dim"]),
        random_state=int(random_state),
        max_iter=int(settings["max_iter"]),
        tol=float(settings["tol"]),
    ).fit(train_features)
    train_counts = spikes[split_masks["train"]][:, :, heldout_indices]
    seconds_per_bin = bin_size_ms / 1000.0
    train_targets = safe_clip_rates(
        _flat(train_counts) / seconds_per_bin, scoring.min_rate_hz, scoring.max_rate_hz
    )
    decoder = fit_ridge_decoder(
        model.transform(train_features),
        train_targets,
        alpha=float(settings["heldout_decoder_alpha"]),
        fit_intercept=True,
    )
    predictions: dict[str, np.ndarray] = {}
    for name, mask in split_masks.items():
        features = (_flat(input_rates[mask]) - feature_stats["mean"]) / feature_stats["std"]
        latents = model.transform(features)
        flat_rates = safe_clip_rates(
            predict_ridge_decoder(latents, decoder), scoring.min_rate_hz, scoring.max_rate_hz
        )
        counts = spikes[mask][:, :, heldout_indices]
        predictions[name] = flat_rates.reshape(counts.shape)
    return predictions


def run_repeated_split_baselines(
    dataset: NeuralDataset,
    split_seeds: list[int],
    methods: list[str],
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    heldout_neuron_fraction: float,
    scoring: ScoringConfig,
    settings: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Recreate splits under many seeds and score CPU-only baselines with the canonical scorer."""
    resolved = dict(FACTOR_LATENT_DEFAULTS) | dict(settings or {})
    spikes = dataset.spikes
    trial_ids = dataset.trial_ids
    rows: list[dict[str, object]] = []
    for split_seed in split_seeds:
        split = create_trial_split(
            trial_ids, train_fraction, validation_fraction, test_fraction, seed=int(split_seed)
        )
        mask = create_neuron_mask(spikes.shape[2], heldout_neuron_fraction, seed=int(split_seed))
        heldin_indices = np.flatnonzero(mask.heldin)
        heldout_indices = np.flatnonzero(mask.heldout)
        train_counts = spikes[np.isin(trial_ids, split.train)][:, :, heldout_indices]
        factor_predictions: dict[str, np.ndarray] | None = None
        if "factor_latent" in methods:
            factor_predictions = factor_latent_heldout_rates(
                dataset, split, heldin_indices, heldout_indices, scoring, resolved
            )
        for method_name in methods:
            scored: dict[str, dict[str, object]] = {}
            for split_name in ("validation", "test"):
                counts = spikes[np.isin(trial_ids, getattr(split, split_name))][
                    :, :, heldout_indices
                ]
                reference = train_heldout_mean_rate_reference(train_counts, counts.shape, scoring)
                if method_name == "train_mean_rate":
                    predicted = reference
                elif method_name == "split_mean_rate":
                    # Fit on the evaluation split itself: an invalid diagnostic control.
                    predicted = train_heldout_mean_rate_reference(counts, counts.shape, scoring)
                elif method_name == "factor_latent":
                    assert factor_predictions is not None
                    predicted = factor_predictions[split_name]
                else:
                    msg = f"unknown repeated-split method: {method_name}"
                    raise ValueError(msg)
                scored[split_name] = score_heldout_prediction(
                    counts,
                    predicted,
                    reference,
                    scoring,
                    method_name,
                    split_name,
                    "repeated_split_audit",
                    method_name != "split_mean_rate",
                )
            notes = (
                "Diagnostic control fit on the evaluation split; invalid model."
                if method_name == "split_mean_rate"
                else "Train-only fit."
            )
            rows.append(
                {
                    "split_seed": int(split_seed),
                    "method_name": method_name,
                    "validation_unified_bits_per_spike": scored["validation"]["bits_per_spike"],
                    "test_unified_bits_per_spike": scored["test"]["bits_per_spike"],
                    "validation_poisson_nll": scored["validation"]["poisson_nll"],
                    "test_poisson_nll": scored["test"]["poisson_nll"],
                    "validation_spike_count": scored["validation"]["spike_count"],
                    "test_spike_count": scored["test"]["spike_count"],
                    "validation_heldout_rate_hz": scored["validation"]["observed_rate_hz"],
                    "test_heldout_rate_hz": scored["test"]["observed_rate_hz"],
                    "notes": notes,
                }
            )
    return pd.DataFrame(rows, columns=REPEATED_SPLIT_COLUMNS)
