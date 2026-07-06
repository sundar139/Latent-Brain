from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from rich.console import Console

from latentbrain.data.io import compute_dataset_hash, load_neural_dataset
from latentbrain.data.schemas import NeuralDataset, NeuronMask, TrialSplit
from latentbrain.data.splits import create_neuron_mask, create_trial_split
from latentbrain.data.validation import (
    validate_neural_dataset,
    validate_neuron_mask,
    validate_trial_split,
)
from latentbrain.eval.behavior import derive_velocity_targets, select_behavior_targets
from latentbrain.eval.decoding import (
    fit_ridge_decoder,
    predict_ridge_decoder,
    regression_metrics,
    standardize_train_apply,
)
from latentbrain.eval.reporting import write_behavior_decoder_outputs
from latentbrain.eval.smoothing import smooth_spike_counts, spike_counts_to_rates_hz
from latentbrain.paths import get_repo_root, resolve_configured_path

console = Console(markup=False)
SplitName = Literal["train", "validation", "test"]
NeuronGroupName = Literal["heldin", "heldout", "all"]


class DatasetSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    processed_path: str = Field(min_length=1)
    expected_hash: str = Field(min_length=1)
    bin_size_ms: int = Field(gt=0)


class SplitSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = Field(ge=0)
    train_fraction: float = Field(gt=0.0, lt=1.0)
    validation_fraction: float = Field(gt=0.0, lt=1.0)
    test_fraction: float = Field(gt=0.0, lt=1.0)
    heldout_neuron_fraction: float = Field(gt=0.0, lt=1.0)


class SmoothingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["gaussian"]
    sigma_ms: float = Field(gt=0.0)
    truncate: float = Field(gt=0.0)


class FeaturesSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    neuron_group: NeuronGroupName
    smoothing: SmoothingSection
    convert_to_hz: bool
    standardize_features: bool


class TargetsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_behavior_prefixes: list[str]
    derive_velocity: bool
    velocity_method: Literal["central_difference"]
    standardize_targets: bool

    @field_validator("source_behavior_prefixes")
    @classmethod
    def prefixes_must_be_nonempty(cls, values: list[str]) -> list[str]:
        if not values or any(not value.strip() for value in values):
            msg = "targets.source_behavior_prefixes must contain non-empty strings"
            raise ValueError(msg)
        return values


class DecoderSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["ridge"]
    alpha: float = Field(ge=0.0)
    fit_intercept: bool
    train_trials_only: bool


class EvaluationSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluate_splits: list[SplitName]
    primary_split: SplitName
    primary_target_prefix: str = Field(min_length=1)
    metrics: list[Literal["r2", "mse", "mae"]]

    @field_validator("evaluate_splits", "metrics")
    @classmethod
    def values_must_be_unique(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            msg = "evaluation lists must not contain duplicate values"
            raise ValueError(msg)
        return values

    @model_validator(mode="after")
    def primary_split_is_evaluated(self) -> EvaluationSection:
        if self.primary_split not in self.evaluate_splits:
            msg = "evaluation.primary_split must be listed in evaluation.evaluate_splits"
            raise ValueError(msg)
        return self


class ReportingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str = Field(min_length=1)


class BehaviorDecoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: DatasetSection
    splits: SplitSection
    features: FeaturesSection
    targets: TargetsSection
    decoder: DecoderSection
    evaluation: EvaluationSection
    reporting: ReportingSection

    @model_validator(mode="after")
    def split_fractions_sum_to_one(self) -> BehaviorDecoderConfig:
        total = (
            self.splits.train_fraction + self.splits.validation_fraction + self.splits.test_fraction
        )
        if abs(total - 1.0) > 1e-8:
            msg = "split fractions must sum to 1.0"
            raise ValueError(msg)
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> BehaviorDecoderConfig:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            msg = f"malformed behavior decoder config: {path}"
            raise ValueError(msg) from exc
        if not isinstance(raw, dict):
            msg = f"behavior decoder config must contain a top-level mapping: {path}"
            raise ValueError(msg)
        return cls.model_validate(raw)


def _parse_args(argv: Sequence[str] | None) -> Path:
    parser = argparse.ArgumentParser(description="Run the local MC_Maze Small behavior decoder.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mc_maze_small_behavior_decoder.yaml"),
    )
    args = parser.parse_args(argv)
    return args.config


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _trial_mask(dataset: NeuralDataset, trial_ids: np.ndarray) -> np.ndarray:
    return np.isin(dataset.trial_ids, trial_ids)


def _split_ids(split: TrialSplit, name: str) -> np.ndarray:
    if name == "train":
        return split.train
    if name == "validation":
        return split.validation
    if name == "test":
        return split.test
    msg = f"unknown split: {name}"
    raise ValueError(msg)


def _neuron_group_mask(neuron_mask: NeuronMask, group: str) -> np.ndarray:
    if group == "heldin":
        return np.asarray(neuron_mask.heldin, dtype=bool)
    if group == "heldout":
        return np.asarray(neuron_mask.heldout, dtype=bool)
    if group == "all":
        return np.asarray(neuron_mask.heldin | neuron_mask.heldout, dtype=bool)
    msg = f"unknown neuron group: {group}"
    raise ValueError(msg)


def _flatten_split(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    subset = values[mask]
    return subset.reshape(subset.shape[0] * subset.shape[1], subset.shape[2])


def _nan_stat(values: pd.Series, statistic: Literal["mean", "median"]) -> float:
    finite = values[np.isfinite(values)]
    if finite.empty:
        return float("nan")
    if statistic == "mean":
        return float(finite.mean())
    return float(finite.median())


def _split_summary_row(
    split_name: str,
    n_trials: int,
    features: np.ndarray,
    target_metrics: pd.DataFrame,
) -> dict[str, float | int | str]:
    return {
        "split": split_name,
        "n_trials": n_trials,
        "n_samples": int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "n_targets": int(len(target_metrics)),
        "mean_r2": _nan_stat(target_metrics["r2"], "mean"),
        "median_r2": _nan_stat(target_metrics["r2"], "median"),
        "mean_mse": float(target_metrics["mse"].mean()),
        "mean_mae": float(target_metrics["mae"].mean()),
    }


def _coefficient_table(coefficients: np.ndarray, target_names: list[str]) -> pd.DataFrame:
    rows = []
    for feature_index in range(coefficients.shape[0]):
        for target_index, target_name in enumerate(target_names):
            rows.append(
                {
                    "feature_index": feature_index,
                    "target_name": target_name,
                    "coefficient": float(coefficients[feature_index, target_index]),
                }
            )
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = get_repo_root()
    config_arg = _parse_args(argv)
    config_path = config_arg if config_arg.is_absolute() else repo_root / config_arg
    config = BehaviorDecoderConfig.from_yaml(config_path)
    processed_path = resolve_configured_path(config.dataset.processed_path, repo_root)
    output_dir = resolve_configured_path(config.reporting.output_dir, repo_root)

    if not processed_path.exists():
        console.print(f"Processed dataset is missing: {_relative(processed_path, repo_root)}")
        console.print(
            "Run: python scripts/prepare_nlb_data.py --config configs/nlb_mc_maze_small.yaml"
        )
        return 2

    dataset = load_neural_dataset(processed_path)
    validate_neural_dataset(dataset)
    if dataset.behavior is None or dataset.behavior_names is None:
        console.print("Processed dataset does not contain behavior targets")
        return 2
    dataset_hash = compute_dataset_hash(dataset)
    if dataset_hash != config.dataset.expected_hash:
        console.print(
            f"Dataset hash mismatch: expected {config.dataset.expected_hash}, got {dataset_hash}"
        )
        return 2

    split = create_trial_split(
        dataset.trial_ids,
        config.splits.train_fraction,
        config.splits.validation_fraction,
        config.splits.test_fraction,
        seed=config.splits.seed,
    )
    neuron_mask = create_neuron_mask(
        dataset.spikes.shape[2], config.splits.heldout_neuron_fraction, seed=config.splits.seed
    )
    validate_trial_split(split, dataset.trial_ids)
    validate_neuron_mask(neuron_mask, dataset.spikes.shape[2])

    feature_mask = _neuron_group_mask(neuron_mask, config.features.neuron_group)
    smoothed = smooth_spike_counts(
        dataset.spikes[:, :, feature_mask],
        dataset.bin_size_ms,
        method=config.features.smoothing.method,
        sigma_ms=config.features.smoothing.sigma_ms,
        truncate=config.features.smoothing.truncate,
    )
    features_3d = (
        spike_counts_to_rates_hz(smoothed, dataset.bin_size_ms)
        if config.features.convert_to_hz
        else smoothed
    )
    positions, position_names = select_behavior_targets(
        dataset.behavior, dataset.behavior_names, config.targets.source_behavior_prefixes
    )
    targets_3d, target_names = derive_velocity_targets(
        positions, position_names, dataset.bin_size_ms, method=config.targets.velocity_method
    )

    train_mask = _trial_mask(dataset, split.train)
    train_features_raw = _flatten_split(features_3d, train_mask)
    all_features_raw = features_3d.reshape(-1, features_3d.shape[2])
    if config.features.standardize_features:
        all_features, feature_stats = standardize_train_apply(train_features_raw, all_features_raw)
    else:
        all_features = all_features_raw
        feature_stats = {}
    features_3d_fit = all_features.reshape(features_3d.shape)

    train_targets_raw = _flatten_split(targets_3d, train_mask)
    all_targets_raw = targets_3d.reshape(-1, targets_3d.shape[2])
    if config.targets.standardize_targets:
        all_targets_fit, target_stats = standardize_train_apply(train_targets_raw, all_targets_raw)
    else:
        all_targets_fit = all_targets_raw
        target_stats = {}
    targets_3d_fit = all_targets_fit.reshape(targets_3d.shape)

    model = fit_ridge_decoder(
        _flatten_split(features_3d_fit, train_mask),
        _flatten_split(targets_3d_fit, train_mask),
        alpha=config.decoder.alpha,
        fit_intercept=config.decoder.fit_intercept,
    )

    split_rows: list[dict[str, float | int | str]] = []
    target_frames = []
    for split_name in config.evaluation.evaluate_splits:
        ids = _split_ids(split, split_name)
        mask = _trial_mask(dataset, ids)
        split_features = _flatten_split(features_3d_fit, mask)
        split_targets = _flatten_split(targets_3d, mask)
        pred_fit = predict_ridge_decoder(split_features, model)
        if config.targets.standardize_targets:
            pred = pred_fit * target_stats["std"] + target_stats["mean"]
        else:
            pred = pred_fit
        metrics = regression_metrics(split_targets, pred, target_names)
        metrics.insert(0, "split", split_name)
        target_frames.append(metrics)
        split_rows.append(_split_summary_row(split_name, len(ids), split_features, metrics))

    split_metrics = pd.DataFrame(split_rows)
    target_metrics = pd.concat(target_frames, ignore_index=True)
    coefficient_table = _coefficient_table(model["coefficients"], target_names)
    primary_row = split_metrics[split_metrics["split"] == config.evaluation.primary_split]
    primary_target_rows = target_metrics[
        (target_metrics["split"] == config.evaluation.primary_split)
        & target_metrics["target_name"].str.startswith(config.evaluation.primary_target_prefix)
    ]
    primary_mean_r2 = float(primary_row["mean_r2"].iloc[0])
    primary_target_r2 = (
        _nan_stat(primary_target_rows["r2"], "mean") if len(primary_target_rows) else None
    )
    metrics_summary = {
        "dataset_name": config.dataset.name,
        "dataset_hash": dataset_hash,
        "processed_path": _relative(processed_path, repo_root),
        "shape": [int(value) for value in dataset.spikes.shape],
        "behavior_shape": [int(value) for value in dataset.behavior.shape],
        "feature_neuron_group": config.features.neuron_group,
        "n_features": int(features_3d.shape[2]),
        "smoothing": config.features.smoothing.model_dump(mode="json"),
        "target_names": target_names,
        "decoder_name": config.decoder.name,
        "decoder_alpha": float(config.decoder.alpha),
        "fit_policy": "train trials only",
        "standardization_policy": "train-only statistics",
        "feature_standardization": dict(feature_stats),
        "target_standardization": dict(target_stats),
        "intercept": model["intercept"].tolist(),
        "primary_split": config.evaluation.primary_split,
        "primary_mean_r2": primary_mean_r2,
        "primary_target_prefix": config.evaluation.primary_target_prefix,
        "primary_target_r2": primary_target_r2,
        "no_neural_network_model_trained": True,
        "official_benchmark_claim": False,
    }
    write_behavior_decoder_outputs(
        output_dir,
        metrics_summary,
        split_metrics,
        target_metrics,
        coefficient_table,
    )

    console.print(f"dataset: {config.dataset.name}")
    console.print(f"shape: {list(dataset.spikes.shape)}")
    console.print(f"behavior_targets: {target_names}")
    console.print(f"feature_neuron_group: {config.features.neuron_group}")
    console.print(f"smoothing_sigma_ms: {config.features.smoothing.sigma_ms}")
    console.print(f"decoder_alpha: {config.decoder.alpha}")
    console.print(f"primary_split: {config.evaluation.primary_split}")
    console.print(f"primary_mean_r2: {primary_mean_r2}")
    console.print(f"primary_hand_velocity_r2: {primary_target_r2}")
    console.print(f"output_dir: {_relative(output_dir, repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
