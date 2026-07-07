from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pandas as pd  # type: ignore[import-untyped]
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from rich.console import Console

from latentbrain.data.io import compute_dataset_hash, load_neural_dataset
from latentbrain.data.splits import create_neuron_mask, create_trial_split
from latentbrain.data.validation import (
    validate_neural_dataset,
    validate_neuron_mask,
    validate_trial_split,
)
from latentbrain.eval.latent_sweep import run_factor_latent_sweep
from latentbrain.eval.reporting import write_factor_latent_sweep_outputs
from latentbrain.eval.sweeps import expand_grid
from latentbrain.paths import get_repo_root, resolve_configured_path

console = Console(markup=False)
SplitName = Literal["train", "validation", "test"]
DEFAULT_MEAN_RATE_BITS = 0.5465273967210786
DEFAULT_FACTOR_BITS = 0.04747691544524409


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


class FeaturesSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_neuron_group: Literal["heldin"]
    target_neuron_group: Literal["heldout"]
    convert_to_hz: bool


class SweepSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latent_dim: list[int]
    smoothing_sigma_ms: list[float]
    heldout_decoder_alpha: list[float]
    standardize_features: list[bool]

    @field_validator("latent_dim")
    @classmethod
    def latent_dims_are_positive(cls, values: list[int]) -> list[int]:
        if not values or any(value <= 0 for value in values):
            msg = "sweep.latent_dim values must be positive"
            raise ValueError(msg)
        return values

    @field_validator("smoothing_sigma_ms")
    @classmethod
    def sigma_values_are_positive(cls, values: list[float]) -> list[float]:
        if not values or any(value <= 0.0 for value in values):
            msg = "sweep.smoothing_sigma_ms values must be positive"
            raise ValueError(msg)
        return values

    @field_validator("heldout_decoder_alpha")
    @classmethod
    def alpha_values_are_non_negative(cls, values: list[float]) -> list[float]:
        if not values or any(value < 0.0 for value in values):
            msg = "sweep.heldout_decoder_alpha values must be non-negative"
            raise ValueError(msg)
        return values

    @field_validator("standardize_features")
    @classmethod
    def bool_sweep_values_are_non_empty(cls, values: list[bool]) -> list[bool]:
        if not values:
            msg = "sweep.standardize_features must be non-empty"
            raise ValueError(msg)
        return values


class LatentModelSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["factor_analysis"]
    random_state: int = Field(ge=0)
    max_iter: int = Field(gt=0)
    tol: float = Field(gt=0.0)
    train_trials_only: bool


class HeldoutDecoderSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["ridge"]
    fit_intercept: bool
    min_rate_hz: float = Field(gt=0.0)
    max_rate_hz: float = Field(gt=0.0)
    train_trials_only: bool

    @model_validator(mode="after")
    def max_rate_exceeds_min_rate(self) -> HeldoutDecoderSection:
        if self.max_rate_hz <= self.min_rate_hz:
            msg = "heldout_decoder.max_rate_hz must exceed heldout_decoder.min_rate_hz"
            raise ValueError(msg)
        return self


class BehaviorDecoderSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    alpha: float = Field(ge=0.0)
    fit_intercept: bool
    target_prefixes: list[str]
    derive_velocity: bool
    velocity_method: Literal["central_difference"]
    standardize_targets: bool
    train_trials_only: bool

    @field_validator("target_prefixes")
    @classmethod
    def target_prefixes_are_nonempty(cls, values: list[str]) -> list[str]:
        if not values or any(not value.strip() for value in values):
            msg = "behavior_decoder.target_prefixes must contain non-empty strings"
            raise ValueError(msg)
        return values


class ReferenceSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["train_mean_rate"]
    fit_train_trials_only: bool


class EvaluationSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluate_splits: list[SplitName]
    primary_split: SplitName
    primary_metric: Literal["bits_per_spike"]
    secondary_metric: Literal["behavior_mean_r2"]

    @model_validator(mode="after")
    def primary_split_is_evaluated(self) -> EvaluationSection:
        if self.primary_split not in self.evaluate_splits:
            msg = "evaluation.primary_split must be listed in evaluation.evaluate_splits"
            raise ValueError(msg)
        if len(set(self.evaluate_splits)) != len(self.evaluate_splits):
            msg = "evaluation.evaluate_splits must not contain duplicates"
            raise ValueError(msg)
        return self


class ReportingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str = Field(min_length=1)


class FactorLatentSweepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: DatasetSection
    splits: SplitSection
    features: FeaturesSection
    sweep: SweepSection
    latent_model: LatentModelSection
    heldout_decoder: HeldoutDecoderSection
    behavior_decoder: BehaviorDecoderSection
    reference: ReferenceSection
    evaluation: EvaluationSection
    reporting: ReportingSection

    @model_validator(mode="after")
    def split_fractions_sum_to_one(self) -> FactorLatentSweepConfig:
        total = (
            self.splits.train_fraction + self.splits.validation_fraction + self.splits.test_fraction
        )
        if abs(total - 1.0) > 1e-8:
            msg = "split fractions must sum to 1.0"
            raise ValueError(msg)
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> FactorLatentSweepConfig:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            msg = f"malformed factor latent sweep config: {path}"
            raise ValueError(msg) from exc
        if not isinstance(raw, dict):
            msg = f"factor latent sweep config must contain a top-level mapping: {path}"
            raise ValueError(msg)
        return cls.model_validate(raw)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local MC_Maze Small factor latent sweep.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mc_maze_small_factor_latent_sweep.yaml"),
    )
    parser.add_argument("--mean-rate-bits", type=float, default=DEFAULT_MEAN_RATE_BITS)
    return parser.parse_args(argv)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _best_primary_row(best_split_metrics: pd.DataFrame, primary_split: str) -> pd.Series:  # type: ignore[type-arg]
    rows = best_split_metrics[best_split_metrics["split"] == primary_split]
    if len(rows) != 1:
        msg = "best configuration did not produce exactly one primary split row"
        raise ValueError(msg)
    return rows.iloc[0]


def _primary_behavior_mean(best_behavior_metrics: pd.DataFrame, primary_split: str) -> float | None:
    if best_behavior_metrics.empty:
        return None
    values = best_behavior_metrics.loc[best_behavior_metrics["split"] == primary_split, "r2"]
    return None if values.empty else float(values.mean())


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = get_repo_root()
    args = _parse_args(argv)
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    config = FactorLatentSweepConfig.from_yaml(config_path)
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
        dataset.spikes.shape[2],
        config.splits.heldout_neuron_fraction,
        seed=config.splits.seed,
    )
    validate_trial_split(split, dataset.trial_ids)
    validate_neuron_mask(neuron_mask, dataset.spikes.shape[2])

    sweep_results, best_config, best_split, best_neuron, best_behavior, best_latent = (
        run_factor_latent_sweep(dataset, split, neuron_mask, config.model_dump(mode="python"))
    )
    if sweep_results.empty:
        console.print("No valid sweep results were produced")
        return 2

    primary = _best_primary_row(best_split, config.evaluation.primary_split)
    behavior_mean_r2 = _primary_behavior_mean(best_behavior, config.evaluation.primary_split)
    n_configurations = len(expand_grid(config.sweep.model_dump(mode="python")))
    summary = {
        "dataset_name": config.dataset.name,
        "dataset_hash": dataset_hash,
        "processed_path": _relative(processed_path, repo_root),
        "shape": [int(value) for value in dataset.spikes.shape],
        "sweep_grid": config.sweep.model_dump(mode="json"),
        "n_configurations": n_configurations,
        "primary_split": config.evaluation.primary_split,
        "primary_metric": config.evaluation.primary_metric,
        "secondary_metric": config.evaluation.secondary_metric,
        "best_config": best_config,
        "best_validation_bits_per_spike": float(primary["bits_per_spike"]),
        "best_validation_poisson_nll": float(primary["poisson_nll"]),
        "best_validation_behavior_mean_r2": behavior_mean_r2,
        "single_factor_latent_validation_bits_per_spike": DEFAULT_FACTOR_BITS,
        "mean_rate_validation_heldout_bits_per_spike": float(args.mean_rate_bits),
        "no_temporal_gp_prior_implemented": True,
        "no_neural_network_model_trained": True,
        "official_benchmark_claim": False,
    }
    write_factor_latent_sweep_outputs(
        output_dir,
        summary,
        sweep_results,
        best_config,
        best_split,
        best_neuron,
        best_behavior,
        best_latent,
    )

    console.print(f"dataset: {config.dataset.name}")
    console.print(f"shape: {list(dataset.spikes.shape)}")
    console.print(f"n_configurations: {n_configurations}")
    console.print(f"primary_split: {config.evaluation.primary_split}")
    console.print(f"best_validation_bits_per_spike: {float(primary['bits_per_spike'])}")
    console.print(f"best_validation_poisson_nll: {float(primary['poisson_nll'])}")
    console.print(f"best_validation_behavior_mean_r2: {behavior_mean_r2}")
    console.print(f"best_latent_dim: {best_config['latent_dim']}")
    console.print(f"best_smoothing_sigma_ms: {best_config['smoothing_sigma_ms']}")
    console.print(f"best_heldout_decoder_alpha: {best_config['heldout_decoder_alpha']}")
    console.print(f"best_standardize_features: {best_config['standardize_features']}")
    console.print(f"output_dir: {_relative(output_dir, repo_root)}")
    if float(primary["bits_per_spike"]) < float(args.mean_rate_bits):
        console.print("warning: best validation bits/spike is below mean-rate reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
