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
from latentbrain.eval.cosmoothing import run_cosmoothing_sweep
from latentbrain.eval.reporting import write_cosmoothing_sweep_outputs
from latentbrain.eval.sweeps import expand_grid
from latentbrain.paths import get_repo_root, resolve_configured_path

console = Console(markup=False)
SplitName = Literal["train", "validation", "test"]


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

    smoothing_sigma_ms: list[float]
    ridge_alpha: list[float]
    standardize_features: list[bool]
    fit_intercept: list[bool]

    @field_validator("smoothing_sigma_ms")
    @classmethod
    def sigma_values_are_positive(cls, values: list[float]) -> list[float]:
        if not values or any(value <= 0.0 for value in values):
            msg = "sweep.smoothing_sigma_ms values must be positive"
            raise ValueError(msg)
        return values

    @field_validator("ridge_alpha")
    @classmethod
    def alpha_values_are_non_negative(cls, values: list[float]) -> list[float]:
        if not values or any(value < 0.0 for value in values):
            msg = "sweep.ridge_alpha values must be non-negative"
            raise ValueError(msg)
        return values

    @field_validator("standardize_features", "fit_intercept")
    @classmethod
    def bool_sweep_values_are_non_empty(cls, values: list[bool]) -> list[bool]:
        if not values:
            msg = "boolean sweep values must be non-empty"
            raise ValueError(msg)
        return values


class TargetsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fit_target_type: Literal["rate_hz"]
    min_rate_hz: float = Field(gt=0.0)
    max_rate_hz: float = Field(gt=0.0)

    @model_validator(mode="after")
    def max_rate_exceeds_min_rate(self) -> TargetsSection:
        if self.max_rate_hz <= self.min_rate_hz:
            msg = "targets.max_rate_hz must exceed targets.min_rate_hz"
            raise ValueError(msg)
        return self


class ReferenceSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["train_mean_rate"]
    fit_train_trials_only: bool


class EvaluationSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_split: SplitName
    primary_metric: Literal["bits_per_spike", "poisson_nll", "mse_rate_hz", "mae_rate_hz"]
    evaluate_splits: list[SplitName]

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


class CosmoothingSweepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: DatasetSection
    splits: SplitSection
    features: FeaturesSection
    sweep: SweepSection
    targets: TargetsSection
    reference: ReferenceSection
    evaluation: EvaluationSection
    reporting: ReportingSection

    @model_validator(mode="after")
    def split_fractions_sum_to_one(self) -> CosmoothingSweepConfig:
        total = (
            self.splits.train_fraction + self.splits.validation_fraction + self.splits.test_fraction
        )
        if abs(total - 1.0) > 1e-8:
            msg = "split fractions must sum to 1.0"
            raise ValueError(msg)
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> CosmoothingSweepConfig:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            msg = f"malformed co-smoothing sweep config: {path}"
            raise ValueError(msg) from exc
        if not isinstance(raw, dict):
            msg = f"co-smoothing sweep config must contain a top-level mapping: {path}"
            raise ValueError(msg)
        return cls.model_validate(raw)


def _parse_args(argv: Sequence[str] | None) -> Path:
    parser = argparse.ArgumentParser(
        description="Run the local MC_Maze Small co-smoothing diagnostic sweep."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mc_maze_small_cosmoothing_sweep.yaml"),
    )
    args = parser.parse_args(argv)
    return args.config


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


def _all_validation_bits_negative(sweep_results: pd.DataFrame) -> bool:
    validation = sweep_results[sweep_results["split"] == "validation"]
    return bool(not validation.empty and (validation["bits_per_spike"] < 0.0).all())


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = get_repo_root()
    config_arg = _parse_args(argv)
    config_path = config_arg if config_arg.is_absolute() else repo_root / config_arg
    config = CosmoothingSweepConfig.from_yaml(config_path)
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

    sweep_config = config.model_dump(mode="python")
    sweep_results, best_config, best_split_metrics, best_neuron_metrics = run_cosmoothing_sweep(
        dataset,
        split,
        neuron_mask,
        sweep_config,
    )
    if sweep_results.empty:
        console.print("No valid sweep results were produced")
        return 2

    primary = _best_primary_row(best_split_metrics, config.evaluation.primary_split)
    all_validation_negative = _all_validation_bits_negative(sweep_results)
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
        "best_config": best_config,
        "best_validation_bits_per_spike": float(primary["bits_per_spike"]),
        "best_validation_poisson_nll": float(primary["poisson_nll"]),
        "all_validation_bits_per_spike_negative": all_validation_negative,
        "no_neural_network_model_trained": True,
        "official_benchmark_claim": False,
    }
    write_cosmoothing_sweep_outputs(
        output_dir,
        summary,
        sweep_results,
        best_config,
        best_split_metrics,
        best_neuron_metrics,
    )

    console.print(f"dataset: {config.dataset.name}")
    console.print(f"shape: {list(dataset.spikes.shape)}")
    console.print(f"n_configurations: {n_configurations}")
    console.print(f"primary_split: {config.evaluation.primary_split}")
    console.print(f"best_validation_bits_per_spike: {float(primary['bits_per_spike'])}")
    console.print(f"best_validation_poisson_nll: {float(primary['poisson_nll'])}")
    console.print(f"best_smoothing_sigma_ms: {best_config['smoothing_sigma_ms']}")
    console.print(f"best_ridge_alpha: {best_config['ridge_alpha']}")
    console.print(f"best_standardize_features: {best_config['standardize_features']}")
    console.print(f"best_fit_intercept: {best_config['fit_intercept']}")
    console.print(f"output_dir: {_relative(output_dir, repo_root)}")
    if all_validation_negative:
        console.print("warning: all validation bits/spike values were negative")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
