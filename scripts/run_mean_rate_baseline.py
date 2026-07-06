from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

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
from latentbrain.eval.baselines import evaluate_mean_rate_baseline
from latentbrain.eval.reporting import write_baseline_outputs
from latentbrain.paths import get_repo_root, resolve_configured_path

console = Console(markup=False)
SplitName = Literal["train", "validation", "test"]
NeuronGroupName = Literal["heldin", "heldout", "all"]


class DatasetSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    processed_path: str = Field(min_length=1)
    expected_hash: str | None = None
    bin_size_ms: int = Field(gt=0)


class SplitSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = Field(ge=0)
    train_fraction: float = Field(gt=0.0, lt=1.0)
    validation_fraction: float = Field(gt=0.0, lt=1.0)
    test_fraction: float = Field(gt=0.0, lt=1.0)
    heldout_neuron_fraction: float = Field(gt=0.0, lt=1.0)


class BaselineSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["mean_rate"]
    min_rate_hz: float = Field(gt=0.0)
    max_rate_hz: float = Field(gt=0.0)
    use_train_trials_only: bool
    predict_constant_rate_per_neuron: bool

    @model_validator(mode="after")
    def max_rate_exceeds_min_rate(self) -> BaselineSection:
        if self.max_rate_hz <= self.min_rate_hz:
            msg = "baseline.max_rate_hz must exceed baseline.min_rate_hz"
            raise ValueError(msg)
        return self


class EvaluationSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluate_splits: list[SplitName]
    evaluate_neuron_groups: list[NeuronGroupName]
    primary_split: SplitName
    primary_neuron_group: NeuronGroupName

    @field_validator("evaluate_splits", "evaluate_neuron_groups")
    @classmethod
    def values_must_be_unique(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            msg = "evaluation lists must not contain duplicate values"
            raise ValueError(msg)
        return values

    @model_validator(mode="after")
    def primary_values_are_evaluated(self) -> EvaluationSection:
        if self.primary_split not in self.evaluate_splits:
            msg = "evaluation.primary_split must be listed in evaluation.evaluate_splits"
            raise ValueError(msg)
        if self.primary_neuron_group not in self.evaluate_neuron_groups:
            msg = (
                "evaluation.primary_neuron_group must be listed in "
                "evaluation.evaluate_neuron_groups"
            )
            raise ValueError(msg)
        return self


class ReportingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str = Field(min_length=1)


class MeanRateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: DatasetSection
    splits: SplitSection
    baseline: BaselineSection
    evaluation: EvaluationSection
    reporting: ReportingSection

    @model_validator(mode="after")
    def split_fractions_sum_to_one(self) -> MeanRateConfig:
        total = (
            self.splits.train_fraction + self.splits.validation_fraction + self.splits.test_fraction
        )
        if abs(total - 1.0) > 1e-8:
            msg = "split fractions must sum to 1.0"
            raise ValueError(msg)
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> MeanRateConfig:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            msg = f"malformed mean-rate config: {path}"
            raise ValueError(msg) from exc
        if not isinstance(raw, dict):
            msg = f"mean-rate config must contain a top-level mapping: {path}"
            raise ValueError(msg)
        return cls.model_validate(raw)


def _parse_args(argv: Sequence[str] | None) -> Path:
    parser = argparse.ArgumentParser(description="Run the local MC_Maze Small mean-rate baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mc_maze_small_mean_rate.yaml"),
    )
    args = parser.parse_args(argv)
    return args.config


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = get_repo_root()
    config_arg = _parse_args(argv)
    config_path = config_arg if config_arg.is_absolute() else repo_root / config_arg
    config = MeanRateConfig.from_yaml(config_path)
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
    if config.dataset.expected_hash and dataset_hash != config.dataset.expected_hash:
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

    split_metrics, neuron_metrics, metadata = evaluate_mean_rate_baseline(
        dataset,
        split,
        neuron_mask,
        config.model_dump(mode="python"),
    )
    primary_row = split_metrics[
        (split_metrics["split"] == config.evaluation.primary_split)
        & (split_metrics["neuron_group"] == config.evaluation.primary_neuron_group)
    ]
    if len(primary_row) != 1:
        console.print("Primary metric row was not produced by baseline evaluation")
        return 2
    primary = primary_row.iloc[0]
    metrics_summary = {
        "dataset_name": config.dataset.name,
        "dataset_hash": dataset_hash,
        "processed_path": _relative(processed_path, repo_root),
        "baseline_name": config.baseline.name,
        "shape": [int(value) for value in dataset.spikes.shape],
        "bin_size_ms": int(dataset.bin_size_ms),
        "split_counts": {
            "train": int(len(split.train)),
            "validation": int(len(split.validation)),
            "test": int(len(split.test)),
        },
        "neuron_mask_counts": {
            "heldin": int(neuron_mask.heldin.sum()),
            "heldout": int(neuron_mask.heldout.sum()),
        },
        "primary_split": config.evaluation.primary_split,
        "primary_neuron_group": config.evaluation.primary_neuron_group,
        "primary_bits_per_spike": float(primary["bits_per_spike"]),
        "primary_poisson_nll": float(primary["poisson_nll"]),
        "train_only_fit": bool(metadata["train_only_fit"]),
        "baseline_metadata": metadata,
    }
    write_baseline_outputs(output_dir, metrics_summary, split_metrics, neuron_metrics)

    console.print(f"dataset: {config.dataset.name}")
    console.print(f"baseline: {config.baseline.name}")
    console.print(f"shape: {list(dataset.spikes.shape)}")
    console.print(f"primary_split: {config.evaluation.primary_split}")
    console.print(f"primary_neuron_group: {config.evaluation.primary_neuron_group}")
    console.print(f"primary_bits_per_spike: {float(primary['bits_per_spike'])}")
    console.print(f"primary_poisson_nll: {float(primary['poisson_nll'])}")
    console.print(f"output_dir: {_relative(output_dir, repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
