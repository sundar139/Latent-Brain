from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rich.console import Console

from latentbrain.data.io import compute_dataset_hash, load_neural_dataset
from latentbrain.data.splits import create_neuron_mask, create_trial_split
from latentbrain.data.validation import (
    validate_neural_dataset,
    validate_neuron_mask,
    validate_trial_split,
)
from latentbrain.eval.cosmoothing import run_cosmoothing_baseline
from latentbrain.eval.reporting import write_cosmoothing_outputs
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


class SmoothingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["gaussian"]
    sigma_ms: float = Field(gt=0.0)
    truncate: float = Field(gt=0.0)


class FeaturesSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_neuron_group: Literal["heldin"]
    target_neuron_group: Literal["heldout"]
    smoothing: SmoothingSection
    convert_to_hz: bool
    standardize_features: bool


class TargetsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_transform: Literal["counts"]
    fit_target_type: Literal["rate_hz"]
    min_rate_hz: float = Field(gt=0.0)
    max_rate_hz: float = Field(gt=0.0)

    @model_validator(mode="after")
    def max_rate_exceeds_min_rate(self) -> TargetsSection:
        if self.max_rate_hz <= self.min_rate_hz:
            msg = "targets.max_rate_hz must exceed targets.min_rate_hz"
            raise ValueError(msg)
        return self


class DecoderSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["ridge"]
    alpha: float = Field(ge=0.0)
    fit_intercept: bool
    train_trials_only: bool


class ReferenceSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["train_mean_rate"]
    fit_train_trials_only: bool


class EvaluationSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluate_splits: list[SplitName]
    primary_split: SplitName
    metrics: list[
        Literal[
            "poisson_nll",
            "poisson_log_likelihood",
            "bits_per_spike",
            "mse_rate_hz",
            "mae_rate_hz",
        ]
    ]

    @model_validator(mode="after")
    def primary_split_is_evaluated(self) -> EvaluationSection:
        if self.primary_split not in self.evaluate_splits:
            msg = "evaluation.primary_split must be listed in evaluation.evaluate_splits"
            raise ValueError(msg)
        if len(set(self.evaluate_splits)) != len(self.evaluate_splits):
            msg = "evaluation.evaluate_splits must not contain duplicates"
            raise ValueError(msg)
        if len(set(self.metrics)) != len(self.metrics):
            msg = "evaluation.metrics must not contain duplicates"
            raise ValueError(msg)
        return self


class ReportingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str = Field(min_length=1)


class CosmoothingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: DatasetSection
    splits: SplitSection
    features: FeaturesSection
    targets: TargetsSection
    decoder: DecoderSection
    reference: ReferenceSection
    evaluation: EvaluationSection
    reporting: ReportingSection

    @model_validator(mode="after")
    def split_fractions_sum_to_one(self) -> CosmoothingConfig:
        total = (
            self.splits.train_fraction + self.splits.validation_fraction + self.splits.test_fraction
        )
        if abs(total - 1.0) > 1e-8:
            msg = "split fractions must sum to 1.0"
            raise ValueError(msg)
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> CosmoothingConfig:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            msg = f"malformed co-smoothing config: {path}"
            raise ValueError(msg) from exc
        if not isinstance(raw, dict):
            msg = f"co-smoothing config must contain a top-level mapping: {path}"
            raise ValueError(msg)
        return cls.model_validate(raw)


def _parse_args(argv: Sequence[str] | None) -> Path:
    parser = argparse.ArgumentParser(
        description="Run the local MC_Maze Small co-smoothing baseline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mc_maze_small_cosmoothing_ridge.yaml"),
    )
    args = parser.parse_args(argv)
    return args.config


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _coefficient_table(metadata: dict[str, object]) -> pd.DataFrame:
    coefficients = np.asarray(metadata["coefficients"], dtype=np.float64)
    input_indices = np.asarray(metadata["input_neuron_indices"], dtype=np.int64)
    target_indices = np.asarray(metadata["target_neuron_indices"], dtype=np.int64)
    rows = []
    for input_rank, input_index in enumerate(input_indices):
        for target_rank, target_index in enumerate(target_indices):
            rows.append(
                {
                    "input_neuron_index": int(input_index),
                    "target_neuron_index": int(target_index),
                    "coefficient": float(coefficients[input_rank, target_rank]),
                }
            )
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = get_repo_root()
    config_arg = _parse_args(argv)
    config_path = config_arg if config_arg.is_absolute() else repo_root / config_arg
    config = CosmoothingConfig.from_yaml(config_path)
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

    split_metrics, neuron_metrics, metadata = run_cosmoothing_baseline(
        dataset,
        split,
        neuron_mask,
        config.model_dump(mode="python"),
    )
    primary_row = split_metrics[split_metrics["split"] == config.evaluation.primary_split]
    if len(primary_row) != 1:
        console.print("Primary metric row was not produced by co-smoothing evaluation")
        return 2
    primary = primary_row.iloc[0]
    metrics_summary = {
        "dataset_name": config.dataset.name,
        "dataset_hash": dataset_hash,
        "processed_path": _relative(processed_path, repo_root),
        "shape": [int(value) for value in dataset.spikes.shape],
        "input_neuron_group": config.features.input_neuron_group,
        "target_neuron_group": config.features.target_neuron_group,
        "input_neuron_count": int(len(metadata["input_neuron_indices"])),
        "target_neuron_count": int(len(metadata["target_neuron_indices"])),
        "smoothing": config.features.smoothing.model_dump(mode="json"),
        "decoder_name": config.decoder.name,
        "decoder_alpha": float(config.decoder.alpha),
        "fit_policy": "train trials only",
        "standardization_policy": "train-only held-in features",
        "reference_policy": "train-only held-out mean rates",
        "intercept": np.asarray(metadata["intercept"], dtype=np.float64).tolist(),
        "reference_rates_hz": np.asarray(metadata["reference_rates_hz"], dtype=np.float64).tolist(),
        "primary_split": config.evaluation.primary_split,
        "primary_bits_per_spike": float(primary["bits_per_spike"]),
        "primary_poisson_nll": float(primary["poisson_nll"]),
        "no_neural_network_model_trained": True,
        "official_benchmark_claim": False,
    }
    write_cosmoothing_outputs(
        output_dir,
        metrics_summary,
        split_metrics,
        neuron_metrics,
        _coefficient_table(metadata),
    )

    console.print(f"dataset: {config.dataset.name}")
    console.print(f"shape: {list(dataset.spikes.shape)}")
    console.print(f"input_neuron_count: {metrics_summary['input_neuron_count']}")
    console.print(f"target_neuron_count: {metrics_summary['target_neuron_count']}")
    console.print(f"smoothing_sigma_ms: {config.features.smoothing.sigma_ms}")
    console.print(f"decoder_alpha: {config.decoder.alpha}")
    console.print(f"primary_split: {config.evaluation.primary_split}")
    console.print(f"primary_bits_per_spike: {float(primary['bits_per_spike'])}")
    console.print(f"primary_poisson_nll: {float(primary['poisson_nll'])}")
    console.print(f"output_dir: {_relative(output_dir, repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
