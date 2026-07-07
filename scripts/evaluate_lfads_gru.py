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
from latentbrain.data.splits import create_neuron_mask, create_trial_split
from latentbrain.data.validation import (
    validate_neural_dataset,
    validate_neuron_mask,
    validate_trial_split,
)
from latentbrain.eval.lfads_eval import run_lfads_gru_evaluation
from latentbrain.eval.reporting import write_lfads_gru_evaluation_outputs
from latentbrain.paths import get_repo_root, resolve_configured_path
from latentbrain.torch.device import resolve_device

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


class DataSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_neuron_group: Literal["heldin"]
    target_neuron_group: Literal["heldout"]
    max_time_bins: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    num_workers: int = Field(ge=0)
    drop_last: bool


class ModelSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["lfads_gru"]
    checkpoint_path: str = Field(min_length=1)
    output_dim: int | Literal["all", "heldin"] | None = None
    encoder_hidden_dim: int = Field(gt=0)
    generator_hidden_dim: int = Field(gt=0)
    latent_dim: int = Field(gt=0)
    factor_dim: int = Field(gt=0)
    dropout: float = Field(ge=0.0, lt=1.0)
    min_rate_hz: float = Field(gt=0.0)
    max_rate_hz: float = Field(gt=0.0)

    @model_validator(mode="after")
    def max_rate_exceeds_min_rate(self) -> ModelSection:
        if self.max_rate_hz <= self.min_rate_hz:
            msg = "model.max_rate_hz must exceed model.min_rate_hz"
            raise ValueError(msg)
        return self


class EvaluationModeSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_direct_model_rates_for_heldout: bool = False
    also_evaluate_factor_decoder: bool = True


class HeldoutDecoderSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["ridge"]
    alpha: float = Field(ge=0.0)
    fit_intercept: bool
    min_rate_hz: float = Field(gt=0.0)
    max_rate_hz: float = Field(gt=0.0)
    standardize_factors: bool
    train_trials_only: bool

    @model_validator(mode="after")
    def max_rate_exceeds_min_rate(self) -> HeldoutDecoderSection:
        if self.max_rate_hz <= self.min_rate_hz:
            msg = "heldout_decoder.max_rate_hz must exceed heldout_decoder.min_rate_hz"
            raise ValueError(msg)
        if not self.train_trials_only:
            msg = "heldout_decoder.train_trials_only must be true"
            raise ValueError(msg)
        return self


class BehaviorDecoderSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    alpha: float = Field(ge=0.0)
    fit_intercept: bool
    standardize_factors: bool
    standardize_targets: bool
    target_prefixes: list[str]
    derive_velocity: bool
    velocity_method: Literal["central_difference"]
    train_trials_only: bool

    @field_validator("target_prefixes")
    @classmethod
    def prefixes_are_nonempty(cls, values: list[str]) -> list[str]:
        if not values or any(not value.strip() for value in values):
            msg = "behavior_decoder.target_prefixes must contain non-empty strings"
            raise ValueError(msg)
        return values

    @model_validator(mode="after")
    def train_trials_only_is_required(self) -> BehaviorDecoderSection:
        if not self.train_trials_only:
            msg = "behavior_decoder.train_trials_only must be true"
            raise ValueError(msg)
        return self


class ReferenceSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["train_mean_rate"]
    fit_train_trials_only: bool

    @model_validator(mode="after")
    def train_only_reference_is_required(self) -> ReferenceSection:
        if not self.fit_train_trials_only:
            msg = "reference.fit_train_trials_only must be true"
            raise ValueError(msg)
        return self


class BaselineReferences(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mean_rate_validation_bits_per_spike: float
    factor_latent_best_validation_bits_per_spike: float
    factor_latent_best_behavior_mean_r2: float | None = None
    previous_lfads_eval_validation_bits_per_spike: float | None = None


class EvaluationSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluate_splits: list[SplitName]
    primary_split: SplitName
    baseline_references: BaselineReferences

    @model_validator(mode="after")
    def primary_split_is_evaluated(self) -> EvaluationSection:
        if self.primary_split not in self.evaluate_splits:
            msg = "evaluation.primary_split must be listed in evaluation.evaluate_splits"
            raise ValueError(msg)
        return self


class ReportingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str = Field(min_length=1)


class RuntimeSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: Literal["cpu", "cuda", "auto"] = "auto"


class LFADSGRUEvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: DatasetSection
    splits: SplitSection
    data: DataSection
    model: ModelSection
    evaluation_mode: EvaluationModeSection = Field(default_factory=EvaluationModeSection)
    heldout_decoder: HeldoutDecoderSection
    behavior_decoder: BehaviorDecoderSection
    reference: ReferenceSection
    evaluation: EvaluationSection
    reporting: ReportingSection
    runtime: RuntimeSection = Field(default_factory=RuntimeSection)

    @model_validator(mode="after")
    def split_fractions_sum_to_one(self) -> LFADSGRUEvalConfig:
        total = (
            self.splits.train_fraction + self.splits.validation_fraction + self.splits.test_fraction
        )
        if abs(total - 1.0) > 1e-8:
            msg = "split fractions must sum to 1.0"
            raise ValueError(msg)
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> LFADSGRUEvalConfig:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            msg = f"malformed LFADS-style GRU eval config: {path}"
            raise ValueError(msg) from exc
        if not isinstance(raw, dict):
            msg = f"LFADS-style GRU eval config must contain a top-level mapping: {path}"
            raise ValueError(msg)
        return cls.model_validate(raw)


def _parse_args(argv: Sequence[str] | None) -> Path:
    parser = argparse.ArgumentParser(description="Evaluate LFADS-style GRU held-out prediction.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/mc_maze_small_lfads_gru_eval.yaml")
    )
    return parser.parse_args(argv).config


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _mean_finite(values: pd.Series) -> float | None:  # type: ignore[type-arg]
    finite = values[np.isfinite(values)]
    return None if finite.empty else float(finite.mean())


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = get_repo_root()
    config_arg = _parse_args(argv)
    config_path = config_arg if config_arg.is_absolute() else repo_root / config_arg
    config = LFADSGRUEvalConfig.from_yaml(config_path)
    processed_path = resolve_configured_path(config.dataset.processed_path, repo_root)
    checkpoint_path = resolve_configured_path(config.model.checkpoint_path, repo_root)
    output_dir = resolve_configured_path(config.reporting.output_dir, repo_root)

    if not processed_path.exists():
        console.print(f"Processed dataset is missing: {_relative(processed_path, repo_root)}")
        console.print(
            "Run: python scripts/prepare_nlb_data.py --config configs/nlb_mc_maze_small.yaml"
        )
        return 2
    if not checkpoint_path.exists():
        console.print(f"LFADS-style checkpoint is missing: {_relative(checkpoint_path, repo_root)}")
        console.print(
            "Run: python scripts/train_lfads_gru.py --config configs/mc_maze_small_lfads_gru.yaml"
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
        dataset.spikes.shape[2], config.splits.heldout_neuron_fraction, seed=config.splits.seed
    )
    validate_trial_split(split, dataset.trial_ids)
    validate_neuron_mask(neuron_mask, dataset.spikes.shape[2])
    try:
        device = resolve_device(config.runtime.device)
    except RuntimeError as exc:
        console.print(str(exc))
        return 2
    config_dict = config.model_dump(mode="python")
    config_dict["model"]["checkpoint_path"] = str(checkpoint_path)
    config_dict["dataset"]["bin_size_ms"] = dataset.bin_size_ms

    split_metrics, neuron_metrics, behavior_metrics, factor_summary, metadata = (
        run_lfads_gru_evaluation(dataset, split, neuron_mask, config_dict, device)
    )
    primary_candidates = split_metrics[split_metrics["split"] == config.evaluation.primary_split]
    direct_candidates = primary_candidates[
        primary_candidates["prediction_source"] == "direct_model"
    ]
    primary = (
        direct_candidates.iloc[0] if not direct_candidates.empty else primary_candidates.iloc[0]
    )
    primary_prediction_source = str(primary["prediction_source"])
    direct_validation = split_metrics[
        (split_metrics["split"] == config.evaluation.primary_split)
        & (split_metrics["prediction_source"] == "direct_model")
    ]
    factor_validation = split_metrics[
        (split_metrics["split"] == config.evaluation.primary_split)
        & (split_metrics["prediction_source"] == "factor_decoder")
    ]
    direct_bits = (
        None if direct_validation.empty else float(direct_validation.iloc[0]["bits_per_spike"])
    )
    factor_bits = (
        None if factor_validation.empty else float(factor_validation.iloc[0]["bits_per_spike"])
    )
    primary_behavior_r2 = None
    if not behavior_metrics.empty:
        primary_behavior_r2 = _mean_finite(
            behavior_metrics.loc[behavior_metrics["split"] == config.evaluation.primary_split, "r2"]
        )
    refs = config.evaluation.baseline_references
    primary_bits = float(primary["bits_per_spike"])
    metrics_summary = {
        "dataset_name": config.dataset.name,
        "dataset_hash": dataset_hash,
        "checkpoint_path": _relative(checkpoint_path, repo_root),
        "checkpoint_epoch": metadata.get("checkpoint_epoch"),
        "model_name": "lfads_gru",
        "factor_dim": int(metadata["factor_dim"]),
        "latent_dim": int(metadata["latent_dim"]),
        "max_time_bins": config.data.max_time_bins,
        "primary_split": config.evaluation.primary_split,
        "primary_bits_per_spike": primary_bits,
        "primary_poisson_nll": float(primary["poisson_nll"]),
        "primary_behavior_mean_r2": primary_behavior_r2,
        "primary_prediction_source": primary_prediction_source,
        "direct_model_available": bool(metadata.get("direct_model_available", False)),
        "factor_decoder_evaluated": bool(metadata.get("factor_decoder_evaluated", False)),
        "direct_model_validation_bits_per_spike": direct_bits,
        "factor_decoder_validation_bits_per_spike": factor_bits,
        "previous_lfads_eval_validation_bits_per_spike": (
            refs.previous_lfads_eval_validation_bits_per_spike
        ),
        "mean_rate_validation_bits_per_spike": refs.mean_rate_validation_bits_per_spike,
        "factor_latent_best_validation_bits_per_spike": (
            refs.factor_latent_best_validation_bits_per_spike
        ),
        "factor_latent_best_behavior_mean_r2": refs.factor_latent_best_behavior_mean_r2,
        "beats_previous_lfads_eval": (
            None
            if refs.previous_lfads_eval_validation_bits_per_spike is None
            else primary_bits > refs.previous_lfads_eval_validation_bits_per_spike
        ),
        "beats_mean_rate_reference": primary_bits > refs.mean_rate_validation_bits_per_spike,
        "beats_factor_latent_reference": primary_bits
        > refs.factor_latent_best_validation_bits_per_spike,
        "heldout_decoder_alpha": float(config.heldout_decoder.alpha),
        "behavior_decoder_alpha": float(config.behavior_decoder.alpha),
        "behavior_decoder_enabled": config.behavior_decoder.enabled,
        "fit_policy": "train trials only",
        "official_benchmark_claim": False,
        "full_lfads_claim": False,
    }
    write_lfads_gru_evaluation_outputs(
        output_dir,
        metrics_summary,
        split_metrics,
        neuron_metrics,
        behavior_metrics,
        factor_summary,
        metadata,
    )

    console.print(f"dataset: {config.dataset.name}")
    console.print(f"shape: {list(dataset.spikes.shape)}")
    console.print(f"device: {device}")
    console.print(f"checkpoint_path: {_relative(checkpoint_path, repo_root)}")
    console.print(f"factor_dim: {metadata['factor_dim']}")
    console.print(f"primary_split: {config.evaluation.primary_split}")
    console.print(f"primary_prediction_source: {metrics_summary['primary_prediction_source']}")
    console.print(f"primary_bits_per_spike: {metrics_summary['primary_bits_per_spike']}")
    console.print(f"primary_poisson_nll: {metrics_summary['primary_poisson_nll']}")
    console.print(
        "direct_model_validation_bits_per_spike: "
        f"{metrics_summary['direct_model_validation_bits_per_spike']}"
    )
    console.print(
        "factor_decoder_validation_bits_per_spike: "
        f"{metrics_summary['factor_decoder_validation_bits_per_spike']}"
    )
    console.print(
        "previous_lfads_eval_validation_bits_per_spike: "
        f"{metrics_summary['previous_lfads_eval_validation_bits_per_spike']}"
    )
    console.print(f"primary_behavior_mean_r2: {primary_behavior_r2}")
    console.print(f"beats_previous_lfads_eval: {metrics_summary['beats_previous_lfads_eval']}")
    console.print(f"beats_mean_rate_reference: {metrics_summary['beats_mean_rate_reference']}")
    console.print(
        f"beats_factor_latent_reference: {metrics_summary['beats_factor_latent_reference']}"
    )
    console.print(f"output_dir: {_relative(output_dir, repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
