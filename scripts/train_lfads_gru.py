from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

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
from latentbrain.eval.reporting import write_lfads_gru_training_report
from latentbrain.models.lfads_gru import LFADSGRU, LFADSGRUConfig
from latentbrain.paths import get_repo_root, resolve_configured_path
from latentbrain.randomness import seed_everything
from latentbrain.torch.datasets import create_dataloaders, create_torch_datasets
from latentbrain.torch.device import resolve_device
from latentbrain.train.lfads_trainer import train_lfads_gru

console = Console(markup=False)
SplitName = Literal["train", "validation", "test"]


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


class DataSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_neuron_group: Literal["heldin"]
    target_neuron_group: Literal["heldout"]
    max_time_bins: int | None = Field(default=None, gt=0)
    batch_size: int = Field(gt=0)
    num_workers: int = Field(ge=0)
    drop_last: bool


class ModelSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["lfads_gru"]
    input_dim: int | None = Field(default=None, gt=0)
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
        if isinstance(self.output_dim, int) and self.output_dim <= 0:
            msg = "model.output_dim must be positive when explicit"
            raise ValueError(msg)
        return self


class TrainingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = Field(ge=0)
    device: Literal["cpu", "cuda", "auto"]
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    gradient_clip_norm: float = Field(gt=0.0)
    kl_warmup_epochs: int = Field(ge=0)
    heldin_loss_weight: float = Field(default=1.0, ge=0.0)
    heldout_loss_weight: float = Field(default=0.0, ge=0.0)
    loss_normalization: Literal["sum", "mean", "batch_mean", "per_observed_spike_bin"] = (
        "batch_mean"
    )
    log_every_batches: int = Field(gt=0)
    checkpoint_metric: str = Field(min_length=1)
    checkpoint_mode: Literal["min", "max"]

    @model_validator(mode="after")
    def at_least_one_observation_loss_is_enabled(self) -> TrainingSection:
        if self.heldin_loss_weight == 0.0 and self.heldout_loss_weight == 0.0:
            msg = "at least one of heldin_loss_weight or heldout_loss_weight must be positive"
            raise ValueError(msg)
        return self


class EvaluationSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluate_splits: list[SplitName]
    primary_split: SplitName
    metrics: list[
        Literal[
            "poisson_nll",
            "bits_per_spike",
            "total_loss",
            "heldin_reconstruction_loss",
            "heldout_prediction_loss",
            "kl_loss",
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
        return self


class ReportingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str = Field(min_length=1)


class LFADSGRUTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: DatasetSection
    splits: SplitSection
    data: DataSection
    model: ModelSection
    training: TrainingSection
    evaluation: EvaluationSection
    reporting: ReportingSection

    @model_validator(mode="after")
    def split_fractions_sum_to_one(self) -> LFADSGRUTrainingConfig:
        total = (
            self.splits.train_fraction + self.splits.validation_fraction + self.splits.test_fraction
        )
        if abs(total - 1.0) > 1e-8:
            msg = "split fractions must sum to 1.0"
            raise ValueError(msg)
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> LFADSGRUTrainingConfig:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            msg = f"malformed LFADS-style GRU config: {path}"
            raise ValueError(msg) from exc
        if not isinstance(raw, dict):
            msg = f"LFADS-style GRU config must contain a top-level mapping: {path}"
            raise ValueError(msg)
        return cls.model_validate(raw)


def _parse_args(argv: Sequence[str] | None) -> Path:
    parser = argparse.ArgumentParser(description="Train the local LFADS-style GRU foundation.")
    parser.add_argument("--config", type=Path, default=Path("configs/mc_maze_small_lfads_gru.yaml"))
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
    config = LFADSGRUTrainingConfig.from_yaml(config_path)
    processed_path = resolve_configured_path(config.dataset.processed_path, repo_root)
    output_dir = resolve_configured_path(config.reporting.output_dir, repo_root)

    if not processed_path.exists():
        console.print(f"Processed dataset is missing: {_relative(processed_path, repo_root)}")
        console.print(
            "Run: python scripts/prepare_nlb_data.py --config configs/nlb_mc_maze_small.yaml"
        )
        return 2

    seed_everything(config.training.seed)
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
        dataset.spikes.shape[2], config.splits.heldout_neuron_fraction, seed=config.splits.seed
    )
    validate_trial_split(split, dataset.trial_ids)
    validate_neuron_mask(neuron_mask, dataset.spikes.shape[2])
    torch_datasets = create_torch_datasets(
        dataset, split, neuron_mask, max_time_bins=config.data.max_time_bins
    )
    dataloaders = create_dataloaders(
        torch_datasets,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        drop_last=config.data.drop_last,
        seed=config.training.seed,
    )
    input_dim = int(neuron_mask.heldin.sum())
    total_neurons = int(dataset.spikes.shape[2])
    if config.model.output_dim == "all":
        output_dim = total_neurons
        training_mode = (
            "cosmoothing" if config.training.heldout_loss_weight > 0.0 else "heldin_reconstruction"
        )
    elif config.model.output_dim == "heldin" or config.model.output_dim is None:
        output_dim = input_dim
        training_mode = "heldin_reconstruction"
    else:
        output_dim = int(config.model.output_dim)
        training_mode = (
            "cosmoothing" if config.training.heldout_loss_weight > 0.0 else "heldin_reconstruction"
        )
    model_config = LFADSGRUConfig(
        input_dim=config.model.input_dim or input_dim,
        output_dim=output_dim,
        encoder_hidden_dim=config.model.encoder_hidden_dim,
        generator_hidden_dim=config.model.generator_hidden_dim,
        latent_dim=config.model.latent_dim,
        factor_dim=config.model.factor_dim,
        dropout=config.model.dropout,
        min_rate_hz=config.model.min_rate_hz,
        max_rate_hz=config.model.max_rate_hz,
    )
    model = LFADSGRU(model_config)
    device = resolve_device(config.training.device)
    config_dict = config.model_dump(mode="python")
    config_dict["dataset"]["bin_size_ms"] = dataset.bin_size_ms
    config_dict["model"]["input_dim"] = model_config.input_dim
    config_dict["model"]["resolved_output_dim"] = model_config.output_dim
    config_dict["training"]["training_mode"] = training_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config_dict, sort_keys=False), encoding="utf-8"
    )

    state = train_lfads_gru(model, dataloaders, config_dict, output_dir, device)
    final = state.history[-1]
    summary = {
        "dataset_name": config.dataset.name,
        "dataset_hash": dataset_hash,
        "model_name": "lfads_gru",
        "input_dim": model_config.input_dim,
        "output_dim": model_config.output_dim,
        "encoder_hidden_dim": model_config.encoder_hidden_dim,
        "generator_hidden_dim": model_config.generator_hidden_dim,
        "factor_dim": model_config.factor_dim,
        "latent_dim": model_config.latent_dim,
        "training_mode": training_mode,
        "output_dim_policy": config.model.output_dim or "heldin",
        "heldin_loss_weight": config.training.heldin_loss_weight,
        "heldout_loss_weight": config.training.heldout_loss_weight,
        "epochs": config.training.epochs,
        "kl_warmup_epochs": config.training.kl_warmup_epochs,
        "best_validation_loss": state.best_metric,
        "best_validation_total_loss": state.best_metric,
        "final_validation_loss": final.get("validation_loss"),
        "final_validation_total_loss": final.get("validation_total_loss"),
        "final_validation_heldout_prediction_loss": final.get("validation_heldout_prediction_loss"),
        "latest_checkpoint": _relative(output_dir / "checkpoints" / "latest.pt", repo_root),
        "best_validation_checkpoint": _relative(
            output_dir / "checkpoints" / "best_validation.pt", repo_root
        ),
    }
    report_name = (
        "lfads_gru_training_report.md" if training_mode == "cosmoothing" else "lfads_gru_report.md"
    )
    write_lfads_gru_training_report(output_dir / report_name, summary)

    console.print(f"dataset: {config.dataset.name}")
    console.print(f"shape: {list(dataset.spikes.shape)}")
    console.print(f"device: {device}")
    console.print(f"epochs: {config.training.epochs}")
    console.print(f"training_mode: {training_mode}")
    console.print(f"model_input_dim: {model_config.input_dim}")
    console.print(f"model_output_dim: {model_config.output_dim}")
    console.print(f"heldin_loss_weight: {config.training.heldin_loss_weight}")
    console.print(f"heldout_loss_weight: {config.training.heldout_loss_weight}")
    console.print(f"latent_dim: {model_config.latent_dim}")
    console.print(f"best_validation_total_loss: {state.best_metric}")
    console.print(f"final_validation_loss: {final.get('validation_loss')}")
    console.print(f"final_reconstruction_loss: {final.get('validation_reconstruction_loss')}")
    console.print(
        f"final_heldout_prediction_loss: {final.get('validation_heldout_prediction_loss')}"
    )
    console.print(f"final_kl_loss: {final.get('validation_kl_loss')}")
    console.print(f"output_dir: {_relative(output_dir, repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
