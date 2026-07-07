from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rich.console import Console

from latentbrain.eval.reporting import write_lfads_tuning_outputs
from latentbrain.eval.tuning import rank_tuning_results
from latentbrain.paths import get_repo_root, resolve_configured_path
from latentbrain.train.lfads_tuning import run_lfads_tuning

console = Console(markup=False)


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


class RuntimeSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: Literal["cuda"]
    fail_if_cuda_unavailable: bool


class SearchSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_runs: int = Field(gt=0)
    selection_metric: Literal["validation_bits_per_spike"]
    selection_mode: Literal["max"]
    run_order: Literal["deterministic"]


class GridSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    encoder_hidden_dim: list[int] = Field(min_length=1)
    generator_hidden_dim: list[int] = Field(min_length=1)
    latent_dim: list[int] = Field(min_length=1)
    factor_dim: list[int] = Field(min_length=1)
    dropout: list[float] = Field(min_length=1)
    learning_rate: list[float] = Field(min_length=1)
    weight_decay: list[float] = Field(min_length=1)
    heldout_loss_weight: list[float] = Field(min_length=1)
    kl_warmup_epochs: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def values_are_valid(self) -> GridSection:
        for name in ("encoder_hidden_dim", "generator_hidden_dim", "latent_dim", "factor_dim"):
            if any(value <= 0 for value in getattr(self, name)):
                msg = f"grid.{name} values must be positive"
                raise ValueError(msg)
        if any(value < 0.0 or value >= 1.0 for value in self.dropout):
            msg = "grid.dropout values must be in [0, 1)"
            raise ValueError(msg)
        if any(value <= 0.0 for value in self.learning_rate):
            msg = "grid.learning_rate values must be positive"
            raise ValueError(msg)
        if any(value < 0.0 for value in self.weight_decay):
            msg = "grid.weight_decay values must be non-negative"
            raise ValueError(msg)
        if any(value < 0.0 for value in self.heldout_loss_weight):
            msg = "grid.heldout_loss_weight values must be non-negative"
            raise ValueError(msg)
        if any(value < 0 for value in self.kl_warmup_epochs):
            msg = "grid.kl_warmup_epochs values must be non-negative"
            raise ValueError(msg)
        return self


class ModelSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["lfads_gru"]
    input_dim: int | None = Field(default=None, gt=0)
    output_dim: Literal["all"]
    min_rate_hz: float = Field(gt=0.0)
    max_rate_hz: float = Field(gt=0.0)

    @model_validator(mode="after")
    def max_rate_exceeds_min_rate(self) -> ModelSection:
        if self.max_rate_hz <= self.min_rate_hz:
            msg = "model.max_rate_hz must exceed model.min_rate_hz"
            raise ValueError(msg)
        return self


class TrainingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = Field(ge=0)
    epochs: int = Field(gt=0)
    gradient_clip_norm: float = Field(gt=0.0)
    heldin_loss_weight: float = Field(ge=0.0)
    loss_normalization: Literal["sum", "mean", "batch_mean", "per_observed_spike_bin"]
    log_every_batches: int = Field(gt=0)
    checkpoint_metric: str = Field(min_length=1)
    checkpoint_mode: Literal["min", "max"]


class BaselineReferences(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_matched_mean_rate_validation_bits_per_spike: float
    window_matched_factor_latent_validation_bits_per_spike: float
    previous_lfads_masked_direct_validation_bits_per_spike: float
    previous_lfads_masked_factor_decoder_validation_bits_per_spike: float | None = None


class EvaluationSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_split: Literal["validation"]
    primary_metric: Literal["bits_per_spike"]
    direct_model_primary: bool
    also_evaluate_factor_decoder: bool
    behavior_decoder_enabled: bool
    baseline_references: BaselineReferences


class ReportingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str = Field(min_length=1)


class LFADSTuningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: DatasetSection
    splits: SplitSection
    data: DataSection
    runtime: RuntimeSection
    search: SearchSection
    grid: GridSection
    model: ModelSection
    training: TrainingSection
    evaluation: EvaluationSection
    reporting: ReportingSection

    @model_validator(mode="after")
    def validate_fixed_contract(self) -> LFADSTuningConfig:
        total = (
            self.splits.train_fraction + self.splits.validation_fraction + self.splits.test_fraction
        )
        if abs(total - 1.0) > 1e-8:
            msg = "split fractions must sum to 1.0"
            raise ValueError(msg)
        if self.data.max_time_bins != 256:
            msg = "data.max_time_bins must be 256 for window-matched tuning"
            raise ValueError(msg)
        if not self.runtime.fail_if_cuda_unavailable:
            msg = "runtime.fail_if_cuda_unavailable must be true"
            raise ValueError(msg)
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> LFADSTuningConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            msg = f"LFADS tuning config must contain a top-level mapping: {path}"
            raise ValueError(msg)
        return cls.model_validate(raw)


def _parse_args(argv: Sequence[str] | None) -> Path:
    parser = argparse.ArgumentParser(description="Run local LFADS-style CUDA tuning.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/mc_maze_small_lfads_gru_tuning.yaml")
    )
    return parser.parse_args(argv).config


def _cuda_diagnostic() -> dict[str, Any]:
    available = torch.cuda.is_available()
    return {
        "torch_version": torch.__version__,
        "cuda_available": available,
        "torch_cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if available else "NONE",
    }


def _best_config_from_disk(output_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    path = output_dir / "best_config.yaml"
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            return loaded
    return dict(summary.get("best_run_params", {}))


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = get_repo_root()
    config_arg = _parse_args(argv)
    config_path = config_arg if config_arg.is_absolute() else repo_root / config_arg
    if not config_path.exists():
        console.print(f"Config file is missing: {config_path}")
        return 2
    try:
        config_model = LFADSTuningConfig.from_yaml(config_path)
    except (OSError, ValueError) as exc:
        console.print(f"Config validation failed: {exc}")
        return 2
    diagnostic = _cuda_diagnostic()
    console.print(f"torch: {diagnostic['torch_version']}")
    console.print(f"cuda_available: {diagnostic['cuda_available']}")
    console.print(f"torch_cuda: {diagnostic['torch_cuda']}")
    console.print(f"gpu_name: {diagnostic['gpu_name']}")
    if not diagnostic["cuda_available"]:
        console.print("CUDA was requested, but torch.cuda.is_available() is False.")
        return 2

    config = config_model.model_dump(mode="python")
    results, summary = run_lfads_tuning(config)
    output_dir = resolve_configured_path(str(config["reporting"]["output_dir"]), repo_root)
    ranked = rank_tuning_results(results, str(config["search"]["selection_metric"]), "max")
    best_config = _best_config_from_disk(output_dir, summary)
    write_lfads_tuning_outputs(output_dir, summary, results, ranked, best_config)

    successful = int(summary.get("successful_runs", 0))
    console.print(f"dataset: {config['dataset']['name']}")
    console.print(f"device: {config['runtime']['device']}")
    console.print(f"gpu_name: {diagnostic['gpu_name']}")
    console.print(f"runs_attempted: {summary.get('runs_attempted')}")
    console.print(f"successful_runs: {successful}")
    console.print(f"best_run_id: {summary.get('best_run_id')}")
    console.print(
        f"best_validation_bits_per_spike: {summary.get('best_validation_bits_per_spike')}"
    )
    console.print(f"best_validation_poisson_nll: {summary.get('best_validation_poisson_nll')}")
    console.print(
        f"best_validation_behavior_mean_r2: {summary.get('best_validation_behavior_mean_r2')}"
    )
    console.print(
        f"beats_window_matched_mean_rate: {summary.get('beats_window_matched_mean_rate')}"
    )
    console.print(
        f"beats_window_matched_factor_latent: {summary.get('beats_window_matched_factor_latent')}"
    )
    console.print(f"output_dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
