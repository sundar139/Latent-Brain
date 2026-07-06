from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


@dataclass(slots=True)
class NeuralDataset:
    """Neural population arrays with trial-major shape conventions."""

    spikes: np.ndarray
    rates: np.ndarray | None
    latents: np.ndarray | None
    trial_ids: np.ndarray
    time_ms: np.ndarray
    bin_size_ms: int
    metadata: dict[str, Any]
    behavior: np.ndarray | None = None
    behavior_names: list[str] | None = None


@dataclass(slots=True)
class TrialSplit:
    """Leakage-safe trial identifiers for train, validation, and test use."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


@dataclass(slots=True)
class NeuronMask:
    """Held-in and held-out neuron masks for future co-smoothing validation."""

    heldin: np.ndarray
    heldout: np.ndarray


@dataclass(slots=True)
class DatasetMetadata:
    """Small serializable dataset metadata record."""

    name: str
    dataset_hash: str
    n_trials: int
    n_time_bins: int
    n_neurons: int
    bin_size_ms: int


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    seed: int = Field(ge=0)
    n_trials: int = Field(gt=0)
    n_time_bins: int = Field(gt=0)
    n_neurons: int = Field(gt=0)
    latent_dim: int = Field(gt=0)
    bin_size_ms: int = Field(gt=0)
    train_fraction: float = Field(gt=0.0, lt=1.0)
    validation_fraction: float = Field(gt=0.0, lt=1.0)
    test_fraction: float = Field(gt=0.0, lt=1.0)
    heldout_neuron_fraction: float = Field(gt=0.0, lt=1.0)


class DynamicsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spectral_radius: float = Field(gt=0.0)
    process_noise_std: float = Field(gt=0.0)


class ObservationsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    log_rate_bias: float
    loading_scale: float = Field(gt=0.0)
    min_rate_hz: float = Field(gt=0.0)
    max_rate_hz: float = Field(gt=0.0)

    @field_validator("max_rate_hz")
    @classmethod
    def max_rate_must_exceed_min_rate(cls, value: float, info: ValidationInfo) -> float:
        min_rate = info.data.get("min_rate_hz")
        if min_rate is not None and value <= min_rate:
            msg = "observations.max_rate_hz must exceed observations.min_rate_hz"
            raise ValueError(msg)
        return value


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    directory: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    metadata_filename: str = Field(min_length=1)


class SyntheticDatasetConfig(BaseModel):
    """Validated configuration for synthetic Poisson LDS generation."""

    model_config = ConfigDict(extra="forbid")

    dataset: DatasetConfig
    dynamics: DynamicsConfig
    observations: ObservationsConfig
    output: OutputConfig

    @model_validator(mode="after")
    def split_fractions_must_sum_to_one(self) -> SyntheticDatasetConfig:
        total = (
            self.dataset.train_fraction
            + self.dataset.validation_fraction
            + self.dataset.test_fraction
        )
        if abs(total - 1.0) > 1e-8:
            msg = "dataset split fractions must sum to 1.0"
            raise ValueError(msg)
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> SyntheticDatasetConfig:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            msg = f"malformed synthetic dataset config: {path}"
            raise ValueError(msg) from exc
        if not isinstance(raw, dict):
            msg = f"synthetic dataset config must contain a mapping: {path}"
            raise ValueError(msg)
        return cls.model_validate(raw)

    def with_seed(self, seed: int) -> SyntheticDatasetConfig:
        return self.model_copy(update={"dataset": self.dataset.model_copy(update={"seed": seed})})
