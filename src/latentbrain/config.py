from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from latentbrain.paths import get_repo_root

T = TypeVar("T")


class ConfigError(RuntimeError):
    """Raised when application configuration cannot be loaded."""


class ProjectConfig(BaseModel):
    """Project metadata and global defaults."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    seed: int

    @field_validator("seed")
    @classmethod
    def seed_must_be_non_negative(cls, value: int) -> int:
        if value < 0:
            msg = "project.seed must be a non-negative integer"
            raise ValueError(msg)
        return value


class PathConfig(BaseModel):
    """Repository-relative or absolute storage locations."""

    model_config = ConfigDict(extra="forbid")

    data_root: str = Field(min_length=1)
    results_root: str = Field(min_length=1)
    models_root: str = Field(min_length=1)
    reports_root: str = Field(min_length=1)
    experiments_root: str = Field(min_length=1)

    @field_validator("data_root", "results_root", "models_root", "reports_root", "experiments_root")
    @classmethod
    def path_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            msg = "configured path values must be non-empty strings"
            raise ValueError(msg)
        return value


class LoggingConfig(BaseModel):
    """Logging behavior."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    level: str = Field(min_length=1)
    json_enabled: bool = Field(default=False, alias="json")


class ReproducibilityConfig(BaseModel):
    """Reproducibility settings shared by executable entrypoints."""

    model_config = ConfigDict(extra="forbid")

    deterministic: bool = True
    benchmark: bool = False


class AppConfig(BaseModel):
    """Validated LatentBrain application configuration."""

    model_config = ConfigDict(extra="forbid")

    project: ProjectConfig
    paths: PathConfig
    logging: LoggingConfig
    reproducibility: ReproducibilityConfig


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    msg = f"cannot parse boolean environment value: {value!r}"
    raise ConfigError(msg)


def _parse_int(value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        msg = f"cannot parse integer environment value: {value!r}"
        raise ConfigError(msg) from exc


def _set_nested(data: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    target = data
    for key in keys[:-1]:
        nested = target.setdefault(key, {})
        if not isinstance(nested, dict):
            msg = f"configuration section {key!r} is not a mapping"
            raise ConfigError(msg)
        target = nested
    target[keys[-1]] = value


def _apply_environment_overrides(data: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, tuple[tuple[str, ...], Callable[[str], Any]]] = {
        "LATENTBRAIN_PROJECT_NAME": (("project", "name"), str),
        "LATENTBRAIN_SEED": (("project", "seed"), _parse_int),
        "LATENTBRAIN_DATA_ROOT": (("paths", "data_root"), str),
        "LATENTBRAIN_RESULTS_ROOT": (("paths", "results_root"), str),
        "LATENTBRAIN_MODELS_ROOT": (("paths", "models_root"), str),
        "LATENTBRAIN_REPORTS_ROOT": (("paths", "reports_root"), str),
        "LATENTBRAIN_EXPERIMENTS_ROOT": (("paths", "experiments_root"), str),
        "LATENTBRAIN_LOG_LEVEL": (("logging", "level"), str),
        "LATENTBRAIN_LOG_JSON": (("logging", "json"), _parse_bool),
        "LATENTBRAIN_DETERMINISTIC": (("reproducibility", "deterministic"), _parse_bool),
        "LATENTBRAIN_BENCHMARK": (("reproducibility", "benchmark"), _parse_bool),
    }
    updated = dict(data)
    for env_name, (keys, parser) in overrides.items():
        raw_value = os.getenv(env_name)
        if raw_value is None or raw_value == "":
            continue
        _set_nested(updated, keys, parser(raw_value))
    return updated


def _default_config_path() -> Path:
    env_path = os.getenv("LATENTBRAIN_CONFIG_PATH")
    if env_path:
        return Path(env_path).expanduser()
    return get_repo_root() / "configs" / "base.yaml"


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load and validate application configuration from YAML and environment."""
    load_dotenv()
    path = config_path.expanduser() if config_path is not None else _default_config_path()

    if not path.exists():
        msg = f"configuration file not found: {path}"
        raise ConfigError(msg)
    if not path.is_file():
        msg = f"configuration path is not a file: {path}"
        raise ConfigError(msg)

    try:
        raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"malformed YAML configuration file: {path}"
        raise ConfigError(msg) from exc
    except OSError as exc:
        msg = f"unable to read configuration file: {path}"
        raise ConfigError(msg) from exc

    if not isinstance(raw_config, dict):
        msg = f"configuration file must contain a top-level mapping: {path}"
        raise ConfigError(msg)

    try:
        return AppConfig.model_validate(_apply_environment_overrides(raw_config))
    except ValidationError as exc:
        msg = f"configuration validation failed for {path}: {exc}"
        raise ConfigError(msg) from exc
