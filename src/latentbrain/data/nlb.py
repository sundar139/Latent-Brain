from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from latentbrain.data.schemas import NeuralDataset
from latentbrain.data.validation import validate_neural_dataset, validate_neural_dataset_minimums

CANDIDATE_FILE_SUFFIXES = {".h5", ".hdf5", ".mat", ".npz", ".nwb"}
LOADABLE_FILE_SUFFIXES = {".h5", ".hdf5", ".nwb"}
OFFICIAL_NLB_DATASETS_URL = "https://neurallatents.github.io/datasets"
MC_MAZE_SMALL_DANDI_URL = "https://gui.dandiarchive.org/#/dandiset/000140"
OPTIONAL_INSTALL_MESSAGE = (
    "NLB loading requires optional neurodata dependencies. Install them with "
    '`python -m pip install -e ".[dev,neurodata]"`. If `nlb-tools` is unavailable '
    "from PyPI in your environment, install it with "
    "`python -m pip install git+https://github.com/neurallatents/nlb_tools.git`."
)
MISSING_DATA_MESSAGE = (
    "MC_Maze/NLB files were not found. LatentBrain does not download real neural "
    "datasets automatically. Open the official Neural Latents Benchmark datasets "
    f"page ({OFFICIAL_NLB_DATASETS_URL}), choose MC_Maze Small "
    f"({MC_MAZE_SMALL_DANDI_URL}), download it legally with DANDI or the web "
    "interface, and place files under the configured dataset_root."
)


class NLBDatasetSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    source: str = Field(min_length=1)
    variant: str = Field(default="standard", min_length=1)
    bin_size_ms: int = Field(gt=0)
    alignment_event: str = Field(min_length=1)
    expected_format: str = Field(min_length=1)
    dataset_root: str = Field(min_length=1)
    processed_root: str = Field(min_length=1)
    output_filename: str = Field(min_length=1)
    metadata_filename: str = Field(min_length=1)
    provenance_filename: str = Field(default="mc_maze_provenance.json", min_length=1)

    @field_validator("expected_format")
    @classmethod
    def expected_format_must_be_nlb(cls, value: str) -> str:
        if value != "nlb":
            msg = "dataset.expected_format must be 'nlb'"
            raise ValueError(msg)
        return value


class NLBValidationSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    require_behavior: bool = False
    require_rates: bool = False
    allow_missing_behavior: bool = True
    min_trials: int = Field(gt=0)
    min_neurons: int = Field(gt=0)
    min_time_bins: int = Field(gt=0)

    @model_validator(mode="after")
    def behavior_flags_must_be_consistent(self) -> NLBValidationSection:
        if self.require_behavior and self.allow_missing_behavior:
            msg = "validation.allow_missing_behavior must be false when behavior is required"
            raise ValueError(msg)
        return self


class NLBSplitSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = Field(ge=0)
    train_fraction: float = Field(gt=0.0, lt=1.0)
    validation_fraction: float = Field(gt=0.0, lt=1.0)
    test_fraction: float = Field(gt=0.0, lt=1.0)
    heldout_neuron_fraction: float = Field(gt=0.0, lt=1.0)


class NLBProvenanceSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hash_size_limit_mb: int = Field(default=256, gt=0)


class NLBConfig(BaseModel):
    """Validated configuration for local NLB-style dataset preparation."""

    model_config = ConfigDict(extra="forbid")

    dataset: NLBDatasetSection
    validation: NLBValidationSection
    splits: NLBSplitSection
    provenance: NLBProvenanceSection = Field(default_factory=NLBProvenanceSection)

    @model_validator(mode="after")
    def split_fractions_must_sum_to_one(self) -> NLBConfig:
        total = (
            self.splits.train_fraction + self.splits.validation_fraction + self.splits.test_fraction
        )
        if abs(total - 1.0) > 1e-8:
            msg = "split fractions must sum to 1.0"
            raise ValueError(msg)
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> NLBConfig:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            msg = f"malformed NLB config: {path}"
            raise ValueError(msg) from exc
        if not isinstance(raw, dict):
            msg = f"NLB config must contain a top-level mapping: {path}"
            raise ValueError(msg)
        return cls.model_validate(raw)

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def resolve_nlb_dataset_root(config: NLBConfig, repo_root: Path) -> Path:
    """Resolve the configured NLB root with an optional environment override."""
    override = os.getenv("LATENTBRAIN_NLB_ROOT")
    root_value = override if override else config.dataset.dataset_root
    candidate = Path(root_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_root / candidate).resolve()


def find_candidate_nlb_files(dataset_root: Path) -> list[Path]:
    """Find local candidate real-data files without downloading anything."""
    resolved_root = dataset_root.expanduser().resolve()
    if not resolved_root.exists() or not resolved_root.is_dir():
        return []
    return sorted(
        path
        for path in resolved_root.rglob("*")
        if path.is_file() and path.suffix.lower() in CANDIDATE_FILE_SUFFIXES
    )


def find_nlb_files(dataset_root: Path) -> list[Path]:
    """Backward-compatible alias for candidate NLB files."""
    return find_candidate_nlb_files(dataset_root)


class NLBDataAdapter:
    """Adapter for local Neural Latents Benchmark-style files."""

    def __init__(self, config: NLBConfig) -> None:
        self.config = config

    def can_load(self, dataset_root: Path) -> bool:
        return bool(find_candidate_nlb_files(dataset_root))

    def load(self, dataset_root: Path) -> NeuralDataset:
        return load_nlb_dataset(dataset_root, self.config)


def _import_nlb_tools() -> object:
    try:
        return importlib.import_module("nlb_tools")
    except ImportError as exc:
        raise ImportError(OPTIONAL_INSTALL_MESSAGE) from exc


def _extract_array(source: object, names: tuple[str, ...]) -> np.ndarray | None:
    if isinstance(source, dict):
        for name in names:
            value = source.get(name)
            if value is not None:
                return np.asarray(value)
    for name in names:
        if hasattr(source, name):
            return np.asarray(getattr(source, name))
    return None


def _integer_spikes(spikes: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(spikes)):
        msg = "spikes must be finite"
        raise ValueError(msg)
    if not np.issubdtype(spikes.dtype, np.integer) and not np.all(spikes == np.floor(spikes)):
        msg = "spikes must contain integer-valued counts"
        raise ValueError(msg)
    return spikes.astype(np.int64, copy=False)


def _dataset_from_mapping_or_object(
    source: object,
    config: NLBConfig,
    source_files: list[Path],
) -> NeuralDataset | None:
    spikes = _extract_array(source, ("spikes", "spike_counts", "heldin_spikes"))
    if spikes is None:
        return None
    rates = _extract_array(source, ("rates", "firing_rates"))
    behavior = _extract_array(source, ("behavior", "behavioral_data", "hand_pos", "cursor_pos"))
    trial_ids = _extract_array(source, ("trial_ids", "trial_id"))
    time_ms = _extract_array(source, ("time_ms", "timestamps_ms", "time"))

    if spikes.ndim != 3:
        return None
    n_trials, n_time_bins, _ = spikes.shape
    if trial_ids is None:
        trial_ids = np.arange(n_trials, dtype=np.int64)
    if time_ms is None:
        time_ms = np.arange(n_time_bins, dtype=np.float64) * config.dataset.bin_size_ms

    metadata: dict[str, Any] = {
        "dataset_name": config.dataset.name,
        "source": config.dataset.source,
        "variant": config.dataset.variant,
        "alignment_event": config.dataset.alignment_event,
        "expected_format": config.dataset.expected_format,
        "bin_size_ms": config.dataset.bin_size_ms,
        "source_files": [path.name for path in source_files],
    }
    if behavior is not None:
        metadata["behavior"] = {
            "shape": list(behavior.shape),
            "dtype": str(behavior.dtype),
            "present": True,
        }

    dataset = NeuralDataset(
        spikes=_integer_spikes(spikes),
        rates=None if rates is None else rates.astype(np.float64, copy=False),
        latents=None,
        trial_ids=trial_ids.astype(np.int64, copy=False),
        time_ms=time_ms.astype(np.float64, copy=False),
        bin_size_ms=config.dataset.bin_size_ms,
        metadata=metadata,
    )
    validate_neural_dataset(dataset)
    return dataset


def _load_with_nlb_tools(
    data_file: Path,
    config: NLBConfig,
    source_files: list[Path],
) -> NeuralDataset:
    _import_nlb_tools()
    try:
        nwb_interface = importlib.import_module("nlb_tools.nwb_interface")
        nwb_dataset_class = nwb_interface.__dict__.get("NWBDataset")
        if nwb_dataset_class is None:
            raise AttributeError("NWBDataset")
    except (ImportError, AttributeError) as exc:
        msg = (
            "nlb_tools is installed, but the expected nlb_tools.nwb_interface.NWBDataset "
            "API is unavailable. Install a compatible nlb_tools release with "
            "`python -m pip install git+https://github.com/neurallatents/nlb_tools.git`."
        )
        raise ImportError(msg) from exc

    nlb_object = nwb_dataset_class(str(data_file))
    candidates = (
        nlb_object,
        getattr(nlb_object, "data", None),
        getattr(nlb_object, "trial_data", None),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        dataset = _dataset_from_mapping_or_object(candidate, config, source_files)
        if dataset is not None:
            return dataset

    detected = ", ".join(path.name for path in source_files[:10])
    msg = (
        "Local NLB files were detected and nlb_tools imported, but LatentBrain could not "
        "extract a [trials, time, neurons] spike tensor from the loaded object. "
        f"Detected files: {detected}. No fake dataset was created."
    )
    raise ValueError(msg)


def _require_candidate_files(dataset_root: Path) -> list[Path]:
    resolved_root = dataset_root.expanduser().resolve()
    if not resolved_root.exists():
        msg = f"dataset root does not exist: {resolved_root}. {MISSING_DATA_MESSAGE}"
        raise FileNotFoundError(msg)
    if not resolved_root.is_dir():
        msg = f"dataset root is not a directory: {resolved_root}"
        raise FileNotFoundError(msg)
    files = find_candidate_nlb_files(resolved_root)
    if not files:
        msg = f"no candidate MC_Maze/NLB files found under {resolved_root}. {MISSING_DATA_MESSAGE}"
        raise FileNotFoundError(msg)
    return files


def load_nlb_dataset(dataset_root: Path, config: NLBConfig) -> NeuralDataset:
    """Load local NLB-style files into the LatentBrain neural dataset contract."""
    files = _require_candidate_files(dataset_root)
    loadable_files = [path for path in files if path.suffix.lower() in LOADABLE_FILE_SUFFIXES]
    if not loadable_files:
        detected = ", ".join(path.name for path in files[:10])
        msg = (
            "unsupported candidate files were detected, but the current loader can only "
            f"attempt .nwb/.h5/.hdf5 through nlb_tools. Detected files: {detected}. "
            "Install/use official NLB/DANDI NWB or HDF5 files; no fake dataset was created."
        )
        raise ValueError(msg)

    dataset = _load_with_nlb_tools(loadable_files[0], config, files)
    validate_neural_dataset_minimums(
        dataset,
        min_trials=config.validation.min_trials,
        min_neurons=config.validation.min_neurons,
        min_time_bins=config.validation.min_time_bins,
    )
    if config.validation.require_rates and dataset.rates is None:
        msg = "NLB config requires rates, but the loaded dataset did not provide rates"
        raise ValueError(msg)
    if config.validation.require_behavior and "behavior" not in dataset.metadata:
        msg = "NLB config requires behavior metadata, but none was loaded"
        raise ValueError(msg)
    return dataset
