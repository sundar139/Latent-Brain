from __future__ import annotations

import importlib
import os
from contextlib import suppress
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from latentbrain.data.schemas import NeuralDataset
from latentbrain.data.validation import validate_neural_dataset, validate_neural_dataset_minimums

CANDIDATE_FILE_SUFFIXES = {".h5", ".hdf5", ".mat", ".npz", ".nwb"}
LOADABLE_FILE_SUFFIXES = {".h5", ".hdf5", ".nwb"}
OFFICIAL_NLB_DATASETS_URL = "https://neurallatents.github.io/datasets"
MC_MAZE_SMALL_DANDI_URL = "https://gui.dandiarchive.org/#/dandiset/000140"
MC_MAZE_LARGE_DANDI_URL = "https://gui.dandiarchive.org/#/dandiset/000138"
KNOWN_VARIANTS = ("small", "medium", "large")
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


class NLBTrializationSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_file_pattern: str = Field(default="*", min_length=1)
    test_file_pattern: str = Field(default="*desc-test_ecephys.nwb", min_length=1)
    start_field: str = Field(default="start_time", min_length=1)
    end_field: str = Field(default="end_time", min_length=1)
    align_field: str | None = None
    align_range_ms: tuple[int | None, int | None] | None = None
    margin_ms: int = Field(default=0, ge=0)
    allow_nans: bool = False
    signal_types: list[str] = Field(default_factory=lambda: ["spikes", "heldout_spikes"])
    combine_heldout_spikes: bool = True
    variable_length_policy: str = "crop_to_min"
    behavior_signal_types: list[str] = Field(default_factory=list)
    require_behavior: bool = False
    behavior_variable_length_policy: str = "crop_to_spike_window"
    allow_behavior_nans: bool = False

    @field_validator("signal_types")
    @classmethod
    def signal_types_must_include_spikes(cls, value: list[str]) -> list[str]:
        if "spikes" not in value:
            msg = "trialization.signal_types must include 'spikes'"
            raise ValueError(msg)
        return value

    @field_validator("variable_length_policy")
    @classmethod
    def variable_length_policy_must_be_supported(cls, value: str) -> str:
        if value != "crop_to_min":
            msg = "trialization.variable_length_policy must be 'crop_to_min'"
            raise ValueError(msg)
        return value

    @field_validator("behavior_variable_length_policy")
    @classmethod
    def behavior_variable_length_policy_must_be_supported(cls, value: str) -> str:
        if value != "crop_to_spike_window":
            msg = "trialization.behavior_variable_length_policy must be 'crop_to_spike_window'"
            raise ValueError(msg)
        return value

    @field_validator("behavior_signal_types")
    @classmethod
    def behavior_signal_types_must_be_nonempty_strings(cls, value: list[str]) -> list[str]:
        if any(not signal_type.strip() for signal_type in value):
            msg = "trialization.behavior_signal_types must contain non-empty strings"
            raise ValueError(msg)
        if len(set(value)) != len(value):
            msg = "trialization.behavior_signal_types must be unique"
            raise ValueError(msg)
        return value


class NLBSourceAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: str | None = None


class NLBSourceSection(BaseModel):
    """Verified upstream identity of a dataset variant. Never used to download."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1)
    dandiset_id: str | None = None
    dandiset_version: str | None = None
    doi: str | None = None
    expected_assets: list[NLBSourceAsset] = Field(default_factory=list)
    automatic_download: bool = False

    @field_validator("automatic_download")
    @classmethod
    def automatic_download_must_be_disabled(cls, value: bool) -> bool:
        if value:
            msg = "source.automatic_download must be false; LatentBrain never downloads raw data"
            raise ValueError(msg)
        return value


class NLBBehaviorSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aliases: dict[str, str] = Field(default_factory=dict)
    required_names: list[str] = Field(default_factory=list)


class NLBConfig(BaseModel):
    """Validated configuration for local NLB-style dataset preparation."""

    model_config = ConfigDict(extra="forbid")

    dataset: NLBDatasetSection
    validation: NLBValidationSection
    splits: NLBSplitSection
    provenance: NLBProvenanceSection = Field(default_factory=NLBProvenanceSection)
    trialization: NLBTrializationSection = Field(default_factory=NLBTrializationSection)
    source: NLBSourceSection | None = None
    behavior: NLBBehaviorSection | None = None

    @model_validator(mode="after")
    def source_assets_must_match_variant(self) -> NLBConfig:
        source = self.source
        if source is None:
            return self
        if source.provider == "dandi" and not source.dandiset_id:
            msg = "source.dandiset_id is required when source.provider is 'dandi'"
            raise ValueError(msg)
        variant = self.dataset.variant
        if variant not in KNOWN_VARIANTS:
            return self
        for asset in source.expected_assets:
            if f"ses-{variant}" not in asset.path:
                msg = (
                    f"source asset {asset.path!r} does not belong to the configured "
                    f"variant {variant!r}"
                )
                raise ValueError(msg)
        return self

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


def _matched_files(candidate_files: list[Path], pattern: str) -> list[Path]:
    return sorted(path for path in candidate_files if fnmatch(path.name, pattern))


def _format_detected_files(candidate_files: list[Path]) -> str:
    return ", ".join(path.name for path in candidate_files) or "none"


def select_train_nwb_file(candidate_files: list[Path], pattern: str) -> Path:
    """Select the local train NWB file used to build supervised local targets."""
    matches = _matched_files(candidate_files, pattern)
    if not matches:
        detected = _format_detected_files(candidate_files)
        msg = (
            f"preferred train NLB file matching {pattern!r} was not found. "
            f"Detected files: {detected}"
        )
        raise FileNotFoundError(msg)
    return matches[0]


def select_test_nwb_files(candidate_files: list[Path], pattern: str) -> list[Path]:
    """Select test NWB files for metadata/provenance only, not target extraction."""
    return _matched_files(candidate_files, pattern)


def _h5_text(handle: Any, key: str) -> str | None:
    node = handle.get(key)
    if node is None:
        return None
    with suppress(AttributeError, TypeError, ValueError, IndexError):
        value = node[()]
        if isinstance(value, np.ndarray):
            value = value.ravel()[0] if value.size else None
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if value is not None:
            return str(value)
    return None


def describe_nwb_file(path: Path) -> dict[str, Any]:
    """Read NWB/HDF5 header metadata without pynwb or nlb_tools."""
    info: dict[str, Any] = {
        "nwb_identifier": None,
        "session_description": None,
        "session_id": None,
        "subject_id": None,
        "available_acquisition_series": [],
        "available_processing_modules": [],
        "trial_table_present": False,
        "trial_count_if_available": None,
        "unit_count_if_available": None,
        "behavior_series_candidates": [],
        "metadata_read": False,
        "metadata_error": None,
    }
    if path.suffix.lower() not in LOADABLE_FILE_SUFFIXES:
        info["metadata_error"] = "unsupported_suffix"
        return info
    try:
        h5py = importlib.import_module("h5py")
    except ImportError:
        info["metadata_error"] = "h5py_unavailable"
        return info

    try:
        with h5py.File(path, "r") as handle:
            info["metadata_read"] = True
            info["nwb_identifier"] = _h5_text(handle, "identifier")
            info["session_description"] = _h5_text(handle, "session_description")
            info["session_id"] = _h5_text(handle, "general/session_id")
            info["subject_id"] = _h5_text(handle, "general/subject/subject_id")
            acquisition = handle.get("acquisition")
            if acquisition is not None:
                info["available_acquisition_series"] = sorted(acquisition)
            processing = handle.get("processing")
            if processing is not None:
                info["available_processing_modules"] = sorted(processing)
            trials = handle.get("intervals/trials")
            if trials is not None:
                info["trial_table_present"] = True
                start_time = trials.get("start_time")
                if start_time is not None:
                    info["trial_count_if_available"] = int(start_time.shape[0])
            units = handle.get("units/id")
            if units is not None:
                info["unit_count_if_available"] = int(units.shape[0])
            behavior = handle.get("processing/behavior")
            if behavior is not None:
                info["behavior_series_candidates"] = sorted(
                    f"{module}/{series}"
                    for module in behavior
                    for series in (behavior[module] if hasattr(behavior[module], "keys") else [])
                )
    except (OSError, KeyError, ValueError, TypeError) as exc:
        info["metadata_error"] = str(exc)
    return info


def detect_variant(info: dict[str, Any], path: Path) -> tuple[str | None, str]:
    """Detect the MC_Maze variant from NWB metadata first, filename only as fallback."""
    # Real NLB MC_Maze assets name the variant in general/session_id ("small"/"large").
    session_id = str(info.get("session_id") or "").strip().lower()
    if session_id in KNOWN_VARIANTS:
        return session_id, "nwb_metadata"
    for field in ("session_description", "nwb_identifier"):
        text = str(info.get(field) or "").lower()
        for variant in KNOWN_VARIANTS:
            if f"mc_maze_{variant}" in text or f"ses-{variant}" in text:
                return variant, "nwb_metadata"
    name = path.name.lower()
    for variant in KNOWN_VARIANTS:
        if f"ses-{variant}" in name or f"mc_maze_{variant}" in name:
            return variant, "filename"
    return None, "unknown"


def inspect_nlb_candidates(dataset_root: Path, config: NLBConfig) -> list[dict[str, Any]]:
    """Describe each local candidate file for the configured dataset variant."""
    records: list[dict[str, Any]] = []
    for path in find_candidate_nlb_files(dataset_root):
        info = describe_nwb_file(path)
        variant, evidence = detect_variant(info, path)
        records.append(
            {
                "dataset_candidate": f"mc_maze_{variant}" if variant else "unknown",
                "configured_dataset": config.dataset.name,
                "variant_candidate": variant,
                "variant_evidence": evidence,
                "path": str(path),
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                **info,
            }
        )
    return records


def enforce_dataset_variant(files: list[Path], config: NLBConfig) -> dict[str, str]:
    """Reject files from another MC_Maze variant, or a mix of incompatible sessions."""
    expected = config.dataset.variant
    if expected not in KNOWN_VARIANTS:
        return {}
    detected: dict[str, str] = {}
    for path in files:
        if path.suffix.lower() not in LOADABLE_FILE_SUFFIXES:
            continue
        variant, _ = detect_variant(describe_nwb_file(path), path)
        if variant is not None:
            detected[path.name] = variant
    variants = sorted(set(detected.values()))
    if not variants:
        return detected
    if len(variants) > 1:
        msg = f"multiple incompatible MC_Maze sessions detected: {detected}"
        raise ValueError(msg)
    if variants[0] != expected:
        msg = (
            f"configured dataset variant is {expected!r}, but the local files belong to "
            f"variant {variants[0]!r}: {sorted(detected)}"
        )
        raise ValueError(msg)
    return detected


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
    if np.any(spikes < 0):
        msg = "spikes must be non-negative"
        raise ValueError(msg)
    if not np.issubdtype(spikes.dtype, np.integer) and not np.all(spikes == np.floor(spikes)):
        msg = "spikes must contain integer-valued counts"
        raise ValueError(msg)
    return spikes.astype(np.int64, copy=False)


def _top_level_columns(dataframe: Any) -> list[str]:
    columns = dataframe.columns
    if isinstance(columns, pd.MultiIndex):
        return sorted({str(value) for value in columns.get_level_values(0)})
    return sorted(str(column) for column in columns)


def _column_key(dataframe: Any, name: str) -> Any:
    columns = dataframe.columns
    if isinstance(columns, pd.MultiIndex):
        matches = [column for column in columns if str(column[0]) == name]
        if matches:
            return matches[0]
    elif name in columns:
        return name
    msg = f"trial dataframe is missing required column {name!r}"
    raise ValueError(msg)


def _column_values(dataframe: Any, name: str) -> np.ndarray:
    values = dataframe.loc[:, _column_key(dataframe, name)]
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]
    return np.asarray(values)


def _signal_frame(dataframe: Any, signal_type: str) -> Any | None:
    columns = dataframe.columns
    if isinstance(columns, pd.MultiIndex):
        mask = [str(column[0]) == signal_type for column in columns]
        if not any(mask):
            return None
        return dataframe.loc[:, mask]
    if signal_type in columns:
        return dataframe.loc[:, [signal_type]]
    return None


def _behavior_metadata(dataframe: Any) -> dict[str, Any]:
    excluded = {
        "spikes",
        "heldout_spikes",
        "trial_id",
        "trial_time",
        "align_time",
        "clock_time",
        "margin",
    }
    columns = dataframe.columns
    behavior_columns: list[dict[str, str]] = []
    if isinstance(columns, pd.MultiIndex):
        for column in columns:
            signal_type = str(column[0])
            if signal_type in excluded:
                continue
            behavior_columns.append({"signal_type": signal_type, "channel": str(column[1])})
    else:
        for column in columns:
            signal_type = str(column)
            if signal_type not in excluded:
                behavior_columns.append({"signal_type": signal_type, "channel": ""})

    column_counts: dict[str, int] = {}
    for column in behavior_columns:
        signal_type = column["signal_type"]
        column_counts[signal_type] = column_counts.get(signal_type, 0) + 1
    return {
        "present": bool(behavior_columns),
        "signal_groups": sorted(column_counts),
        "column_counts": column_counts,
        "columns": behavior_columns,
    }


def _milliseconds(value: object) -> float:
    if isinstance(value, pd.Timedelta | np.timedelta64):
        return float(pd.to_timedelta(value).total_seconds() * 1000.0)
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(pd.to_timedelta(value).total_seconds() * 1000.0)


def make_trial_dataframe(nwb_dataset: Any, config: NLBConfig) -> Any:
    """Trialize a continuous NWBDataset dataframe with nlb_tools metadata."""
    make_trial_data = getattr(nwb_dataset, "make_trial_data", None)
    if make_trial_data is None:
        msg = (
            "loaded NWBDataset does not provide make_trial_data for continuous "
            "dataframe trialization"
        )
        raise RuntimeError(msg)

    data = getattr(nwb_dataset, "data", None)
    if data is not None and hasattr(data, "index") and not data.index.is_monotonic_increasing:
        nwb_dataset.data = data.sort_index(kind="stable")

    trialization = config.trialization
    align_range = trialization.align_range_ms or (None, None)
    try:
        trial_data = make_trial_data(
            start_field=trialization.start_field,
            end_field=trialization.end_field,
            align_field=trialization.align_field,
            align_range=align_range,
            margin=trialization.margin_ms,
            allow_nans=trialization.allow_nans,
        )
    except Exception as exc:
        msg = f"NWBDataset continuous dataframe could not be trialized: {exc}"
        raise RuntimeError(msg) from exc

    if trial_data is None:
        trial_data = getattr(nwb_dataset, "trial_data", None)
    if trial_data is None or getattr(trial_data, "empty", False):
        msg = "NWBDataset.make_trial_data returned no trial rows"
        raise RuntimeError(msg)
    return trial_data


def dataframe_to_trial_tensor(
    trial_data: Any,
    signal_types: list[str],
    combine_heldout_spikes: bool,
    variable_length_policy: str,
    bin_size_ms: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Convert a long NLB trial dataframe to [trials, time, neurons] spikes."""
    if variable_length_policy != "crop_to_min":
        msg = "only variable_length_policy='crop_to_min' is supported"
        raise ValueError(msg)

    available_signal_types = _top_level_columns(trial_data)
    spike_frame = _signal_frame(trial_data, "spikes")
    if spike_frame is None:
        msg = f"spikes signal is missing; available signal types: {available_signal_types}"
        raise ValueError(msg)

    frames = [spike_frame]
    spike_column_counts = {"spikes": int(spike_frame.shape[1])}
    heldout_frame = _signal_frame(trial_data, "heldout_spikes")
    heldout_present = heldout_frame is not None
    if combine_heldout_spikes and "heldout_spikes" in signal_types and heldout_frame is not None:
        frames.append(heldout_frame)
        spike_column_counts["heldout_spikes"] = int(heldout_frame.shape[1])
    combined_spikes = pd.concat(frames, axis=1)

    trial_id_values = _column_values(trial_data, "trial_id")
    unique_trial_ids = pd.unique(trial_id_values)
    trial_time_key = None
    with suppress(ValueError):
        trial_time_key = _column_key(trial_data, "trial_time")

    trial_matrices: list[np.ndarray] = []
    trial_lengths: list[int] = []
    trial_time_min_ms: float | None = None
    trial_time_max_ms: float | None = None
    for trial_id in unique_trial_ids:
        mask = trial_id_values == trial_id
        group_spikes = combined_spikes.loc[mask]
        if trial_time_key is not None:
            group_times = trial_data.loc[mask, trial_time_key]
            if isinstance(group_times, pd.DataFrame):
                group_times = group_times.iloc[:, 0]
            order = np.argsort(np.asarray(group_times))
            group_spikes = group_spikes.iloc[order]
            time_values = np.asarray(group_times.iloc[order])
            if len(time_values):
                first_time_ms = _milliseconds(time_values[0])
                last_time_ms = _milliseconds(time_values[-1])
                trial_time_min_ms = (
                    first_time_ms
                    if trial_time_min_ms is None
                    else min(trial_time_min_ms, first_time_ms)
                )
                trial_time_max_ms = (
                    last_time_ms
                    if trial_time_max_ms is None
                    else max(trial_time_max_ms, last_time_ms)
                )
        matrix = group_spikes.to_numpy(dtype=np.float64)
        trial_matrices.append(matrix)
        trial_lengths.append(int(matrix.shape[0]))

    positive_lengths = [length for length in trial_lengths if length > 0]
    if not positive_lengths or len(positive_lengths) != len(trial_lengths):
        msg = "all trialized trials must contain at least one time bin"
        raise ValueError(msg)
    min_length = min(positive_lengths)
    max_length = max(positive_lengths)
    cropped = [_integer_spikes(matrix[:min_length]) for matrix in trial_matrices]
    spikes = np.stack(cropped, axis=0)
    trial_ids = np.asarray(unique_trial_ids)
    if not np.all(np.isfinite(trial_ids.astype(np.float64, copy=False))):
        msg = "trial_id values must be finite"
        raise ValueError(msg)
    if not np.all(trial_ids == np.floor(trial_ids.astype(np.float64, copy=False))):
        msg = "trial_id values must be integer-valued"
        raise ValueError(msg)
    time_ms = np.arange(min_length, dtype=np.float64) * bin_size_ms

    raw_total = int(np.nansum(combined_spikes.to_numpy(dtype=np.float64)))
    kept_total = int(spikes.sum())
    conservation = {
        "raw_spike_count": raw_total,
        "trialized_spike_count": kept_total,
        "excluded_spike_count": raw_total - kept_total,
        "excluded_bins": int(sum(trial_lengths) - len(trial_lengths) * min_length),
        "conserved": raw_total == kept_total,
        "exclusion_reason": None if raw_total == kept_total else variable_length_policy,
    }

    metadata: dict[str, Any] = {
        # ingestion_summary is excluded from the dataset hash payload; see data/io.py.
        "ingestion_summary": {"spike_conservation": conservation},
        "available_signal_types": available_signal_types,
        "signal_types_requested": signal_types,
        "spike_column_counts": spike_column_counts,
        "heldout_spikes_present": heldout_present,
        "heldout_spikes_combined": bool(combine_heldout_spikes and heldout_present),
        "behavior": _behavior_metadata(trial_data),
        "trialization": {
            "original_trial_lengths": trial_lengths,
            "min_length": min_length,
            "max_length": max_length,
            "n_trials": len(trial_lengths),
            "variable_length_policy": variable_length_policy,
            "cropping_occurred": min_length != max_length,
            "trial_time_range_ms": None
            if trial_time_min_ms is None or trial_time_max_ms is None
            else [trial_time_min_ms, trial_time_max_ms],
        },
    }
    return spikes, trial_ids.astype(np.int64, copy=False), time_ms, metadata


def _behavior_name(signal_type: str, column: object, index: int) -> str:
    suffix = ""
    if isinstance(column, tuple):
        parts = [str(value) for value in column[1:] if str(value)]
        suffix = "_".join(parts)
    return f"{signal_type}_{suffix or index}"


def _unique_names(names: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique: list[str] = []
    for name in names:
        count = counts.get(name, 0)
        counts[name] = count + 1
        unique.append(name if count == 0 else f"{name}_{count}")
    return unique


def dataframe_to_behavior_tensor(
    trial_data: Any,
    behavior_signal_types: list[str],
    trial_ids: np.ndarray,
    n_time_bins: int,
    require_behavior: bool,
    allow_behavior_nans: bool,
    behavior_variable_length_policy: str,
) -> tuple[np.ndarray | None, list[str] | None, dict[str, Any]]:
    """Extract trial-major behavior aligned to the already-cropped spike window."""
    if behavior_variable_length_policy != "crop_to_spike_window":
        msg = "only behavior_variable_length_policy='crop_to_spike_window' is supported"
        raise ValueError(msg)

    found_frames: list[pd.DataFrame] = []
    names: list[str] = []
    groups_found: list[str] = []
    for signal_type in behavior_signal_types:
        frame = _signal_frame(trial_data, signal_type)
        if frame is None:
            continue
        groups_found.append(signal_type)
        found_frames.append(frame)
        names.extend(
            _behavior_name(signal_type, column, index) for index, column in enumerate(frame.columns)
        )

    missing = [
        signal_type for signal_type in behavior_signal_types if signal_type not in groups_found
    ]
    if missing and require_behavior:
        msg = f"required behavior signals missing: {missing}"
        raise ValueError(msg)
    if not found_frames:
        return (
            None,
            None,
            {
                "present": False,
                "groups_requested": behavior_signal_types,
                "groups_found": [],
                "missing_groups": missing,
                "extraction_policy": behavior_variable_length_policy,
            },
        )

    behavior_frame = pd.concat(found_frames, axis=1)
    trial_id_values = _column_values(trial_data, "trial_id")
    trial_time_key = None
    with suppress(ValueError):
        trial_time_key = _column_key(trial_data, "trial_time")

    matrices: list[np.ndarray] = []
    lengths: list[int] = []
    cropped = False
    for trial_id in trial_ids:
        mask = trial_id_values == trial_id
        group_behavior = behavior_frame.loc[mask]
        if trial_time_key is not None:
            group_times = trial_data.loc[mask, trial_time_key]
            if isinstance(group_times, pd.DataFrame):
                group_times = group_times.iloc[:, 0]
            order = np.argsort(np.asarray(group_times))
            group_behavior = group_behavior.iloc[order]
        matrix = group_behavior.to_numpy(dtype=np.float64)
        if matrix.shape[0] < n_time_bins:
            msg = f"behavior for trial {trial_id} has fewer bins than spikes"
            raise ValueError(msg)
        cropped = cropped or matrix.shape[0] != n_time_bins
        matrices.append(matrix[:n_time_bins])
        lengths.append(int(matrix.shape[0]))

    behavior = np.stack(matrices, axis=0)
    nan_count = int(np.isnan(behavior).sum())
    inf_count = int(np.isinf(behavior).sum())
    if not allow_behavior_nans and nan_count:
        msg = f"behavior contains NaN values: {nan_count}"
        raise ValueError(msg)
    if inf_count:
        msg = f"behavior contains Inf values: {inf_count}"
        raise ValueError(msg)

    unique_names = _unique_names(names)
    metadata = {
        "present": True,
        "groups_requested": behavior_signal_types,
        "groups_found": groups_found,
        "missing_groups": missing,
        "dimensions": int(behavior.shape[2]),
        "behavior_names": unique_names,
        "extraction_policy": behavior_variable_length_policy,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "contains_nans": nan_count > 0,
        "original_trial_lengths": lengths,
        "cropped_to_spike_window": cropped,
    }
    return behavior, unique_names, metadata


def apply_behavior_aliases(
    names: list[str],
    behavior: NLBBehaviorSection | None,
) -> tuple[list[str], dict[str, str]]:
    """Rename source behavior channels to canonical names and enforce required channels."""
    if behavior is None:
        return names, {name: name for name in names}
    mapping = {name: behavior.aliases.get(name, name) for name in names}
    canonical = list(mapping.values())
    if len(set(canonical)) != len(canonical):
        msg = f"behavior aliases produced duplicate canonical names: {canonical}"
        raise ValueError(msg)
    missing = [name for name in behavior.required_names if name not in canonical]
    if missing:
        msg = (
            f"required behavior channels are missing after alias mapping: {missing}. "
            f"Available channels: {canonical}"
        )
        raise ValueError(msg)
    return canonical, mapping


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
        behavior = behavior.astype(np.float64, copy=False)
        metadata["behavior"] = {
            "shape": list(behavior.shape),
            "dtype": str(behavior.dtype),
            "present": True,
        }
    if behavior is not None and behavior.ndim == 3:
        behavior_names = [f"behavior_{index}" for index in range(int(behavior.shape[2]))]
    else:
        behavior = None
        behavior_names = None

    dataset = NeuralDataset(
        spikes=_integer_spikes(spikes),
        rates=None if rates is None else rates.astype(np.float64, copy=False),
        latents=None,
        trial_ids=trial_ids.astype(np.int64, copy=False),
        time_ms=time_ms.astype(np.float64, copy=False),
        bin_size_ms=config.dataset.bin_size_ms,
        metadata=metadata,
        behavior=behavior,
        behavior_names=behavior_names,
    )
    validate_neural_dataset(dataset)
    return dataset


def _load_with_nlb_tools(
    data_file: Path,
    config: NLBConfig,
    source_files: list[Path],
) -> NeuralDataset:
    try:
        from nlb_tools.nwb_interface import NWBDataset
    except ImportError as exc:
        raise ImportError(OPTIONAL_INSTALL_MESSAGE) from exc

    try:
        nlb_object = NWBDataset(str(data_file))
    except TypeError:
        try:
            nlb_object = NWBDataset(str(data_file), split_heldout=True)
        except Exception as exc:
            detected = _format_detected_files(source_files)
            msg = (
                "nlb_tools imported and the preferred train file was detected, but "
                "NWBDataset could not be constructed. "
                f"Train file: {data_file.name}. Detected files: {detected}. Error: {exc}"
            )
            raise RuntimeError(msg) from exc
    except Exception as exc:
        detected = _format_detected_files(source_files)
        msg = (
            "nlb_tools imported and the preferred train file was detected, but "
            "NWBDataset could not be loaded. "
            f"Train file: {data_file.name}. Detected files: {detected}. Error: {exc}"
        )
        raise RuntimeError(msg) from exc

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

    trial_data = make_trial_dataframe(nlb_object, config)
    spikes, trial_ids, time_ms, trial_metadata = dataframe_to_trial_tensor(
        trial_data,
        signal_types=config.trialization.signal_types,
        combine_heldout_spikes=config.trialization.combine_heldout_spikes,
        variable_length_policy=config.trialization.variable_length_policy,
        bin_size_ms=config.dataset.bin_size_ms,
    )
    behavior, behavior_names, behavior_metadata = dataframe_to_behavior_tensor(
        trial_data,
        behavior_signal_types=config.trialization.behavior_signal_types,
        trial_ids=trial_ids,
        n_time_bins=spikes.shape[1],
        require_behavior=config.trialization.require_behavior,
        allow_behavior_nans=config.trialization.allow_behavior_nans,
        behavior_variable_length_policy=config.trialization.behavior_variable_length_policy,
    )
    behavior_mapping: dict[str, str] = {}
    if behavior_names is not None:
        behavior_names, behavior_mapping = apply_behavior_aliases(behavior_names, config.behavior)
        behavior_metadata["behavior_names"] = behavior_names
    elif config.behavior is not None and config.behavior.required_names:
        msg = f"required behavior channels are missing: {config.behavior.required_names}"
        raise ValueError(msg)

    trial_metadata["behavior"] = behavior_metadata
    trial_metadata["ingestion_summary"]["behavior_mapping"] = behavior_mapping
    metadata: dict[str, Any] = {
        "dataset_name": config.dataset.name,
        "source": config.dataset.source,
        "variant": config.dataset.variant,
        "alignment_event": config.dataset.alignment_event,
        "expected_format": config.dataset.expected_format,
        "bin_size_ms": config.dataset.bin_size_ms,
        "source_files": [path.name for path in source_files],
        "source_file": data_file.name,
        "processed_target_source_file": data_file.name,
        "trialization_settings": config.trialization.model_dump(mode="json"),
        **trial_metadata,
    }
    dataset = NeuralDataset(
        spikes=spikes,
        rates=None,
        latents=None,
        trial_ids=trial_ids,
        time_ms=time_ms,
        bin_size_ms=config.dataset.bin_size_ms,
        metadata=metadata,
        behavior=behavior,
        behavior_names=behavior_names,
    )
    validate_neural_dataset(dataset)
    return dataset


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

    detected_variants = enforce_dataset_variant(loadable_files, config)
    train_file = select_train_nwb_file(loadable_files, config.trialization.train_file_pattern)
    test_files = select_test_nwb_files(files, config.trialization.test_file_pattern)
    dataset = _load_with_nlb_tools(train_file, config, files)
    summary = dataset.metadata.setdefault("ingestion_summary", {})
    summary["detected_variants"] = detected_variants
    dataset.metadata["test_source_files"] = [path.name for path in test_files]
    dataset.metadata["test_files_used_for_targets"] = False
    validate_neural_dataset_minimums(
        dataset,
        min_trials=config.validation.min_trials,
        min_neurons=config.validation.min_neurons,
        min_time_bins=config.validation.min_time_bins,
    )
    if config.validation.require_rates and dataset.rates is None:
        msg = "NLB config requires rates, but the loaded dataset did not provide rates"
        raise ValueError(msg)
    if config.validation.require_behavior and dataset.behavior is None:
        msg = "NLB config requires behavior, but none was loaded"
        raise ValueError(msg)
    return dataset
