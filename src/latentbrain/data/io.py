from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from latentbrain.data.schemas import NeuralDataset
from latentbrain.data.validation import validate_neural_dataset


def compute_array_hash(array: np.ndarray) -> str:
    """Compute a SHA-256 hash over array dtype, shape, and bytes."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(json.dumps(contiguous.shape).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _stable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    # ingestion_summary holds derived, non-identifying descriptions of the same arrays,
    # so it stays out of the hash payload and never invalidates an expected_hash.
    excluded = {"dataset_hash", "generated_at_utc", "provenance", "ingestion_summary"}
    return {key: value for key, value in metadata.items() if key not in excluded}


def compute_dataset_hash(dataset: NeuralDataset) -> str:
    """Compute a stable dataset hash from arrays and important metadata."""
    validate_neural_dataset(dataset)
    payload = {
        "spikes_hash": compute_array_hash(dataset.spikes),
        "metadata": _stable_metadata(dataset.metadata),
        "shape": list(dataset.spikes.shape),
        "bin_size_ms": dataset.bin_size_ms,
    }
    if dataset.behavior is not None:
        payload["behavior_hash"] = compute_array_hash(dataset.behavior)
        payload["behavior_names"] = dataset.behavior_names
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_neural_dataset(
    dataset: NeuralDataset,
    output_path: Path,
    metadata_path: Path | None = None,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> None:
    """Validate and save a neural dataset as compressed NumPy arrays.

    extra_arrays are stored alongside the dataset (e.g. split and mask indices) and
    are not part of the dataset hash payload.
    """
    validate_neural_dataset(dataset)
    if dataset.behavior is not None:
        dataset.metadata["dataset_hash_includes_behavior"] = True
    dataset.metadata["dataset_hash"] = compute_dataset_hash(dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_json = json.dumps(dataset.metadata, sort_keys=True)
    arrays: dict[str, Any] = {
        "spikes": dataset.spikes,
        "rates": np.array([], dtype=np.float64) if dataset.rates is None else dataset.rates,
        "latents": np.array([], dtype=np.float64) if dataset.latents is None else dataset.latents,
        "trial_ids": dataset.trial_ids,
        "time_ms": dataset.time_ms,
        "bin_size_ms": np.array(dataset.bin_size_ms, dtype=np.int64),
        "metadata": np.array(metadata_json),
    }
    if dataset.behavior is not None:
        arrays["behavior"] = dataset.behavior
        arrays["behavior_names"] = np.asarray(dataset.behavior_names, dtype=np.str_)
    if extra_arrays:
        conflicts = sorted(set(extra_arrays) & set(arrays))
        if conflicts:
            msg = f"extra_arrays must not override dataset arrays: {conflicts}"
            raise ValueError(msg)
        arrays.update(extra_arrays)
    np.savez_compressed(output_path, **arrays)
    if metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(metadata_json + "\n", encoding="utf-8")


def load_neural_dataset(path: Path) -> NeuralDataset:
    """Load and validate a neural dataset saved by save_neural_dataset."""
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        rates = data["rates"]
        latents = data["latents"]
        behavior = data["behavior"] if "behavior" in data.files else None
        behavior_names = (
            [str(value) for value in data["behavior_names"].tolist()]
            if "behavior_names" in data.files
            else None
        )
        dataset = NeuralDataset(
            spikes=data["spikes"],
            rates=None if rates.size == 0 else rates,
            latents=None if latents.size == 0 else latents,
            trial_ids=data["trial_ids"],
            time_ms=data["time_ms"],
            bin_size_ms=int(data["bin_size_ms"].item()),
            metadata=metadata,
            behavior=behavior,
            behavior_names=behavior_names,
        )
    validate_neural_dataset(dataset)
    expected_hash = compute_dataset_hash(dataset)
    if dataset.metadata.get("dataset_hash") != expected_hash:
        msg = "dataset hash does not match loaded content"
        raise ValueError(msg)
    return dataset
