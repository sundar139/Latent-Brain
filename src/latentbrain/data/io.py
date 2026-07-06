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
    excluded = {"dataset_hash", "generated_at_utc", "provenance"}
    return {key: value for key, value in metadata.items() if key not in excluded}


def compute_dataset_hash(dataset: NeuralDataset) -> str:
    """Compute a stable dataset hash from spikes and important metadata."""
    validate_neural_dataset(dataset)
    payload = {
        "spikes_hash": compute_array_hash(dataset.spikes),
        "metadata": _stable_metadata(dataset.metadata),
        "shape": list(dataset.spikes.shape),
        "bin_size_ms": dataset.bin_size_ms,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_neural_dataset(
    dataset: NeuralDataset,
    output_path: Path,
    metadata_path: Path | None = None,
) -> None:
    """Validate and save a neural dataset as compressed NumPy arrays."""
    validate_neural_dataset(dataset)
    dataset.metadata["dataset_hash"] = compute_dataset_hash(dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_json = json.dumps(dataset.metadata, sort_keys=True)
    np.savez_compressed(
        output_path,
        spikes=dataset.spikes,
        rates=np.array([], dtype=np.float64) if dataset.rates is None else dataset.rates,
        latents=np.array([], dtype=np.float64) if dataset.latents is None else dataset.latents,
        trial_ids=dataset.trial_ids,
        time_ms=dataset.time_ms,
        bin_size_ms=np.array(dataset.bin_size_ms, dtype=np.int64),
        metadata=np.array(metadata_json),
    )
    if metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(metadata_json + "\n", encoding="utf-8")


def load_neural_dataset(path: Path) -> NeuralDataset:
    """Load and validate a neural dataset saved by save_neural_dataset."""
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        rates = data["rates"]
        latents = data["latents"]
        dataset = NeuralDataset(
            spikes=data["spikes"],
            rates=None if rates.size == 0 else rates,
            latents=None if latents.size == 0 else latents,
            trial_ids=data["trial_ids"],
            time_ms=data["time_ms"],
            bin_size_ms=int(data["bin_size_ms"].item()),
            metadata=metadata,
        )
    validate_neural_dataset(dataset)
    expected_hash = compute_dataset_hash(dataset)
    if dataset.metadata.get("dataset_hash") != expected_hash:
        msg = "dataset hash does not match loaded content"
        raise ValueError(msg)
    return dataset
