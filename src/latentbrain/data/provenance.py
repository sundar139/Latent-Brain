from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from latentbrain import __version__

DEFAULT_HASH_SIZE_LIMIT_BYTES = 100 * 1024 * 1024


def compute_file_sha256(path: Path) -> str:
    """Compute a SHA-256 digest for a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_file_manifest(
    root: Path,
    max_hash_size_bytes: int = DEFAULT_HASH_SIZE_LIMIT_BYTES,
) -> list[dict[str, str | int]]:
    """Collect relative file names, sizes, and size-limited hashes for a local dataset root."""
    resolved_root = root.expanduser().resolve()
    if not resolved_root.exists():
        msg = f"dataset root does not exist: {resolved_root}"
        raise FileNotFoundError(msg)
    if not resolved_root.is_dir():
        msg = f"dataset root is not a directory: {resolved_root}"
        raise NotADirectoryError(msg)

    manifest: list[dict[str, str | int]] = []
    for path in sorted(item for item in resolved_root.rglob("*") if item.is_file()):
        size_bytes = path.stat().st_size
        entry: dict[str, str | int] = {
            "relative_path": path.relative_to(resolved_root).as_posix(),
            "size_bytes": size_bytes,
        }
        if size_bytes <= max_hash_size_bytes:
            entry["sha256"] = compute_file_sha256(path)
        else:
            entry["sha256"] = "skipped:size_exceeds_limit"
            entry["hash_skipped_reason"] = "size_exceeds_limit"
            entry["hash_size_limit_bytes"] = max_hash_size_bytes
        manifest.append(entry)
    return manifest


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _dataset_config(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config.get("dataset", {})
    return dataset if isinstance(dataset, dict) else {}


def _split_config(config: dict[str, Any]) -> dict[str, Any]:
    splits = config.get("splits", {})
    return splits if isinstance(splits, dict) else {}


def _trialization_config(config: dict[str, Any]) -> dict[str, Any]:
    trialization = config.get("trialization", {})
    return trialization if isinstance(trialization, dict) else {}


def _dandiset_id(dataset_root: Path, manifest: list[dict[str, str | int]]) -> str | None:
    for part in dataset_root.parts:
        if part.isdigit() and len(part) == 6:
            return part
    for entry in manifest:
        first_part = str(entry["relative_path"]).split("/", maxsplit=1)[0]
        if first_part.isdigit() and len(first_part) == 6:
            return first_part
    return None


def write_provenance(
    dataset_name: str,
    dataset_root: Path,
    output_path: Path,
    config: dict[str, Any],
    max_hash_size_bytes: int = DEFAULT_HASH_SIZE_LIMIT_BYTES,
    dataset_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a provenance JSON document for a local dataset preparation run."""
    manifest = collect_file_manifest(dataset_root, max_hash_size_bytes=max_hash_size_bytes)
    dataset_config = _dataset_config(config)
    split_config = _split_config(config)
    trialization_config = _trialization_config(config)
    dataset_metadata = dataset_metadata or {}
    provenance: dict[str, Any] = {
        "dataset_name": dataset_name,
        "variant": dataset_config.get("variant"),
        "source": dataset_config.get("source"),
        "dandiset_id": _dandiset_id(dataset_root, manifest),
        "dataset_root": str(dataset_root.expanduser().resolve()),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "file_count": len(manifest),
        "files": manifest,
        "config": config,
        "package_version": __version__,
        "git_commit": _git_commit(),
        "split_seed": split_config.get("seed"),
        "heldout_mask_seed": split_config.get("seed"),
        "bin_size_ms": dataset_config.get("bin_size_ms"),
        "alignment_event": dataset_config.get("alignment_event"),
        "hash_size_limit_bytes": max_hash_size_bytes,
        "train_file_used": dataset_metadata.get("processed_target_source_file"),
        "test_files_detected": dataset_metadata.get("test_source_files", []),
        "test_files_used_for_targets": dataset_metadata.get("test_files_used_for_targets", False),
        "trialization": trialization_config,
        "variable_length_policy": trialization_config.get("variable_length_policy"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return provenance
