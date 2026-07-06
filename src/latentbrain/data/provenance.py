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
    """Collect relative file names, sizes, and hashes for a local dataset root."""
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


def write_provenance(
    dataset_name: str,
    dataset_root: Path,
    output_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Write a provenance JSON document for a local dataset preparation run."""
    manifest = collect_file_manifest(dataset_root)
    provenance: dict[str, Any] = {
        "dataset_name": dataset_name,
        "dataset_root": str(dataset_root.expanduser().resolve()),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "file_count": len(manifest),
        "files": manifest,
        "config": config,
        "package_version": __version__,
        "git_commit": _git_commit(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return provenance
