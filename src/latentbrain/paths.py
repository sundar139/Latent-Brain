from __future__ import annotations

import os
from pathlib import Path


def get_repo_root() -> Path:
    """Resolve the repository root from an override or installed source path."""
    override = os.getenv("LATENTBRAIN_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() and (parent / "configs").is_dir():
            return parent

    return current.parents[2]


def resolve_configured_path(path_value: str | Path, repo_root: Path | None = None) -> Path:
    """Resolve a configured path relative to the repository root unless absolute."""
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    root = get_repo_root() if repo_root is None else repo_root
    return (root / candidate).resolve()


def ensure_directory(path: Path) -> Path:
    """Create a directory when it does not exist and return the resolved path."""
    resolved = path.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
