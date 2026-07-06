from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.table import Table

from latentbrain.data.nlb import CANDIDATE_FILE_SUFFIXES, LOADABLE_FILE_SUFFIXES
from latentbrain.paths import get_repo_root

stdout = Console()
stderr = Console(stderr=True)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _candidate_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in CANDIDATE_FILE_SUFFIXES
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect local NLB candidate files.")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)

    repo_root = get_repo_root()
    root = args.root if args.root.is_absolute() else repo_root / args.root
    root = root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        stderr.print(f"NLB directory does not exist: {_relative(root, repo_root)}")
        stderr.print("Create it and place legally downloaded MC_Maze Small files there first.")
        return 2

    files = _candidate_files(root)
    if not files:
        stderr.print(f"No candidate NLB files found under {_relative(root, repo_root)}")
        stderr.print("Expected one of: .nwb, .h5, .hdf5, .mat, .npz")
        stderr.print("Start with MC_Maze Small from https://gui.dandiarchive.org/#/dandiset/000140")
        return 2

    table = Table(title="NLB candidate files")
    table.add_column("relative_path")
    table.add_column("size_mb", justify="right")
    table.add_column("extension")
    table.add_column("will_load")
    total_bytes = 0
    for path in files:
        size_bytes = path.stat().st_size
        total_bytes += size_bytes
        suffix = path.suffix.lower()
        table.add_row(
            path.relative_to(root).as_posix(),
            f"{size_bytes / (1024 * 1024):.3f}",
            suffix,
            str(suffix in LOADABLE_FILE_SUFFIXES),
        )
    stdout.print(table)
    stdout.print(f"total_files: {len(files)}")
    stdout.print(f"total_size_mb: {total_bytes / (1024 * 1024):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
