from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console

from latentbrain.data.nlb import (
    LOADABLE_FILE_SUFFIXES,
    NLBConfig,
    find_candidate_nlb_files,
    inspect_nlb_candidates,
    resolve_nlb_dataset_root,
)
from latentbrain.paths import get_repo_root

stdout = Console()
stderr = Console(stderr=True)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _print_missing(config: NLBConfig | None, root: Path, repo_root: Path) -> None:
    source = None if config is None or config.source is None else config.source.model_dump()
    if not root.exists() or not root.is_dir():
        stderr.print(f"NLB directory does not exist: {_relative(root, repo_root)}")
    stderr.print(
        json.dumps(
            {
                "status": "missing_raw_data",
                "dataset": None if config is None else config.dataset.name,
                "expected_raw_dir": _relative(root, repo_root),
                "verified_source_metadata": source,
                "candidate_assets": []
                if source is None
                else [asset["path"] for asset in source["expected_assets"]],
                "automatic_download_performed": False,
            },
            indent=2,
        )
    )
    stderr.print("No data was downloaded and no fake data was created.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect local NLB candidate files.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args(argv)
    if args.config is None and args.root is None:
        parser.error("one of --config or --root is required")

    repo_root = get_repo_root()
    config = None
    if args.config is not None:
        config_path = args.config if args.config.is_absolute() else repo_root / args.config
        config = NLBConfig.from_yaml(config_path)

    if args.root is not None:
        root = (args.root if args.root.is_absolute() else repo_root / args.root).expanduser()
        root = root.resolve()
    else:
        assert config is not None
        root = resolve_nlb_dataset_root(config, repo_root)

    if not root.exists() or not root.is_dir() or not find_candidate_nlb_files(root):
        _print_missing(config, root, repo_root)
        return 2

    if config is None:
        files = find_candidate_nlb_files(root)
        for path in files:
            stdout.print(
                json.dumps(
                    {
                        "path": _relative(path, repo_root),
                        "filename": path.name,
                        "size_bytes": path.stat().st_size,
                        "will_load": path.suffix.lower() in LOADABLE_FILE_SUFFIXES,
                    }
                )
            )
        stdout.print(f"total_files: {len(files)}")
        return 0

    records = inspect_nlb_candidates(root, config)
    for record in records:
        record["path"] = _relative(Path(record["path"]), repo_root)
        stdout.print(json.dumps(record, indent=2, sort_keys=True))
    stdout.print(f"total_files: {len(records)}")
    stdout.print(f"total_size_bytes: {sum(int(record['size_bytes']) for record in records)}")
    mismatched = sorted(
        {
            str(record["variant_candidate"])
            for record in records
            if record["variant_candidate"] not in (None, config.dataset.variant)
        }
    )
    if mismatched:
        stderr.print(f"variant mismatch: configured {config.dataset.variant}, found {mismatched}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
