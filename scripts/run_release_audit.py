from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from latentbrain.eval.release_audit import run_release_audit
from latentbrain.paths import get_repo_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate final LatentBrain release evidence.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.config.exists():
        print(f"Release config is missing: {args.config}")
        return 2
    try:
        loaded = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("release config must contain a mapping")
        result = run_release_audit(dict(loaded), get_repo_root())
    except (FileNotFoundError, KeyError, ValueError, yaml.YAMLError) as error:
        print(str(error))
        return 2
    print(json.dumps({**result, "output_dir": str(result["output_dir"])}, indent=2))
    return 0 if result["readiness"]["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
