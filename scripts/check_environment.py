from __future__ import annotations

import platform
import sys
from typing import NoReturn

from rich.console import Console
from rich.table import Table

from latentbrain import __version__
from latentbrain.config import ConfigError, load_config
from latentbrain.paths import get_repo_root, resolve_configured_path
from latentbrain.randomness import seed_everything

console = Console()


def _fail(message: str) -> NoReturn:
    console.print(f"[red]Environment check failed:[/red] {message}", stderr=True)
    raise SystemExit(1)


def main() -> int:
    try:
        config = load_config()
    except ConfigError as exc:
        _fail(str(exc))

    try:
        seed_status = seed_everything(
            config.project.seed,
            deterministic=config.reproducibility.deterministic,
        )
    except ValueError as exc:
        _fail(str(exc))

    repo_root = get_repo_root()

    table = Table(title="LatentBrain Environment")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("package_version", __version__)
    table.add_row("python_version", sys.version.split()[0])
    table.add_row("python_implementation", platform.python_implementation())
    table.add_row("repository_root", str(repo_root))
    table.add_row("config_seed", str(config.project.seed))
    table.add_row("data_root", str(resolve_configured_path(config.paths.data_root, repo_root)))
    table.add_row(
        "results_root",
        str(resolve_configured_path(config.paths.results_root, repo_root)),
    )
    table.add_row("deterministic", str(seed_status["deterministic"]))
    table.add_row("torch_seeded", str(seed_status["torch"]))

    console.print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
