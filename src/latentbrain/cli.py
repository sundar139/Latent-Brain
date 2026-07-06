from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from latentbrain import __version__
from latentbrain.config import ConfigError, load_config
from latentbrain.logging_utils import configure_logging
from latentbrain.paths import get_repo_root, resolve_configured_path

app = typer.Typer(help="LatentBrain command line interface.")
console = Console()
error_console = Console(stderr=True)

ConfigPathOption = Annotated[
    Path | None,
    typer.Option(
        "--config",
        "-c",
        help="Optional path to a YAML configuration file.",
    ),
]


def _exit_with_config_error(exc: ConfigError) -> None:
    error_console.print(f"[red]Configuration error:[/red] {exc}")
    raise typer.Exit(code=1) from exc


@app.command("validate-config")
def validate_config_command(config_path: ConfigPathOption = None) -> None:
    """Validate the active configuration."""
    try:
        load_config(config_path)
    except ConfigError as exc:
        _exit_with_config_error(exc)
    console.print("Configuration is valid.")


@app.command("info")
def info_command(config_path: ConfigPathOption = None) -> None:
    """Display safe project metadata."""
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        _exit_with_config_error(exc)

    configure_logging(level=config.logging.level, json=config.logging.json_enabled)
    repo_root = get_repo_root()

    table = Table(title="LatentBrain Project Information")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value")
    table.add_row("project_name", config.project.name)
    table.add_row("package_version", __version__)
    table.add_row("python_version", sys.version.split()[0])
    table.add_row("repository_root", str(repo_root))
    table.add_row("configured_seed", str(config.project.seed))
    table.add_row("data_root", str(resolve_configured_path(config.paths.data_root, repo_root)))
    table.add_row(
        "results_root",
        str(resolve_configured_path(config.paths.results_root, repo_root)),
    )
    table.add_row("deterministic_mode", str(config.reproducibility.deterministic))
    console.print(table)


if __name__ == "__main__":
    app()
