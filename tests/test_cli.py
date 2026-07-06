from __future__ import annotations

from typer.testing import CliRunner

from latentbrain.cli import app

runner = CliRunner()


def test_validate_config_exits_successfully() -> None:
    result = runner.invoke(app, ["validate-config"])

    assert result.exit_code == 0, result.output
    assert "Configuration is valid" in result.output


def test_info_exits_successfully_and_hides_env_secrets() -> None:
    result = runner.invoke(
        app,
        ["info"],
        env={"WANDB_API_KEY": "secret-value-that-must-not-appear"},
    )

    assert result.exit_code == 0, result.output
    assert "LatentBrain" in result.output
    assert "secret-value-that-must-not-appear" not in result.output
    assert "WANDB_API_KEY" not in result.output
