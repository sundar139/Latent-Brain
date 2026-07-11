from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _script_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_latent_interpretability.py"
    spec = importlib.util.spec_from_file_location("run_latent_interpretability_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_missing_config_returns_two(tmp_path: Path) -> None:
    assert _script_module().main(["--config", str(tmp_path / "missing.yaml")]) == 2
