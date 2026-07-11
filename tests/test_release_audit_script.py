from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_release_audit.py"
    spec = importlib.util.spec_from_file_location("run_release_audit_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_release_config_returns_two(tmp_path: Path) -> None:
    assert _module().main(["--config", str(tmp_path / "missing.yaml")]) == 2
