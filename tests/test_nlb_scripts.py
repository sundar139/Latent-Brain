from __future__ import annotations

import importlib.util
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_script(path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _main(path: str) -> Callable[[Sequence[str] | None], int]:
    return _load_script(path).main


def test_inspect_nlb_files_missing_root_fails_gracefully(
    tmp_path: Path,
    capsys: Any,
) -> None:
    inspect_main = _main("scripts/inspect_nlb_files.py")
    exit_code = inspect_main(["--root", str(tmp_path / "missing")])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "directory does not exist" in captured.err


def test_inspect_nlb_files_lists_candidates(tmp_path: Path, capsys: Any) -> None:
    inspect_main = _main("scripts/inspect_nlb_files.py")
    (tmp_path / "session.nwb").write_bytes(b"abc")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    exit_code = inspect_main(["--root", str(tmp_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "session.nwb" in captured.out
    assert "will_load" in captured.out
    assert "total_files: 1" in captured.out


def test_prepare_nlb_data_missing_small_data_fails_gracefully(capsys: Any) -> None:
    prepare_main = _main("scripts/prepare_nlb_data.py")
    argv: Sequence[str] = ["--config", "configs/nlb_mc_maze_small.yaml"]
    exit_code = prepare_main(argv)
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "MC_Maze Small" in captured.err
    assert "data/raw/nlb/mc_maze_small" in captured.err
    assert "No fake data or processed output was created" in captured.err
