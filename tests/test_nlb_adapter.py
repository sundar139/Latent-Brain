from __future__ import annotations

import sys
from pathlib import Path

import pytest

from latentbrain.data.adapters import NeuralDataAdapter
from latentbrain.data.nlb import NLBConfig, NLBDataAdapter, load_nlb_dataset


def test_missing_dataset_root_raises_clear_file_not_found(tmp_path: Path) -> None:
    config = NLBConfig.from_yaml(Path("configs/nlb_mc_maze.yaml"))
    missing_root = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="MC_Maze/NLB files were not found"):
        load_nlb_dataset(missing_root, config)


def test_missing_optional_dependency_raises_helpful_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "nlb"
    dataset_root.mkdir()
    (dataset_root / "mc_maze.h5").write_bytes(b"not a real dataset")
    monkeypatch.setitem(sys.modules, "nlb_tools", None)

    config = NLBConfig.from_yaml(Path("configs/nlb_mc_maze.yaml"))
    with pytest.raises(ImportError, match="python -m pip install -e"):
        load_nlb_dataset(dataset_root, config)


def test_adapter_does_not_silently_create_fake_data(tmp_path: Path) -> None:
    config = NLBConfig.from_yaml(Path("configs/nlb_mc_maze.yaml"))
    adapter: NeuralDataAdapter = NLBDataAdapter(config)

    assert not adapter.can_load(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        adapter.load(tmp_path / "missing")
    assert not (tmp_path / "missing").exists()
