from __future__ import annotations

import sys
from pathlib import Path

import pytest

from latentbrain.data.adapters import NeuralDataAdapter
from latentbrain.data.nlb import (
    NLBConfig,
    NLBDataAdapter,
    find_candidate_nlb_files,
    load_nlb_dataset,
)


def test_missing_dataset_root_raises_clear_file_not_found(tmp_path: Path) -> None:
    config = NLBConfig.from_yaml(Path("configs/nlb_mc_maze.yaml"))
    missing_root = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="dataset root does not exist"):
        load_nlb_dataset(missing_root, config)


def test_empty_dataset_root_raises_clear_file_not_found(tmp_path: Path) -> None:
    config = NLBConfig.from_yaml(Path("configs/nlb_mc_maze.yaml"))

    with pytest.raises(FileNotFoundError, match="no candidate MC_Maze/NLB files"):
        load_nlb_dataset(tmp_path, config)


def test_find_candidate_nlb_files_detects_supported_extensions(tmp_path: Path) -> None:
    for name in ["a.nwb", "b.h5", "c.hdf5", "d.mat", "e.npz", "ignore.txt"]:
        (tmp_path / name).write_bytes(b"x")

    candidates = [path.name for path in find_candidate_nlb_files(tmp_path)]

    assert candidates == ["a.nwb", "b.h5", "c.hdf5", "d.mat", "e.npz"]


def test_missing_optional_dependency_raises_helpful_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "mc_maze.h5").write_bytes(b"not a real dataset")
    monkeypatch.setitem(sys.modules, "nlb_tools", None)

    config = NLBConfig.from_yaml(Path("configs/nlb_mc_maze.yaml"))
    with pytest.raises(ImportError, match="git\+https://github.com/neurallatents/nlb_tools.git"):
        load_nlb_dataset(tmp_path, config)


def test_adapter_does_not_silently_create_fake_data(tmp_path: Path) -> None:
    config = NLBConfig.from_yaml(Path("configs/nlb_mc_maze.yaml"))
    adapter: NeuralDataAdapter = NLBDataAdapter(config)

    assert not adapter.can_load(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        adapter.load(tmp_path / "missing")
    assert not (tmp_path / "missing").exists()


def test_loader_rejects_unsupported_candidate_without_fake_success(tmp_path: Path) -> None:
    (tmp_path / "mc_maze.mat").write_bytes(b"real file, unsupported by current loader")
    config = NLBConfig.from_yaml(Path("configs/nlb_mc_maze_small.yaml"))

    with pytest.raises(ValueError, match="unsupported candidate files"):
        load_nlb_dataset(tmp_path, config)
