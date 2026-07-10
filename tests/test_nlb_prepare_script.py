from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import yaml

from latentbrain.data.nlb import NLBConfig
from latentbrain.data.schemas import NeuralDataset

LARGE_CONFIG = Path("configs/nlb_mc_maze_large.yaml")


def _script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "prepare_nlb_data", Path("scripts/prepare_nlb_data.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _mock_dataset() -> NeuralDataset:
    rng = np.random.default_rng(0)
    spikes = rng.integers(0, 3, size=(8, 6, 5)).astype(np.int64)
    behavior = rng.normal(size=(8, 6, 4))
    return NeuralDataset(
        spikes=spikes,
        rates=None,
        latents=None,
        trial_ids=np.arange(8, dtype=np.int64),
        time_ms=np.arange(6, dtype=np.float64) * 5.0,
        bin_size_ms=5,
        metadata={
            "dataset_name": "mc_maze_large",
            "source_files": ["sub-Jenkins_ses-large_desc-train_behavior+ecephys.nwb"],
            "processed_target_source_file": (
                "sub-Jenkins_ses-large_desc-train_behavior+ecephys.nwb"
            ),
            "ingestion_summary": {
                "spike_conservation": {
                    "raw_spike_count": int(spikes.sum()),
                    "trialized_spike_count": int(spikes.sum()),
                    "excluded_spike_count": 0,
                    "excluded_bins": 0,
                    "conserved": True,
                    "exclusion_reason": None,
                }
            },
        },
        behavior=behavior,
        behavior_names=["hand_pos_x", "hand_pos_y", "cursor_pos_x", "cursor_pos_y"],
    )


def metadata_hash(processed: Path) -> str:
    text = (processed / "mc_maze_large_metadata.json").read_text("utf-8")
    return str(json.loads(text)["dataset_hash"])


def _local_config(tmp_path: Path) -> Path:
    raw = yaml.safe_load(LARGE_CONFIG.read_text(encoding="utf-8"))
    raw["dataset"]["processed_root"] = str(tmp_path / "processed")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_missing_raw_data_exits_with_documented_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    module = _script_module()
    monkeypatch.setenv("LATENTBRAIN_NLB_ROOT", str(tmp_path / "missing"))

    assert module.main(["--config", str(LARGE_CONFIG)]) == 2

    captured = capsys.readouterr()
    assert '"status": "missing_raw_data"' in captured.err
    assert '"automatic_download_performed": false' in captured.err
    assert "000138" in captured.err
    assert "No data was downloaded" in captured.err
    assert not (tmp_path / "missing").exists()


def test_missing_raw_data_check_precedes_optional_dependency_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script_module()
    monkeypatch.setenv("LATENTBRAIN_NLB_ROOT", str(tmp_path / "missing"))
    monkeypatch.setitem(sys.modules, "nlb_tools", None)
    monkeypatch.setitem(sys.modules, "h5py", None)

    assert module.main(["--config", str(LARGE_CONFIG)]) == 2


def test_mock_large_ingestion_writes_outputs_and_prints_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    module = _script_module()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "sub-Jenkins_ses-large_desc-train_behavior+ecephys.nwb").write_bytes(b"x")
    monkeypatch.setenv("LATENTBRAIN_NLB_ROOT", str(raw_dir))
    monkeypatch.setattr(module, "load_nlb_dataset", lambda _root, _config: _mock_dataset())
    config_path = _local_config(tmp_path)

    assert module.main(["--config", str(config_path)]) == 0

    out = capsys.readouterr().out
    for field in (
        "status",
        "dataset",
        "variant",
        "source_files_used",
        "spikes_shape",
        "behavior_shape",
        "behavior_names",
        "bin_size_ms",
        "trial_count",
        "neuron_count",
        "time_bins",
        "train_trials",
        "validation_trials",
        "test_trials",
        "heldin_neurons",
        "heldout_neurons",
        "dataset_hash",
        "output",
        "metadata",
        "provenance",
        "warnings",
    ):
        assert f"{field}:" in out

    processed = tmp_path / "processed"
    npz_path = processed / "mc_maze_large_processed.npz"
    assert npz_path.exists()
    assert (processed / "mc_maze_large_metadata.json").exists()
    provenance = json.loads((processed / "mc_maze_large_provenance.json").read_text("utf-8"))
    assert provenance["source_provider"] == "dandi"
    assert provenance["dandiset_id"] == "000138"
    assert provenance["automatic_download_performed"] is False
    assert provenance["config_digest"]
    assert provenance["processed_dataset_hash"] == metadata_hash(processed)
    assert provenance["creation_command"].startswith("python scripts/prepare_nlb_data.py")

    with np.load(npz_path, allow_pickle=False) as data:
        for key in (
            "spikes",
            "behavior",
            "behavior_names",
            "trial_ids",
            "bin_size_ms",
            "train_indices",
            "validation_indices",
            "test_indices",
            "heldin_indices",
            "heldout_indices",
        ):
            assert key in data.files
        train = set(data["train_indices"].tolist())
        validation = set(data["validation_indices"].tolist())
        test = set(data["test_indices"].tolist())
        heldin = set(data["heldin_indices"].tolist())
        heldout = set(data["heldout_indices"].tolist())

    assert train | validation | test == set(range(8))
    assert not (train & validation) and not (train & test) and not (validation & test)
    assert heldin | heldout == set(range(5))
    assert not heldin & heldout

    metadata = json.loads((processed / "mc_maze_large_metadata.json").read_text("utf-8"))
    summary = metadata["ingestion_summary"]
    for field in (
        "dataset_name",
        "dataset_family",
        "variant",
        "trial_count",
        "time_bins",
        "neuron_count",
        "behavior_dim",
        "behavior_names",
        "bin_size_ms",
        "split_counts",
        "heldin_neuron_count",
        "heldout_neuron_count",
        "source_files",
        "source_identifiers",
        "trialization_policy",
        "warnings",
    ):
        assert field in summary
    assert summary["spike_conservation"]["conserved"] is True
    assert metadata["dataset_hash"]


def test_repeated_preparation_reproduces_dataset_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script_module()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "sub-Jenkins_ses-large_desc-train_behavior+ecephys.nwb").write_bytes(b"x")
    monkeypatch.setenv("LATENTBRAIN_NLB_ROOT", str(raw_dir))
    monkeypatch.setattr(module, "load_nlb_dataset", lambda _root, _config: _mock_dataset())
    config_path = _local_config(tmp_path)
    metadata_path = tmp_path / "processed" / "mc_maze_large_metadata.json"

    assert module.main(["--config", str(config_path)]) == 0
    first = json.loads(metadata_path.read_text("utf-8"))["dataset_hash"]
    assert module.main(["--config", str(config_path)]) == 0
    second = json.loads(metadata_path.read_text("utf-8"))["dataset_hash"]

    assert first == second


def test_config_rejects_small_source_for_large_dataset(tmp_path: Path) -> None:
    raw = yaml.safe_load(LARGE_CONFIG.read_text(encoding="utf-8"))
    raw["source"]["expected_assets"] = [
        {
            "path": "sub-Jenkins/sub-Jenkins_ses-small_desc-test_ecephys.nwb",
            "size_bytes": 100,
        }
    ]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="does not belong to the configured variant"):
        NLBConfig.from_yaml(path)
