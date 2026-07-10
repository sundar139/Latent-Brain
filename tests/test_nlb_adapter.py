from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from latentbrain.data.adapters import NeuralDataAdapter
from latentbrain.data.nlb import (
    NLBBehaviorSection,
    NLBConfig,
    NLBDataAdapter,
    apply_behavior_aliases,
    describe_nwb_file,
    detect_variant,
    enforce_dataset_variant,
    find_candidate_nlb_files,
    inspect_nlb_candidates,
    load_nlb_dataset,
)

LARGE_CONFIG = Path("configs/nlb_mc_maze_large.yaml")
SMALL_TRAIN = "sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb"
LARGE_TRAIN = "sub-Jenkins_ses-large_desc-train_behavior+ecephys.nwb"


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


def test_large_config_carries_verified_dandi_source() -> None:
    config = NLBConfig.from_yaml(LARGE_CONFIG)

    assert config.dataset.variant == "large"
    assert config.source is not None
    assert config.source.provider == "dandi"
    assert config.source.dandiset_id == "000138"
    assert config.source.automatic_download is False
    assert [asset.path for asset in config.source.expected_assets] == [
        "sub-Jenkins/sub-Jenkins_ses-large_desc-train_behavior+ecephys.nwb",
        "sub-Jenkins/sub-Jenkins_ses-large_desc-test_ecephys.nwb",
    ]


def test_config_rejects_source_assets_from_another_variant() -> None:
    raw = yaml.safe_load(LARGE_CONFIG.read_text(encoding="utf-8"))
    raw["source"]["expected_assets"][0]["path"] = f"sub-Jenkins/{SMALL_TRAIN}"

    with pytest.raises(ValueError, match="does not belong to the configured variant"):
        NLBConfig.model_validate(raw)


def test_config_rejects_automatic_download() -> None:
    raw = yaml.safe_load(LARGE_CONFIG.read_text(encoding="utf-8"))
    raw["source"]["automatic_download"] = True

    with pytest.raises(ValueError, match="never downloads raw data"):
        NLBConfig.model_validate(raw)


def test_variant_detection_prefers_nwb_metadata_over_filename(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "session_without_variant_token.nwb"
    with h5py.File(path, "w") as handle:
        handle["identifier"] = "jenkins-session"
        handle["session_description"] = "MC_Maze_Large delayed reaching"
        handle.create_dataset("general/subject/subject_id", data="Jenkins")
        handle.create_group("acquisition/ElectricalSeries")
        handle.create_dataset("intervals/trials/start_time", data=[0.0, 1.0, 2.0])
        handle.create_dataset("units/id", data=[0, 1])
        handle.create_group("processing/behavior/Position/hand_pos")

    info = describe_nwb_file(path)
    variant, evidence = detect_variant(info, path)

    assert info["metadata_read"] is True
    assert info["subject_id"] == "Jenkins"
    assert info["trial_table_present"] is True
    assert info["trial_count_if_available"] == 3
    assert info["unit_count_if_available"] == 2
    assert info["available_acquisition_series"] == ["ElectricalSeries"]
    assert info["behavior_series_candidates"] == ["Position/hand_pos"]
    assert (variant, evidence) == ("large", "nwb_metadata")


def test_variant_detection_falls_back_to_filename(tmp_path: Path) -> None:
    small = tmp_path / SMALL_TRAIN
    large = tmp_path / LARGE_TRAIN
    small.write_bytes(b"x")
    large.write_bytes(b"x")

    assert detect_variant(describe_nwb_file(small), small) == ("small", "filename")
    assert detect_variant(describe_nwb_file(large), large) == ("large", "filename")
    assert detect_variant({}, tmp_path / "unknown.nwb") == (None, "unknown")


def test_large_request_rejects_small_files(tmp_path: Path) -> None:
    (tmp_path / SMALL_TRAIN).write_bytes(b"x")
    config = NLBConfig.from_yaml(LARGE_CONFIG)

    with pytest.raises(ValueError, match="files belong to variant 'small'"):
        enforce_dataset_variant([tmp_path / SMALL_TRAIN], config)


def test_mixed_incompatible_sessions_fail_clearly(tmp_path: Path) -> None:
    files = [tmp_path / SMALL_TRAIN, tmp_path / LARGE_TRAIN]
    for path in files:
        path.write_bytes(b"x")
    config = NLBConfig.from_yaml(LARGE_CONFIG)

    with pytest.raises(ValueError, match="multiple incompatible MC_Maze sessions"):
        enforce_dataset_variant(files, config)


def test_inspection_records_expected_fields(tmp_path: Path) -> None:
    (tmp_path / LARGE_TRAIN).write_bytes(b"x")
    config = NLBConfig.from_yaml(LARGE_CONFIG)

    records = inspect_nlb_candidates(tmp_path, config)

    assert len(records) == 1
    record = records[0]
    for field in (
        "dataset_candidate",
        "variant_candidate",
        "path",
        "filename",
        "size_bytes",
        "nwb_identifier",
        "session_description",
        "subject_id",
        "available_acquisition_series",
        "available_processing_modules",
        "trial_table_present",
        "trial_count_if_available",
        "unit_count_if_available",
        "behavior_series_candidates",
    ):
        assert field in record
    assert record["dataset_candidate"] == "mc_maze_large"
    assert record["variant_candidate"] == "large"


def test_behavior_aliases_map_source_names_to_canonical() -> None:
    behavior = NLBBehaviorSection(
        aliases={"hand_position_x": "hand_pos_x"},
        required_names=["hand_pos_x"],
    )

    names, mapping = apply_behavior_aliases(["hand_position_x"], behavior)

    assert names == ["hand_pos_x"]
    assert mapping == {"hand_position_x": "hand_pos_x"}


def test_missing_required_behavior_fails_clearly() -> None:
    behavior = NLBBehaviorSection(required_names=["cursor_pos_x"])

    with pytest.raises(ValueError, match="required behavior channels are missing"):
        apply_behavior_aliases(["hand_pos_x"], behavior)
