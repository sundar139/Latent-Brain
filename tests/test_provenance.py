from __future__ import annotations

from pathlib import Path

from latentbrain.data.provenance import (
    collect_file_manifest,
    compute_file_sha256,
    write_provenance,
)


def test_file_manifest_captures_names_and_sizes(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    nested = tmp_path / "nested"
    nested.mkdir()
    second = nested / "second.txt"
    first.write_text("abc", encoding="utf-8")
    second.write_text("abcd", encoding="utf-8")

    manifest = collect_file_manifest(tmp_path)

    assert {entry["relative_path"] for entry in manifest} == {"first.txt", "nested/second.txt"}
    assert {entry["size_bytes"] for entry in manifest} == {3, 4}


def test_sha256_changes_when_content_changes(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("one", encoding="utf-8")
    first_hash = compute_file_sha256(path)
    path.write_text("two", encoding="utf-8")

    assert compute_file_sha256(path) != first_hash


def test_write_provenance_includes_core_metadata(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "source.h5").write_bytes(b"source")
    output_path = tmp_path / "metadata.json"

    provenance = write_provenance(
        dataset_name="mc_maze",
        dataset_root=dataset_root,
        output_path=output_path,
        config={"dataset": {"name": "mc_maze"}},
    )

    assert output_path.exists()
    assert provenance["dataset_name"] == "mc_maze"
    assert provenance["file_count"] == 1
    assert provenance["config"] == {"dataset": {"name": "mc_maze"}}
    assert "generated_at_utc" in provenance
    assert "git_commit" in provenance
