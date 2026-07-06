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


def test_manifest_records_hash_skip_reason_for_large_files(tmp_path: Path) -> None:
    path = tmp_path / "large.h5"
    path.write_bytes(b"abcdef")

    manifest = collect_file_manifest(tmp_path, max_hash_size_bytes=3)

    assert manifest[0]["sha256"] == "skipped:size_exceeds_limit"
    assert manifest[0]["hash_skipped_reason"] == "size_exceeds_limit"
    assert manifest[0]["hash_size_limit_bytes"] == 3


def test_write_provenance_includes_core_metadata(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "source.h5").write_bytes(b"source")
    output_path = tmp_path / "metadata.json"
    config = {
        "dataset": {
            "name": "mc_maze_small",
            "variant": "small",
            "source": "neural_latents_benchmark",
            "bin_size_ms": 5,
            "alignment_event": "movement",
        },
        "splits": {"seed": 2027},
    }

    provenance = write_provenance(
        dataset_name="mc_maze_small",
        dataset_root=dataset_root,
        output_path=output_path,
        config=config,
        max_hash_size_bytes=16,
    )

    assert output_path.exists()
    assert provenance["dataset_name"] == "mc_maze_small"
    assert provenance["variant"] == "small"
    assert provenance["source"] == "neural_latents_benchmark"
    assert provenance["split_seed"] == 2027
    assert provenance["bin_size_ms"] == 5
    assert provenance["alignment_event"] == "movement"
    assert provenance["file_count"] == 1
    assert provenance["config"] == config
    assert "generated_at_utc" in provenance
    assert "git_commit" in provenance
