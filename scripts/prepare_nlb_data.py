from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from rich.console import Console

from latentbrain.data.io import compute_dataset_hash, save_neural_dataset
from latentbrain.data.nlb import (
    MC_MAZE_LARGE_DANDI_URL,
    MC_MAZE_SMALL_DANDI_URL,
    MISSING_DATA_MESSAGE,
    OFFICIAL_NLB_DATASETS_URL,
    NLBConfig,
    find_candidate_nlb_files,
    load_nlb_dataset,
    resolve_nlb_dataset_root,
)
from latentbrain.data.provenance import write_provenance
from latentbrain.data.splits import create_neuron_mask, create_trial_split
from latentbrain.data.validation import (
    validate_neural_dataset,
    validate_neuron_mask,
    validate_source_bin_size,
    validate_trial_split,
)
from latentbrain.paths import get_repo_root, resolve_configured_path

console = Console(stderr=True, markup=False)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _parse_args(argv: Sequence[str] | None) -> Path:
    import argparse

    parser = argparse.ArgumentParser(description="Prepare local NLB-style neural data.")
    parser.add_argument("--config", type=Path, default=Path("configs/nlb_mc_maze.yaml"))
    args = parser.parse_args(argv)
    return args.config


def _print_missing_data(config: NLBConfig, dataset_root: Path, repo_root: Path) -> None:
    source = None if config.source is None else config.source.model_dump()
    console.print(
        json.dumps(
            {
                "status": "missing_raw_data",
                "dataset": config.dataset.name,
                "expected_raw_dir": _relative(dataset_root, repo_root),
                "verified_source_metadata": source,
                "candidate_assets": []
                if source is None
                else [asset["path"] for asset in source["expected_assets"]],
                "automatic_download_performed": False,
            },
            indent=2,
        )
    )
    console.print(MISSING_DATA_MESSAGE)
    console.print(f"Configured dataset_root: {_relative(dataset_root, repo_root)}")
    console.print(f"Official datasets page: {OFFICIAL_NLB_DATASETS_URL}")
    console.print(f"MC_Maze Small DANDI: {MC_MAZE_SMALL_DANDI_URL}")
    console.print(f"MC_Maze Large DANDI: {MC_MAZE_LARGE_DANDI_URL}")
    if config.source is not None and config.source.dandiset_id:
        total_mb = sum(asset.size_bytes for asset in config.source.expected_assets) / (1024 * 1024)
        console.print(f"Expected download size: {total_mb:.1f} MB")
        console.print(
            f"Manual download: dandi download "
            f"DANDI:{config.source.dandiset_id}/{config.source.dandiset_version}"
        )
        console.print(f"Then place the sub-* directory under {config.dataset.dataset_root}")
    console.print("No data was downloaded. No fake data or processed output was created.")


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = get_repo_root()
    config_arg = _parse_args(argv)
    config_path = config_arg if config_arg.is_absolute() else repo_root / config_arg
    config = NLBConfig.from_yaml(config_path)

    dataset_root = resolve_nlb_dataset_root(config, repo_root)
    if not find_candidate_nlb_files(dataset_root):
        _print_missing_data(config, dataset_root, repo_root)
        return 2

    try:
        dataset = load_nlb_dataset(dataset_root, config)
    except (ImportError, RuntimeError, ValueError, FileNotFoundError) as exc:
        console.print(str(exc))
        console.print("No fake data or processed output was created.")
        return 2
    validate_neural_dataset(dataset)
    validate_source_bin_size(dataset, config.dataset.bin_size_ms)
    split = create_trial_split(
        dataset.trial_ids,
        config.splits.train_fraction,
        config.splits.validation_fraction,
        config.splits.test_fraction,
        config.splits.seed,
    )
    mask = create_neuron_mask(
        dataset.spikes.shape[2],
        config.splits.heldout_neuron_fraction,
        config.splits.seed,
    )
    validate_trial_split(split, dataset.trial_ids)
    validate_neuron_mask(mask, dataset.spikes.shape[2])

    split_counts = {
        "train": int(len(split.train)),
        "validation": int(len(split.validation)),
        "test": int(len(split.test)),
    }
    dataset.metadata["split_counts"] = split_counts
    dataset.metadata["neuron_mask_counts"] = {
        "heldin": int(mask.heldin.sum()),
        "heldout": int(mask.heldout.sum()),
    }

    n_trials, n_time_bins, n_neurons = dataset.spikes.shape
    summary = dataset.metadata.setdefault("ingestion_summary", {})
    warnings: list[str] = []
    conservation = summary.get("spike_conservation", {})
    if conservation and not conservation.get("conserved", True):
        warnings.append(
            f"{conservation['excluded_spike_count']} spikes in "
            f"{conservation['excluded_bins']} bins were excluded by "
            f"{conservation['exclusion_reason']}"
        )
    if dataset.metadata.get("trialization", {}).get("cropping_occurred"):
        warnings.append("variable-length trials were cropped to the minimum trial length")
    summary.update(
        {
            "dataset_name": config.dataset.name,
            "dataset_family": "mc_maze",
            "variant": config.dataset.variant,
            "trial_count": int(n_trials),
            "time_bins": int(n_time_bins),
            "neuron_count": int(n_neurons),
            "behavior_dim": 0 if dataset.behavior is None else int(dataset.behavior.shape[2]),
            "behavior_names": dataset.behavior_names or [],
            "bin_size_ms": dataset.bin_size_ms,
            "split_counts": split_counts,
            "heldin_neuron_count": int(mask.heldin.sum()),
            "heldout_neuron_count": int(mask.heldout.sum()),
            "source_files": dataset.metadata.get("source_files", []),
            "source_identifiers": None if config.source is None else config.source.model_dump(),
            "trialization_policy": config.trialization.variable_length_policy,
            "warnings": warnings,
        }
    )

    processed_root = resolve_configured_path(config.dataset.processed_root, repo_root)
    output_path = processed_root / config.dataset.output_filename
    metadata_path = processed_root / config.dataset.metadata_filename
    provenance_path = processed_root / config.dataset.provenance_filename
    hash_limit_bytes = config.provenance.hash_size_limit_mb * 1024 * 1024
    dataset_hash = compute_dataset_hash(dataset)
    relative_config = _relative(config_path, repo_root)
    provenance = write_provenance(
        config.dataset.name,
        dataset_root,
        provenance_path,
        config.as_dict(),
        max_hash_size_bytes=hash_limit_bytes,
        dataset_metadata=dataset.metadata,
        config_path=relative_config,
        dataset_hash=dataset_hash,
        creation_command=f"python scripts/prepare_nlb_data.py --config {relative_config}",
    )
    dataset.metadata["provenance"] = provenance
    save_neural_dataset(
        dataset,
        output_path,
        metadata_path,
        extra_arrays={
            "train_indices": np.asarray(split.train, dtype=np.int64),
            "validation_indices": np.asarray(split.validation, dtype=np.int64),
            "test_indices": np.asarray(split.test, dtype=np.int64),
            "heldin_indices": np.flatnonzero(mask.heldin).astype(np.int64),
            "heldout_indices": np.flatnonzero(mask.heldout).astype(np.int64),
        },
    )

    report = {
        "status": "prepared",
        "dataset": config.dataset.name,
        "variant": config.dataset.variant,
        "source_files_used": dataset.metadata.get("source_files", []),
        "train_file_used": dataset.metadata.get("processed_target_source_file"),
        "spikes_shape": list(dataset.spikes.shape),
        "behavior_shape": None if dataset.behavior is None else list(dataset.behavior.shape),
        "behavior_names": dataset.behavior_names or [],
        "bin_size_ms": dataset.bin_size_ms,
        "trial_count": int(n_trials),
        "neuron_count": int(n_neurons),
        "time_bins": int(n_time_bins),
        "train_trials": split_counts["train"],
        "validation_trials": split_counts["validation"],
        "test_trials": split_counts["test"],
        "heldin_neurons": int(mask.heldin.sum()),
        "heldout_neurons": int(mask.heldout.sum()),
        "spike_conservation": conservation,
        "dataset_hash": dataset.metadata["dataset_hash"],
        "output": _relative(output_path, repo_root),
        "metadata": _relative(metadata_path, repo_root),
        "provenance": _relative(provenance_path, repo_root),
        "warnings": warnings,
    }
    output_console = Console()
    for key, value in report.items():
        output_console.print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
