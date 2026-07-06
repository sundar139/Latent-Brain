from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console

from latentbrain.data.io import save_neural_dataset
from latentbrain.data.nlb import (
    MISSING_DATA_MESSAGE,
    NLBConfig,
    find_nlb_files,
    load_nlb_dataset,
    resolve_nlb_dataset_root,
)
from latentbrain.data.provenance import write_provenance
from latentbrain.data.splits import create_neuron_mask, create_trial_split
from latentbrain.data.validation import (
    validate_neural_dataset,
    validate_neuron_mask,
    validate_trial_split,
)
from latentbrain.paths import get_repo_root, resolve_configured_path

console = Console(stderr=True)


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


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = get_repo_root()
    config_arg = _parse_args(argv)
    config_path = config_arg if config_arg.is_absolute() else repo_root / config_arg
    config = NLBConfig.from_yaml(config_path)

    dataset_root = resolve_nlb_dataset_root(config, repo_root)
    if not find_nlb_files(dataset_root):
        console.print("NLB/MC_Maze local data is missing.")
        console.print(MISSING_DATA_MESSAGE)
        console.print(f"Configured dataset_root: {_relative(dataset_root, repo_root)}")
        console.print("Expected local files: .nwb, .h5, or .hdf5 files under that directory.")
        console.print("No fake data or processed output was created.")
        return 2

    try:
        dataset = load_nlb_dataset(dataset_root, config)
    except (ImportError, ValueError, FileNotFoundError) as exc:
        console.print(str(exc))
        console.print("No fake data or processed output was created.")
        return 2
    validate_neural_dataset(dataset)
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

    dataset.metadata["split_counts"] = {
        "train": int(len(split.train)),
        "validation": int(len(split.validation)),
        "test": int(len(split.test)),
    }
    dataset.metadata["neuron_mask_counts"] = {
        "heldin": int(mask.heldin.sum()),
        "heldout": int(mask.heldout.sum()),
    }

    processed_root = resolve_configured_path(config.dataset.processed_root, repo_root)
    output_path = processed_root / config.dataset.output_filename
    metadata_path = processed_root / config.dataset.metadata_filename
    provenance_path = metadata_path.with_name(f"{metadata_path.stem}_provenance.json")
    provenance = write_provenance(
        config.dataset.name,
        dataset_root,
        provenance_path,
        config.as_dict(),
    )
    dataset.metadata["provenance"] = provenance
    save_neural_dataset(dataset, output_path, metadata_path)

    summary = {
        "dataset": config.dataset.name,
        "spikes_shape": list(dataset.spikes.shape),
        "bin_size_ms": dataset.bin_size_ms,
        "trial_count": dataset.spikes.shape[0],
        "neuron_count": dataset.spikes.shape[2],
        "time_bins": dataset.spikes.shape[1],
        "train_trials": len(split.train),
        "validation_trials": len(split.validation),
        "test_trials": len(split.test),
        "heldin_neurons": int(mask.heldin.sum()),
        "heldout_neurons": int(mask.heldout.sum()),
        "dataset_hash": dataset.metadata["dataset_hash"],
        "metadata": _relative(metadata_path, repo_root),
    }
    output_console = Console()
    for key, value in summary.items():
        output_console.print(f"{key}: {value}")
    output_console.print(f"provenance: {_relative(provenance_path, repo_root)}")
    output_console.print(f"output: {_relative(output_path, repo_root)}")
    output_console.print(json.dumps({"status": "prepared"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
