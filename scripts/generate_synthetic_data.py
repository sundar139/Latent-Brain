from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console

from latentbrain.data.io import save_neural_dataset
from latentbrain.data.schemas import SyntheticDatasetConfig
from latentbrain.data.splits import create_neuron_mask, create_trial_split
from latentbrain.data.synthetic import generate_poisson_lds
from latentbrain.data.validation import (
    validate_neural_dataset,
    validate_neuron_mask,
    validate_trial_split,
)
from latentbrain.paths import get_repo_root

console = Console()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate local synthetic LatentBrain data.")
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic_poisson_lds.yaml"))
    args = parser.parse_args(argv)

    repo_root = get_repo_root()
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    config = SyntheticDatasetConfig.from_yaml(config_path)
    dataset = generate_poisson_lds(config)
    split = create_trial_split(
        dataset.trial_ids,
        config.dataset.train_fraction,
        config.dataset.validation_fraction,
        config.dataset.test_fraction,
        seed=config.dataset.seed,
    )
    mask = create_neuron_mask(
        config.dataset.n_neurons,
        config.dataset.heldout_neuron_fraction,
        seed=config.dataset.seed,
    )

    validate_neural_dataset(dataset)
    validate_trial_split(split, dataset.trial_ids)
    validate_neuron_mask(mask, config.dataset.n_neurons)

    dataset.metadata["split_counts"] = {
        "train": int(len(split.train)),
        "validation": int(len(split.validation)),
        "test": int(len(split.test)),
    }
    dataset.metadata["neuron_mask_counts"] = {
        "heldin": int(mask.heldin.sum()),
        "heldout": int(mask.heldout.sum()),
    }

    output_dir = Path(config.output.directory)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_path = output_dir / config.output.filename
    metadata_path = output_dir / config.output.metadata_filename
    save_neural_dataset(dataset, output_path, metadata_path)

    rates_shape = dataset.rates.shape if dataset.rates is not None else ()
    latents_shape = dataset.latents.shape if dataset.latents is not None else ()
    summary = {
        "dataset": config.dataset.name,
        "spikes_shape": list(dataset.spikes.shape),
        "rates_shape": list(rates_shape),
        "latents_shape": list(latents_shape),
        "bin_size_ms": dataset.bin_size_ms,
        "train_trials": len(split.train),
        "validation_trials": len(split.validation),
        "test_trials": len(split.test),
        "heldin_neurons": int(mask.heldin.sum()),
        "heldout_neurons": int(mask.heldout.sum()),
        "dataset_hash": dataset.metadata["dataset_hash"],
        "output": _relative(output_path, repo_root),
        "metadata": _relative(metadata_path, repo_root),
    }
    for key, value in summary.items():
        console.print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
