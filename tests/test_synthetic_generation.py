from __future__ import annotations

from pathlib import Path

import numpy as np

from latentbrain.data.io import load_neural_dataset, save_neural_dataset
from latentbrain.data.schemas import SyntheticDatasetConfig
from latentbrain.data.synthetic import generate_poisson_lds
from latentbrain.data.validation import validate_neural_dataset


def _config(seed: int = 2027) -> SyntheticDatasetConfig:
    config = SyntheticDatasetConfig.from_yaml(Path("configs/synthetic_poisson_lds.yaml"))
    return config.with_seed(seed)


def test_generate_poisson_lds_shapes_and_ranges() -> None:
    dataset = generate_poisson_lds(_config())

    validate_neural_dataset(dataset)
    assert dataset.spikes.shape == (64, 80, 32)
    assert dataset.rates is not None
    assert dataset.rates.shape == (64, 80, 32)
    assert dataset.latents is not None
    assert dataset.latents.shape == (64, 80, 4)
    assert dataset.spikes.dtype.kind in {"i", "u"}
    assert np.min(dataset.rates) >= 0.1
    assert np.max(dataset.rates) <= 120.0


def test_generate_poisson_lds_is_reproducible_for_same_seed() -> None:
    first = generate_poisson_lds(_config(seed=2027))
    second = generate_poisson_lds(_config(seed=2027))

    np.testing.assert_array_equal(first.spikes, second.spikes)
    np.testing.assert_allclose(first.rates, second.rates)
    np.testing.assert_allclose(first.latents, second.latents)


def test_generate_poisson_lds_changes_with_different_seed() -> None:
    first = generate_poisson_lds(_config(seed=2027))
    second = generate_poisson_lds(_config(seed=2028))

    assert not np.array_equal(first.spikes, second.spikes)


def test_save_and_load_neural_dataset_roundtrip(tmp_path: Path) -> None:
    dataset = generate_poisson_lds(_config())
    output_path = tmp_path / "synthetic.npz"
    metadata_path = tmp_path / "metadata.json"

    save_neural_dataset(dataset, output_path, metadata_path)
    loaded = load_neural_dataset(output_path)

    np.testing.assert_array_equal(dataset.spikes, loaded.spikes)
    assert loaded.metadata["dataset_hash"] == dataset.metadata["dataset_hash"]
    assert metadata_path.exists()
