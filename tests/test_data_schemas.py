from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from latentbrain.data.schemas import NeuralDataset, SyntheticDatasetConfig
from latentbrain.data.validation import validate_neural_dataset


def test_neural_dataset_schema_accepts_valid_arrays() -> None:
    dataset = NeuralDataset(
        spikes=np.zeros((2, 3, 4), dtype=np.int64),
        rates=np.ones((2, 3, 4), dtype=np.float64),
        latents=np.zeros((2, 3, 2), dtype=np.float64),
        trial_ids=np.array([10, 11], dtype=np.int64),
        time_ms=np.array([0.0, 20.0, 40.0], dtype=np.float64),
        bin_size_ms=20,
        metadata={"name": "synthetic_poisson_lds"},
    )

    validate_neural_dataset(dataset)
    assert dataset.spikes.shape == (2, 3, 4)


def test_dataset_validation_rejects_negative_spikes() -> None:
    dataset = NeuralDataset(
        spikes=np.array([[[-1]]], dtype=np.int64),
        rates=np.ones((1, 1, 1), dtype=np.float64),
        latents=np.zeros((1, 1, 1), dtype=np.float64),
        trial_ids=np.array([0], dtype=np.int64),
        time_ms=np.array([0.0], dtype=np.float64),
        bin_size_ms=20,
        metadata={},
    )

    with pytest.raises(ValueError, match="non-negative"):
        validate_neural_dataset(dataset)


def test_synthetic_config_validates_split_fractions() -> None:
    with pytest.raises(ValidationError, match="sum"):
        SyntheticDatasetConfig.model_validate(
            {
                "dataset": {
                    "name": "synthetic_poisson_lds",
                    "seed": 2027,
                    "n_trials": 64,
                    "n_time_bins": 80,
                    "n_neurons": 32,
                    "latent_dim": 4,
                    "bin_size_ms": 20,
                    "train_fraction": 0.5,
                    "validation_fraction": 0.2,
                    "test_fraction": 0.2,
                    "heldout_neuron_fraction": 0.25,
                },
                "dynamics": {"spectral_radius": 0.92, "process_noise_std": 0.15},
                "observations": {
                    "log_rate_bias": -1.25,
                    "loading_scale": 0.35,
                    "min_rate_hz": 0.1,
                    "max_rate_hz": 120.0,
                },
                "output": {
                    "directory": "data/synthetic",
                    "filename": "synthetic_poisson_lds.npz",
                    "metadata_filename": "synthetic_poisson_lds_metadata.json",
                },
            }
        )
