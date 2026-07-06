from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from latentbrain import __version__
from latentbrain.data.io import compute_dataset_hash
from latentbrain.data.schemas import NeuralDataset, SyntheticDatasetConfig
from latentbrain.data.validation import validate_neural_dataset


def _stable_matrix(rng: np.random.Generator, latent_dim: int, spectral_radius: float) -> np.ndarray:
    matrix = rng.normal(size=(latent_dim, latent_dim))
    eigenvalues = np.linalg.eigvals(matrix)
    radius = float(np.max(np.abs(eigenvalues)))
    if radius == 0.0:
        return np.eye(latent_dim) * spectral_radius
    return matrix * (spectral_radius / radius)


def generate_poisson_lds(config: SyntheticDatasetConfig) -> NeuralDataset:
    """Generate a reproducible synthetic Poisson LDS neural population dataset."""
    dataset_config = config.dataset
    dynamics = config.dynamics
    observations = config.observations
    rng = np.random.default_rng(dataset_config.seed)

    n_trials = dataset_config.n_trials
    n_time_bins = dataset_config.n_time_bins
    n_neurons = dataset_config.n_neurons
    latent_dim = dataset_config.latent_dim

    a_matrix = _stable_matrix(rng, latent_dim, dynamics.spectral_radius)
    loadings = rng.normal(scale=observations.loading_scale, size=(latent_dim, n_neurons))
    latents = np.zeros((n_trials, n_time_bins, latent_dim), dtype=np.float64)

    for trial_index in range(n_trials):
        latents[trial_index, 0] = rng.normal(size=latent_dim)
        for time_index in range(1, n_time_bins):
            noise = rng.normal(scale=dynamics.process_noise_std, size=latent_dim)
            previous = latents[trial_index, time_index - 1]
            latents[trial_index, time_index] = a_matrix @ previous + noise

    log_rates = observations.log_rate_bias + latents @ loadings
    rates = np.clip(
        np.exp(log_rates),
        observations.min_rate_hz,
        observations.max_rate_hz,
    ).astype(np.float64)
    expected_counts = rates * (dataset_config.bin_size_ms / 1000.0)
    spikes = rng.poisson(expected_counts).astype(np.int64)
    time_ms = np.arange(n_time_bins, dtype=np.float64) * dataset_config.bin_size_ms

    metadata = {
        "dataset_name": dataset_config.name,
        "seed": dataset_config.seed,
        "n_trials": n_trials,
        "n_time_bins": n_time_bins,
        "n_neurons": n_neurons,
        "latent_dim": latent_dim,
        "bin_size_ms": dataset_config.bin_size_ms,
        "generator": "poisson_lds",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "package_version": __version__,
    }
    dataset = NeuralDataset(
        spikes=spikes,
        rates=rates,
        latents=latents,
        trial_ids=np.arange(n_trials, dtype=np.int64),
        time_ms=time_ms,
        bin_size_ms=dataset_config.bin_size_ms,
        metadata=metadata,
    )
    validate_neural_dataset(dataset)
    dataset.metadata["dataset_hash"] = compute_dataset_hash(dataset)
    return dataset
