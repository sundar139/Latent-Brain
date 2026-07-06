# Synthetic Data

LatentBrain includes a local synthetic Poisson linear dynamical system dataset generator so data contracts can be tested before real neural datasets are integrated. The generated files are engineering fixtures, not benchmark results.

## Purpose

The synthetic generator provides a deterministic neural population dataset with known latent trajectories, firing rates, spike counts, trial identifiers, and time bins. It supports validation of shapes, dtypes, split leakage checks, held-in and held-out neuron masks, and save/load behavior.

## Generation process

For each trial, latent states follow a stable linear dynamical system:

```text
z[t + 1] = A z[t] + epsilon
```

The observation model maps latent states to firing rates in Hz:

```text
rate[t] = exp(bias + C z[t])
```

Rates are clipped to configured minimum and maximum firing rates. Spike counts are sampled from a Poisson distribution using the bin duration:

```text
spikes[t] ~ Poisson(rate[t] * bin_size_ms / 1000)
```

The random generator is local to the configured seed, so the same configuration and seed reproduce the same arrays.

## Shape conventions

```text
spikes: [n_trials, n_time_bins, n_neurons]
rates: [n_trials, n_time_bins, n_neurons]
latents: [n_trials, n_time_bins, latent_dim]
trial_ids: [n_trials]
time_ms: [n_time_bins]
```

Spikes are integer spike counts per bin. Rates are firing rates in Hz.

## Validation checks

Validation rejects invalid neural arrays before they reach future modeling code. Checks include:

- Spike rank, integer dtype, and non-negative values
- Rate shape, positivity, and finite values
- Latent shape and finite values
- Unique trial identifiers
- Strictly increasing time bins
- Positive bin size
- Complete train, validation, and test coverage
- No trial leakage across splits
- Held-in and held-out mask coverage with no overlap

## Leakage prevention

Trial splits are generated from trial identifiers using a local random generator and validated so every trial appears exactly once across train, validation, and test splits. Held-out neurons are masked independently from trial splits for future co-smoothing evaluation.

## Limitations

This dataset is intentionally simple. It does not represent a real recording session, task structure, behavior, dataset-specific preprocessing, or benchmark protocol. It must not be reported as a model result or benchmark score.
