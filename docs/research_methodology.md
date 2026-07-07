# Research Methodology

LatentBrain is a research codebase for studying latent dynamical structure in neural population activity. The repository currently contains engineering infrastructure only; model implementations, datasets, experiments, benchmark scores, and results are not available yet.

## Problem formulation

The planned scientific objective is to infer compact latent states that explain high-dimensional neural observations over time while preserving interpretable dynamics and uncertainty where appropriate.

## Planned generative modeling direction

Future work may compare latent variable models, dynamical systems, and probabilistic sequence models for neural trajectories. Candidate modeling families will be introduced only after the data contract, preprocessing pipeline, and evaluation protocol are defined.

## Initial latent-variable baseline

The first latent baseline is Factor Analysis over smoothed held-in firing rates. It is used as a transparent, non-neural precursor to stronger GPFA/LFADS/SDE models and is GPFA-style only: no temporal GP prior is implemented.

Latent-dimension, smoothing, and regularization sweeps tune this transparent baseline before any neural model comparison, so future LFADS/SDE work is not compared against an avoidably weak untuned baseline.

## LFADS-style sequential VAE foundation

The first neural model is a minimal LFADS-style sequential VAE foundation, not a full LFADS implementation. A bidirectional GRU encoder reads held-in spike counts and parameterizes a Gaussian posterior over an initial latent condition. A GRU generator is initialized from that latent sample, emits factor trajectories, and maps factors to positive Poisson firing rates. Training uses a Poisson reconstruction term, a Gaussian KL term to a standard normal prior, KL warmup, gradient clipping, deterministic seeding, and local checkpointing.

The current target is held-in reconstruction. This is intentional: it verifies PyTorch device handling, tensor datasets, model shapes, finite losses, gradients, checkpoints, and short MC_Maze Small training before adding held-out readouts or making co-smoothing claims. Held-out neurons remain masked from inputs and are available as targets for later neural co-smoothing work.

## Planned evaluation direction

Evaluation is expected to include predictive quality, latent structure diagnostics, reproducibility checks, and benchmark compatibility when the project reaches that scope. No evaluation claims are made in the current repository state.
