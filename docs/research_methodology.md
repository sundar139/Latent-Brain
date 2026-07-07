# Research Methodology

LatentBrain is a research codebase for studying latent dynamical structure in neural population activity. The repository currently contains engineering infrastructure only; model implementations, datasets, experiments, benchmark scores, and results are not available yet.

## Problem formulation

The planned scientific objective is to infer compact latent states that explain high-dimensional neural observations over time while preserving interpretable dynamics and uncertainty where appropriate.

## Planned generative modeling direction

Future work may compare latent variable models, dynamical systems, and probabilistic sequence models for neural trajectories. Candidate modeling families will be introduced only after the data contract, preprocessing pipeline, and evaluation protocol are defined.

## Initial latent-variable baseline

The first latent baseline is Factor Analysis over smoothed held-in firing rates. It is used as a transparent, non-neural precursor to stronger GPFA/LFADS/SDE models and is GPFA-style only: no temporal GP prior is implemented.

## Planned evaluation direction

Evaluation is expected to include predictive quality, latent structure diagnostics, reproducibility checks, and benchmark compatibility when the project reaches that scope. No evaluation claims are made in the current repository state.
