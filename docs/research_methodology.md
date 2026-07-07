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

The initial target was held-in reconstruction. This was intentional: it verified PyTorch device handling, tensor datasets, model shapes, finite losses, gradients, checkpoints, and short MC_Maze Small training before adding held-out readouts.

## LFADS-style masked co-smoothing objective

The next architectural improvement keeps the same minimal LFADS-style GRU foundation but changes the readout and objective. Held-in spike counts remain the only sequence input. The model readout can predict all-neuron firing rates, and the training objective combines three terms:

- held-in Poisson reconstruction loss on held-in targets;
- held-out Poisson prediction loss on masked held-out targets from train trials only;
- Gaussian KL regularization on the inferred initial latent condition, with warmup.

Held-out spikes are never concatenated into the model input. They are used as supervised targets for the train split when the masked co-smoothing objective is enabled. Validation and test held-out spikes are evaluation-only; they may be used to report local validation/test losses and select the local checkpoint by validation total loss, but they do not update model parameters.

This is the natural next improvement after held-in reconstruction because it tests whether the sequential latent representation can directly support held-out neural prediction, rather than requiring a separate post hoc factor decoder. It is still LFADS-style only: there is no full LFADS controller input pathway, no AutoLFADS/PBT, and no official NLB benchmark claim.

Real MC_Maze LFADS-style training and evaluation configs request CUDA explicitly so these local neural runs do not silently fall back to CPU. Synthetic smoke configs may remain CPU-compatible for ordinary tests.

## Metric comparability

Local method rankings are meaningful only when every method uses the same dataset hash, train/validation/test split, held-in and held-out neuron mask, time window, Poisson likelihood convention, bits/spike reference model, and behavior target convention. Earlier full-window baseline metrics are useful historical references, but they are not directly comparable to LFADS-style metrics produced on a 256-bin crop.

The window-matched comparison pipeline recomputes the transparent baselines and evaluates existing LFADS-style checkpoints on the same cropped tensor before building a validation leaderboard. This matters before tuning neural models because apparent improvements can come from different spike totals, reference likelihoods, or time windows rather than a better latent representation. The comparison remains local: it is not an official NLB leaderboard result, and LFADS-style checkpoints are not full LFADS.

## Controlled LFADS-style CUDA tuning

Neural tuning comes after the window-matched comparison so every candidate is compared against references computed on the same local scoreboard. The controlled LFADS-style tuning workflow fixes the MC_Maze Small dataset hash, deterministic trial split, deterministic held-in/held-out neuron mask, 256-bin crop, Poisson likelihood convention, bits/spike reference, and behavior target convention. The grid order is deterministic and capped by `search.max_runs`, so local compute stays practical and reproducible.

Each candidate trains only on train trials, uses validation loss for checkpoint selection, and reports validation held-out prediction metrics with the existing direct-rate and factor-decoder evaluation logic. The selected run is the best local validation bits/spike run, with lower validation Poisson NLL, higher behavior mean R², smaller parameter-count estimate, and lower run index as deterministic tie-breakers. These tuning results are local validation artifacts only, not official NLB leaderboard performance, and the model remains LFADS-style only.

## LFADS-style factor evaluation

The next local evaluation uses the trained LFADS-style GRU checkpoint without training a new neural network. Held-in spike counts are the only model inputs. The checkpointed model produces factor trajectories for train, validation, and test trials, and those factors become features for a train-only ridge decoder that predicts held-out neuron rates. Held-out spikes are targets only; they are never fed into the LFADS-style model or used to fit validation/test standardization statistics.

Poisson likelihood, log-likelihood, and bits/spike are computed against a train-only held-out mean-rate reference. A behavior velocity decoder from the same factors is a secondary diagnostic when behavior is available, and it also uses train trials only for fitting and standardization. This is local co-smoothing-style evaluation of a minimal LFADS-style foundation, not a full LFADS implementation and not an official NLB leaderboard result.

## Planned evaluation direction

Evaluation is expected to include predictive quality, latent structure diagnostics, reproducibility checks, and benchmark compatibility when the project reaches that scope. No evaluation claims are made in the current repository state.
