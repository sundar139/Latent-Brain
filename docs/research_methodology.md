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

## LFADS-style diagnostic audit

When a tuned LFADS-style checkpoint remains below the window-matched mean-rate and factor-latent references, the next scientific question is not whether a larger architecture can win. The next question is whether the current objective, normalization, reference likelihood, and calibration are internally consistent. The diagnostic audit therefore comes before adding neural SDEs, controller inputs, switching dynamics, or larger tuning policies.

The audit holds the dataset hash, deterministic split, held-in/held-out mask, 256-bin crop, and references fixed. It measures loss scale and bits/spike agreement, predicted-rate calibration against observed held-out rates and train-only references, factor usage and possible posterior underuse, direct-model versus factor-decoder disagreement through the existing evaluator, held-out target sparsity, and whether a tiny train subset can be overfit with the same masked co-smoothing objective. A failure to overfit the tiny subset is treated as stronger evidence for an implementation or objective issue than ordinary validation underperformance.

Audit outputs are local reproducibility artifacts only. They are intended to identify likely issue flags and next debugging targets, not to report official NLB benchmark performance. The audited model remains LFADS-style only, not full LFADS.

## Temporal binning diagnostics

The first follow-up to the target-sparsity audit is temporal rebinning rather than a new architecture. Spike-count conservation is the primary preprocessing invariant: coarser bins are formed by summing adjacent 5 ms spike-count bins, while behavior samples are averaged over the same grouped bins so neural and behavioral arrays remain aligned. Any trailing incomplete groups are trimmed only when needed and recorded in metadata.

All bin-size comparisons preserve a fixed 1.28-second window, so 5 ms, 10 ms, and 20 ms diagnostics differ in temporal resolution rather than trial duration. Mean-rate and factor-latent references are recomputed at each bin size, and LFADS-style runs are compared only against the same-bin references. Bits/spike values across different bin sizes are diagnostic and should not be treated as direct benchmark comparisons because the event count, reference likelihood, and time discretization change together.

This check comes before neural SDEs, switching models, controller inputs, or larger tuning because it tests whether the current masked co-smoothing objective is starved by sparse 5 ms held-out targets. The generated tables, figures, and checkpoints are local artifacts only; they are not official NLB leaderboard results, and the neural model remains LFADS-style only.

## LFADS-style rate calibration and readout anchoring

After temporal rebinning, the next diagnostic keeps the 20 ms bin size fixed and asks whether the LFADS-style direct-rate output is poorly anchored rather than immediately adding model capacity. The workflow fits post-hoc calibrators on train trials only: per-neuron multiplicative Poisson rate scaling, the equivalent log-rate bias, and convex blending between direct model rates and train-only mean held-out rates.

The blend parameter diagnoses whether factors add held-out information beyond a mean-rate anchor. If the best train-selected alpha is near 0, the mean-rate component explains the held-out targets better than the model dynamics. If multiplicative or log-bias calibration helps, output scale is a likely issue. If readout bias initialization from train-only all-neuron firing rates helps the newly trained 20 ms model, poor output anchoring is a likely issue.

All validation and test metrics remain evaluation-only. Same-bin mean-rate and factor-latent references are required for interpretation because the 20 ms likelihood scale, spike totals, and target sparsity differ from 5 ms diagnostics. The generated results and checkpoints are local diagnostic artifacts only, not official NLB leaderboard results, and the model remains LFADS-style only.

## LFADS-style coordinated input masking

Because post-hoc calibration and readout bias initialization did not improve the 20 ms LFADS-style held-out prediction, the next diagnostic changes the training signal rather than the rate scale. Coordinated input masking randomly drops held-in input neurons during training while preserving the original held-in and held-out spike-count targets for loss computation. This probes whether the model can learn shared latent structure that predicts missing population activity from partial observations.

The workflow keeps the 20 ms bin size, fixed 1.28-second window, deterministic split, and neuron mask from the previous diagnostics. Dropout is applied to training inputs only; validation and test inputs stay unmasked unless explicitly configured for a diagnostic experiment. Same-bin mean-rate, factor-latent, previous raw LFADS-style, and calibration references are required so any apparent improvement is interpreted relative to the current local baseline family.

This diagnostic follows rate calibration because it addresses a different failure mode. If mild dropout helps, the model may benefit from robustness to partial observations. If high dropout hurts, the held-in input information may already be too limited. If none of the rates help, underfitting or objective mismatch may still dominate. Outputs are local artifacts only, not official NLB leaderboard results, and the model remains LFADS-style only.

## Metric consistency and control baselines

The metric audit fixes the scoring convention before any additional model work. For every audited method, the model log-likelihood is the summed Poisson log-likelihood of held-out spike counts under predicted rates. The reference log-likelihood is the summed Poisson log-likelihood of the same held-out spike counts under train-only held-out mean rates. Bits/spike is `(model_log_likelihood - reference_log_likelihood) / (log(2) * spike_count)`, using the same held-out spike-count denominator for every method.

This same-reference requirement matters because changing the reference can inflate or deflate bits/spike without changing model predictions. A train-only held-out mean-rate predictor scored against the train-only held-out mean-rate reference should score near zero. If an older mean-rate headline number is positive under a different reference, it is useful history but not directly comparable to LFADS-style or factor-latent values using the unified reference.

Oracle and shuffled controls make the audit interpretable. Smoothed true held-out rates are an upper-bound diagnostic because they use evaluation targets directly; they are not valid models. Random-rate and trial-shuffled controls are negative controls for whether arbitrary rates or target-like activity can appear competitive. Together these checks test whether held-in activity adds predictive information beyond train-only held-out mean rates without creating an official NLB benchmark claim.

## Canonical local scoring standard

Future local MC_Maze Small comparisons use the unified train-reference convention by default. The reference model is the train-only held-out mean rate computed from training trials and broadcast to the evaluated split. Bits/spike is `(model_log_likelihood - reference_log_likelihood) / (log(2) * spike_count)`, where both log-likelihoods use the same held-out counts, bin size, Poisson constant convention, and spike-count denominator.

This convention makes the reference-as-model score exactly `0.0` bits/spike. Positive values beat train-heldout mean rate under the same convention; negative values trail it. The current local tuning targets are the `0.0` train-mean reference, the factor-latent unified validation value, and the oracle diagnostic upper bound. Oracle scores remain invalid as models because they use held-out targets directly.

Historical positive mean-rate values from incompatible references are historical-only and must not be used as direct targets. A model comparison is valid only if the reference log-likelihood, held-out spike-count denominator, split, neuron mask, bin size, and time window all match the canonical scoreboard convention. These local scoreboards are not official NLB leaderboard results.

## Canonical LFADS-style model selection

LFADS-family tuning uses the canonical scoreboard convention for model comparison. Each candidate keeps the MC_Maze Small dataset hash, 20 ms bin size, 1.28-second window, deterministic train/validation/test split, deterministic held-in/held-out mask, train-heldout mean-rate reference, and Poisson likelihood convention fixed. The selected run is the best validation unified bits/spike run.

Validation loss remains useful for checkpointing inside a run, but it is not the primary cross-run model-comparison metric. Cross-run ranking uses unified validation bits/spike first, then deterministic tie-breakers. Factor-latent remains the current valid local target to beat; oracle scores are diagnostic upper bounds only, and old incompatible mean-rate values are not tuning targets. The unified scoreboard can read `inputs.lfads_unified_tuning_summary` and include the latest canonical LFADS-style tuning winner when that ignored local artifact exists; otherwise it remains based on the static known LFADS-family references in the config.

## LFADS-style factor evaluation

The next local evaluation uses the trained LFADS-style GRU checkpoint without training a new neural network. Held-in spike counts are the only model inputs. The checkpointed model produces factor trajectories for train, validation, and test trials, and those factors become features for a train-only ridge decoder that predicts held-out neuron rates. Held-out spikes are targets only; they are never fed into the LFADS-style model or used to fit validation/test standardization statistics.

Poisson likelihood, log-likelihood, and bits/spike are computed against a train-only held-out mean-rate reference. A behavior velocity decoder from the same factors is a secondary diagnostic when behavior is available, and it also uses train trials only for fitting and standardization. This is local co-smoothing-style evaluation of a minimal LFADS-style foundation, not a full LFADS implementation and not an official NLB leaderboard result.

## Planned evaluation direction

Evaluation is expected to include predictive quality, latent structure diagnostics, reproducibility checks, and benchmark compatibility when the project reaches that scope. No evaluation claims are made in the current repository state.
