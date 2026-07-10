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

Validation loss remains useful for checkpointing inside a run, but it is not the primary cross-run model-comparison metric. Cross-run ranking uses unified validation bits/spike first, then deterministic tie-breakers. Factor-latent remains the current valid local target to beat; oracle scores are diagnostic upper bounds only, and old incompatible mean-rate values are not tuning targets. The unified scoreboard reads local LFADS-family summaries, including canonical LFADS tuning, controller-style LFADS tuning, coordinated dropout, and raw LFADS rate-calibration summaries when those ignored artifacts exist. If local summaries are absent on a fresh clone, it falls back to configured known LFADS-family values.

## Controller-style inferred inputs

The controller-style LFADS-family model adds time-varying inferred inputs because a single initial latent condition can be too weak for trial-to-trial perturbations and within-trial dynamics that are visible in held-in spiking. A bidirectional encoder summarizes held-in activity, a controller GRU infers per-bin input posteriors, and the generator uses those inferred inputs to produce factors and all-neuron rates.

Inferred-input KL is read as a usage diagnostic. Near-zero KL may indicate posterior underuse; very large KL may indicate overfitting or weak prior regularization. This comes before neural SDE or rSLDS work because it is the smallest LFADS-family architecture change that directly tests whether time-varying latent drive helps held-out prediction under the existing canonical scorer.

Selection remains validation unified bits/spike with the train-heldout mean-rate reference, fixed 20 ms bins, fixed 1.28-second window, deterministic split, and deterministic held-in/held-out mask. Validation loss still checkpoints runs only; it is not the tuning target.

## Neural-SDE-style latent dynamics

The neural-SDE-style latent generator tests continuous-time latent dynamics after controller-style LFADS-family tuning and before rSLDS switching. The encoder still reads held-in spikes only and infers an initial latent posterior. Instead of a discrete GRU generator, drift and diffusion networks evolve the latent state with an Euler/Euler-Maruyama update at the same 20 ms bin width used by the canonical scorer.

`diffusion_scale: 0.0` is the deterministic neural ODE-style limit. Nonzero diffusion injects seedable Brownian noise during training, while evaluation uses the deterministic mean path by default. Near-zero learned diffusion can indicate that deterministic dynamics are enough for this local co-smoothing objective; high diffusion with worse validation bits/spike can indicate noisy latent paths rather than useful stochastic dynamics.

Selection remains validation unified bits/spike with train-heldout mean rate as the reference, fixed 20 ms bins, fixed 1.28-second window, deterministic split, and deterministic held-in/held-out mask. The immediate valid local target is the factor-latent unified score, with the previous controller-style LFADS-family score as a dynamics-family reference. This comes before rSLDS switching because it tests smooth continuous latent evolution without adding discrete state identities or switching-transition interpretation. Outputs are local artifacts only, not official NLB leaderboard results.

## LFADS-style factor evaluation

The next local evaluation uses the trained LFADS-style GRU checkpoint without training a new neural network. Held-in spike counts are the only model inputs. The checkpointed model produces factor trajectories for train, validation, and test trials, and those factors become features for a train-only ridge decoder that predicts held-out neuron rates. Held-out spikes are targets only; they are never fed into the LFADS-style model or used to fit validation/test standardization statistics.

Poisson likelihood, log-likelihood, and bits/spike are computed against a train-only held-out mean-rate reference. A behavior velocity decoder from the same factors is a secondary diagnostic when behavior is available, and it also uses train trials only for fitting and standardization. This is local co-smoothing-style evaluation of a minimal LFADS-style foundation, not a full LFADS implementation and not an official NLB leaderboard result.

## Planned evaluation direction

Evaluation is expected to include predictive quality, latent structure diagnostics, reproducibility checks, and benchmark compatibility when the project reaches that scope. No evaluation claims are made in the current repository state.

## Deterministic latent dynamics tuning

The previous neural-SDE-style local tuning selected a zero-diffusion run, so the next focused comparison uses deterministic neural-ODE-style latent dynamics. The workflow keeps the same MC_Maze Small dataset hash, 20 ms bins, 1.28-second window, train/validation/test split, held-in/held-out mask, and train-heldout mean-rate reference. Validation unified bits/spike remains the primary model-selection metric.

Checkpoint selection is aligned with the scorer by evaluating saved validation-loss and latest checkpoints after training, copying the best validation unified bits/spike checkpoint to `best_unified.pt`, and writing `checkpoint_selection.csv`. Validation loss remains a training diagnostic, not the cross-run selection target.

Decision rule before rSLDS switching: deterministic latent dynamics should first beat the current factor-latent unified reference under this same-bin scorer. If it does, run multiple-seed robustness checks before adding switching dynamics. If it does not, treat the gap as evidence that the factor-latent baseline remains the local valid target. Old incompatible mean-rate values remain historical-only and are not tuning targets.

## Deterministic objective refinement

After switching neural-ODE-style tuning collapsed to one dominant regime and underperformed, the lean next neural workflow returns to deterministic latent dynamics rather than adding architecture. The refinement tunes objective and schedule controls: held-out loss weight, KL scale/warmup, input dropout, drift regularization, cosine learning-rate scheduling, and post-training unified checkpoint selection.

Drift regularization adds `drift_regularization * mean(drift ** 2)` to the training loss and reports both drift norm and the scaled regularization loss. Scheduler traces are logged per epoch. Validation unified bits/spike remains the cross-run selection metric; validation loss remains an internal checkpoint diagnostic and `best_unified.pt` is selected by reranking saved checkpoints under the canonical scorer.

Decision rule before further architecture: if deterministic refinement beats factor-latent, run multi-seed robustness before any claims. If it does not, investigate objective redesign or multiple datasets before adding more model classes. Old incompatible mean-rate values remain historical-only and are not tuning targets.

## rSLDS-style switching dynamics

The switching neural-ODE-style latent generator extends deterministic latent dynamics with soft discrete regimes, not full Bayesian rSLDS inference. A bidirectional encoder reads held-in spikes and infers the initial latent posterior. At each 20 ms bin, a regime network maps the current latent state and encoder context to regime probabilities. Regime-specific drift networks produce candidate latent updates, and the model evolves using the probability-weighted drift. Diffusion remains disabled.

Regime occupancy and categorical entropy are diagnostics, not claims of discovered discrete brain states. A useful switching result should show multiple active regimes, non-degenerate occupancy, and improved validation unified bits/spike under the same canonical scorer. If one regime dominates, switching did not add meaningful dynamics. If switching beats factor-latent, the next step is multi-seed robustness before any scientific or benchmark-style claims.

Selection remains validation unified bits/spike with train-heldout mean rate as the reference, fixed 20 ms bins, fixed 1.28-second window, deterministic split, and deterministic held-in/held-out mask. Old incompatible mean-rate values remain historical-only and are not tuning targets. Outputs are local artifacts only, not official NLB leaderboard results.

## Objective redesign diagnostics for deterministic latent dynamics

Switching dynamics collapsed to one dominant regime and deterministic refinement produced only marginal gains, so the next controlled workflow asks whether the deterministic neural-ODE family is limited by the training objective rather than the architecture. The model class, dataset hash, 20 ms bins, 1.28-second window, deterministic split/mask, and train-heldout mean-rate reference are all held fixed. `diffusion_scale` stays exactly `0.0`.

Held-out loss weighting varies the ratio between the held-in reconstruction term and the held-out prediction term. If held-out-heavy objectives improve validation unified bits/spike, the previous model was underweighting the co-smoothing task it is scored on.

Sparse-count weighting assigns `positive_count_weight` to bins with at least one spike and `zero_count_weight` to empty bins before summing the Poisson negative log likelihood. MC_Maze Small at 20 ms is dominated by zero bins, so this variant tests whether that imbalance limits training. If zero-downweighting helps, sparse-count imbalance was a real constraint.

The rate-calibration auxiliary loss adds `rate_calibration_loss_weight * mean((mean_predicted_rate - mean_observed_train_rate) ** 2)` using per-neuron train-only observed rates. Validation and test targets are never used for calibration. If calibration hurts, the output scale was already adequate.

Drift regularization adds `drift_regularization * mean(drift ** 2)` and reports both drift norm and the scaled regularization loss. Cosine learning-rate scheduling is retained and the learning rate is logged per epoch.

Unified checkpoint selection is preserved: saved validation-loss and latest checkpoints are re-ranked after training under the canonical scorer, the winner is copied to `best_unified.pt`, and `checkpoint_scores.csv` records the comparison.

Evaluation stays canonical and unweighted. Weighted losses shape training only; validation unified bits/spike is computed with the same unweighted scorer used by every other method on the scoreboard, so cross-method comparison remains valid.

Decision rule before further architecture: if no objective variant beats the factor-latent unified reference, the next step is multi-seed and local robustness analysis of the best dynamics model plus expanded baselines and datasets, not more model classes. Old incompatible mean-rate values remain historical-only and are not tuning targets. Outputs are local artifacts only, not official NLB leaderboard results.

### Seed control in objective diagnostics

Every objective variant trains under one shared seed. The grid workflows seed with `seed + run_index`, which is acceptable when each grid point is a different model, but it would confound the objective with initialization here. Locally, re-running the identical `refined_baseline` objective under two seed offsets moved validation unified bits/spike from `-0.0038` to `0.0283514699322505` — a swing larger than any objective effect measured in this workflow. Objective variants are therefore only compared against the same-seed `refined_baseline` row, never against references produced under a different seed.

This also means the stored `previous_neural_ode_refinement_validation_bits_per_spike` reference (`0.0283514699322505`) was produced at a different seed than these runs. A `beats_previous_neural_ode_refinement` value of `false` is not by itself evidence that an objective is worse; the same-seed baseline row is the controlled comparison. Cross-seed reference comparisons remain in the outputs for continuity with the scoreboard, and should be read as uncontrolled.

## Multi-seed robustness and seed-controlled comparison

Objective diagnostics surfaced a seed confound. The original objective workflow seeded with `seed + run_index`, so the objective variant under test was inseparable from its initialization. Re-running the identical `refined_baseline` objective at two seed offsets produced `-0.003776` and `0.0283514699322505` validation unified bits/spike — a swing of roughly `0.032`, larger than every objective effect the workflow was built to measure. Any single-seed leaderboard over stochastic training is therefore an unreliable basis for claims.

The robustness workflow enforces an explicit seed policy. The trial split and the held-in/held-out neuron mask are constructed once from a fixed `split_seed` and reused by every method and every seed, so no method ever sees a different partition of trials or neurons. The initialization and training seed varies over an explicit `seeds` list, and that same list is used by every method. No seed is derived from a run index, a method index, or any other positional quantity; when a seed is derived at all, the derivation is written to the outputs. A regression test asserts that reordering the method list leaves every method's seeds unchanged.

Factor analysis fits with randomized SVD, so the initialization seed is passed through as its `random_state`. Its spread across seeds is consequently small but not exactly zero, which is the honest behavior for this baseline under a fixed split.

Comparisons are paired by seed. For each seed both methods see the identical split, mask, and data, so the per-seed difference removes the shared split effect and isolates the method contrast. Uncertainty on each method's mean is a percentile bootstrap confidence interval, deterministic given `bootstrap_seed`.

Decision rule for carrying a method forward. If no neural method beats factor-latent across seeds, stop adding architecture on this dataset and window; invest instead in rigorous reporting, expanded baselines, or additional datasets. If a neural method beats the factor-latent mean but its own confidence-interval lower bound does not clear that mean, the result is suggestive but not established: run more seeds before making any claim. Only when a neural method clears the factor-latent mean at its CI lower bound should the work advance to held-out test reporting and additional datasets. Evaluation remains canonical and unweighted throughout, and old incompatible mean-rate values are never tuning targets. All outputs are local artifacts, not official NLB leaderboard results.

## Validation/test generalization audit

Multi-seed robustness produced a result that invalidates naive reporting: factor-latent,
deterministic neural-ODE refinement, and the best objective variant all score positive on the
validation split and negative on the test split. A method that beats the train-mean reference on
one held-out split and loses to it on another has not demonstrated generalization, and its
validation number is not a performance figure.

The audit separates three candidate explanations. The first is distribution shift: trial-level
spike counts, population and held-out firing rates, zero fractions, and behavior summaries
(endpoint displacement, endpoint angle, path distance, mean speed) are computed per trial and
aggregated per split, then compared between validation and test as a standardized mean difference
using the pooled trial-level spread. A large standardized difference in held-out rate or reach
geometry means the two splits are not exchangeable and the gap is partly a property of the data,
not the model.

The second is sampling noise. MC_Maze Small yields 15 validation and 15 test trials at a 70/15/15
split. The validation-minus-test gap is bootstrapped over seeds, paired within seed so that the
shared split effect cancels, and reported as a percentile confidence interval that is
deterministic given the bootstrap seed.

The third is split-specific luck. A factor-latent baseline is refit from scratch under ten
independent trial splits and neuron masks, alongside a train-mean control (which must score
exactly `0.0` bits/spike by construction, and serves as a scorer self-check) and a split-mean
control (fit on the evaluation split itself, therefore an invalid model retained only as a
diagnostic ceiling). If the test score is negative under most splits, the pattern is a property
of the dataset and window rather than of the accepted split.

Risk is labelled `high` whenever the validation mean is positive and the test mean is negative,
`moderate` when the paired gap confidence interval excludes zero but the test mean remains
positive, `low` otherwise, and `unresolved_missing_data` when the inputs needed to decide are
absent. The unified scoreboard ingests this label; under high risk its report states that every
ranking in it must be read as a validation-only diagnostic.

Decision rule before any reporting. Under high risk, no performance claim may be made, and the
current MC_Maze Small split must be described as unstable rather than conclusive. If repeated
splits show high variance, the dataset is underpowered for strong conclusions and the correct
response is cross-validation or more data, not more architecture and not a better-looking seed.
Evaluation remains canonical and unweighted throughout, and old incompatible mean-rate values are
never tuning targets. All outputs are local artifacts, not official NLB leaderboard results.

## Cross-validated rate-offset audit

The split audit left two facts that block reporting. Repeated trial splits move factor-latent's
test score across the sign boundary, so the accepted split's number is one draw from a wide
distribution rather than a measurement. And an invalid control — predicting each evaluation split
by its own held-out mean rate — outscores every valid model, which means a large, trivially
capturable split-level rate offset sits unmodeled inside the metric.

Repeated-split evaluation replaces single-split interpretation. Factor-latent is refit under every
combination of trial split seed and sklearn `FactorAnalysis` random state. Crossing the two lets
the variance be attributed: the variance of per-split means measures the trial-split effect, while
the mean of within-split variances measures the estimator's randomized-SVD effect. Both are
reported. Any claim that survives only one particular pairing of the two is not a result.

Valid controls use train data and model inputs only. `train_mean_rate` is the canonical reference
and therefore scores exactly `0.0` against itself, which doubles as a scorer self-check.
`train_per_neuron_mean_rate` is, by construction, that same per-neuron train mean; it is retained
precisely to make the degeneracy explicit rather than to imply headroom that does not exist.
`train_population_scaled_mean_rate` rescales the train held-out profile by a population factor read
from **held-in** spikes, which are legal model inputs at evaluation time — this is the honest test
of whether the split-level rate offset can be recovered without touching evaluation targets.
`train_rate_calibrated_factor_latent` fits one scalar calibration on train held-out counts and
train predictions only, then applies it unchanged to validation and test.

Invalid controls are diagnostics, never performance. `split_mean_rate_invalid` fits the evaluation
split's own held-out targets. `oracle_split_scaled_factor_latent_invalid` rescales factor-latent to
match the evaluation split's observed mean rate. Both carry `valid_model: false` and an
`invalid_reason`, and both are excluded from best-valid-model selection at every layer, including
the unified scoreboard.

The rate-offset decomposition asks how much of the invalid split-mean advantage a pure rescaling
recovers. If the oracle-rescaled factor-latent recovers at least half of that advantage, the gap is
dominated by a split-level rate offset rather than by trial structure. If the train-only
calibration reproduces the gain, it can be carried forward as a valid baseline. If only the invalid
controls gain, the effect is evaluation-split mean leakage and not a deployable model improvement —
the correct response is a better reference or per-split rate handling, not a better model.

Reporting rules that follow. Single-split results are not reportable as final performance. The
recommended reporting mode is repeated split. Factor-latent, not any neural model, is carried
forward as the reporting baseline. Evaluation stays canonical and unweighted, and old incompatible
mean-rate values are never tuning targets. All outputs are local artifacts, not official NLB
leaderboard results.

## Diagnostic reporting and claim safety

The final artifact for a dataset is not the best score it produced; it is a report stating what the
evidence supports and what it forbids. The MC_Maze Small diagnostic report freezes the accepted
findings and is regenerated deterministically from the stored audit summaries, with no timestamps
and no randomness, so that two runs on the same inputs produce byte-identical output.

The accepted evidence hierarchy is explicit. A repeated-split result with its spread outranks a
single-split number. A result that survives multi-seed evaluation outranks one that does not. A
paired comparison outranks an unpaired one. A single-seed leaderboard entry is a diagnostic, never
a performance figure — the objective workflow's earlier `seed + run_index` scheme confounded the
method under test with its initialization, and every grid workflow that still seeds that way
inherits the same caveat.

Valid models and invalid controls are separated structurally, not by convention. The method registry
records, per method, whether it is a valid model, whether it is reportable as model performance, and
if invalid, why. Invalid controls carry an `invalid_reason`, are excluded from best-valid-model
selection at every layer, and cannot be marked carried forward. Validation refuses to emit a report
in which an invalid control is reportable, the carried-forward method is invalid, single-split
reporting is recommended, an official leaderboard claim is set, or the split-mean advantage is
attributed to a global rate offset rather than to target leakage.

Three disclosures are mandatory in the report text and are checked mechanically. The seed-confound
disclosure states that the neural-ODE near-win was seed-specific. The split-instability disclosure
states that the 15-trial evaluation splits are unstable and that single-split numbers are not final
performance. The rate-leakage disclosure states that invalid controls use evaluation split target
information and cannot be reported as model performance. A fourth check keeps old incompatible
mean-rate values labelled historical-only.

The claim safety checklist accompanies every report: no official leaderboard claim, no invalid
control reported as model performance, no single-split result reported as final performance,
canonical unified metric used, old mean-rate values excluded from current targets, generated outputs
not committed, negative neural results included, seed confound disclosed, split instability
disclosed. The report builder exits non-zero if any item fails.

## Behavior-stratified cross-validation

Repeated random splits fixed the wrong half of the problem. They average over the accident of which
15 trials land in an evaluation split, but any individual fold can still be dominated by one reach
direction or one firing-rate regime. On a centre-out reaching task that is not a cosmetic concern:
a fold missing an entire direction asks the model to extrapolate, and the resulting score measures
the split rather than the method.

Each trial is summarized by endpoint displacement, endpoint direction, endpoint distance, mean
speed, population firing rate, and held-out firing rate. Direction is binned into equal-width
sectors of the circle so opposite reaches can never share a bin; distance, speed, and both rates are
binned by rank so each bin holds an equal share of trials regardless of the underlying distribution.
The stratum label is the tuple of enabled bins. Strata with fewer trials than
`min_trials_per_stratum` are pooled rather than dropped, because the alternative is a fold that never
observes a rare reach at all. When behavior is absent, the behavior-derived terms fall away and the
protocol degrades explicitly to rate-only stratification.

Assignment is greedy and balanced: within each stratum, trials are visited in a seeded random order
and placed into whichever fold is currently smallest. This keeps fold sizes within one trial of each
other while spreading every stratum across folds. Assignment is deterministic given the seed, and
each repeat uses a fresh seed. Within a repeat, the held-in/held-out neuron mask is fixed, so folds
differ only in which trials they hold out.

Fold balance is measured, not asserted. Per fold we record trial count, mean and spread of
population rate, held-out rate, endpoint distance, and speed, plus the Shannon entropy of the reach
directions it contains. Per repeat we report the min, max, range, and coefficient of variation of
each of those quantities across folds, and warn when trial counts deviate from the mean by more than
a quarter or when a rate metric spans more than a fifth of its mean across folds.

Random-versus-stratified comparison is run under matched conditions: the same fold count, the same
number of folds, the same scorer, the same methods — only the assignment differs. The variance
reduction fraction is `1 - var(stratified) / var(random)`, and it is reported with its sign. If
stratification does not reduce variance, that is the finding.

The scoring protocol is unchanged in every other respect. The train-held-out mean-rate reference is
recomputed from the training folds alone for every evaluation fold, and scores exactly `0.0`
bits/spike against itself, which serves as a per-fold scorer self-check. Evaluation is canonical and
unweighted. `split_mean_rate_invalid` is scored on every fold as a leakage diagnostic and is excluded
from valid-model selection, as is the reference itself: neither is reportable as model performance.
Stratified cross-validation is the recommended reporting mode for MC_Maze Small; single-split numbers
remain unreportable, and old incompatible mean-rate values remain historical-only.

## Movement-window and alignment audit

A window is an experimental choice, not a formatting detail. Stratified cross-validation on the
`from_start` 1.28-second crop showed endpoint-direction entropy far below its ceiling, which is the
signature of a window that ends before the reach has expressed itself. If most bins precede
movement, then behavior-related latent structure is largely absent from the data being scored, and
every model comparison run on that window is answering a narrower question than it appears to.

Three families of candidate window are compared. From-start windows keep the current convention and
vary only in length, isolating the effect of duration. Peak-speed-centred windows place the crop
symmetrically around each trial's fastest hand movement, which guarantees the reach is inside the
window at the cost of aligning trials to an outcome of the movement. Movement-onset windows begin a
configurable interval before the first bin whose speed reaches a per-trial quantile, which aligns to
the start of the movement rather than its peak.

Onset detection needs care. On a trial that is static and then moves, a fixed speed quantile can
coincide with the minimum speed, placing "onset" at the first bin — the opposite of the intent. The
implementation detects that degeneracy and falls back to a fraction of the trial's own peak speed,
so onset always lands where movement actually begins.

Windows are applied per trial, so different trials may be cropped at different offsets. Every window
keeps a fixed length; trials whose ideal crop would run off either edge are clipped to the nearest
valid position and counted, so the report can state how many trials were shifted rather than
silently misaligning them. Behavior features, stratified folds, and the train-held-out mean-rate
reference are all recomputed inside each candidate window; nothing is carried across windows.

Coverage is measured, not assumed. For each window we record the behavior source actually used
(hand position, falling back to cursor position when hand position is absent), the mean and peak
hand speed, the mean endpoint distance, the Shannon entropy of endpoint directions, and the fraction
of bins in which speed clears a fraction of that trial's peak. A window whose moving-bin fraction
falls below a floor is treated as pre-movement rather than as a reach window.

Selection rules are deliberately conservative and use valid models only. A candidate is eligible if
it has usable behavior coverage, produces no fold-balance warning, and preserves factor-latent
against the current window's confidence-interval lower bound, so that ordinary fold noise cannot
disqualify it. Among eligible candidates, a challenger replaces the current window only if it
carries strictly more reach-direction diversity *and* more moving bins. Invalid controls are scored
on every window and excluded from the decision entirely: a window that raises the split-mean control
has increased the leakage available in the metric, not the quality of the evaluation. When no
candidate clears both bars, the current window is retained and must be reported as an early-window
diagnostic. Reporting remains stratified cross-validation, old incompatible mean-rate values remain
historical-only, and none of this is an official benchmark result.

## Recommended movement-window cross-validation

The carried-forward MC_Maze Small target is the per-trial
`behavior_speed_peak_centered_1p28s` crop. It is applied after rebinning to 20 ms and spans 64 bins
around each trial's peak hand speed. Behavior is mandatory: a missing hand/cursor position signal is
an error rather than a silent rate-only fallback. Movement coverage and endpoint-direction entropy
are reported with every run so the crop remains auditable as a reach-dynamics window.

The frozen protocol uses five repeats of five-fold greedy-balanced stratified cross-validation. Trial
strata include endpoint direction, endpoint distance, mean speed, population rate, and held-out rate.
The held-out neuron mask is fixed within a repeat, and the train-held-out mean-rate reference is
recomputed from each training fold. Factor Analysis, feature standardization, and held-out decoding
are also fit on training trials only.

`factor_latent` is the carried-forward valid baseline. `train_mean_rate` is the canonical zero-bit
reference, not a competitor. `split_mean_rate_invalid` reads evaluation-fold target counts and is
included only to re-check whether target-leakage dominance persists; it is excluded from valid-model
selection regardless of its score. The generated protocol YAML freezes the dataset hash, window,
binning, scorer, folds, stratification, methods, and bootstrap settings.

Claim safety is unchanged. The original `from_start_1p28s` and recommended-window scores answer
different prediction problems and cannot be described as model-performance improvements. Single-split
results are unreportable, old incompatible mean-rate values are not tuning targets, invalid controls
are not model performance, and all outputs are local rather than official NLB leaderboard results.

## MC_Maze Small diagnostic evidence chain

The final local diagnostic bundle preserves the full evidence chain instead of presenting the last
number without context. Multi-seed evaluation established the seed confound behind the neural-ODE
near-win. The split audit then exposed instability in the small validation and test sets. The rate
audit showed that split-mean advantages came from evaluation-target leakage. Behavior-stratified
cross-validation reduced dependence on accidental fold composition, and the movement-window audit
showed that `from_start_1p28s` was early/pre-movement. Recommended-window cross-validation finally
evaluated `behavior_speed_peak_centered_1p28s` using five folds by five repeats at 20 ms.

The report therefore carries forward `factor_latent` only under
`recommended_window_stratified_cross_validation`. Old neural results remain historical or
early/pre-movement diagnostics until rerun under that protocol. Invalid split-mean controls remain
leakage diagnostics regardless of score. The report validator rejects a different carried-forward
window or reporting mode, reportable invalid controls, final single-split reporting, direct
old-window versus recommended-window improvement claims, and any official leaderboard claim.
