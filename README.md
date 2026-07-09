# LatentBrain

LatentBrain is a Python research and engineering codebase for studying latent dynamical structure in neural population activity.

## Research goal

The project aims to support rigorous investigation of latent variable and dynamical systems methods for neural time series. The current repository contains only the reproducible engineering foundation needed before data access, preprocessing, model development, training, and evaluation are introduced.

## Current repository status

Initialized foundation:

- `src` package layout with the `latentbrain` package
- Validated YAML configuration loading with environment overrides
- Safe `.env.example` contract without committed secrets
- Standard logging utilities
- Deterministic seeding utilities for Python, NumPy, and optional PyTorch
- Synthetic Poisson LDS data generation for validating data contracts
- Local MC_Maze Small ingestion that trializes continuous NLB NWB dataframes into spike and behavior tensors when behavior is available
- Typer-based CLI sanity commands
- Ruff, mypy, pytest, pre-commit, and GitHub Actions quality checks

Not implemented yet:

- Automatic dataset download or final benchmark preprocessing pipelines
- Full LFADS, full neural SDE, or full Bayesian rSLDS inference
- Neural Latents Benchmark evaluation
- Official benchmark scores, checkpoints, or model artifacts

## Local setup

Use Windows PowerShell from the repository root:

```powershell
cd "C:\Users\rohit\Documents\Personal Projects\Latent Brain"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Python 3.11 is the intended runtime for continuous integration. If multiple Python versions are installed, create the virtual environment with the Python 3.11 executable available on your system.

## Quality checks

Run these commands before committing changes:

```powershell
ruff check .
ruff format --check .
mypy src
pytest -q
python -m latentbrain.cli validate-config
python -m latentbrain.cli info
python scripts/check_environment.py
```

Optional neural-data tooling for local real dataset preparation can be installed with:

```powershell
python -m pip install -e ".[dev,neurodata]"
```

If `nlb-tools` is unavailable from pip in your environment, install it from the official Neural Latents Benchmark GitHub repository:

```powershell
python -m pip install git+https://github.com/neurallatents/nlb_tools.git
```

## Cross-validated rate audit

The split generalization audit showed two things that make single-split numbers unreportable: the
accepted 70/15/15 split's test-negative result is split-specific luck, and an **invalid** control
that reads the evaluation split's own mean firing rate beats every valid model by a wide margin.
This workflow replaces single-split interpretation with repeated-split reporting and quantifies
that rate offset:

```powershell
python scripts/run_cv_rate_audit.py --config configs/mc_maze_small_cv_rate_audit.yaml
```

It is CPU-only and trains no neural networks. Factor-latent is scored across ten trial splits
crossed with five sklearn `FactorAnalysis` random states, which separates trial-split variance
from the estimator's own randomized-SVD variance — the latter alone moves the metric enough to
matter. Alongside it run valid train-only controls (per-neuron train mean, held-in population
rescaling, a train-only rate calibration) and clearly-labelled **invalid** diagnostic controls
(`split_mean_rate_invalid`, `oracle_split_scaled_factor_latent_invalid`) that read evaluation
targets.

**Invalid controls use evaluation split targets and can never be reported as model performance.**
They never compete for best valid model; they exist only to measure how much of the gap is a
split-level rate offset. Single-split numbers are not reportable as final performance — report
factor-latent as a repeated-split baseline instead. Old mean-rate values remain historical-only.
Outputs and figures are local ignored artifacts under `results/mc_maze_small/cv_rate_audit/`, not
official NLB leaderboard results.

## Validation/test generalization audit

Seed robustness established that no neural method beats factor-latent, but it also surfaced a
more serious problem: every carried-forward method scores **positive on validation and negative
on test**. Before any result is reported, that has to be explained. This audit does it:

```powershell
python scripts/run_split_audit.py --config configs/mc_maze_small_split_audit.yaml
```

The audit is CPU-only and trains no neural networks. It compares trial spike rates, held-out
neuron activity, and behavior distributions across splits; bootstraps the validation/test gap;
and re-runs a factor-latent baseline plus train-mean and split-mean controls under ten
independent trial splits to see whether the test-negative pattern is specific to the accepted
split or persists everywhere.

MC_Maze Small has only 15 validation and 15 test trials, so a single split is weak evidence
either way. **No model performance claim should be made until the validation/test instability is
resolved.** If the audit reports high generalization risk, every score in this repository must be
read as a validation-only diagnostic. Old mean-rate values remain historical-only. Outputs and
figures are local ignored artifacts under `results/mc_maze_small/split_audit/`, not official NLB
leaderboard results.

## Multi-seed robustness

Objective diagnostics uncovered a seed confound: the earlier workflow seeded with `seed + run_index`, so each method effectively trained from a different initialization, and re-running one identical objective under two seed offsets moved validation unified bits/spike by roughly 0.032 — more than any effect being measured. Single-seed leaderboards are therefore not sufficient for claims. This workflow re-compares the strongest methods under an explicit seed policy:

```powershell
python scripts/run_seed_robustness.py --config configs/mc_maze_small_multiseed_robustness.yaml
```

The trial split and held-in/held-out neuron mask are held **fixed** across every method and seed (`split_seed_mode: fixed`), while the **initialization/training seed varies** over the configured `seeds` list. Every method receives the identical seed list, and no seed is ever derived from a run index. Score spread across seeds therefore reflects initialization and training variance only.

Selection uses canonical unified validation bits/spike over 20 ms MC_Maze Small bins, a 1.28-second window, and train-heldout mean rate as the reference; evaluation stays canonical and unweighted. Each method reports mean, standard deviation, a bootstrap 95% confidence interval, and paired per-seed differences against factor-latent. A method must beat factor-latent by mean *and* by CI lower bound before it is carried forward. Old mean-rate values remain historical-only. Results, figures, and checkpoints are local ignored artifacts under `results/mc_maze_small/seed_robustness/`, not official NLB leaderboard results.

## Deterministic neural-ODE objective diagnostics

Switching dynamics collapsed to one dominant regime and deterministic refinement gained only marginally, so the next workflow interrogates the training objective rather than the architecture:

```powershell
python scripts/tune_neural_ode_objectives.py --config configs/mc_maze_small_neural_ode_objectives.yaml
```

Controlled objective variants around the best deterministic neural-ODE refinement setting vary held-in/held-out loss weighting, zero/positive spike-count weighting, an optional train-only rate-calibration auxiliary loss, drift regularization, KL schedule, and input dropout. The model class is unchanged and `diffusion_scale` is forced to `0.0`.

The canonical scoring target is validation unified bits/spike over 20 ms MC_Maze Small bins, a 1.28-second window, the deterministic split/mask, and train-heldout mean rate as the reference. Training losses may be weighted, but evaluation always uses the canonical unweighted unified metric. Factor-latent (0.0316438194429199) remains the current valid local target to beat; old mean-rate values are historical-only and are never tuning targets. Outputs, figures, and checkpoints are local ignored artifacts under `results/mc_maze_small/neural_ode_objectives/`, not official NLB leaderboard results.

## Deterministic neural-ODE refinement

Switching dynamics collapsed to one dominant regime locally, so the next local neural workflow refines deterministic latent dynamics instead of adding regimes:

```powershell
python scripts/refine_neural_ode.py --config configs/mc_maze_small_neural_ode_refinement.yaml
```

The refinement keeps diffusion disabled and tunes objective/schedule choices: held-out loss weight, KL scale/warmup, input dropout, drift regularization, cosine learning-rate scheduling, and unified-metric checkpoint selection. Selection uses canonical unified validation bits/spike: 20 ms MC_Maze Small bins, a 1.28-second window, deterministic split/mask, and train-heldout mean rate as the reference. Factor-latent remains the current valid local target to beat; old mean-rate values are historical-only. Outputs and checkpoints are local ignored artifacts under `results/mc_maze_small/neural_ode_refinement/`, not official NLB leaderboard results.

## Switching deterministic latent dynamics

Run local rSLDS-style switching neural-ODE-style tuning with:

```powershell
python scripts/tune_switching_ode.py --config configs/mc_maze_small_switching_ode_tuning.yaml
```

The model keeps diffusion disabled, infers soft regime probabilities over time, mixes a small number of learned drift fields, and reports regime occupancy/entropy diagnostics. Selection uses canonical unified validation bits/spike: 20 ms MC_Maze Small bins, a 1.28-second window, the deterministic split/mask, and train-heldout mean rate as the reference. The factor-latent unified score is the current valid local target to beat; old incompatible mean-rate values are historical-only and are not tuning targets. Results, figures, and checkpoints are local ignored artifacts under `results/mc_maze_small/switching_ode_tuning/`, not official NLB leaderboard results.

## Configuration and environment

The default configuration lives in `configs/base.yaml`. Local machine-specific values may be supplied through environment variables or a local `.env` file, but `.env` files must never be committed.

Copy `.env.example` only when local overrides are needed, then keep any real values private.

## Synthetic data

LatentBrain includes a synthetic Poisson LDS generator for testing the data stack before real datasets are integrated:

```powershell
python scripts/generate_synthetic_data.py --config configs/synthetic_poisson_lds.yaml
```

The generated files are local validation artifacts, not benchmark results. Real neural datasets are not integrated yet, and generated synthetic files under `data/` are ignored by Git.

## Real data

LatentBrain includes a local MC_Maze Small ingestion path:

```powershell
python scripts/inspect_nlb_files.py --root data/raw/nlb/mc_maze_small
python scripts/prepare_nlb_data.py --config configs/nlb_mc_maze_small.yaml
```

The script does not download data. Start with MC_Maze Small from the official Neural Latents Benchmark datasets page and DANDI repository `https://gui.dandiarchive.org/#/dandiset/000140`. Place legally obtained files under `data/raw/nlb/mc_maze_small` or set `LATENTBRAIN_NLB_ROOT`. The real NWB train file loads as a continuous `NWBDataset.data` pandas dataframe; LatentBrain uses NLB trial metadata to trialize it into `spikes: [trials, time, neurons]`, concatenating `heldout_spikes` after `spikes` when available. When train-file behavior columns are available, `hand_pos` and `cursor_pos` are trialized with the same trial IDs and cropped time window into `behavior: [trials, time, behavior_dims]` with named dimensions. The initial policy crops variable-length trials to the minimum trial length for a clean validation tensor. This is validation-oriented preprocessing, not final benchmark preprocessing. If files are missing, the script exits with guidance and creates no fake data. No trained model, benchmark score, EvalAI submission, or leaderboard claim exists. Behavior is stored for validation and future decoding work; velocity decoding is not implemented yet.

## Data validation report

After preparing MC_Maze Small, generate a local data-quality report with:

```powershell
python scripts/analyze_mc_maze.py --config configs/mc_maze_small_eda.yaml
```

The analysis writes JSON, CSV, Markdown, and PNG files under ignored `reports/mc_maze_small/` paths. It checks the processed dataset hash, split coverage, held-in and held-out masks, spike statistics, behavior availability, and trialization metadata. The report is exploratory validation only; no model training, behavior decoding metric, or benchmark evaluation is performed.

## Mean-rate baseline

After real-data preparation, run the first local sanity baseline with:

```powershell
python scripts/run_mean_rate_baseline.py --config configs/mc_maze_small_mean_rate.yaml
```

The baseline fits one constant firing rate per neuron using train trials only, then evaluates Poisson negative log-likelihood and bits/spike-style improvement on train, validation, and test splits for held-in, held-out, and all neurons. Outputs are written under ignored `results/mc_maze_small/mean_rate/` paths. This validates the local metric pipeline; it is not an official NLB benchmark result and does not train a neural network model.

## Behavior decoder baseline

After real-data preparation, run the local behavior-decoding sanity baseline with:

```powershell
python scripts/run_behavior_decoder.py --config configs/mc_maze_small_behavior_decoder.yaml
```

The baseline smooths binned spikes within each trial, uses held-in neurons by default, derives velocity targets from `hand_pos` and `cursor_pos`, and fits a train-only ridge decoder with train-only feature and target standardization. Outputs are written under ignored `results/mc_maze_small/behavior_decoder/` paths. This is not official benchmark performance, and no neural network model is trained.

## Co-smoothing ridge baseline

Run the local held-in to held-out neural co-smoothing sanity baseline with:

```powershell
python scripts/run_cosmoothing_baseline.py --config configs/mc_maze_small_cosmoothing_ridge.yaml
```

The baseline uses held-in neurons as inputs, held-out neurons as targets, smooths held-in spikes within each trial, and fits a train-only ridge decoder against a train-only held-out mean-rate reference. Outputs are written under ignored `results/mc_maze_small/cosmoothing_ridge/` paths. This is not an official NLB leaderboard result, and no neural network model is trained.

## Co-smoothing ridge diagnostic sweep

The first single ridge co-smoothing run underperformed the train-only mean-rate reference, so LatentBrain includes a local diagnostic sweep before any GPFA, LFADS, neural SDE, switching, or neural-network model work:

```powershell
python scripts/run_cosmoothing_sweep.py --config configs/mc_maze_small_cosmoothing_sweep.yaml
```

The sweep varies smoothing sigma, ridge alpha, feature standardization, and intercept use. Every fit uses train trials only, held-in neurons as inputs, held-out neurons as targets, train-only feature standardization when enabled, and a train-only held-out mean-rate reference. It evaluates train, validation, and test splits, selects the best configuration by validation held-out bits/spike, and writes local CSV/JSON/Markdown outputs under ignored `results/mc_maze_small/cosmoothing_sweep/` paths. These files are diagnostic artifacts ignored by Git, not official benchmark performance, and no neural network model is trained.

## Factor latent baseline

Run the first non-neural latent-variable sanity baseline with:

```powershell
python scripts/run_factor_latent_baseline.py --config configs/mc_maze_small_factor_latent.yaml
```

This baseline smooths held-in spike counts into firing rates, fits train-only Factor Analysis latents, decodes held-out neuron rates from those latents, and decodes behavior velocity when behavior is available. It is GPFA-style only: no temporal GP prior is implemented, so it is not a full GPFA claim. Outputs are local artifacts under ignored `results/mc_maze_small/factor_latent/` paths, not official benchmark performance, and no neural network model is trained.

## Factor latent diagnostic sweep

Tune the transparent non-neural latent baseline before LFADS/SDE work with:

```powershell
python scripts/run_factor_latent_sweep.py --config configs/mc_maze_small_factor_latent_sweep.yaml
```

The sweep varies latent dimension, smoothing sigma, held-out decoder alpha, and feature standardization. It selects by validation held-out bits/spike, reports behavior R² as a secondary metric, compares to the mean-rate sanity reference, and writes local ignored outputs under `results/mc_maze_small/factor_latent_sweep/`. This is not official benchmark performance, not full GPFA because no temporal GP prior is implemented, and no neural network model is trained.

## LFADS-style GRU training foundation

Run the first PyTorch neural modeling foundation with:

```powershell
python scripts/train_lfads_gru.py --config configs/mc_maze_small_lfads_gru.yaml
```

For a small synthetic smoke run, generate synthetic data first and use the synthetic training config:

```powershell
python scripts/generate_synthetic_data.py --config configs/synthetic_poisson_lds.yaml
python scripts/train_lfads_gru.py --config configs/synthetic_lfads_gru.yaml
```

This model is an LFADS-style sequential VAE foundation, not a full LFADS implementation. It uses held-in neurons as input and reconstructs held-in activity with a Poisson observation model. Metrics, checkpoints, config snapshots, and reports are local outputs under ignored `results/` paths. No official NLB leaderboard result is reported.
The real MC_Maze LFADS-style configs request CUDA explicitly and fail fast if a CUDA-enabled PyTorch build is unavailable; synthetic configs may still run on CPU.

## LFADS-style masked co-smoothing training

Train the LFADS-style GRU with a masked co-smoothing objective using held-in spikes as the only model input and all-neuron rates as the model output:

```powershell
python scripts/train_lfads_gru.py --config configs/mc_maze_small_lfads_gru_cosmoothing.yaml
```

In this run the model input dimension is the held-in neuron count, while the readout predicts rates for all neurons. Held-in reconstruction loss is computed on held-in targets, and held-out prediction loss is computed only from train-trial held-out targets during optimization. Validation and test held-out spikes remain evaluation-only. The run writes local metrics, reports, and checkpoints under ignored `results/mc_maze_small/lfads_gru_cosmoothing/` paths.

Evaluate the checkpoint with direct model held-out rates and the factor-decoder diagnostic with:

```powershell
python scripts/evaluate_lfads_gru.py --config configs/mc_maze_small_lfads_gru_cosmoothing_eval.yaml
```

The evaluation script creates no new neural-network checkpoint. It reports direct model held-out prediction when available, optionally also reports a train-only factor decoder, and compares local validation bits/spike to the previous LFADS-style factor evaluation, factor-latent, and mean-rate references. These outputs are local artifacts under ignored `results/mc_maze_small/lfads_gru_cosmoothing_eval/` paths, not official benchmark performance, and not a full LFADS claim.
The real MC_Maze co-smoothing training and evaluation configs request CUDA explicitly and do not silently fall back to CPU.

## LFADS-style held-out evaluation

After the short LFADS-style GRU training command has created a local checkpoint, evaluate held-out neural prediction from its factors with:

```powershell
python scripts/evaluate_lfads_gru.py --config configs/mc_maze_small_lfads_gru_eval.yaml
```

The evaluation script loads the existing checkpoint and does not train a new neural network. It feeds held-in spikes through the LFADS-style model, extracts factor trajectories, and fits a train-only ridge decoder from those factors to held-out neuron rates. Validation and test samples are evaluation-only, and the train-only held-out mean rate remains the reference for bits/spike. Behavior velocity decoding from factors is reported as a secondary local diagnostic when behavior is available.

Evaluation JSON, CSV, and Markdown outputs are local artifacts under ignored `results/mc_maze_small/lfads_gru_eval/` paths. This is a local held-out evaluation, not an official NLB leaderboard result, and it is not a full LFADS claim.

## Window-matched local comparison

Full-window baseline metrics and short-window LFADS-style metrics should not be read as direct comparisons. The LFADS-style MC_Maze runs use a 256-bin crop for fast local iteration, while earlier mean-rate and factor-latent numbers were produced on the full processed trial window. Recompute local methods on the same dataset hash, train/validation/test split, held-in/held-out mask, time crop, Poisson likelihood convention, bits/spike convention, and behavior target convention with:

```powershell
python scripts/run_window_matched_comparison.py --config configs/mc_maze_small_window_matched_comparison.yaml
```

The comparison script evaluates windowed mean-rate, ridge co-smoothing, factor-latent, and existing LFADS-style checkpoints without training a new neural network. It writes `comparison_summary.json`, `comparison_metrics.csv`, `validation_leaderboard.csv`, `behavior_comparison.csv`, and `comparison_report.md` under ignored `results/mc_maze_small/window_matched_comparison/` paths. These are local comparison artifacts only, not official benchmark outputs, and neural methods remain LFADS-style only rather than full LFADS.

## LFADS-style CUDA tuning

Run the controlled CUDA tuning workflow for the LFADS-style masked co-smoothing model with:

```powershell
python scripts/tune_lfads_gru.py --config configs/mc_maze_small_lfads_gru_tuning.yaml
```

The workflow runs a small deterministic grid capped by `search.max_runs`, keeps the MC_Maze Small dataset hash, split, held-in/held-out mask, and 256-bin crop fixed, and selects by local validation bits/spike. It compares runs only against the 256-bin window-matched scoreboard references, including the window-matched mean-rate and factor-latent baselines. Tuning outputs, reports, and checkpoints are written under ignored `results/mc_maze_small/lfads_gru_tuning/` paths and must stay local. No official NLB benchmark or leaderboard result is reported, and the model remains LFADS-style only, not full LFADS.

## LFADS-style diagnostic audit

After tuning, diagnose why the LFADS-style masked co-smoothing model trails simple window-matched references with:

```powershell
python scripts/audit_lfads_gru.py --config configs/mc_maze_small_lfads_audit.yaml
```

The audit reloads the same MC_Maze Small processed dataset hash, deterministic split, held-in/held-out mask, and 256-bin crop, then audits the tuned local checkpoint for loss scale, bits/spike reference agreement, rate calibration, factor usage, direct-model behavior, held-out sparsity, and tiny-subset overfit behavior. It writes local CSV/JSON/Markdown outputs and matplotlib figures under ignored `results/mc_maze_small/lfads_audit/` paths. This is only a local diagnostic audit for deciding what to fix before adding larger architecture changes; it is not an official benchmark result and the checkpoint remains LFADS-style only, not full LFADS.

## Temporal rebinning diagnostic

The audit found that 5 ms held-out MC_Maze targets are extremely sparse. Test whether coarser temporal bins reduce target sparsity before changing architectures with:

```powershell
python scripts/run_temporal_rebinning_diagnostic.py --config configs/mc_maze_small_temporal_rebinning.yaml
```

The diagnostic rebins the 5 ms spike counts to 10 ms and 20 ms by summing grouped spike bins, averages behavior over the same groups, keeps the comparison window fixed at 1.28 seconds, recomputes same-bin mean-rate and factor-latent references, and runs small CUDA LFADS-style masked co-smoothing jobs at 10 ms and 20 ms. Outputs, checkpoints, plots, and reports are local artifacts under ignored `results/mc_maze_small/temporal_rebinning/` paths. Bits/spike values across bin sizes are diagnostic and should not be treated as direct benchmark comparisons; no official NLB result is reported.

## LFADS-style rate calibration diagnostic

The temporal rebinning diagnostic identified 20 ms as the best local LFADS-style bin size so far, but the 20 ms model still trails same-bin references. Test whether the direct-rate output is poorly anchored with:

```powershell
python scripts/run_lfads_rate_calibration.py --config configs/mc_maze_small_lfads_rate_calibration.yaml
```

The diagnostic reloads the existing 20 ms LFADS-style checkpoint, fits train-only post-hoc rate calibration on direct held-out predictions, evaluates per-neuron multiplicative scaling, log-rate bias, and mean-rate blending, then trains a small CUDA LFADS-style masked co-smoothing model whose output readout bias is initialized from train-only firing rates. It compares all LFADS-family results only against same-bin 20 ms mean-rate and factor-latent references.

Generated metrics, figures, reports, checkpoints, and config snapshots are local artifacts under ignored `results/mc_maze_small/lfads_rate_calibration/` paths. This is local diagnostic work for output scale and mean-rate anchoring, not an official NLB leaderboard result, and the model remains LFADS-style only, not full LFADS.

## LFADS-style coordinated dropout diagnostic

After rate calibration and readout bias initialization failed to close the held-out prediction gap, test whether the 20 ms LFADS-style masked co-smoothing model benefits from input robustness and shared-population prediction:

```powershell
python scripts/run_lfads_coordinated_dropout.py --config configs/mc_maze_small_lfads_coordinated_dropout.yaml
```

The workflow keeps the same 1.28-second 20 ms window, deterministic trial split, and held-in/held-out neuron mask. During training only, it randomly masks a configured fraction of held-in input neurons before the LFADS-style forward pass. Held-in and held-out targets remain unmasked for loss computation, so the model is forced to infer neural activity from partial population observations without corrupting the supervised targets. Validation and test evaluation use the original unmasked held-in inputs by default.

Generated metrics, dropout diagnostics, figures, reports, config snapshots, and checkpoints are local artifacts under ignored `results/mc_maze_small/lfads_coordinated_dropout/` paths. The report compares dropout runs against same-bin mean-rate, same-bin factor-latent, previous raw 20 ms LFADS, and rate-calibration references. No official benchmark result is reported, and the model remains LFADS-style only, not full LFADS.

## Metric/reference audit

Before adding new model families, audit whether local MC_Maze Small scores share the same Poisson likelihood and bits/spike reference convention:

```powershell
python scripts/run_metric_audit.py --config configs/mc_maze_small_metric_audit.yaml
```

The audit uses 20 ms bins and the same 1.28-second window as the recent LFADS-style diagnostics. It scores the train-only held-out mean-rate predictor against that same train-only held-out mean-rate reference, so its validation bits/spike should be near zero. Global-mean, split-mean, oracle-smoothed, random, and trial-shuffled controls are written alongside any existing reported metrics that can be loaded safely.

Unified bits/spike references are necessary because a model log-likelihood only becomes comparable after subtracting the same reference log-likelihood and dividing by the same held-out spike count. A mean-rate model scored against itself should not look like a strong positive baseline; if it does, the reference convention is different. Oracle controls use held-out targets directly and are not valid models. Outputs are local audit artifacts under ignored `results/mc_maze_small/metric_audit/` paths and are not official NLB leaderboard results.

## Unified local scoreboard

After the metric audit, use the canonical train-reference scoreboard for future local MC_Maze Small comparisons:

```powershell
python scripts/run_unified_scoreboard.py --config configs/mc_maze_small_unified_scoreboard.yaml
```

The scoreboard uses 20 ms bins, the fixed 1.28-second window, deterministic train/validation/test split, and the train-only held-out mean-rate reference. Bits/spike is always `(model_log_likelihood - reference_log_likelihood) / (log(2) * spike_count)`, so the train-heldout mean-rate predictor scores `0.0` against itself. Future tuning should optimize against the unified scoreboard: the `0.0` train-mean reference, the current factor-latent unified local reference, and the oracle diagnostic upper bound.

Older positive mean-rate values from incompatible reference conventions are historical-only and must not be used as direct model targets. Oracle controls are invalid models because they use held-out targets directly. The scoreboard reads local LFADS/dynamics-family summaries when present, including `inputs.lfads_unified_tuning_summary_path`, `inputs.lfads_controller_tuning_summary_path`, `inputs.neural_sde_tuning_summary_path`, `inputs.neural_ode_tuning_summary_path`, the coordinated-dropout summary, and the raw LFADS rate-calibration summary. If those ignored local summaries are absent on a fresh clone, it falls back to configured known LFADS-family values. Generated CSVs, figures, and the report live under ignored `results/mc_maze_small/unified_scoreboard/` paths and are local artifacts, not official NLB leaderboard results.

## Canonical LFADS-style unified tuning

Tune LFADS-family runs under the canonical train-reference scorer with:

```powershell
python scripts/tune_lfads_unified.py --config configs/mc_maze_small_lfads_unified_tuning.yaml
```

This workflow uses the 20 ms MC_Maze Small tensor, the fixed 1.28-second window, and train-heldout mean rate as the bits/spike reference. It selects runs by validation unified bits/spike, not validation loss. The current valid local target to beat is the factor-latent unified score; the previous coordinated-dropout LFADS-family score is the LFADS-family reference. Old incompatible mean-rate values remain historical-only and are not tuning targets.

Outputs, run reports, figures, config snapshots, and checkpoints are local artifacts under ignored `results/mc_maze_small/lfads_unified_tuning/` paths. This is local model selection only, not an official NLB leaderboard result, and the model remains LFADS-style only, not full LFADS.

## Controller-style LFADS-family tuning

Run the inferred-input controller workflow with:

```powershell
python scripts/tune_lfads_controller.py --config configs/mc_maze_small_lfads_controller_tuning.yaml
```

This model adds a controller GRU that reads held-in activity and generator state to infer time-varying latent inputs. Runs still use 20 ms MC_Maze Small bins, the fixed 1.28-second window, train-heldout mean-rate as the canonical reference, and validation unified bits/spike as the selection metric. The current local target to beat is the factor-latent unified score; the previous best LFADS-family score is a secondary LFADS-family reference. Old incompatible mean-rate values are historical-only and are not tuning targets.

Outputs, reports, figures, config snapshots, and checkpoints are local artifacts under ignored `results/mc_maze_small/lfads_controller_tuning/` paths. This is local controller-style LFADS-family tuning, not an official NLB leaderboard result, and the model is LFADS-style with inferred inputs, not full LFADS.

## Neural-SDE-style latent generator tuning

Run the compact Euler/Euler-Maruyama latent generator workflow with:

```powershell
python scripts/tune_neural_sde.py --config configs/mc_maze_small_neural_sde_tuning.yaml
```

The model replaces the discrete GRU generator with continuous-time latent dynamics integrated directly in PyTorch. A bidirectional GRU encoder infers the initial latent state, drift and diffusion networks evolve the latent trajectory, and a factor readout maps latents to all-neuron Poisson rates. `diffusion_scale: 0.0` is the deterministic neural ODE-style limit; nonzero diffusion tests stochastic latent paths through Euler-Maruyama noise. No `torchsde` dependency is used.

This workflow uses 20 ms MC_Maze Small bins, the fixed 1.28-second window, train-heldout mean-rate as the canonical reference, and validation unified bits/spike as the model-selection metric. The current valid local target to beat is the factor-latent unified score; the previous controller-style LFADS-family score is the dynamics-family reference. Old incompatible mean-rate values remain historical-only and are not tuning targets.

Outputs, reports, figures, config snapshots, and checkpoints are local artifacts under ignored `results/mc_maze_small/neural_sde_tuning/` paths. This is local neural-SDE-style tuning, not an official NLB leaderboard result, and it is a compact Euler/Euler-Maruyama latent generator rather than a full benchmarked neural SDE system.

## Deterministic neural-ODE-style latent dynamics tuning

Focus the best neural-SDE-style setting, where diffusion scale zero won, with:

```powershell
python scripts/tune_neural_ode.py --config configs/mc_maze_small_neural_ode_tuning.yaml
```

This reuses the compact Euler latent generator with `diffusion_scale: 0.0` forced for every run. It uses 20 ms MC_Maze Small bins, the fixed 1.28-second window, train-heldout mean-rate as the canonical reference, and validation unified bits/spike as the selection metric. The current valid local target to beat is the factor-latent unified score; the previous neural-SDE-style score is the dynamics-family reference. Old incompatible mean-rate values remain historical-only and are not tuning targets.

The workflow saves `best_validation.pt`, `latest.pt`, and `best_unified.pt`, then records checkpoint re-ranking in `checkpoint_selection.csv`. Outputs, reports, figures, config snapshots, and checkpoints are local artifacts under ignored `results/mc_maze_small/neural_ode_tuning/` paths and are not committed. This is local deterministic neural-ODE-style tuning, not an official NLB leaderboard result, and not a full benchmarked neural ODE/SDE system.

## Data policy

Raw neural datasets are not committed to this repository. Data files, derived datasets, model checkpoints, generated results, local logs, and experiment artifacts are ignored by Git. Future data ingestion must follow dataset licenses, access terms, and ethical requirements.

## Reproducibility principles

LatentBrain uses validated configuration, explicit random seeds, isolated environments, and automated quality checks. Future experiments should record the Git commit, configuration, environment summary, data provenance, split seed, and artifact metadata needed to audit results.

## Safety warning

Do not commit `.env`, raw data, checkpoints, generated results, W&B keys, cloud credentials, API tokens, private absolute paths, or local experiment artifacts.
