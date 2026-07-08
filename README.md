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
- Full LFADS, neural SDE, or switching dynamical system logic
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

## Data policy

Raw neural datasets are not committed to this repository. Data files, derived datasets, model checkpoints, generated results, local logs, and experiment artifacts are ignored by Git. Future data ingestion must follow dataset licenses, access terms, and ethical requirements.

## Reproducibility principles

LatentBrain uses validated configuration, explicit random seeds, isolated environments, and automated quality checks. Future experiments should record the Git commit, configuration, environment summary, data provenance, split seed, and artifact metadata needed to audit results.

## Safety warning

Do not commit `.env`, raw data, checkpoints, generated results, W&B keys, cloud credentials, API tokens, private absolute paths, or local experiment artifacts.
