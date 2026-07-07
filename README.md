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
- Model training or inference
- LFADS, neural SDE, or switching dynamical system logic
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

## Data policy

Raw neural datasets are not committed to this repository. Data files, derived datasets, model checkpoints, generated results, local logs, and experiment artifacts are ignored by Git. Future data ingestion must follow dataset licenses, access terms, and ethical requirements.

## Reproducibility principles

LatentBrain uses validated configuration, explicit random seeds, isolated environments, and automated quality checks. Future experiments should record the Git commit, configuration, environment summary, data provenance, split seed, and artifact metadata needed to audit results.

## Safety warning

Do not commit `.env`, raw data, checkpoints, generated results, W&B keys, cloud credentials, API tokens, private absolute paths, or local experiment artifacts.
