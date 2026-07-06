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
- Local MC_Maze Small ingestion that trializes continuous NLB NWB dataframes
- Typer-based CLI sanity commands
- Ruff, mypy, pytest, pre-commit, and GitHub Actions quality checks

Not implemented yet:

- Automatic dataset download or final benchmark preprocessing pipelines
- Model training or inference
- LFADS, neural SDE, or switching dynamical system logic
- Neural Latents Benchmark evaluation
- Experiment runs, benchmark scores, checkpoints, or model artifacts

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

The script does not download data. Start with MC_Maze Small from the official Neural Latents Benchmark datasets page and DANDI repository `https://gui.dandiarchive.org/#/dandiset/000140`. Place legally obtained files under `data/raw/nlb/mc_maze_small` or set `LATENTBRAIN_NLB_ROOT`. The real NWB train file loads as a continuous `NWBDataset.data` pandas dataframe; LatentBrain uses NLB trial metadata to trialize it into `spikes: [trials, time, neurons]`, concatenating `heldout_spikes` after `spikes` when available. The initial policy crops variable-length trials to the minimum trial length for a clean validation tensor. This is validation-oriented preprocessing, not final benchmark preprocessing. If files are missing, the script exits with guidance and creates no fake data. No trained model, benchmark score, EvalAI submission, or leaderboard claim exists. Future EDA should inspect alignment, behavior, trial lengths, and spike statistics.

## Data validation report

After preparing MC_Maze Small, generate a local data-quality report with:

```powershell
python scripts/analyze_mc_maze.py --config configs/mc_maze_small_eda.yaml
```

The analysis writes JSON, CSV, Markdown, and PNG files under ignored `reports/mc_maze_small/` paths. It checks the processed dataset hash, split coverage, held-in and held-out masks, spike statistics, and trialization metadata. The report is exploratory validation only; no model training or benchmark evaluation is performed.

## Data policy

Raw neural datasets are not committed to this repository. Data files, derived datasets, model checkpoints, generated results, local logs, and experiment artifacts are ignored by Git. Future data ingestion must follow dataset licenses, access terms, and ethical requirements.

## Reproducibility principles

LatentBrain uses validated configuration, explicit random seeds, isolated environments, and automated quality checks. Future experiments should record the Git commit, configuration, environment summary, data provenance, split seed, and artifact metadata needed to audit results.

## Safety warning

Do not commit `.env`, raw data, checkpoints, generated results, W&B keys, cloud credentials, API tokens, private absolute paths, or local experiment artifacts.
