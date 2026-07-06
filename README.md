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
- Local NLB/MC_Maze ingestion scaffold with provenance capture
- Typer-based CLI sanity commands
- Ruff, mypy, pytest, pre-commit, and GitHub Actions quality checks

Not implemented yet:

- Automatic dataset download or full preprocessing pipelines
- Model training or inference
- LFADS, neural SDE, or switching dynamical system logic
- Neural Latents Benchmark evaluation
- Experiment runs, benchmark scores, plots, checkpoints, or reports

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

If `nlb-tools` is unavailable from pip in your environment, install it from the official Neural Latents Benchmark GitHub repository.

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

LatentBrain includes an NLB/MC_Maze-style local ingestion scaffold:

```powershell
python scripts/prepare_nlb_data.py --config configs/nlb_mc_maze.yaml
```

The script does not download data. Place legally obtained local files under `data/raw/nlb` or set `LATENTBRAIN_NLB_ROOT`. If files are missing, the script exits with guidance and creates no fake data. Real-data support is scaffolded only; no trained model, benchmark result, or leaderboard claim exists.

## Data policy

Raw neural datasets are not committed to this repository. Data files, derived datasets, model checkpoints, generated results, local logs, and experiment artifacts are ignored by Git. Future data ingestion must follow dataset licenses, access terms, and ethical requirements.

## Reproducibility principles

LatentBrain uses validated configuration, explicit random seeds, isolated environments, and automated quality checks. Future experiments should record the Git commit, configuration, environment summary, data provenance, split seed, and artifact metadata needed to audit results.

## Safety warning

Do not commit `.env`, raw data, checkpoints, generated results, W&B keys, cloud credentials, API tokens, private absolute paths, or local experiment artifacts.
