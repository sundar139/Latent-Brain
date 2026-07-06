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
- Typer-based CLI sanity commands
- Ruff, mypy, pytest, pre-commit, and GitHub Actions quality checks

Not implemented yet:

- Dataset download or preprocessing
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

## Configuration and environment

The default configuration lives in `configs/base.yaml`. Local machine-specific values may be supplied through environment variables or a local `.env` file, but `.env` files must never be committed.

Copy `.env.example` only when local overrides are needed, then keep any real values private.

## Data policy

Raw neural datasets are not committed to this repository. Data files, derived datasets, model checkpoints, generated results, local logs, and experiment artifacts are ignored by Git. Future data ingestion must follow dataset licenses, access terms, and ethical requirements.

## Reproducibility principles

LatentBrain uses validated configuration, explicit random seeds, isolated environments, and automated quality checks. Future experiments should record the Git commit, configuration, environment summary, data provenance, split seed, and artifact metadata needed to audit results.

## Safety warning

Do not commit `.env`, raw data, checkpoints, generated results, W&B keys, cloud credentials, API tokens, private absolute paths, or local experiment artifacts.
