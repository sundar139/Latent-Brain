# LatentBrain reproducibility

## Reproducibility levels

- **Unit-test reproducibility:** CPU-only, no raw data, internet, CUDA, or ignored results required.
- **Analysis reproducibility:** requires legally acquired DANDI assets and local processed data; non-neural analyses are CPU-capable.
- **GPU training reproducibility:** LFADS-style and deterministic neural-ODE pilot training require CUDA and reproduce expensive feasibility artifacts, not ordinary tests.

## Environment

Supported Python: 3.11 or newer within project constraints.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,neurodata]"
```

Development pins `mypy==2.2.0`. Verify setup:

```powershell
mypy --version
python -m latentbrain.cli validate-config
python scripts/check_environment.py
```

## Raw data and hashes

LatentBrain never downloads data automatically. Place DANDI assets under:

```text
data/raw/nlb/mc_maze_small/000140/
data/raw/nlb/mc_maze_large/000138/
```

MC_Maze Small: DANDI 000140, accepted version 0.220113.0408.

```text
train NWB SHA-256: dcaf3a524e2b2f65f163ee3b07789b8474bdfc6ca66098bc542ab93dff489884
test NWB SHA-256:  ca8a0bef5f189eafb9db1961d617e065cb461360718fde1c16a538922a6fa5fe
processed hash:    7ed048df5fab3cb8e7c82957c24619a29154800364231467af2deaba65fb6d9f
```

MC_Maze Large: DANDI 000138, version 0.220113.0407.

```text
train NWB SHA-256: 1188ddf5b822dd9ac49afb8916f4b42a85867613cf7f169d0b04a1d1aeecd700
test NWB SHA-256:  c65e12e9f484653ec8d85b265fc7e0e93789fd5f760c6adddd54759ac48974f1
processed hash:    074f6d693ba59b23c7e3449633d7c66171c9b52b22379047b414067036830c84
```

## Small workflow

```powershell
python scripts/inspect_nlb_files.py --root data/raw/nlb/mc_maze_small
python scripts/prepare_nlb_data.py --config configs/nlb_mc_maze_small.yaml
python scripts/run_window_audit.py --config configs/mc_maze_small_window_audit.yaml
python scripts/run_recommended_window_cv.py --config configs/mc_maze_small_recommended_window_cv.yaml
python scripts/run_unified_scoreboard.py --config configs/mc_maze_small_unified_scoreboard.yaml
```

Small is frozen. Expensive historical neural workflows are not required to reproduce its accepted report.

## Large workflow

Ingestion and protocol:

```powershell
python scripts/inspect_nlb_files.py --root data/raw/nlb/mc_maze_large
python scripts/prepare_nlb_data.py --config configs/nlb_mc_maze_large.yaml
python scripts/run_window_audit.py --config configs/mc_maze_large_window_audit.yaml
python scripts/run_recommended_window_cv.py --config configs/mc_maze_large_recommended_window_cv.yaml
python scripts/run_baseline_suite.py --config configs/mc_maze_large_baseline_suite.yaml
```

GPU feasibility pilots and CPU-capable frozen-checkpoint diagnostics:

```powershell
python scripts/run_lfads_pilot.py --config configs/mc_maze_large_lfads_pilot.yaml
python scripts/run_lfads_diagnostics.py --config configs/mc_maze_large_lfads_diagnostics.yaml
python scripts/run_neural_ode_pilot.py --config configs/mc_maze_large_neural_ode_pilot.yaml
python scripts/run_neural_ode_diagnostics.py --config configs/mc_maze_large_neural_ode_diagnostics.yaml
```

The two pilot commands require a CUDA-enabled PyTorch build and compatible NVIDIA GPU. Diagnostics load accepted checkpoints and can run without CUDA where configured. Neither retired pilot needs rerunning for unit tests.

Interpretability, scoreboard, and release:

```powershell
python scripts/run_latent_interpretability.py --config configs/mc_maze_large_latent_interpretability.yaml
python scripts/run_unified_scoreboard.py --config configs/mc_maze_large_unified_scoreboard.yaml
python scripts/run_release_audit.py --config configs/latentbrain_release.yaml
```

These analysis workflows are non-neural and do not schedule model training. The interpretability run can be computationally expensive but does not require CUDA.

## Seed policy

Every stochastic workflow records explicit seeds. Trial-fold and neuron-mask seeds are separated from FactorAnalysis initialization states. Neuron masks remain fixed within a repeat. Inner selection uses outer-training trials only. Neural pilots use the same five initialization seeds per fold and checkpoint selection on inner validation.

## Development checks

Bash-equivalent commands are shown because this agent environment uses Git Bash:

```bash
mypy --version
ruff check .
ruff format --check .
mypy src
pytest -q
python -m latentbrain.cli validate-config
python scripts/check_environment.py
git diff --check
```

Tests are designed to run without internet, CUDA, raw Large data, or existing ignored results.

## Artifact policy

`data/`, `results/`, `reports/`, checkpoints, fitted latents, figures, caches, and `.env` are ignored. Generated release-audit artifacts stay under `results/release_audit/`. Commit source, configs, tests, and documentation only. Never present local results as official benchmark submissions.
