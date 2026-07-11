# LatentBrain

LatentBrain is a Python 3.11 research codebase for leakage-safe latent-variable analysis of motor-cortical population activity from the Neural Latents Benchmark MC_Maze datasets.

## What the project studies

The project asks whether low-dimensional states inferred from single-trial spiking can predict held-out neurons and preserve meaningful relationships to hand and cursor movement. It emphasizes reproducible data provenance, train-only model selection, repeated cross-validation, invalid-control separation, and claim safety.

## Main findings

- Initial fixed-split and globally cropped evaluations were insufficient. Final reporting uses trial-aware, peak-speed-centered 1.28-second windows and 5-fold × 5-repeat stratified cross-validation.
- Factor latents are the carried-forward valid model on MC_Maze Small.
- Nested-selected factor latents are the strongest valid tested MC_Maze Large model under the frozen protocol.
- LFADS-style and deterministic neural-ODE single-repeat feasibility pilots were positive and stable but did not pass predeclared gates for full multi-repeat evaluation. Both branches are retired; neural-model search is closed.
- Out-of-fold Large factor latents predict hand/cursor kinematics, decode endpoint direction, preserve relational trajectory geometry, and contain predictive structure beyond a scalar population-rate signal.

Findings are local, associative, and predictive. They are not causal claims or official NLB leaderboard results. Small and Large score differences are not interpreted as direct model-performance improvement.

## Evaluation protocol

Final protocols use:

- train-heldout per-neuron mean rate as the unified bits/spike reference;
- trial-aware event-centered extraction before rebinning;
- exact repeated stratified folds and fixed neuron masks within each repeat;
- nested hyperparameter selection using outer-training trials only;
- repeat-level paired uncertainty rather than treating correlated folds as independent;
- invalid split-mean target-reading controls excluded from model ranking;
- out-of-fold latent interpretation and train-only Procrustes alignment.

MC_Maze Large final evaluation shape is `[500, 64, 162]` at 20 ms, with 122 held-in and 40 held-out neurons.

## Reproducing the work

Install:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,neurodata]"
```

Validate:

```powershell
python -m latentbrain.cli validate-config
python scripts/check_environment.py
pytest -q
```

Data are never downloaded automatically. See [the reproducibility guide](docs/latentbrain_reproducibility.md) for DANDI assets, hashes, CPU/GPU requirements, and complete workflow commands.

## Repository structure

```text
configs/              validated experiment and release contracts
scripts/              thin command-line entry points
src/latentbrain/data  ingestion, validation, splits, rebinning, provenance
src/latentbrain/eval  scoring, baselines, audits, interpretability, release checks
src/latentbrain/models and train  tested neural feasibility implementations
src/latentbrain/reporting        claim-safe report helpers
tests/                CPU-safe unit and script tests
docs/                 research, methodology, provenance, and release documents
```

Generated `data/`, `results/`, `reports/`, checkpoints, latents, figures, and caches remain ignored.

## Research reports

- [Final research report](docs/latentbrain_research_report.md)
- [Claim registry](docs/latentbrain_claim_registry.md)
- [Reproducibility guide](docs/latentbrain_reproducibility.md)
- [Release notes](docs/latentbrain_release_notes.md)
- [Real-data record](docs/real_data.md)
- [Research methodology](docs/research_methodology.md)

## Limitations and claim safety

Single-split results are non-reportable. Invalid controls never rank as models. Neural pilots use one held-out-neuron mask and cannot establish final multi-repeat superiority. Latent axes are rotationally non-identifiable. Temporal autocorrelation explains part of continuous decoding. Observational analysis cannot establish a biological mechanism. Cross-dataset score differences are not direct performance comparisons.

## Development checks

```powershell
mypy --version
ruff check .
ruff format --check .
mypy src
pytest -q
python -m latentbrain.cli validate-config
python scripts/check_environment.py
git diff --check
```

Expected mypy version: `2.2.0`.
