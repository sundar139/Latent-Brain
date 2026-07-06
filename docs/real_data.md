# Real Data

LatentBrain targets MC_Maze-style Neural Latents Benchmark data as the first real neural dataset family because it exercises trial-aligned population spiking tensors, held-in and held-out neuron contracts, and reproducible split handling without requiring model training.

## Current capability

Real-data support is a local ingestion scaffold. It validates configuration, checks for local NLB-style files, attempts to adapt local files into the existing `NeuralDataset` schema, records provenance, and saves processed arrays only when real local data can be loaded.

No model has been trained. No benchmark result exists. No public leaderboard submission is claimed or planned here; the public NLB challenge has ended, so future evaluation should be local and reproducible.

## Synthetic validation versus real-data validation

Synthetic data validates shape contracts, split leakage checks, masks, hashing, and save/load behavior with deterministic arrays.

Real-data validation applies the same core contracts to externally obtained recordings. It must additionally preserve provenance, dataset version details, file manifests, local configuration snapshots, and session identifiers when available.

## Manual dataset policy

LatentBrain does not download MC_Maze automatically. Obtain data only from public resources or institutions that grant you access under their published terms. Store the files locally and keep them out of Git.

Default local path:

```text
data/raw/nlb
```

You may override the local root with:

```text
LATENTBRAIN_NLB_ROOT=
```

Expected local files currently include `.nwb`, `.h5`, or `.hdf5` files under the configured root.

## Optional dependencies

The base development install does not require neural-data tooling:

```powershell
python -m pip install -e ".[dev]"
```

For local NLB preparation, use:

```powershell
python -m pip install -e ".[dev,neurodata]"
```

If `nlb-tools` is not available from pip in your environment, install it manually from the official Neural Latents Benchmark GitHub repository, then rerun the preparation script.

## Local preparation

```powershell
python scripts/prepare_nlb_data.py --config configs/nlb_mc_maze.yaml
```

If local data is missing, the script exits nonzero with manual download guidance and creates no fake outputs. If local data is present and readable, processed outputs are written under ignored `data/processed/nlb` paths.

## Storage and version control

Do not commit real dataset files, processed arrays, credentials, checkpoints, generated metrics, or experiment outputs. The repository tracks code, configs, tests, and documentation only.

## Future local evaluation

Future work may add local reproducible evaluation against validated splits. Such evaluation must record the Git commit, config snapshot, dataset provenance, split seed, artifact hashes, and all relevant environment details before any result is reported.
