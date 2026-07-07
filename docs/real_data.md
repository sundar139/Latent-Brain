# Real Data

LatentBrain targets MC_Maze-style Neural Latents Benchmark data as the first real neural dataset family because it exercises trial-aligned population spiking tensors, held-in and held-out neuron contracts, and reproducible split handling without requiring model training.

## Current capability

Real-data support is a local ingestion path for MC_Maze Small. It validates configuration, checks for local NLB-style files, trializes continuous `NWBDataset.data` dataframes into the existing `NeuralDataset` schema, records provenance, and saves processed arrays only when real local data can be loaded.

No model has been trained. No benchmark result exists. No EvalAI or public leaderboard submission is claimed or planned here; the public NLB challenge has ended, so future evaluation should be local and reproducible.

## Synthetic validation versus real-data validation

Synthetic data validates shape contracts, split leakage checks, masks, hashing, and save/load behavior with deterministic arrays.

Real-data validation applies the same core contracts to externally obtained recordings. It additionally preserves provenance, dataset variant details, file manifests, local configuration snapshots, trialization settings, behavior column names, and session identifiers when available.

## Manual dataset policy

LatentBrain does not download MC_Maze automatically. Obtain data only from public resources or institutions that grant you access under their published terms. Store the files locally and keep them out of Git.

Recommended first target:

```text
MC_Maze Small DANDI: https://gui.dandiarchive.org/#/dandiset/000140
Official NLB datasets page: https://neurallatents.github.io/datasets
```

Local path for the first run:

```text
data/raw/nlb/mc_maze_small
```

You may override the local root with:

```text
LATENTBRAIN_NLB_ROOT=
```

Expected candidate files include `.nwb`, `.h5`, `.hdf5`, `.mat`, or `.npz`. MC_Maze Small uses the real train NWB file matching `*desc-train_behavior+ecephys.nwb` for target extraction. Test files such as `*desc-test_ecephys.nwb` are recorded in metadata and provenance but are not used to create supervised local targets.

## MC_Maze Small trialization

The real MC_Maze Small NWB loaded by `nlb_tools.NWBDataset` exposes continuous pandas dataframe data, not a direct `[trials, time, neurons]` tensor. LatentBrain calls NLB trial metadata through `NWBDataset.make_trial_data`, then extracts MultiIndex `spikes` columns and concatenates `heldout_spikes` after held-in spikes when available. It also extracts configured behavior groups from the train file, currently `hand_pos` and `cursor_pos` when present.

Trials may have different lengths. The current validation-oriented preprocessing policy is `crop_to_min`: crop every spike trial to the minimum positive trial length and record the original length distribution in metadata. Behavior uses `crop_to_spike_window`, so behavior arrays use the same trial IDs, ordering, and number of time bins as the spike tensor. Zero padding is intentionally not used because it can contaminate later likelihood calculations.

The output contract is:

```text
spikes: [n_trials, n_time_bins, n_neurons]
behavior: [n_trials, n_time_bins, n_behavior_dims] when available
behavior_names: [n_behavior_dims] when behavior is available
trial_ids: [n_trials]
time_ms: [n_time_bins]
```

Behavior arrays are saved only from train files that include supervised behavior. Test files such as `*desc-test_ecephys.nwb` are recorded in metadata and provenance but are not used as supervised behavior targets. This prevents test-target leakage while keeping local validation reproducible.

This fixed-length tensor is for validation-oriented local analysis. Velocity decoding and other behavior-decoding metrics are intentionally not implemented yet. Later baseline and evaluation work may need a more benchmark-faithful preprocessing path with alignment choices and held-out targets reviewed explicitly.

## Optional dependencies

The base development install does not require neural-data tooling:

```powershell
python -m pip install -e ".[dev]"
```

For local NLB preparation, use:

```powershell
python -m pip install -e ".[dev,neurodata]"
```

If `nlb-tools` is not available from pip in your environment, install it manually from the official Neural Latents Benchmark GitHub repository:

```powershell
python -m pip install git+https://github.com/neurallatents/nlb_tools.git
```

## Local preparation

1. Open the official Neural Latents Benchmark datasets page.
2. Go to MC_Maze.
3. Choose the Small DANDI repository first.
4. Download the dataset legally and ethically using DANDI or the DANDI web interface.
5. Place files under `data/raw/nlb/mc_maze_small`.
6. Inspect local files.
7. Prepare validated local arrays.

```powershell
python scripts/inspect_nlb_files.py --root data/raw/nlb/mc_maze_small
python scripts/prepare_nlb_data.py --config configs/nlb_mc_maze_small.yaml
```

If local data is missing, both commands exit nonzero with guidance and create no fake outputs. If local data is present and readable, processed `.npz`, metadata JSON, and provenance JSON outputs are written under ignored `data/processed/nlb/mc_maze_small` paths.

## MC_Maze Small validation report

After local preparation succeeds, run:

```powershell
python scripts/analyze_mc_maze.py --config configs/mc_maze_small_eda.yaml
```

The analysis reads the processed `.npz`, verifies the configured dataset hash, recreates deterministic train/validation/test splits and held-in/held-out masks, computes spike-count, firing-rate, and behavior availability summaries, and writes local JSON, CSV, Markdown, and PNG outputs under ignored `reports/mc_maze_small/` paths.

The generated Markdown report states that no model training or benchmark evaluation was performed. Behavior reporting is validation-only: no velocity R², decoder, or benchmark claim is produced.

## Local mean-rate baseline

The first local baseline is a mean-rate Poisson sanity check:

```powershell
python scripts/run_mean_rate_baseline.py --config configs/mc_maze_small_mean_rate.yaml
```

It reuses the deterministic trial split and held-in/held-out neuron mask from ingestion. The baseline fits one constant firing rate per neuron on train trials only, so validation and test spikes cannot affect fitted rates. It then reports Poisson negative log-likelihood and bits/spike-style improvement against a train-only global-rate reference for train, validation, and test splits across held-in, held-out, and all-neuron groups.

Held-in neurons are the local training-observed group; held-out neurons are reserved for co-smoothing-style sanity checks. This baseline matters because it validates split reuse, masking, likelihood math, and report generation before introducing LFADS, SDE, or other stronger modeling approaches. The output is a local sanity baseline, not an official NLB leaderboard result.

## Local behavior decoder baseline

The behavior decoder sanity baseline derives velocity targets from saved `hand_pos` and `cursor_pos` position channels using central differences in behavior units per second:

```powershell
python scripts/run_behavior_decoder.py --config configs/mc_maze_small_behavior_decoder.yaml
```

It smooths spike counts within each trial, uses held-in neurons by default, computes feature standardization from train trials only, computes target standardization from train trials only, and fits a ridge decoder on train samples only. Validation and test samples are used only for evaluation. Generated JSON, CSV, and Markdown outputs are local artifacts under ignored `results/mc_maze_small/behavior_decoder/` paths.

This baseline is a bridge toward future LFADS/SDE behavior-decoding work. It is not an official NLB leaderboard result and does not train a neural network model.

## Local co-smoothing ridge baseline

The co-smoothing sanity baseline predicts held-out neuron rates from held-in neuron activity:

```powershell
python scripts/run_cosmoothing_baseline.py --config configs/mc_maze_small_cosmoothing_ridge.yaml
```

It smooths held-in spikes within each trial, converts them to rates, computes train-only feature standardization, and fits a ridge decoder on train trials only. Held-out spikes from validation and test trials are used only for evaluation. A train-only held-out mean-rate reference supplies the bits/spike comparison.

This baseline is a bridge toward future GPFA/LFADS/SDE models. It is not an official NLB leaderboard result and does not train a neural network model.

## Local co-smoothing diagnostic sweep

The initial single co-smoothing ridge configuration produced validation held-out bits/spike below the train-only mean-rate reference. Before adding GPFA, LFADS, neural SDE, switching, or neural-network models, run a transparent diagnostic sweep:

```powershell
python scripts/run_cosmoothing_sweep.py --config configs/mc_maze_small_cosmoothing_sweep.yaml
```

The sweep evaluates smoothing sigma, ridge alpha, feature standardization, and intercept choices while keeping the leakage policy fixed: train trials only for fitting, held-in neurons as inputs, held-out neurons as targets, train-only feature standardization when enabled, and train-only held-out mean rates as the reference. It reports train, validation, and test metrics, selects the best configuration by validation held-out bits/spike, and writes local outputs under ignored `results/mc_maze_small/cosmoothing_sweep/` paths.

This sweep helps select and diagnose a transparent baseline before more complex latent-dynamics models are considered. It is not an official NLB leaderboard result and does not train a neural network model.

## Local factor latent baseline

After the mean-rate, behavior-decoder, co-smoothing ridge, and co-smoothing sweep checks pass, run the first non-neural latent-variable sanity baseline:

```powershell
python scripts/run_factor_latent_baseline.py --config configs/mc_maze_small_factor_latent.yaml
```

The baseline smooths held-in spikes into firing-rate features, computes feature standardization from train trials only, fits Factor Analysis on train held-in samples only, and transforms train/validation/test held-in samples into latent trajectories. Held-out rate decoding, behavior velocity decoding, and train-only held-out mean-rate references are fit from train trials only; validation and test samples are evaluation-only.

This is a transparent precursor to stronger GPFA/LFADS/SDE models. It is GPFA-style only because no temporal GP prior is implemented. It is not an official NLB leaderboard result and does not train a neural network model.

## Local factor latent diagnostic sweep

Tune the factor latent baseline with:

```powershell
python scripts/run_factor_latent_sweep.py --config configs/mc_maze_small_factor_latent_sweep.yaml
```

The sweep selects the best configuration by validation held-out bits/spike. Behavior mean R² is reported as a secondary metric, not as the selector. The train-only mean-rate held-out baseline remains the sanity reference for neural prediction, so the report includes that comparison alongside the previous single factor latent result.

The same leakage policy applies: train held-in features fit standardization and Factor Analysis, train latents fit held-out and behavior decoders, and validation/test samples are evaluation-only. This is local diagnostic output, not official benchmark performance, not full GPFA without a temporal GP prior, and no neural network model is trained.

## Storage and version control

Do not commit real dataset files, processed arrays, metadata generated from real data, credentials, checkpoints, generated metrics, or experiment outputs. The repository tracks code, configs, tests, and documentation only.

## Future local evaluation

The first real-data run is validation only. The mean-rate baseline is the initial local metric sanity check and is not an official benchmark score. No EvalAI submission is planned. Future work should inspect alignment choices, behavior extraction, trial length distribution, and spike statistics before stronger local reproducible evaluation is added.
