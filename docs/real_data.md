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

## Deterministic neural-ODE refinement

MC_Maze Small deterministic neural-ODE refinement uses 20 ms bins and the fixed 1.28-second window:

```powershell
python scripts/refine_neural_ode.py --config configs/mc_maze_small_neural_ode_refinement.yaml
```

Switching dynamics collapsed to one dominant regime and did not improve local validation bits/spike, so this workflow refines deterministic objective/schedule choices instead of adding regimes. The current local goal remains beating the factor-latent unified score. If factor-latent is beaten, the next action should be multi-seed robustness before any claims.

## Switching deterministic latent dynamics

MC_Maze Small switching tuning uses 20 ms bins and a fixed 1.28-second window:

```powershell
python scripts/tune_switching_ode.py --config configs/mc_maze_small_switching_ode_tuning.yaml
```

The current local goal is to beat the factor-latent unified validation score under canonical train-heldout mean-rate scoring. The model is a soft switching neural-ODE-style generator, not full Bayesian rSLDS inference. If it beats factor-latent, the next step should be multi-seed robustness before any claims. Generated outputs and checkpoints remain ignored under `results/mc_maze_small/switching_ode_tuning/`.

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

## Local LFADS-style GRU smoke training

MC_Maze Small is now also used for short neural training smoke runs:

```powershell
python scripts/train_lfads_gru.py --config configs/mc_maze_small_lfads_gru.yaml
```

The run trains an LFADS-style sequential VAE foundation on held-in reconstruction only. It verifies that the PyTorch dataloaders, GRU encoder/generator, Poisson loss, KL warmup, gradient clipping, checkpointing, and report writing work on the real processed tensor. It is not full LFADS, does not claim held-out co-smoothing performance, and reports no official NLB leaderboard result.

Generated neural metrics, reports, and checkpoints are local artifacts under ignored `results/mc_maze_small/lfads_gru/` paths and must stay out of Git.
The real MC_Maze LFADS-style training and evaluation configs request CUDA explicitly; if CUDA is unavailable, scripts fail with a clear runtime message instead of falling back to CPU.

## Local LFADS-style held-out evaluation

After the short training smoke run has produced `results/mc_maze_small/lfads_gru/checkpoints/best_validation.pt`, evaluate held-out prediction from checkpointed LFADS-style factors with:

```powershell
python scripts/evaluate_lfads_gru.py --config configs/mc_maze_small_lfads_gru_eval.yaml
```

The evaluation loads the checkpoint, extracts factors from held-in spikes, and fits train-only ridge decoders for held-out neuron rates and optional behavior velocity targets. Validation and test trials are never used for decoder fitting or standardization. The script writes JSON, CSV, and Markdown outputs under ignored `results/mc_maze_small/lfads_gru_eval/` paths and creates no new checkpoints.

This MC_Maze Small evaluation is local only. It is not a full LFADS implementation, not a neural network retraining command, and not an official NLB leaderboard result.

## Local LFADS-style masked co-smoothing training

After held-in reconstruction and factor-based held-out evaluation pass, MC_Maze Small can run a masked co-smoothing training configuration:

```powershell
python scripts/train_lfads_gru.py --config configs/mc_maze_small_lfads_gru_cosmoothing.yaml
```

The model receives held-in spikes only and predicts all-neuron rates. Held-in reconstruction loss is computed on held-in targets, and held-out prediction loss is computed from train-trial held-out targets only. Validation and test held-out targets are evaluation-only: they can appear in local validation/test metrics but never in optimizer updates or train-only decoder fits. Generated metrics, reports, and checkpoints are local artifacts under ignored `results/mc_maze_small/lfads_gru_cosmoothing/` paths.

Evaluate the masked co-smoothing checkpoint with:

```powershell
python scripts/evaluate_lfads_gru.py --config configs/mc_maze_small_lfads_gru_cosmoothing_eval.yaml
```

The evaluation reports direct model held-out rates when the checkpoint output covers all neurons, and can also report the factor-decoder diagnostic for continuity with the previous LFADS-style evaluation. It creates JSON, CSV, and Markdown outputs under ignored `results/mc_maze_small/lfads_gru_cosmoothing_eval/` paths and creates no new neural-network checkpoint.

This MC_Maze Small masked co-smoothing run is local validation only. It is not full LFADS, not official NLB leaderboard performance, and all generated checkpoints and result files must remain out of Git.
The real MC_Maze masked co-smoothing configs also request CUDA explicitly and are intended for a CUDA-enabled PyTorch environment.

## Local window-matched comparison

The current MC_Maze Small neural runs use a 256-bin time window for fast local iteration. Earlier transparent baselines were originally run on the full processed trial window, so their headline numbers should not be treated as direct comparisons to cropped LFADS-style evaluations. Full-window neural evaluation is future work.

Run the local comparison pipeline with:

```powershell
python scripts/run_window_matched_comparison.py --config configs/mc_maze_small_window_matched_comparison.yaml
```

The pipeline reloads the processed dataset, verifies the configured dataset hash, applies the same 256-bin crop, recreates the deterministic split and neuron mask, recomputes mean-rate/ridge/factor-latent baselines on the cropped tensor, and evaluates existing LFADS-style checkpoints without training a new neural network. Generated tables and the Markdown report are written under ignored `results/mc_maze_small/window_matched_comparison/` paths. They are local reproducibility artifacts only, not official NLB leaderboard results.

## Local LFADS-style CUDA tuning

Run the controlled masked co-smoothing tuning workflow with:

```powershell
python scripts/tune_lfads_gru.py --config configs/mc_maze_small_lfads_gru_tuning.yaml
```

The workflow keeps the MC_Maze Small dataset hash, deterministic split, held-in/held-out mask, and 256-bin crop fixed while training a small deterministic CUDA grid. It compares validation bits/spike only to the window-matched mean-rate, factor-latent, and previous LFADS-style masked references. Results, reports, and checkpoints are written under ignored `results/mc_maze_small/lfads_gru_tuning/` paths. This is local validation tuning only, not official NLB leaderboard performance, and the model is LFADS-style only, not full LFADS.

CUDA is required for real MC_Maze LFADS-style tuning workflows. If a CUDA-enabled PyTorch build or GPU is unavailable, tuning fails before model training rather than falling back to CPU.

## Local LFADS-style diagnostic audit

After the controlled tuning workflow, audit the tuned checkpoint with:

```powershell
python scripts/audit_lfads_gru.py --config configs/mc_maze_small_lfads_audit.yaml
```

The audit uses the same MC_Maze Small processed dataset hash, deterministic split, held-in/held-out mask, and 256-bin window as the window-matched comparison and tuning runs. It checks loss scale, bits/spike reference agreement, held-out rate calibration, factor usage, target sparsity, direct-model diagnostics, and a tiny-subset overfit run. The tiny-subset overfit run is a local diagnostic only: it is used to test whether the current objective can reduce training loss on a very small slice, not to produce a benchmark result.

Generated audit tables, figures, reports, and optional checkpoints are local artifacts under ignored `results/mc_maze_small/lfads_audit/` paths. The audit is not official NLB leaderboard performance and the model remains LFADS-style only, not full LFADS.

## Local temporal rebinning diagnostic

MC_Maze Small is stored at 5 ms resolution, which can make validation held-out targets nearly all zero inside the short local comparison window. Rebin the processed data for diagnostics with:

```powershell
python scripts/run_temporal_rebinning_diagnostic.py --config configs/mc_maze_small_temporal_rebinning.yaml
```

The diagnostic verifies the processed dataset hash, recreates the deterministic split and neuron mask, evaluates 5 ms, 10 ms, and 20 ms versions over the same 1.28-second window, recomputes same-bin mean-rate and factor-latent baselines, and trains small CUDA LFADS-style masked co-smoothing runs at 10 ms and 20 ms. Rebinned arrays, metrics, figures, checkpoints, and reports remain local under ignored `results/mc_maze_small/temporal_rebinning/` paths.

These outputs are meant to decide whether coarser bins reduce sparsity enough to improve neural learning or merely alter metric scale. They are not official benchmark results, and bits/spike across different bin sizes should be read only as bin-specific diagnostics.

## Local LFADS-style rate calibration diagnostic

The next MC_Maze Small diagnostic uses the 20 ms bin size because temporal rebinning gave the best local LFADS-style validation result at that bin width. It keeps the 1.28-second window fixed and requires same-bin 20 ms references, because mean-rate, factor-latent, and LFADS-style values should only be interpreted when they share the same bin size, split, held-in/held-out mask, and likelihood convention.

Run the local diagnostic with:

```powershell
python scripts/run_lfads_rate_calibration.py --config configs/mc_maze_small_lfads_rate_calibration.yaml
```

The workflow calibrates existing direct held-out predictions using train trials only, tests mean-rate blending, and trains a small CUDA LFADS-style model with its output bias initialized from train-only firing rates. Validation and test targets are used only for evaluation. Outputs and checkpoints are written under ignored `results/mc_maze_small/lfads_rate_calibration/` paths and are not official benchmark results.

## Local LFADS-style coordinated dropout diagnostic

The coordinated dropout diagnostic also uses 20 ms binning because that bin size gave the best LFADS-style diagnostic result so far. The 1.28-second window, deterministic split, and held-in/held-out neuron mask are held fixed, and same-bin references are required for every comparison.

Run the local diagnostic with:

```powershell
python scripts/run_lfads_coordinated_dropout.py --config configs/mc_maze_small_lfads_coordinated_dropout.yaml
```

During training only, the workflow masks a random subset of held-in input neurons and feeds the masked held-in activity to the LFADS-style model. The original unmasked held-in and held-out spike counts remain the loss targets. Evaluation uses unmasked held-in inputs and compares direct held-out predictions and factor-decoder predictions against same-bin mean-rate, same-bin factor-latent, previous raw 20 ms LFADS, and rate-calibration references.

Outputs, figures, reports, and checkpoints are written under ignored `results/mc_maze_small/lfads_coordinated_dropout/` paths. They are local diagnostics only and are not official benchmark results.

## Local metric/reference audit

Run the MC_Maze Small metric audit with:

```powershell
python scripts/run_metric_audit.py --config configs/mc_maze_small_metric_audit.yaml
```

The audit rebins the processed 5 ms MC_Maze Small tensor to 20 ms bins and crops the same 1.28-second window used by the recent LFADS-style diagnostics. It recreates the deterministic trial split and held-in/held-out neuron mask, then scores mean-rate references, oracle controls, random controls, shuffled controls, and safely loadable existing reported metrics against a unified train-only held-out mean-rate reference.

Existing reported scores may not be directly comparable unless their reference log-likelihood, held-out spike-count denominator, split, neuron mask, bin size, and time window match the unified audit convention. The audit report explicitly marks reported-only metrics and missing output directories instead of treating them as re-scored predictions. Oracle controls use held-out targets directly and are not valid models. Generated files are local artifacts under ignored `results/mc_maze_small/metric_audit/` paths, not official leaderboard results.

## Local unified scoreboard

Run the canonical MC_Maze Small local scoreboard with:

```powershell
python scripts/run_unified_scoreboard.py --config configs/mc_maze_small_unified_scoreboard.yaml
```

The scoreboard uses the same 20 ms bins and 1.28-second window as the metric audit. It fixes train-heldout mean rate as the default reference, so the train-mean-as-model validation score is `0.0` bits/spike. Factor-latent is the current best valid local model under unified scoring, while LFADS-style models trail but are now evaluated against the same reference convention.

The old mean-rate values are historical-only because they used incompatible reference conventions. The oracle diagnostic remains an invalid-model upper bound. The scoreboard includes local LFADS/dynamics-family summaries when available, including canonical tuning, controller tuning, neural-SDE-style tuning, coordinated dropout, and raw LFADS rate-calibration summaries; if ignored summaries are absent on a fresh clone, configured known LFADS-family values are used as fallback. Generated files are local artifacts under ignored `results/mc_maze_small/unified_scoreboard/` paths, not official leaderboard results.

## Local canonical LFADS-style tuning

Run the canonical MC_Maze Small LFADS-family tuning workflow with:

```powershell
python scripts/tune_lfads_unified.py --config configs/mc_maze_small_lfads_unified_tuning.yaml
```

The workflow uses 20 ms bins and the same 1.28-second window as the unified scoreboard. It trains a small deterministic CUDA grid, evaluates every run with the train-heldout mean-rate reference, and selects by validation unified bits/spike. The immediate local goal is to beat the factor-latent unified score before moving to neural SDE, rSLDS, controller-input, or larger tuning work.

Generated tuning tables, reports, figures, config snapshots, and checkpoints live under ignored `results/mc_maze_small/lfads_unified_tuning/` paths. They are local artifacts only, not official NLB leaderboard results, and the model remains LFADS-style only.

## Local controller-style LFADS-family tuning

Run the controller-style MC_Maze Small workflow with:

```powershell
python scripts/tune_lfads_controller.py --config configs/mc_maze_small_lfads_controller_tuning.yaml
```

The workflow uses 20 ms bins and the same fixed 1.28-second window as the unified scoreboard. It trains a small CUDA grid with inferred inputs, evaluates every run with the train-heldout mean-rate reference, and selects by validation unified bits/spike. The current goal is to beat the factor-latent unified score before moving to neural SDE or rSLDS models.

Generated controller tuning tables, reports, figures, config snapshots, and checkpoints live under ignored `results/mc_maze_small/lfads_controller_tuning/` paths. They are local artifacts only, not official NLB leaderboard results, and the model is LFADS-style with inferred inputs, not full LFADS.

## Local neural-SDE-style latent generator tuning

Run the MC_Maze Small Euler/Euler-Maruyama latent generator workflow with:

```powershell
python scripts/tune_neural_sde.py --config configs/mc_maze_small_neural_sde_tuning.yaml
```

The workflow uses 20 ms bins and the same fixed 1.28-second window as the unified scoreboard. It trains a small CUDA grid, evaluates every run with the train-heldout mean-rate reference, and selects by validation unified bits/spike. The current goal is to beat the factor-latent unified score before adding rSLDS switching; the previous controller-style LFADS-family score is a secondary dynamics-family reference.

Generated neural-SDE-style tuning tables, reports, figures, config snapshots, and checkpoints live under ignored `results/mc_maze_small/neural_sde_tuning/` paths. They are local artifacts only, not official NLB leaderboard results. This is a compact Euler/Euler-Maruyama latent generator, not a full benchmarked neural SDE system, and old incompatible mean-rate values are not tuning targets.

## Storage and version control

Do not commit real dataset files, processed arrays, metadata generated from real data, credentials, checkpoints, generated metrics, or experiment outputs. The repository tracks code, configs, tests, and documentation only.

## Future local evaluation

The first real-data run is validation only. The mean-rate baseline is the initial local metric sanity check and is not an official benchmark score. No EvalAI submission is planned. Future work should inspect alignment choices, behavior extraction, trial length distribution, and spike statistics before stronger local reproducible evaluation is added.

## Deterministic neural-ODE-style MC_Maze Small tuning

The deterministic neural-ODE-style workflow uses the real MC_Maze Small processed tensor rebinned from 5 ms to 20 ms and cropped to the fixed 1.28-second window. It forces `diffusion_scale: 0.0`, uses train-heldout mean-rate as the canonical reference, and selects runs by validation unified bits/spike.

The current goal is to beat the factor-latent unified score before adding rSLDS switching or other architectural complexity. The previous neural-SDE-style score is a dynamics-family reference, while the oracle remains an invalid diagnostic because it uses held-out truth. Old incompatible mean-rate values are historical-only and are not tuning targets.

If deterministic tuning beats the factor-latent reference, the next local milestone should be multi-seed robustness on the same dataset/bin/window/split before adding more architecture. Generated outputs, checkpoints, reports, and figures stay under ignored `results/mc_maze_small/neural_ode_tuning/` paths.

## MC_Maze Small deterministic neural-ODE objective diagnostics

Run local objective redesign diagnostics with:

```powershell
python scripts/tune_neural_ode_objectives.py --config configs/mc_maze_small_neural_ode_objectives.yaml
```

These diagnostics use 20 ms rebinned MC_Maze Small data and the fixed 1.28-second window, the same deterministic split and held-in/held-out neuron mask as the other dynamics workflows, and train-heldout mean rate as the canonical reference. Diffusion stays disabled at exactly `0.0`.

Switching latent dynamics collapsed to one dominant regime locally and did not improve validation unified bits/spike. Deterministic neural-ODE refinement improved on the switching and neural-SDE runs, but only marginally, and still trailed the factor-latent unified reference. That is why this workflow varies the objective — held-out loss weighting, zero/positive spike-count weighting, a train-only rate-calibration auxiliary loss, drift regularization, and the KL schedule — rather than adding architecture.

Training losses may be weighted; evaluation always uses the canonical unweighted unified bits/spike metric. If no objective variant beats factor-latent, the next step should not be more architecture. It should be multi-seed robustness of the best dynamics model plus expanded baselines and datasets. Generated outputs, figures, and checkpoints remain ignored under `results/mc_maze_small/neural_ode_objectives/` and are not official NLB leaderboard results.

Objective variants all train under one shared seed so that score differences reflect the objective rather than initialization. The stored refinement reference was produced at a different seed, so compare objective variants against the same-seed `refined_baseline` row; the cross-seed reference comparison is uncontrolled and is retained only for scoreboard continuity.

## MC_Maze Small multi-seed robustness

Run the seed-controlled local comparison with:

```powershell
python scripts/run_seed_robustness.py --config configs/mc_maze_small_multiseed_robustness.yaml
```

This benchmark uses 20 ms rebinned MC_Maze Small data and the fixed 1.28-second window, with train-heldout mean rate as the canonical reference. The trial split and held-in/held-out neuron mask are fixed across all methods and seeds; only the initialization and training seed varies. Every method uses the same seed list, and factor-latent, deterministic neural-ODE refinement, and the best controlled objective variant are all evaluated on it.

Each method reports mean, standard deviation, a bootstrap 95% confidence interval, and paired per-seed differences against factor-latent. A single-seed near-win is not evidence: the earlier deterministic neural-ODE result sat within the seed noise band measured here.

If neural ODE does not beat factor-latent across seeds, the next step should be rigorous reporting or additional datasets, not more architecture. If neural ODE beats factor-latent by the mean but not the confidence-interval lower bound, run more seeds before any claim. If neural ODE beats factor-latent robustly at the CI lower bound, the next step is held-out test reporting and additional datasets. Generated outputs, figures, and checkpoints remain ignored under `results/mc_maze_small/seed_robustness/` and are not official NLB leaderboard results.

## MC_Maze Small split and generalization audit

Run the CPU-only split audit with:

```powershell
python scripts/run_split_audit.py --config configs/mc_maze_small_split_audit.yaml
```

MC_Maze Small is small where it matters most. At the accepted 70/15/15 split there are 70 train
trials but only 15 validation and 15 test trials, so each evaluation split is a thin sample and a
single split's score carries little evidence on its own.

Multi-seed robustness found that every carried-forward method — factor-latent, deterministic
neural-ODE refinement, and the best controlled objective variant — is validation-positive and
test-negative under the canonical train-heldout mean-rate reference. Nothing measured so far
generalizes from the validation split to the test split.

This audit decides whether reporting is valid at all. It checks whether the validation and test
splits differ in trial spike rates, held-out neuron activity, or reach behavior; bootstraps the
paired validation/test gap; and refits factor-latent under ten independent splits with train-mean
and split-mean controls. If the test-negative pattern persists across splits, the conclusion is
that the dataset and window are underpowered, and the next step is cross-validation, additional
datasets, or more data — not more architecture. Under high generalization risk no performance
claim may be made, and results here are validation-only diagnostics. Generated outputs and figures
remain ignored under `results/mc_maze_small/split_audit/` and are not official NLB leaderboard
results.

## MC_Maze Small cross-validated rate audit

Run the CPU-only cross-validated rate audit with:

```powershell
python scripts/run_cv_rate_audit.py --config configs/mc_maze_small_cv_rate_audit.yaml
```

MC_Maze Small's 15-trial validation and 15-trial test splits are unstable. Across repeated trial
splits, factor-latent's test score changes sign, and its validation score spans a range wider than
any difference between the methods compared in this repository. A single-split number is therefore
not a final result and should not be treated as one. Report the repeated-split factor-latent
baseline instead, with its spread.

The estimator adds its own noise: sklearn `FactorAnalysis` uses randomized SVD, so changing only
its `random_state` moves the score. This workflow crosses split seeds with random states so the two
sources of variance can be told apart.

`split_mean_rate_invalid` predicts each evaluation split from that split's own held-out mean rate.
It is an invalid diagnostic — it reads evaluation targets — and it exists solely to size the
split-level rate effect, which is large. It can never be reported as model performance and never
competes for best valid model. Generated outputs and figures remain ignored under
`results/mc_maze_small/cv_rate_audit/` and are not official NLB leaderboard results.

## MC_Maze Small diagnostic status

Build the consolidated diagnostic report with:

```powershell
python scripts/build_mc_maze_diagnostic_report.py --config configs/mc_maze_small_diagnostic_report.yaml
```

The accepted status for MC_Maze Small at 20 ms bins and a 1.28-second window is as follows.

Repeated-split reporting is required. The 15-trial validation and test splits are unstable, and a
single 70/15/15 split carries little evidence in either direction. Report the repeated-split
factor-latent baseline with its spread, never a single-split number.

Factor-latent is the carried-forward valid baseline. No neural method beats it by mean or by
confidence-interval lower bound. The LFADS-family models, neural-SDE, deterministic neural-ODE
refinement, its objective variants, and switching latent dynamics are all negative or historical
diagnostics; the deterministic neural-ODE near-win was seed-specific and did not survive multi-seed
evaluation.

The `split_mean_rate_invalid` control is invalid. It predicts each evaluation split from that
split's own held-out mean rate, and its large advantage is per-neuron evaluation-target leakage
rather than a global rate offset a valid model could learn — oracle rescaling recovers almost none
of it, and a train-only rate calibration gains essentially nothing. It is a leakage diagnostic and
is never model performance.

The recommended next work is larger or additional datasets, or cross-validated reporting on this
one. Generated report bundles remain ignored under `reports/mc_maze_small_diagnostic/` and are not
official NLB leaderboard results.

## MC_Maze Small stratified cross-validation status

Run the CPU-only stratified cross-validation with:

```powershell
python scripts/run_stratified_cv.py --config configs/mc_maze_small_stratified_cv.yaml
```

Random 70/15/15 splits of MC_Maze Small are unstable. With only 15 validation and 15 test trials, a
single split's score depends heavily on which trials it happened to draw, and repeated random splits
average that away without ever guaranteeing that a fold contains a balanced spread of reach
directions, distances, speeds, or firing rates.

Behavior-stratified cross-validation is the preferred protocol. Folds are built so that endpoint
direction, endpoint distance, speed, population rate, and held-out rate are comparable across folds,
and fold balance is reported alongside every score.

Factor-latent remains the carried-forward valid baseline and is reported as a stratified
cross-validation mean with its spread and confidence interval, never as a single-split number. The
`split_mean_rate_invalid` control reads each evaluation fold's own held-out targets; it remains a
leakage diagnostic only, is never model performance, and never competes for best valid model.
Generated outputs and figures remain ignored under `results/mc_maze_small/stratified_cv/` and are not
official NLB leaderboard results.

## MC_Maze Small movement-window status

Run the CPU-only movement-window and alignment audit with:

```powershell
python scripts/run_window_audit.py --config configs/mc_maze_small_window_audit.yaml
```

The audit found a moving-bin fraction of zero in `from_start_1p28s`. Previous factor-latent and neural
results on that crop are therefore early/pre-movement diagnostics, not reach-dynamics results. The
carried-forward MC_Maze Small window is `behavior_speed_peak_centered_1p28s`.

Confirm the recommended window under the frozen CPU-only stratified protocol with:

```powershell
python scripts/run_recommended_window_cv.py --config configs/mc_maze_small_recommended_window_cv.yaml
```

Factor-latent remains the carried-forward valid baseline. Five repeats of five-fold stratified
cross-validation re-check whether the invalid split-mean control still dominates while reporting
movement coverage, endpoint-direction entropy, and fold balance. If confirmed, lack of invalid-control
dominance is a protocol diagnostic only: `split_mean_rate_invalid` still uses evaluation-fold targets,
never becomes model performance, and never participates in valid-model selection.

Recommended-window scores and `from_start` scores are different prediction targets, so their absolute
difference is not a model-performance improvement. Generated outputs remain ignored under
`results/mc_maze_small/recommended_window_cv/`; no official NLB leaderboard claim is made.

## MC_Maze Small final local status

The local diagnostic report now carries forward `behavior_speed_peak_centered_1p28s` under
`recommended_window_stratified_cross_validation`. At 20 ms with five folds by five repeats,
`factor_latent` has mean unified bits/spike `0.07707984048489147` and CI95
`[0.07143536625695274, 0.08251744011449201]`. It is the only carried-forward valid model.

The invalid `split_mean_rate_invalid` leakage control has mean `0.07110368937717054`, leaving
factor-latent minus invalid at `0.005976151107720928`; leakage dominance therefore does not persist
on the recommended window. This does not validate the control or establish a causal explanation:
the control still reads evaluation targets and is excluded from model performance.

Every earlier `from_start_1p28s` score remains an early/pre-movement diagnostic. Those values are not
directly comparable as performance improvements because the prediction target changed. Single-split
results remain unreportable, and there is no official leaderboard claim. The next major move is
transfer of this frozen protocol to MC_Maze Large, not additional MC_Maze Small model tuning.

## MC_Maze Large ingestion

The frozen MC_Maze Small protocol above is unchanged. MC_Maze Large is a separate ingestion target
that reuses the same adapter, trialization, validation, provenance, and serialization code with a
variant-specific config, `configs/nlb_mc_maze_large.yaml`.

Verified source identifiers, read from the DANDI API on 2026-07-10 rather than assumed:

```text
provider: dandi
dandiset_id: 000138
dandiset_version: 0.220113.0407
doi: 10.48324/dandi.000138/0.220113.0407
name: MC_Maze_Large: macaque primary motor and dorsal premotor cortex spiking activity during delayed reaching
web: https://gui.dandiarchive.org/#/dandiset/000138
```

Verified assets, downloaded with `dandi download DANDI:000138/0.220113.0407` and confirmed locally by
chunked SHA-256 against the digests recorded in the config:

```text
data/raw/nlb/mc_maze_large/000138/sub-Jenkins/sub-Jenkins_ses-large_desc-train_behavior+ecephys.nwb
  148590536 bytes  sha256 1188ddf5b822dd9ac49afb8916f4b42a85867613cf7f169d0b04a1d1aeecd700
data/raw/nlb/mc_maze_large/000138/sub-Jenkins/sub-Jenkins_ses-large_desc-test_ecephys.nwb
  802352 bytes     sha256 c65e12e9f484653ec8d85b265fc7e0e93789fd5f760c6adddd54759ac48974f1
```

The DANDI CLI writes a `000138/` dandiset directory; recursive discovery handles that layout, so the
files are not flattened. The raw assets are read-only inputs and are unchanged after ingestion
(digests re-verified post-run).

Local paths, both gitignored:

```text
data/raw/nlb/mc_maze_large
data/processed/nlb/mc_maze_large
```

Nothing is downloaded automatically; `source.automatic_download` is validated to be false. Downloads
are a manual operator action. When the raw directory is empty, `scripts/inspect_nlb_files.py` and
`scripts/prepare_nlb_data.py` both print `status: missing_raw_data` with the expected raw directory,
the verified source metadata, the candidate assets, `automatic_download_performed: false`, the
expected 142.5 MB download size, and the manual command `dandi download DANDI:000138/0.220113.0407`.
Both exit with code 2 and create nothing.

### Variant detection

`describe_nwb_file` reads NWB/HDF5 header metadata directly with `h5py` (no pynwb, no nlb_tools):
identifier, session description, session id, subject, acquisition series, processing modules, trial
table and trial count, unit count, and behavior series candidates. `detect_variant` reads
`general/session_id` first, which the real NLB MC_Maze assets set to exactly `small` or `large`; the
session description does not name the variant. It then falls back to `mc_maze_<variant>` or
`ses-<variant>` tokens in other metadata, and only to the filename when metadata is unreadable,
recording which evidence was used. `enforce_dataset_variant` then rejects a
Small file requested under the Large config, and rejects a directory that mixes incompatible
sessions. Detection is skipped for configs whose variant is not one of `small`, `medium`, `large`.

### Behavior mapping

Required canonical behavior channels are `hand_pos_x`, `hand_pos_y`, `cursor_pos_x`, `cursor_pos_y`,
declared in `behavior.required_names`. `behavior.aliases` maps a differently named source channel to
a canonical name, and the resulting source-to-canonical mapping is recorded in metadata under
`ingestion_summary.behavior_mapping`. Missing required channels are a hard error. Missing behavior
channels are never fabricated.

### Spike conservation

Trialization records a conservation report in `ingestion_summary.spike_conservation`: raw spike
count over the trialized dataframe, retained count, excluded count, excluded bins, and the exclusion
reason. Exact conservation only holds when all trials share a length. Under the documented
`crop_to_min` policy, variable-length trials lose their tail bins, and the report quantifies exactly
how many bins and spikes were excluded. For reference, MC_Maze Small excludes 36288 of 131669 spikes
across 82610 cropped bins under this policy. This is a documented, quantified exclusion, not silent
data loss.

### Determinism and hashing

Trial splits and held-in/held-out neuron masks are generated deterministically from `splits.seed`
and are validated as disjoint and exhaustive. Split and mask indices are written into the processed
`.npz` as `train_indices`, `validation_indices`, `test_indices`, `heldin_indices`, and
`heldout_indices`. Derived, non-identifying descriptions live in the `ingestion_summary` metadata
key, which is excluded from the dataset hash payload; the MC_Maze Small hash
`7ed048df5fab3cb8e7c82957c24619a29154800364231467af2deaba65fb6d9f` is therefore unchanged and still
reproduces exactly through the updated ingestion path. Provenance records the source provider,
verified DANDI identifiers, asset paths, sizes, raw file hashes, package version, git commit, config
path, config digest, processed dataset hash, and the creation command.

### Real MC_Maze Large ingestion result

Produced by `python scripts/prepare_nlb_data.py --config configs/nlb_mc_maze_large.yaml` from the
verified assets above. The train asset supplies all targets; the test asset is recognized, recorded
in metadata and provenance, and never used as a supervised behavior or spike target.

```text
source_files_used:  sub-Jenkins_ses-large_desc-train_behavior+ecephys.nwb (targets)
                    sub-Jenkins_ses-large_desc-test_ecephys.nwb (metadata/provenance only)
spikes:             [500, 2006, 162]
behavior:           [500, 2006, 4]
behavior_names:     hand_pos_x, hand_pos_y, cursor_pos_x, cursor_pos_y
trial_count:        500
time_bins:          2006
neuron_count:       162
source_bin_size_ms: 5
split_counts:       train 350, validation 75, test 75
heldin_neurons:     122
heldout_neurons:    40
trialization:       crop_to_min
dataset_hash:       074f6d693ba59b23c7e3449633d7c66171c9b52b22379047b414067036830c84
```

Behavior contains no NaN or Inf values, so the `allow_behavior_nans: false` policy holds with no
missing-value handling required. The behavior alias map is the identity
(`hand_pos_x -> hand_pos_x`, and likewise for the other three channels): the source already names the
four required canonical channels, so no alias was needed and nothing was fabricated. The train file
also exposes `eye_pos` and `hand_vel` series, which are not part of the configured behavior contract
and are not extracted.

Spike conservation under `crop_to_min`:

```text
raw_spike_count:       916892
trialized_spike_count: 620097
excluded_spike_count:  296795
excluded_bins:         450340
conserved:             false
exclusion_reason:      crop_to_min
```

Warnings emitted by preparation:

```text
296795 spikes in 450340 bins were excluded by crop_to_min
variable-length trials were cropped to the minimum trial length
```

This exclusion is real and material: about 32.4 percent of raw spikes fall outside the common cropped
window, because Large trials vary in length and every trial is cropped to the shortest one. It is
quantified rather than silent. The `crop_to_min` policy is validation-oriented preprocessing, not a
benchmark-faithful preprocessing decision, and the upcoming movement-window work will re-window the
data rather than inherit this crop.

Preparation was run twice with the same config. The dataset hash, spikes, behavior, trial ids, split
indices, neuron masks, shapes, and deterministic metadata fields were identical across both runs, and
no extra processed files were created. Provenance records the same
`processed_dataset_hash` the artifact itself carries.

### Differences from Small

| Field | Small | Large |
|---|---|---|
| dandiset | 000140 | 000138 |
| trials | 100 | 500 |
| time bins | 2051 | 2006 |
| neurons | 142 | 162 |
| held-in / held-out | 107 / 35 | 122 / 40 |
| train / validation / test | 70 / 15 / 15 | 350 / 75 / 75 |
| source bin size | 5 ms | 5 ms |
| behavior channels | same four | same four |
| excluded spikes under crop_to_min | 36288 of 131669 | 296795 of 916892 |

Large has five times the trials, twenty more units, and a slightly shorter common cropped window.
Small's dimensions were not assumed anywhere; every Large number above is measured.

### Status and next phase

Real MC_Maze Large ingestion is complete and verified: assets downloaded and digest-checked, variant
identified from NWB metadata, spikes and behavior trialized and aligned, splits and neuron masks
deterministic, hash reproducible across two runs, and the frozen MC_Maze Small hash
`7ed048df5fab3cb8e7c82957c24619a29154800364231467af2deaba65fb6d9f` still reproducing unchanged.

No model was trained. No cross-validation was run. No official benchmark claim is made; these are
local ignored artifacts. Movement-window validation followed and is recorded below; it confirmed
that this globally cropped artifact must not supply event-centered windows, while leaving the
artifact and its hash unchanged as the model-input contract.

## MC_Maze Large movement-window audit

Run with:

```powershell
python scripts/run_window_audit.py --config configs/mc_maze_large_window_audit.yaml
```

Status: complete. No model was trained, scored, or cross-validated, no window was frozen into a
reporting protocol yet, and no official NLB leaderboard result is claimed. Outputs and figures are
local ignored artifacts under `results/mc_maze_large/window_audit/`.

### Trial-aware source

The audit does not read event-centered windows out of the globally cropped processed artifact. It
rebuilds trials from the verified raw train asset using the ingestion config recorded in the
processed provenance, producing a ragged `list[np.ndarray]` per trial with exact lengths, aligned
behavior, and deterministic trial identifiers. The representation conserves every raw spike.

```text
trial count:            500
neuron count:           162
trial length range:     2006 to 4141 source bins (10.03 s to 20.71 s at 5 ms)
source bin size:        5 ms
reporting bin size:     20 ms
behavior source:        hand_pos
behavior alias mapping: identity for hand_pos_x, hand_pos_y, cursor_pos_x, cursor_pos_y
raw spikes conserved:   true
```

### Crop-to-min impact on behavioral events

```text
raw spike count:                          916892
global crop retained spike count:         620097
fraction of raw spikes excluded:          0.3237
fraction of raw bins excluded:            0.3099
trials whose peak speed is inside crop:   0.600
trials whose movement onset is inside:    0.764
global crop suitable for window audit:    false
```

The global `crop_to_min` keeps the first 2006 bins of every trial. Median peak-speed time is 9.63 s
and median movement-onset time is 8.97 s, while the retained prefix ends at 10.03 s. The crop
therefore removes the peak-speed bin for 40% of trials and the movement-onset bin for 23.6% of
trials. It does **not** merely discard post-event tail data: it deletes the alignment events
themselves on a large minority of trials. Windows were extracted from the trial-aware
representation, so this audit does not depend on the global crop being adequate.

This is the concrete reason the accepted `[500, 2006, 162]` array must not be used as the source of
event-centered windows. It remains valid as the frozen model-input contract and its hash is
unchanged.

### Candidate windows

Behavior coverage only. No model score contributed to any column.

| window | duration | clipped | moving-bin | peak coverage | onset coverage | direction entropy | usable |
|---|---|---|---|---|---|---|---|
| behavior_speed_peak_centered_1p28s | 1.28 s | 0.000 | 0.856 | 1.000 | 0.756 | 1.819 | yes |
| behavior_speed_peak_centered_2p56s | 2.56 s | 0.000 | 0.534 | 1.000 | 0.948 | 1.578 | yes |
| behavior_movement_onset_1p28s | 1.28 s | 0.004 | 0.792 | 0.922 | 1.000 | 1.859 | no |
| behavior_movement_onset_2p56s | 2.56 s | 0.004 | 0.548 | 0.996 | 1.000 | 1.539 | yes |
| from_start_1p28s | 1.28 s | 0.000 | 0.0001 | 0.000 | 0.004 | 1.029 | no |

No candidate required padding on any trial. `behavior_movement_onset_1p28s` was rejected because it
misses the peak-speed bin on 7.8% of trials: 1.28 s after onset is not always long enough to reach
the peak. `from_start_1p28s` was rejected as a pre-movement window: it contains essentially no
moving bins and never contains the peak. It remains an early-window diagnostic, exactly as on Small.

### Recommended window

```text
recommended window:            behavior_speed_peak_centered_1p28s
duration:                      1.28 s
clipped trial fraction:        0.000
moving bin fraction:           0.8555625
peak speed coverage:           1.000
movement onset coverage:       0.756
endpoint direction entropy:    1.819469420007445 nats
reporting mode:                recommended_window_stratified_cross_validation
selection used model scores:   false
```

### Transfer from MC_Maze Small

The frozen MC_Maze Small window `behavior_speed_peak_centered_1p28s` **transfers to Large**. It
passes every behavior gate: zero clipped trials, moving-bin fraction 0.856 against a 0.25 floor,
peak-speed coverage 1.000 against a 0.95 floor, and direction entropy 1.819 nats against a
1.456-nat floor. Because the transferred window satisfies the gates, it is preferred over the
longer peak-centered window and over the longer onset window, both of which are also usable but
neither shorter nor better aligned.

Differences from Small worth recording: Large's moving-bin fraction inside the recommended window is
higher (0.856 versus 0.577) and its endpoint-direction entropy is lower (1.819 versus 2.028 nats),
so Large reaches occupy more of the window while covering reach directions slightly less evenly.

### Movement-onset detection correction

On full-length Large trials the reach occupies a small minority of bins, so a pure 70th-percentile
speed threshold falls inside the resting jitter and movement onset collapsed to bin 0 for more than
75% of trials, even though hand speed at trial start is about 2.5% of peak. The onset threshold is
now floored at a fraction of each trial's speed range, the same criterion that defines a moving bin.
The floor defaults to zero, so MC_Maze Small's validated behavior is unchanged.

### Warnings

```text
global crop_to_min excludes peak speed or movement onset for at least one trial;
  event-centered windows were taken from the trial-aware representation
behavior_movement_onset_1p28s rejected: peak_speed_coverage_fraction=0.9220 below 0.95
from_start_1p28s rejected: moving_bin_fraction_mean=0.0001 below 0.25;
  endpoint_direction_entropy=1.0286 below 1.4556; peak_speed_coverage_fraction=0.0000 below 0.95
```

### Next phase

Recommended-window stratified cross-validation on MC_Maze Large, carrying
`behavior_speed_peak_centered_1p28s` forward at 20 ms bins. Single-split results remain
unreportable. No model has been run on Large and no benchmark claim is made.

## MC_Maze Large recommended-window cross-validation

Run with:

```powershell
python scripts/run_recommended_window_cv.py --config configs/mc_maze_large_recommended_window_cv.yaml
python scripts/run_unified_scoreboard.py --config configs/mc_maze_large_unified_scoreboard.yaml
```

Status: complete and frozen. No neural model was trained or tuned. No official NLB leaderboard
result is claimed. Outputs are local ignored artifacts under
`results/mc_maze_large/recommended_window_cv/`.

### Evaluation source and protocol

Event-centered windows were extracted from the trial-aware raw representation, never from the
globally crop-to-min processed array. Extraction happens at the 5 ms source bin size and the result
is rebinned to 20 ms afterwards, so summing counts preserves every spike inside the window.

```text
dataset_hash:        074f6d693ba59b23c7e3449633d7c66171c9b52b22379047b414067036830c84
trial_source:        trial_aware_raw (sub-Jenkins_ses-large_desc-train_behavior+ecephys.nwb)
trial length range:  2006 to 4141 source bins
window:              behavior_speed_peak_centered_1p28s (1.28 s, zero clipped, zero padded)
evaluation array:    [500, 64, 162]  (20 ms bins)
held-in / held-out:  122 / 40 neurons
folds x repeats:     5 x 5 = 25 fold evaluations
train / eval trials: 400 / 100 per fold (every fold held exactly 100 trials)
heldout_mask_policy: fixed_within_repeat
```

Every trial appears exactly once as evaluation per repeat. The held-out neuron mask is drawn once per
repeat and held fixed across that repeat's five folds, so folds differ only in trials. The
train-heldout mean-rate reference is recomputed from training trials only on every fold and scores
exactly `0.0` unified bits/spike against itself on all 25 folds.

### Factor-latent baseline

```text
mean:                 0.12271672423988657
std:                  0.025404656817326
CI95:                 [0.11323877517640364, 0.13279417856463843]
positive fold fraction: 1.0  (25 of 25 folds)
between-repeat std:   0.026885468208411117
within-repeat std:    0.0067354761061974855
```

The confidence interval lies entirely above the train-mean reference of `0.0`. Factor-latent is
therefore the first valid MC_Maze Large baseline under this protocol. Between-repeat variation
exceeds within-repeat variation, so the neuron mask, not the trial fold, drives most of the spread.

### FactorAnalysis random-state sensitivity

The FactorAnalysis random state is configured explicitly and is never derived from the fold index or
repeat index. All 25 folds were rescored at each of five states.

```text
random states:  0, 2027, 2028, 2029, 2030
mean by state:  0.122717, 0.121704, 0.121694, 0.121956, 0.122013
range:          0.0010229403760970285
std:            0.00041710575451157797
warning:        none  (tolerance 0.005 bits/spike)
```

The baseline is stable with respect to the FactorAnalysis seed: the spread is about 40 times smaller
than the fold-to-fold standard deviation.

### Invalid leakage control

```text
split_mean_invalid_mean:  0.008967375250262581
split_mean_invalid_std:   0.002695464410988248
factor_latent_minus_invalid: 0.113749348989624
factor beats invalid on the mean:  true
factor beats invalid on folds:     1.0  (25 of 25)
leakage_dominance_persists:        false
```

The split-mean control reads each evaluation fold's own held-out-neuron means, so it can never be
reported as model performance. Beating it is a leakage diagnostic, not a benchmark result. On Large
the control is much weaker than on Small (0.0090 versus 0.0711 bits/spike): averaging targets over
100 evaluation trials leaks far less than averaging over Small's 20.

### Fold balance

```text
trial count per fold:        100 for every fold (perfectly balanced)
strata per fold:             65 to 68
population-rate fold range:  0.0177 Hz
held-out-rate fold range:    0.0232 Hz
endpoint-distance fold range: 2.1201
mean-speed fold range:       2.2840
endpoint-direction entropy:  1.8060 (max 2.0794), not concentrated
fold_balance_warning:        none
```

Fold quality was judged on behavior and rate balance only. Invalid-control scores were never used to
judge fold quality.

### Comparison with MC_Maze Small

Small and Large differ in trials, neurons, firing rates, and target distributions. Only protocol
stability and leakage diagnostics are compared. **No cross-dataset model-performance improvement is
claimed, and the Large factor-latent mean must not be read as better than Small's.**

| field | mc_maze_small | mc_maze_large |
|---|---|---|
| fold count x repeats | 5 x 5 | 5 x 5 |
| eval trials per fold | 20 | 100 |
| factor-latent mean | 0.077080 | 0.122717 |
| factor-latent std | 0.014665 | 0.025405 |
| factor-latent CI95 | [0.071435, 0.082517] | [0.113239, 0.132794] |
| positive fold fraction | 1.0 | 1.0 |
| split-mean invalid mean | 0.071104 | 0.008967 |
| factor minus invalid | 0.005976 | 0.113749 |
| moving-bin fraction | 0.576875 | 0.855563 |
| endpoint-direction entropy | 2.028389 | 1.819469 |

Permitted conclusions, as generated:

- Large fold-to-fold variance is wider than Small under this protocol.
- Large and Small have the same positive-fold fraction.
- Leakage dominance does not persist on Small.
- Leakage dominance does not persist on Large.

### Frozen protocol

`results/mc_maze_large/recommended_window_cv/recommended_window_protocol.yaml` freezes the source
dataset hash, trial-aware extraction, the window name and duration, the extract-before-rebin policy,
the 20 ms target bin size, 5 folds x 5 repeats, the stratification settings, the fixed-within-repeat
held-out-neuron policy, the train-heldout mean-rate reference, the factor-latent settings, and the
claim-safety flags (`single_split_results_reportable: false`,
`old_mean_rate_values_used_as_targets: false`, `official_leaderboard_claim: false`,
`invalid_controls_excluded_from_model_selection: true`).

### Next phase

Valid Large baseline expansion, then reevaluation of the LFADS-style, neural-ODE, and neural-SDE
models under this frozen protocol. No neural model has been run on Large.

## MC_Maze Large valid baseline suite

Run with:

```powershell
python scripts/run_baseline_suite.py --config configs/mc_maze_large_baseline_suite.yaml
python scripts/run_unified_scoreboard.py --config configs/mc_maze_large_unified_scoreboard.yaml
```

Status: complete, baseline frozen. **No neural model was trained, tuned, or scored.** No official NLB
leaderboard result is claimed. Outputs are local ignored artifacts under
`results/mc_maze_large/baseline_suite/`.

### Protocol reuse and nested selection

The 25 accepted outer folds were read verbatim from
`recommended_window_fold_assignments.csv`, and each repeat's held-out neuron mask was recreated from
the frozen base seed `2027 + repeat_index`. Folds were never regenerated. Hyperparameters were chosen
by 3 inner folds cut only from each outer fold's 400 training trials; the winner was refit on those
400 trials and the outer-evaluation fold was scored exactly once. 2475 inner evaluations ran in total
(12, 12 and 9 configurations for the three train-selected methods).

`factor_latent_fixed` reproduced the accepted recommended-window mean exactly:

```text
reproduced mean: 0.12271672423988657
accepted mean:   0.12271672423988657
difference:      0.0
```

That exact reproduction is the evidence that the folds and masks are the accepted ones.

### Valid baselines

| method | family | mean | std | CI95 | positive fraction | between-repeat std | within-repeat std |
|---|---|---|---|---|---|---|---|
| factor_latent_train_selected | factor_latent | 0.135545 | 0.026166 | [0.125743, 0.146088] | 1.0 | 0.027284 | 0.008189 |
| factor_latent_fixed | factor_latent | 0.122717 | 0.025405 | [0.113239, 0.132794] | 1.0 | 0.026885 | 0.006735 |
| smoothed_cosmoothing_ridge | ridge | 0.121562 | 0.025822 | [0.111850, 0.131790] | 1.0 | 0.027718 | 0.005434 |
| reduced_rank_cosmoothing | reduced_rank | 0.087901 | 0.023851 | [0.078951, 0.097026] | 1.0 | 0.019528 | 0.016032 |

`reduced_rank_cosmoothing` is linear reduced-rank ridge regression. It carries no temporal or
dynamical assumptions and is neither GPFA nor a latent dynamical model.

Selected hyperparameters were identical on all 25 outer folds for every train-selected method, which
is itself a stability result:

```text
factor_latent_train_selected: latent_dim 16, smoothing_sigma_ms 200.0, heldout_decoder_alpha 10000.0
smoothed_cosmoothing_ridge:   smoothing_sigma_ms 240.0, alpha 100000.0
reduced_rank_cosmoothing:     smoothing_sigma_ms 160.0, alpha 10000.0, rank 8
```

Both co-smoothing families selected the heaviest smoothing and the strongest regularization offered,
which suggests the grids sit at their conservative edge rather than at an interior optimum.

### Paired comparisons against factor_latent_fixed

Comparison unit is the repeat, with a hierarchical paired bootstrap over repeats and folds. The 25
folds are not independent.

| comparison | mean paired difference | CI95 | positive repeat fraction | supported |
|---|---|---|---|---|
| factor_latent_train_selected | +0.012828 | [0.011205, 0.014242] | 1.0 | yes |
| smoothed_cosmoothing_ridge | -0.001155 | [-0.003228, 0.000951] | 0.2 | no |
| reduced_rank_cosmoothing | -0.034816 | [-0.044943, -0.026459] | 0.0 | no |

Answering the milestone questions directly: a direct co-smoothing ridge does **not** outperform
factor-latent (its interval straddles zero and it wins on one repeat in five); a reduced-rank model
does **not** outperform full-rank ridge; and train-only nested selection **does** improve
factor-latent, by a small but perfectly consistent margin.

### Invalid leakage control

`split_mean_rate_invalid` scored 0.008967 mean (std 0.002695). It reads each evaluation fold's own
held-out targets, is excluded from selection, ranking and superiority testing, and never appears in
`paired_method_comparisons.csv`. It cannot become the baseline to beat. `train_mean_rate` scored
exactly 0.0 on all 25 folds, as the reference must.

### Baseline to beat

```text
baseline_to_beat:                factor_latent_train_selected
baseline_replaced:              true
baseline_replacement_supported: true
baseline mean:                  0.13554470127397905
baseline CI95:                  [0.12574272544462758, 0.14608796822823822]
```

`factor_latent_train_selected` cleared every declared gate: positive paired mean, bootstrap interval
excluding zero, and 100 percent positive repeats. Neural models must beat this, not the previous
`factor_latent_fixed` value.

### A leakage defect the real run exposed

The first real run reported `smoothed_cosmoothing_ridge` at 0.9459 bits/spike and replaced the
baseline. That was wrong. The smoothing cache in the baseline suite keyed on the held-in set's size
and first index rather than the set itself. Repeats 1 through 4 all produce a mask whose first
held-in index is 0 and whose size is 122, so they collided; repeats 2, 3 and 4 silently reused repeat
1's held-in columns, 29 of which are their own held-out target neurons. Only the co-smoothing
families used that cache, which is exactly why factor-latent still reproduced its accepted mean and
why the corruption was confined to three repeats. The cache now keys on the full held-in index array,
and a regression test constructs two same-size masks sharing a first index and asserts their features
differ. Every number above is from the corrected run.

### Neural reevaluation readiness

`neural_reevaluation_readiness.json` reports `ready: true` with no blockers. It is a plan artifact:
no neural experiment was run during this milestone. It fixes the dataset hash, window, bin size, fold
and mask sources, the baseline to beat and its interval, the repeat-level comparison unit, at least
five controlled neural seeds, five outer repeats, the claim-safety rules, checkpoint selection on
inner-training folds, and the forbidden old protocols (`from_start_1p28s`, `single_70_15_15_split`,
`seed_plus_run_index`, `evaluation_target_calibration`, `invalid_split_mean_as_model`).

### Next phase

Controlled neural-model reevaluation: prepare and approve the reevaluation manifest, then reevaluate
LFADS-style and deterministic neural-ODE models under this exact frozen protocol.

## MC_Maze Large LFADS feasibility pilot

Command:

```powershell
python scripts/run_lfads_pilot.py --config configs/mc_maze_large_lfads_pilot.yaml
```

The pilot is restricted to outer repeat 0, all five accepted folds, and initialization seeds
`2027-2031`, giving 25 LFADS-style runs under one fixed held-out-neuron mask. Each run receives only
the 122 held-in neurons and predicts positive rates for all 162 neurons; the 40 held-out neurons are
targets only. Event-centered inputs are extracted from the trial-aware 5 ms source before rebinning to
the accepted 20 ms, 64-bin movement window.

Checkpoint selection maximizes inner-validation unified bits/spike. The inner split is built only from
the 400 outer-training trials; no outer-evaluation trial enters checkpoint selection, early stopping,
normalization, calibration, or hyperparameter selection. The outer reference is recomputed from all
400 outer-training trials. The comparison target remains `factor_latent_train_selected`; fold-paired
differences are descriptive diagnostics, not a final superiority test.

Generated score, seed-stability, fold-stability, checkpoint, runtime, memory, recommendation, report,
figure, and checkpoint artifacts live under ignored `results/mc_maze_large/lfads_pilot/`. The generated
report records observed seed variation, paired baseline difference, compute, peak CUDA memory, and the
full-evaluation recommendation. This one-repeat pilot cannot support a final model claim and is not an
official NLB leaderboard result.

### Pilot result

All 25 runs completed with finite scores and losses. Mean unified bits/spike across fold-seed runs was
`0.02925965290281923` (run-level standard deviation `0.003563618625820026`). The five seed means ranged
from `0.02787411504791416` to `0.030709969818508542`; their standard deviation was
`0.001022292447963269`, and every seed mean was positive. The pilot-repeat
`factor_latent_train_selected` mean was `0.17392712874670385`, giving a descriptive mean paired
difference of `-0.1446674758438845`. No fold-seed run beat the baseline.

Training took 23.8 minutes in aggregate on the local RTX 4070 Laptop GPU. Individual runs averaged
57.1 seconds and allocated at most 54.3 MiB of CUDA memory; the observed-pilot estimate for 125 full
evaluation runs is about 1.98 hours, not a completion-time promise. Every selected checkpoint came
from inner validation and all explicit leakage checks passed.

Full evaluation is **not recommended** because the mean paired difference is worse than the
predeclared `-0.02` margin. The corrected movement window produced positive, initialization-stable
LFADS-style scores, but it did not resolve the earlier failure mode of trailing the valid
factor-latent baseline. Later neural-model phases were not started.
