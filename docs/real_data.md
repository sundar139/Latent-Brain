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
