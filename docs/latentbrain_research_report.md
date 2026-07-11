# LatentBrain

## Executive summary

LatentBrain asks whether low-dimensional structure inferred from single-trial motor-cortical spiking can predict held-out neural activity and retain behaviorally meaningful population geometry. The principal methodological result is corrective: initial fixed-split and globally cropped evaluations were insufficient. Final reporting uses trial-aware, peak-speed-centered 1.28-second windows and repeated stratified cross-validation.

Under the frozen MC_Maze Large protocol, nested-selected factor latents remained the strongest valid tested model: 0.135545 mean unified bits/spike, 95% CI [0.125743, 0.146088], positive in every outer fold. Single-repeat LFADS-style and deterministic neural-ODE feasibility pilots were positive and seed-stable but failed predeclared gates for full five-repeat evaluation; both branches are retired. Out-of-fold factor latents predicted hand and cursor kinematics, decoded endpoint direction, and retained structure beyond a scalar population-rate signal.

These are local associative and predictive findings. They are not causal claims or official NLB leaderboard results. MC_Maze Small and Large scores are not interpreted as measurements on interchangeable benchmark populations.

## Research question

Can a leakage-safe latent-variable model compress motor-cortical population activity while predicting held-out neurons and preserving interpretable relationships to movement on unseen trials?

## Contributions

- Reproducible NWB ingestion with provenance, hashes, and trial-aware Large sequences.
- A canonical train-heldout mean-rate reference and unified bits/spike scorer.
- Behavior-aligned repeated cross-validation with fixed neuron masks inside repeats.
- Nested train-only selection across transparent non-neural baselines.
- Controlled one-mask LFADS-style and deterministic neural-ODE feasibility pilots with frozen gates.
- Out-of-fold kinematic decoding, trajectory geometry, stability, rate-confound, and shuffle analyses.
- Machine-readable claim and release audits separating valid evidence from invalid controls.

## Datasets and provenance

| Dataset | DANDI/version | Processed hash | Full processed shape | Behavior shape |
|---|---|---|---|---|
| MC_Maze Small | 000140 / 0.220113.0408 | `7ed048df5fab3cb8e7c82957c24619a29154800364231467af2deaba65fb6d9f` | [100, 2051, 142] | [100, 2051, 4] |
| MC_Maze Large | 000138 / 0.220113.0407 | `074f6d693ba59b23c7e3449633d7c66171c9b52b22379047b414067036830c84` | [500, 2006, 162] globally cropped | [500, 2006, 4] |

Large raw trials span 2006–4141 source bins at 5 ms. Global crop-to-min excluded 32.37% of raw spikes and 30.99% of raw bins and removed the peak-speed event from 40% of trials. Event-centered Large windows therefore come from the trial-aware raw source, not the cropped tensor.

## Leakage-safe evaluation design

Unified bits/spike is `(model log likelihood - train-reference log likelihood) / (log(2) × held-out spikes)`. The reference predicts each held-out neuron's outer-training mean rate. The split-mean control reads evaluation targets and is explicitly invalid: it is useful only as a leakage diagnostic and cannot enter valid ranking.

Final protocol:

1. Extract each peak-speed-centered 1.28-second window at 5 ms, then rebin to 20 ms.
2. Assign trials to five behavior/rate-stratified folds across five repeats.
3. Draw one held-in/held-out neuron mask per repeat; keep it fixed across that repeat's folds.
4. Reuse exact folds and masks across methods.
5. Select hyperparameters on inner folds cut only from each outer-training set.
6. Refit on all outer-training trials; evaluate the outer fold once.
7. Aggregate paired comparisons at repeat level.

Folds within one repeat overlap in training trials and share a neuron mask, so they are correlated, not 25 independent samples. Single-split performance is non-reportable.

## Why the movement window changed

The original from-start window contained little or no reach movement on Small. On Large, global crop-to-min also deleted behaviorally relevant alignment events. The final peak-speed-centered window captures movement while preserving a fixed 64-bin evaluation shape. Cross-window bits/spike differences are not interpreted as model improvements because each window defines a different prediction problem.

## MC_Maze Small findings

Small uses the same peak-speed-centered 1.28-second window, 20 ms bins, and 5 × 5 repeated stratified protocol. Factor latents achieved mean 0.077080 unified bits/spike, 95% CI [0.071435, 0.082517], positive fraction 1.0. The invalid split-mean control scored 0.071104; factor-minus-invalid was 0.005976, so the earlier pre-movement leakage dominance did not persist. Single-split results remain non-reportable.

## MC_Maze Large findings

Large evaluation shape is [500, 64, 162], with 122 held-in and 40 held-out neurons. The fixed factor-latent baseline scored 0.122717 mean unified bits/spike. Nested selection increased the repeat-paired mean by 0.012828, 95% paired CI [0.011205, 0.014242], positive on all five repeats, satisfying the predeclared replacement rule.

Small and Large differ in trial count, neurons, masks, behavior, endpoint distributions, firing rates, durations, and targets. Cross-dataset score differences are not interpreted as model-performance improvement. Comparisons are limited to protocol behavior, robustness, positive-fold fraction, leakage-control behavior, and qualitative conclusions.

## Valid baseline comparison

| Method | Valid/reportable | Mean | SD | 95% CI | Positive fraction | Paired difference vs fixed |
|---|---:|---:|---:|---:|---:|---:|
| factor_latent_fixed | yes | 0.122717 | 0.025405 | [0.113239, 0.132794] | 1.0 | reference |
| factor_latent_train_selected | yes | 0.135545 | 0.026166 | [0.125743, 0.146088] | 1.0 | +0.012828 |
| smoothed_cosmoothing_ridge | yes | 0.121562 | 0.025822 | [0.111850, 0.131790] | 1.0 | -0.001155 |
| reduced_rank_cosmoothing | yes | 0.087901 | 0.023851 | [0.078951, 0.097026] | 1.0 | -0.034816 |
| split_mean_rate_invalid | no | 0.008967 | 0.002695 | [0.007988, 0.010065] | n/a | excluded |
| train_mean_rate reference | no | 0.000000 | 0.000000 | [0, 0] | n/a | reference, not competitor |

Nested-selected factor latents replaced the fixed baseline through repeat-level paired criteria. A smoothing cache once keyed masks too weakly, allowing collisions across held-out masks. The corrected cache keys the complete held-in index set; corrected scores replaced corrupted results because the latter violated input/target isolation.

## LFADS feasibility and retirement decision

Scope: repeat 0 only, five outer folds × five initialization seeds = 25 runs on one neuron mask. Checkpoints were isolated and selected on inner validation. Mean was 0.029260, seed-level SD 0.001022, positive-seed fraction 1.0. Matched pilot-repeat factor-latent mean was 0.173927; paired deficit was -0.144667.

Audit reproduced all checkpoints and scores. LFADS-style predictions collapsed near peak speed (0.002647 bits/spike versus 0.036369 before and 0.031985 after), were extremely smooth, and used factor effective rank 1.216 of 32. No frozen repair recovered the predeclared gap. Full evaluation is disallowed and the tested LFADS-style branch is retired. This is not evidence that LFADS fails universally.

## Deterministic neural-ODE feasibility and retirement decision

Scope matched LFADS: repeat 0, five folds × five seeds, one neuron mask. The pilot inherited the frozen Small configuration and used deterministic Euler dynamics (`diffusion_scale=0`). Solver checks passed. Mean was 0.141294, seed-mean SD 0.003410, positive-seed fraction 1.0. The matched baseline was 0.173927; paired deficit was -0.032633.

Within this matched pilot only, neural-ODE exceeded the LFADS descriptive reference by 0.112034, avoided the LFADS near-peak collapse (0.143166 near peak), and used effective rank 3.561 of 32. Frozen diagnostics found only 0.005223 estimated recoverable gap versus 0.012633 required. No targeted repair was supported; full evaluation remains disallowed and the tested branch is retired. This is not a five-repeat model comparison or universal architecture ranking.

## Latent interpretability and behavioral validity

All 25 outer scores reproduced within 8.33e-17. Each evaluation latent tensor was [100, 64, 16] and generated out of fold.

| Continuous target | Mean outer-fold R² | Repeat-level 95% CI |
|---|---:|---:|
| hand_pos_x | 0.652276 | [0.625924, 0.678252] |
| hand_pos_y | 0.481943 | [0.459566, 0.504321] |
| hand_vel_x | 0.589250 | [0.565192, 0.612399] |
| hand_vel_y | 0.416875 | [0.405615, 0.428136] |
| hand_speed | 0.198549 | [0.181171, 0.216125] |
| cursor_pos_x | 0.644877 | [0.619443, 0.670396] |
| cursor_pos_y | 0.470609 | [0.448159, 0.493059] |
| cursor_vel_x | 0.571397 | [0.546668, 0.593903] |
| cursor_vel_y | 0.395660 | [0.386016, 0.405217] |
| cursor_speed | 0.209273 | [0.190609, 0.227937] |

Mean across targets was 0.463071. Across-trial permutation control mean was 0.000077; circular temporal-shift control mean was 0.341148, both below observed 0.463645 with empirical p = 1/101. Smooth behavior and temporal autocorrelation therefore explain part, but not all, of continuous decoding. Across-trial controls show the full result is not generic temporal smoothness. With 100 permutations, resolution is limited to approximately 1/101; more digits do not imply stronger precision.

Eight-way direction decoding reached accuracy 0.784000, balanced accuracy 0.626584, macro F1 0.590085, versus 0.125 chance. Across-trial and label-permutation balanced-accuracy means were 0.132861 and 0.133308; empirical p = 1/101.

Geometry: effective dimension 13.216 of 16; mean direction-separability ratio 0.607461; direction-centroid path length 3.343757; maximum displacement 2.463079; pre-movement displacement progressed monotonically in the aggregate. Distance modulation remains descriptive only.

| Stability comparison | Aligned centroid correlation | RSA correlation | Subspace cosine |
|---|---:|---:|---:|
| folds within repeat | 0.742856 | 0.762796 | 0.540958 |
| repeats/neuron masks | 0.828037 | 0.930925 | 0.503923 |
| FactorAnalysis states | 0.898524 | 0.998053 | 0.778817 |

Relational geometry is more stable than exact subspace orientation. Factor latents exceeded the scalar population-rate diagnostic by mean R² difference 0.463352. This supports structure beyond one scalar rate signal; it does not make firing-rate magnitude irrelevant. Rate-regressed variants are diagnostic, not accepted performance estimates.

## Supported scientific claims

Supported claims are predictive or relational: factor latents predict hand position, velocity, and speed; encode endpoint direction; organize separable direction trajectories; remain stable across folds and neuron masks; contain information beyond scalar population rate; and use substantially fewer dimensions than neurons. Exact wording and limitations are frozen in [the claim registry](latentbrain_claim_registry.md).

## Descriptive-only findings

Distance modulation, neural-ODE factor-spectrum breadth versus LFADS, attenuation of the LFADS near-peak failure, and apparent cross-dataset robustness differences remain descriptive. None establishes direct Small-versus-Large performance superiority.

## Unsupported claims

No evidence establishes causal movement generation, a true biological neural manifold, universal superiority of factor latents over dynamical models, direct Small-versus-Large improvement, or official benchmark standing.

## Limitations

- Observational decoding cannot establish causality or biological mechanism.
- Neural pilots use one held-out-neuron mask and are feasibility studies, not five-repeat comparisons.
- FactorAnalysis axes are rotationally non-identifiable; only aligned relational comparisons are meaningful.
- Circular-shift controls retain smooth temporal structure and remain substantially above zero.
- One hundred permutations limit empirical p-value resolution to 1/101.
- Local DANDI datasets and protocols do not represent all tasks, animals, areas, or architectures.
- Invalid controls diagnose leakage; they are never valid competitors.

## Reproducibility

Commands, hashes, CPU/GPU requirements, and artifact policy are in [the reproducibility guide](latentbrain_reproducibility.md). The release audit maps central findings to machine-readable sources and fails on conflicting evidence.

## Conclusions

Under the corrected leakage-safe protocol, nested factor latents were the strongest valid tested Large model and preserved meaningful out-of-fold motor associations. Controlled neural pilots were informative but did not earn expensive full evaluation. The project supports a compact predictive account of motor-cortical population structure, bounded by explicit non-causal and non-leaderboard claims.

## Future research directions

Future work should test additional sessions, animals, brain areas, and tasks; pre-register larger multi-mask neural comparisons; increase permutation resolution; and test mechanism-specific hypotheses with interventions. Those are new projects, not extensions of this frozen release.
