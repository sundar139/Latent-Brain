# LatentBrain release notes

## Scope

LatentBrain is a Python 3.11 research/engineering project for leakage-safe latent modeling of motor-cortical population activity. This release freezes MC_Maze Small and Large ingestion, evaluation, baseline, feasibility-pilot, interpretability, and reporting evidence.

## Accepted datasets and protocols

- MC_Maze Small, DANDI 000140: peak-speed-centered 1.28-second window, 20 ms bins, 5 folds × 5 repeats.
- MC_Maze Large, DANDI 000138: trial-aware 5 ms extraction before 20 ms rebinning, the same window, [500, 64, 162], 5 folds × 5 repeats, 122/40 held-in/held-out neurons.

Single fixed splits are non-reportable. Small and Large scores are not compared as direct performance measurements.

## Best valid models

- Small: factor latent, mean 0.077080 unified bits/spike, 95% CI [0.071435, 0.082517].
- Large: nested-selected factor latent, mean 0.135545, 95% CI [0.125743, 0.146088].

The Large selected model replaced the fixed factor latent through predeclared repeat-level paired criteria. Invalid target-reading controls remain excluded from ranking.

## Closed neural branches

LFADS-style repeat-0 feasibility mean: 0.029260; paired deficit to matched baseline: -0.144667. Full evaluation disallowed; branch retired.

Deterministic neural-ODE repeat-0 feasibility mean: 0.141294; paired deficit: -0.032633. Frozen diagnostics found no adequate targeted repair. Full evaluation disallowed; branch retired.

Both are one-neuron-mask feasibility pilots, not final five-repeat comparisons. Neural-model search is closed.

## Interpretability

Out-of-fold Large latents achieved mean continuous-target R² 0.463071 and eight-way direction balanced accuracy 0.626584 versus 0.125 chance. Effective dimension was 13.216 of 16; direction separability was 0.607461. Geometry was stable relationally across folds, masks, and FactorAnalysis states, and behavior prediction exceeded a scalar population-rate diagnostic.

Across-trial controls were near zero; temporal shifts retained substantial smooth-behavior signal but remained below observed. With 100 permutations, empirical resolution is 1/101.

## Known limitations and claim restrictions

- Associative and predictive evidence only; no causal mechanism established.
- No official NLB leaderboard claim.
- No direct Small-versus-Large performance-improvement claim.
- Neural pilots cover one held-out-neuron mask.
- Distance modulation is descriptive only.
- Exact latent axes are rotationally non-identifiable.

## Quality and readiness

The release audit verifies hashes, protocols, metrics, pilot retirement decisions, interpretability completion, claim safety, ignored generated outputs, and documentation coverage. Standard gates are Ruff, Ruff format, mypy 2.2.0, pytest, config validation, environment inspection, and `git diff --check`.

No Git tag or GitHub release is created by this milestone. Release readiness is recorded locally in ignored `results/release_audit/release_readiness.json`.
