# LatentBrain claim registry

All claims refer to local, leakage-safe analyses of the tested implementations and frozen protocols. “Supported” means supported by these data and controls, not established causally or universally.

## Supported

| Claim | Evidence | Allowed wording | Required limitation | Prohibited stronger wording |
|---|---|---|---|---|
| Factor latents predict hand position | Out-of-fold x/y R² 0.652/0.482; permutation control near zero | “Factor latents predicted hand position on unseen trials.” | Associative prediction under frozen Large protocol | Causes hand position; decodes all motor behavior generally |
| Factor latents predict hand velocity | Out-of-fold x/y R² 0.589/0.417 | “Latents predicted hand velocity.” | Observational and task-specific | Latents generate movement |
| Factor latents predict movement speed | Hand-speed R² 0.199; controls below observed | “Latents carried predictive speed information.” | Smooth temporal behavior contributes | Speed mechanism discovered |
| Factor latents encode endpoint direction | Accuracy 0.784; balanced accuracy 0.627; controls near chance | “Latents encoded endpoint-direction information.” | Linear decoding association, not causal code | Direction is caused by latent state |
| Reach directions occupy separable latent trajectories | Separability ratio 0.607 | “Direction conditions occupied separable trajectories.” | Geometry is aligned and descriptive | Distinct attractors or dynamical regimes proven |
| Latent geometry is stable across folds | Aligned centroid correlation 0.743 | “Relational geometry was stable across folds.” | Exact axes are non-identifiable | Identical latent axes recovered |
| Latent geometry is stable across neuron masks | Aligned centroid correlation 0.828 | “Relational geometry was stable across tested masks.” | Five deterministic repeat masks only | Stable under arbitrary neuron loss |
| Factor latents contain information beyond population rate | Mean R² advantage 0.463 over scalar-rate diagnostic | “Latents contained predictive structure beyond one scalar rate signal.” | Rate magnitude remains relevant; diagnostic comparison | Firing rate is irrelevant |
| Latent dimensionality is substantially lower than neuron count | Effective dimension 13.216 of 16 versus 122 held-in neurons | “Population activity used a lower-dimensional predictive representation.” | Participation ratio depends on model and window | True intrinsic biological dimensionality discovered |

## Descriptive only

| Finding | Evidence | Allowed wording | Why not promoted |
|---|---|---|---|
| Distance modulates latent trajectory magnitude | Distance-centroid path/displacement summaries | “Trajectory magnitude varied descriptively with distance.” | No completed control-backed inferential criterion |
| Neural-ODE uses a broader factor spectrum than LFADS | Effective rank 3.561 versus 1.216 in matched repeat-0 pilots | “The tested neural-ODE pilot used a broader factor spectrum.” | One mask; different trained implementations |
| Deterministic neural dynamics attenuate the LFADS near-peak failure | Neural-ODE near-peak 0.143 versus LFADS 0.00265 in matched pilot context | “The tested neural-ODE pilot did not show the LFADS near-peak collapse.” | Pilot-only descriptive comparison |
| Large results appear more robust in some diagnostics than earlier Small results | Positive folds and lower leakage-control dominance under final protocols | “Some Large diagnostics appeared more stable.” | Dataset/protocol differences forbid direct performance superiority |

## Unsupported

- Inferred latents causally generate movement.
- The latent space is the true biological neural manifold.
- Factor-latent is universally superior to neural dynamical models.
- Official NLB leaderboard performance.
- State-of-the-art performance.
- Any direct Small-versus-Large improvement claim.

Allowed replacement: “The tested representation predicts behavior and held-out neural activity under local repeated cross-validation.”
