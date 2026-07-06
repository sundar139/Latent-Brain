# Data Governance

LatentBrain will work with neural datasets only when they can be obtained legally, ethically, and under their published access terms. Dataset acquisition is not implemented yet.

## Local data policy

Raw data files stay on local or approved institutional storage and are ignored by Git. The repository tracks code, configuration, and documentation only. Derived datasets, checkpoints, generated reports, and experiment outputs are also excluded from version control unless a future governance review explicitly approves a small metadata artifact.

Synthetic data may be regenerated locally from tracked configuration for validating data contracts. Generated synthetic `.npz` and metadata JSON files stay under ignored `data/` paths and must not be committed.

Real neural data must be downloaded manually by an authorized user and stored under ignored local paths such as `data/raw/nlb/mc_maze_small`. Processed local arrays belong under ignored derived-data paths such as `data/processed/nlb/mc_maze_small`. No real dataset file, processed tensor, metadata generated from real data, credential, or private data path should be committed.

## Dataset provenance

Future data pipelines must record enough metadata to make analyses auditable:

- Dataset source and version
- Dataset variant, such as MC_Maze Small
- Acquisition date and access terms
- Configuration used for preprocessing
- Split seed and split definition
- Hashes or integrity checks for immutable inputs
- Preprocessing software version and parameters
- Recording or session identifiers when real neural sessions are introduced
- File manifest with sizes and SHA-256 hashes when files are small enough to hash locally
- Config snapshot used for preparation
- Git commit used to prepare local artifacts

Large files may skip hashing when a configured size limit would make hashing impractical, but the manifest must record that hashing was skipped because of file size.

## Leakage prevention

Training, validation, and test boundaries must be defined before model selection. No information from held-out evaluation data may influence preprocessing statistics, hyperparameter choices, feature selection, early stopping, or model comparison decisions.

For MC_Maze/NLB-style data, trial IDs and held-in or held-out neuron masks must be validated before future modeling code consumes them. The public NLB challenge has ended, so future comparisons should be described as local reproducible evaluation rather than public leaderboard or EvalAI submissions.
