# Data Governance

LatentBrain will work with neural datasets only when they can be obtained legally, ethically, and under their published access terms. Dataset acquisition is not implemented yet.

## Local data policy

Raw data files stay on local or approved institutional storage and are ignored by Git. The repository tracks code, configuration, and documentation only. Derived datasets, checkpoints, generated reports, and experiment outputs are also excluded from version control unless a future governance review explicitly approves a small metadata artifact.

Synthetic data may be regenerated locally from tracked configuration for validating data contracts. Generated synthetic `.npz` and metadata JSON files stay under ignored `data/` paths and must not be committed.

## Dataset provenance

Future data pipelines must record enough metadata to make analyses auditable:

- Dataset source and version
- Acquisition date and access terms
- Configuration used for preprocessing
- Split seed and split definition
- Hashes or integrity checks for immutable inputs
- Preprocessing software version and parameters
- Recording or session identifiers when real neural sessions are introduced

## Leakage prevention

Training, validation, and test boundaries must be defined before model selection. No information from held-out evaluation data may influence preprocessing statistics, hyperparameter choices, feature selection, early stopping, or model comparison decisions.
