# Reproducibility

LatentBrain is initialized with reproducible engineering practices before data and modeling code are added.

## Environment reproducibility

The Python package declares runtime and development dependencies in `pyproject.toml`. Local contributors should create an isolated virtual environment, install the package in editable mode, and run the quality checks before committing changes.

## Random seeds

The base configuration defines a default seed. The package includes a central seeding utility for Python, NumPy, and optional PyTorch installations. Future modeling code should receive seeds from validated configuration instead of hardcoded constants.

## Config-driven runs

Configuration values should live in tracked YAML files or in local environment variables for machine-specific paths. Secrets and private paths must not be committed.

## Git commit tracking

Future experiment records should capture the Git commit hash, repository cleanliness, configuration file, environment summary, and data provenance metadata.

## Experiment tracking expectations

When experiment tracking is introduced, runs should store parameters, metrics, code version, data identifiers, and artifacts in an approved local or remote tracking system. No benchmark results are available yet.
