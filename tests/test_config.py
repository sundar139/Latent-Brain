from __future__ import annotations

from pathlib import Path

import pytest

from latentbrain.config import ConfigError, load_config


def test_base_config_loads() -> None:
    config = load_config()

    assert config.project.name == "LatentBrain"
    assert config.project.seed >= 0
    assert config.paths.data_root
    assert config.paths.results_root
    assert config.paths.models_root
    assert config.paths.reports_root
    assert config.paths.experiments_root


def test_invalid_negative_seed_raises_error(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_config.yaml"
    config_path.write_text(
        """
project:
  name: LatentBrain
  seed: -1
paths:
  data_root: data
  results_root: results
  models_root: models
  reports_root: reports
  experiments_root: experiments
logging:
  level: INFO
  json: false
reproducibility:
  deterministic: true
  benchmark: false
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="non-negative"):
        load_config(config_path)
