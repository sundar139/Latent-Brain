from __future__ import annotations

from pathlib import Path

from latentbrain.data.nlb import NLBConfig
from latentbrain.paths import get_repo_root, resolve_configured_path


def test_nlb_config_loads_and_validates() -> None:
    config = NLBConfig.from_yaml(Path("configs/nlb_mc_maze.yaml"))

    assert config.dataset.name == "mc_maze"
    assert config.dataset.source == "neural_latents_benchmark"
    assert config.validation.min_trials == 1


def test_real_data_config_split_fractions_are_valid() -> None:
    config = NLBConfig.from_yaml(Path("configs/nlb_mc_maze.yaml"))
    total = (
        config.splits.train_fraction
        + config.splits.validation_fraction
        + config.splits.test_fraction
    )

    assert abs(total - 1.0) < 1e-8
    assert 0.0 < config.splits.heldout_neuron_fraction < 1.0


def test_processed_output_paths_resolve_under_ignored_data_folder() -> None:
    config = NLBConfig.from_yaml(Path("configs/nlb_mc_maze.yaml"))
    repo_root = get_repo_root()
    processed_root = resolve_configured_path(config.dataset.processed_root, repo_root)
    output_path = processed_root / config.dataset.output_filename

    assert processed_root.is_relative_to(repo_root / "data")
    assert output_path.name == "mc_maze_processed.npz"


def test_real_data_contract_tests_do_not_create_real_outputs() -> None:
    processed_path = get_repo_root() / "data" / "processed" / "nlb" / "mc_maze_processed.npz"

    assert not processed_path.exists()
