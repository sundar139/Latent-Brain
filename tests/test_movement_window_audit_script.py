from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import yaml

import latentbrain.data.nlb as nlb_module
import latentbrain.eval.stratified_cv as stratified_cv
from latentbrain.data.io import save_neural_dataset
from latentbrain.data.schemas import NeuralDataset, TrialSequences

NAMES = ["hand_pos_x", "hand_pos_y", "cursor_pos_x", "cursor_pos_y"]
LENGTHS = (32, 40, 48, 40, 32, 48, 40, 32)
GLOBAL_BINS = min(LENGTHS)


def _script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_window_audit", Path("scripts/run_window_audit.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _trial(length: int, angle: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    spikes = generator.poisson(0.3, size=(length, 6)).astype(np.int64)
    times = np.arange(length, dtype=np.float64)
    position = np.cumsum(np.exp(-(((times - 0.5 * length) / (0.15 * length)) ** 2)))
    behavior = np.zeros((length, 4))
    behavior[:, 0] = position * np.cos(angle)
    behavior[:, 1] = position * np.sin(angle)
    behavior[:, 2] = behavior[:, 0]
    behavior[:, 3] = behavior[:, 1]
    return spikes, behavior


def _sequences() -> TrialSequences:
    angles = np.linspace(-np.pi, np.pi, len(LENGTHS), endpoint=False)
    spikes, behavior = [], []
    for index, length in enumerate(LENGTHS):
        trial_spikes, trial_behavior = _trial(length, angles[index], index)
        spikes.append(trial_spikes)
        behavior.append(trial_behavior)
    return TrialSequences(
        spikes=spikes,
        behavior=behavior,
        behavior_names=NAMES,
        trial_ids=np.arange(len(LENGTHS), dtype=np.int64),
        trial_lengths=np.asarray(LENGTHS, dtype=np.int64),
        bin_size_ms=5,
        metadata={"spikes_conserved": True, "source_file": "toy_train.nwb"},
    )


def _write_processed(path: Path, sequences: TrialSequences) -> str:
    spikes = np.stack([trial[:GLOBAL_BINS] for trial in sequences.spikes])
    behavior = np.stack([trial[:GLOBAL_BINS] for trial in sequences.behavior or []])
    dataset = NeuralDataset(
        spikes=spikes,
        rates=None,
        latents=None,
        trial_ids=sequences.trial_ids,
        time_ms=np.arange(GLOBAL_BINS, dtype=np.float64) * 5.0,
        bin_size_ms=5,
        metadata={"dataset_name": "toy_large"},
        behavior=behavior,
        behavior_names=NAMES,
    )
    save_neural_dataset(dataset, path)
    return str(dataset.metadata["dataset_hash"])


def _config(tmp_path: Path, processed: Path, dataset_hash: str) -> dict[str, Any]:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(exist_ok=True)
    provenance = tmp_path / "provenance.json"
    nlb_config = yaml.safe_load(Path("configs/nlb_mc_maze_large.yaml").read_text(encoding="utf-8"))
    provenance.write_text(json.dumps({"config": nlb_config}), encoding="utf-8")
    return {
        "dataset": {
            "name": "toy_large",
            "processed_path": str(processed),
            "metadata_path": str(tmp_path / "metadata.json"),
            "provenance_path": str(provenance),
            "raw_dir": str(raw_dir),
            "expected_hash": dataset_hash,
            "original_bin_size_ms": 5,
        },
        "binning": {"target_bin_size_ms": 20},
        "behavior": {
            "preferred_source": "hand",
            "fallback_source": "cursor",
            "required_channels": NAMES,
        },
        "trial_source": {
            "prefer_trial_aware_raw_extraction": True,
            "allow_global_crop_to_min_fallback": False,
            "compare_against_processed_crop_to_min": True,
        },
        "window_candidates": [
            {
                "name": "behavior_speed_peak_centered_1p28s",
                "crop_policy": "behavior_speed_peak_centered",
                "duration_seconds": 0.08,
                "report_label": "Small transferred window",
            },
            {
                "name": "behavior_movement_onset_1p28s",
                "crop_policy": "behavior_movement_onset",
                "duration_seconds": 0.08,
                "pre_event_seconds": 0.02,
                "speed_threshold_quantile": 0.70,
                "report_label": "Movement-onset aligned window",
            },
            {
                "name": "from_start_1p28s",
                "crop_policy": "from_start",
                "duration_seconds": 0.08,
                "start_seconds": 0.0,
                "report_label": "Early-window diagnostic",
            },
        ],
        "selection": {
            "maximum_clipped_trial_fraction": 0.05,
            "minimum_moving_bin_fraction": 0.25,
            "minimum_endpoint_direction_entropy_fraction": 0.70,
            "prioritize_behavior_coverage": True,
            "use_model_scores_for_selection": False,
        },
        "statistics": {
            "confidence_interval": 0.95,
            "bootstrap_repeats": 100,
            "bootstrap_seed": 1337,
        },
        "references": {
            "small_recommended_window": "behavior_speed_peak_centered_1p28s",
            "small_moving_bin_fraction": 0.576875,
            "small_endpoint_direction_entropy": 2.0283893834346562,
        },
        "reporting": {"output_dir": str(tmp_path / "out")},
    }


def _write_config(tmp_path: Path, config: dict[str, Any]) -> Path:
    path = tmp_path / "audit.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.fixture
def toy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, dict[str, Any]]:
    module = _script_module()
    sequences = _sequences()
    processed = tmp_path / "processed.npz"
    dataset_hash = _write_processed(processed, sequences)
    monkeypatch.setattr(nlb_module, "load_trial_sequences", lambda _root, _config: _sequences())
    return module, _config(tmp_path, processed, dataset_hash)


def test_missing_processed_data_fails_clearly(toy: Any, tmp_path: Path, capsys: Any) -> None:
    module, config = toy
    config["dataset"]["processed_path"] = str(tmp_path / "absent.npz")

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 2
    assert "Processed dataset is missing" in capsys.readouterr().out


def test_missing_raw_data_fails_clearly(toy: Any, tmp_path: Path, capsys: Any) -> None:
    module, config = toy
    config["dataset"]["raw_dir"] = str(tmp_path / "absent_raw")

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 2
    out = capsys.readouterr().out
    assert "Raw dataset directory is missing" in out
    assert "must not be used instead" in out


def test_hash_mismatch_fails_clearly(toy: Any, tmp_path: Path, capsys: Any) -> None:
    module, config = toy
    config["dataset"]["expected_hash"] = "0" * 64

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 2
    assert "Dataset hash mismatch" in capsys.readouterr().out


def test_global_crop_fallback_must_stay_disabled(toy: Any, tmp_path: Path, capsys: Any) -> None:
    module, config = toy
    config["trial_source"]["allow_global_crop_to_min_fallback"] = True

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 2
    assert "allow_global_crop_to_min_fallback must be" in capsys.readouterr().out


def test_toy_variable_length_run_writes_every_output(
    toy: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    module, config = toy

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("the movement-window audit must not score any model")

    monkeypatch.setattr(stratified_cv, "score_folds", _forbidden)

    assert module.main(["--config", str(_write_config(tmp_path, config))]) == 0

    out = tmp_path / "out"
    for name in (
        "window_audit_summary.json",
        "window_candidate_statistics.csv",
        "window_behavior_statistics.csv",
        "window_trial_coverage.csv",
        "crop_to_min_impact.csv",
        "window_recommendations.json",
        "window_audit_report.md",
    ):
        assert (out / name).exists()
    for name in (
        "movement_speed_profile.png",
        "peak_speed_time_distribution.png",
        "movement_onset_time_distribution.png",
        "movement_coverage_by_window.png",
        "clipping_fraction_by_window.png",
        "endpoint_direction_entropy_by_window.png",
    ):
        assert (out / "figures" / name).exists()

    printed = capsys.readouterr().out
    for key in (
        "dataset_name",
        "dataset_hash",
        "trial_count",
        "neuron_count",
        "raw_spike_count",
        "global_crop_retained_spike_count",
        "fraction_trials_peak_inside_global_crop",
        "fraction_trials_onset_inside_global_crop",
        "recommended_window_name",
        "small_window_transfers",
        "global_crop_suitable_for_movement_window_audit",
        "warnings",
    ):
        assert f"{key}:" in printed

    summary = json.loads((out / "window_audit_summary.json").read_text("utf-8"))
    assert summary["models_trained"] is False
    assert summary["models_scored"] is False
    assert summary["cross_validation_run"] is False
    assert summary["official_benchmark_claim"] is False
    assert summary["global_crop_used_for_event_centered_windows"] is False
    assert summary["trial_length_min"] == min(LENGTHS)
    assert summary["trial_length_max"] == max(LENGTHS)

    report = (out / "window_audit_report.md").read_text("utf-8")
    assert "not model performance" in report
    assert "No official NLB leaderboard result is claimed." in report
    assert "was not silently used as the source of event-centered windows" in report

    recommendations = json.loads((out / "window_recommendations.json").read_text("utf-8"))
    assert recommendations["selection_used_model_scores"] is False
