from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rich.console import Console

from latentbrain.analysis.figures import (
    plot_neuron_firing_rates,
    plot_population_rate_over_time,
    plot_split_activity_summary,
    plot_trial_spike_counts,
    plot_zero_fraction_by_neuron,
)
from latentbrain.analysis.quality import (
    compute_dataset_summary,
    compute_neuron_activity,
    compute_quality_flags,
    compute_split_activity_summary,
    compute_time_activity,
    compute_trial_activity,
)
from latentbrain.analysis.reporting import write_json_report, write_markdown_validation_report
from latentbrain.data.io import compute_dataset_hash, load_neural_dataset
from latentbrain.data.splits import create_neuron_mask, create_trial_split
from latentbrain.data.validation import (
    validate_neural_dataset,
    validate_neural_dataset_minimums,
    validate_neuron_mask,
    validate_trial_split,
)
from latentbrain.paths import get_repo_root, resolve_configured_path

console = Console(markup=False)


class DatasetSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    processed_path: str = Field(min_length=1)
    metadata_path: str = Field(min_length=1)
    provenance_path: str = Field(min_length=1)
    expected_hash: str | None = None
    bin_size_ms: int = Field(gt=0)


class SplitSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = Field(ge=0)
    train_fraction: float = Field(gt=0.0, lt=1.0)
    validation_fraction: float = Field(gt=0.0, lt=1.0)
    test_fraction: float = Field(gt=0.0, lt=1.0)
    heldout_neuron_fraction: float = Field(gt=0.0, lt=1.0)


class QualityThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_trials: int = Field(gt=0)
    min_time_bins: int = Field(gt=0)
    min_neurons: int = Field(gt=0)
    max_nan_count: int = Field(ge=0)
    max_inf_count: int = Field(ge=0)
    min_total_spikes: int = Field(ge=0)
    max_zero_fraction_warning: float = Field(ge=0.0, le=1.0)
    inactive_neuron_rate_hz_threshold: float = Field(ge=0.0)
    high_rate_warning_hz: float = Field(gt=0.0)


class ReportingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str = Field(min_length=1)
    figures_dir: str = Field(min_length=1)


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: DatasetSection
    splits: SplitSection
    quality_thresholds: QualityThresholds
    reporting: ReportingSection

    @model_validator(mode="after")
    def split_fractions_sum_to_one(self) -> AnalysisConfig:
        total = (
            self.splits.train_fraction + self.splits.validation_fraction + self.splits.test_fraction
        )
        if abs(total - 1.0) > 1e-8:
            msg = "split fractions must sum to 1.0"
            raise ValueError(msg)
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> AnalysisConfig:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            msg = f"malformed analysis config: {path}"
            raise ValueError(msg) from exc
        if not isinstance(raw, dict):
            msg = f"analysis config must contain a top-level mapping: {path}"
            raise ValueError(msg)
        return cls.model_validate(raw)


def _parse_args(argv: Sequence[str] | None) -> Path:
    parser = argparse.ArgumentParser(description="Analyze processed MC_Maze Small data.")
    parser.add_argument("--config", type=Path, default=Path("configs/mc_maze_small_eda.yaml"))
    args = parser.parse_args(argv)
    return args.config


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = get_repo_root()
    config_arg = _parse_args(argv)
    config_path = config_arg if config_arg.is_absolute() else repo_root / config_arg
    config = AnalysisConfig.from_yaml(config_path)

    processed_path = resolve_configured_path(config.dataset.processed_path, repo_root)
    metadata_path = resolve_configured_path(config.dataset.metadata_path, repo_root)
    provenance_path = resolve_configured_path(config.dataset.provenance_path, repo_root)
    output_dir = resolve_configured_path(config.reporting.output_dir, repo_root)
    figures_dir = resolve_configured_path(config.reporting.figures_dir, repo_root)

    if not processed_path.exists():
        console.print(f"Processed dataset is missing: {_relative(processed_path, repo_root)}")
        console.print(
            "Run: python scripts/prepare_nlb_data.py --config configs/nlb_mc_maze_small.yaml"
        )
        return 2

    dataset = load_neural_dataset(processed_path)
    validate_neural_dataset(dataset)
    validate_neural_dataset_minimums(
        dataset,
        min_trials=config.quality_thresholds.min_trials,
        min_neurons=config.quality_thresholds.min_neurons,
        min_time_bins=config.quality_thresholds.min_time_bins,
    )
    dataset_hash = compute_dataset_hash(dataset)
    if config.dataset.expected_hash and dataset_hash != config.dataset.expected_hash:
        console.print(
            f"Dataset hash mismatch: expected {config.dataset.expected_hash}, got {dataset_hash}"
        )
        return 2

    split = create_trial_split(
        dataset.trial_ids,
        config.splits.train_fraction,
        config.splits.validation_fraction,
        config.splits.test_fraction,
        seed=config.splits.seed,
    )
    mask = create_neuron_mask(
        dataset.spikes.shape[2],
        config.splits.heldout_neuron_fraction,
        seed=config.splits.seed,
    )
    validate_trial_split(split, dataset.trial_ids)
    validate_neuron_mask(mask, dataset.spikes.shape[2])

    metadata = _read_json(metadata_path)
    provenance = _read_json(provenance_path)
    summary = compute_dataset_summary(dataset, dataset_hash=dataset_hash)
    summary["processed_path"] = _relative(processed_path, repo_root)
    summary["split_counts"] = {
        "train": int(len(split.train)),
        "validation": int(len(split.validation)),
        "test": int(len(split.test)),
    }
    summary["neuron_mask_counts"] = {
        "heldin": int(mask.heldin.sum()),
        "heldout": int(mask.heldout.sum()),
    }
    thresholds = config.quality_thresholds.model_dump(mode="json")
    quality_flags = compute_quality_flags(dataset, summary, thresholds)
    summary["quality_flags"] = quality_flags

    neuron_activity = compute_neuron_activity(dataset)
    trial_activity = compute_trial_activity(dataset)
    time_activity = compute_time_activity(dataset)
    split_summary = compute_split_activity_summary(
        dataset,
        split.train,
        split.validation,
        split.test,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    table_paths = {
        "neuron_activity": output_dir / "neuron_activity.csv",
        "trial_activity": output_dir / "trial_activity.csv",
        "time_activity": output_dir / "time_activity.csv",
        "split_activity_summary": output_dir / "split_activity_summary.csv",
    }
    neuron_activity.to_csv(table_paths["neuron_activity"], index=False)
    trial_activity.to_csv(table_paths["trial_activity"], index=False)
    time_activity.to_csv(table_paths["time_activity"], index=False)
    split_summary.to_csv(table_paths["split_activity_summary"], index=False)

    figure_paths = {
        "neuron_firing_rates": plot_neuron_firing_rates(
            neuron_activity, figures_dir / "neuron_firing_rates.png"
        ),
        "trial_spike_counts": plot_trial_spike_counts(
            trial_activity, figures_dir / "trial_spike_counts.png"
        ),
        "population_rate_over_time": plot_population_rate_over_time(
            time_activity, figures_dir / "population_rate_over_time.png"
        ),
        "zero_fraction_by_neuron": plot_zero_fraction_by_neuron(
            neuron_activity, figures_dir / "zero_fraction_by_neuron.png"
        ),
        "split_activity_summary": plot_split_activity_summary(
            split_summary, figures_dir / "split_activity_summary.png"
        ),
    }
    summary_path = write_json_report(summary, output_dir / "data_quality_summary.json")
    report_path = write_markdown_validation_report(
        output_path=output_dir / "validation_report.md",
        dataset_name=config.dataset.name,
        summary=summary,
        quality_flags=quality_flags,
        generated_tables={key: _relative(path, output_dir) for key, path in table_paths.items()},
        generated_figures={key: _relative(path, output_dir) for key, path in figure_paths.items()},
        metadata=metadata,
        provenance=provenance,
    )

    error_count = sum(flag["severity"] == "error" for flag in quality_flags)
    warning_count = sum(flag["severity"] == "warning" for flag in quality_flags)
    console.print(f"dataset: {config.dataset.name}")
    console.print(f"shape: {list(dataset.spikes.shape)}")
    console.print(f"total_spikes: {summary['total_spikes']}")
    console.print(f"mean_population_rate_hz: {summary['mean_population_rate_hz']}")
    console.print(f"zero_fraction: {summary['zero_fraction']}")
    console.print(f"quality_errors: {error_count}")
    console.print(f"quality_warnings: {warning_count}")
    for flag in quality_flags:
        console.print(f"quality_{flag['severity']}: {flag['code']} - {flag['message']}")
    console.print(f"summary: {_relative(summary_path, repo_root)}")
    console.print(f"report: {_relative(report_path, repo_root)}")
    console.print(f"figures_dir: {_relative(figures_dir, repo_root)}")
    return 1 if error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
