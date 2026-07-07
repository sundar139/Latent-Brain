from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from rich.console import Console

from latentbrain.data.io import compute_dataset_hash, load_neural_dataset
from latentbrain.data.splits import create_neuron_mask, create_trial_split
from latentbrain.data.validation import (
    validate_neural_dataset,
    validate_neuron_mask,
    validate_trial_split,
)
from latentbrain.eval.baselines import evaluate_mean_rate_baseline
from latentbrain.eval.comparison import (
    COMPARISON_COLUMNS,
    build_comparison_row,
    rank_validation_methods,
    summarize_comparison,
)
from latentbrain.eval.cosmoothing import run_cosmoothing_baseline
from latentbrain.eval.latent_baseline import run_factor_latent_baseline
from latentbrain.eval.lfads_eval import run_lfads_gru_evaluation
from latentbrain.eval.reporting import write_window_matched_comparison_outputs
from latentbrain.eval.windowing import crop_neural_dataset_time, describe_time_window
from latentbrain.paths import get_repo_root, resolve_configured_path
from latentbrain.torch.device import resolve_device

console = Console(markup=False)
SplitName = Literal["train", "validation", "test"]
MethodName = Literal[
    "mean_rate_windowed",
    "ridge_cosmoothing_windowed",
    "factor_latent_windowed",
    "lfads_gru_factor_decoder",
    "lfads_gru_cosmoothing_direct",
    "lfads_gru_cosmoothing_factor_decoder",
]
KNOWN_METHODS = {
    "mean_rate_windowed",
    "ridge_cosmoothing_windowed",
    "factor_latent_windowed",
    "lfads_gru_factor_decoder",
    "lfads_gru_cosmoothing_direct",
    "lfads_gru_cosmoothing_factor_decoder",
}
BEHAVIOR_COLUMNS = [
    "method_name",
    "split",
    "prediction_source",
    "behavior_mean_r2",
    "behavior_mean_mse",
    "behavior_mean_mae",
    "target_count",
]
METRIC_COLUMNS = {
    "spike_count",
    "poisson_nll",
    "poisson_log_likelihood",
    "reference_log_likelihood",
    "bits_per_spike",
    "mse_rate_hz",
    "mae_rate_hz",
}


class DatasetSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    processed_path: str = Field(min_length=1)
    expected_hash: str = Field(min_length=1)
    bin_size_ms: int = Field(gt=0)


class SplitSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = Field(ge=0)
    train_fraction: float = Field(gt=0.0, lt=1.0)
    validation_fraction: float = Field(gt=0.0, lt=1.0)
    test_fraction: float = Field(gt=0.0, lt=1.0)
    heldout_neuron_fraction: float = Field(gt=0.0, lt=1.0)


class WindowSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_time_bins: int = Field(gt=0)
    crop_policy: Literal["from_start"]


class ReferenceSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    full_window_mean_rate_bits_per_spike: float
    full_window_factor_latent_best_bits_per_spike: float
    lfads_heldin_checkpoint_path: str = Field(min_length=1)
    lfads_cosmoothing_checkpoint_path: str = Field(min_length=1)


class MethodsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include: list[MethodName]

    @field_validator("include")
    @classmethod
    def methods_are_unique_and_known(cls, values: list[str]) -> list[str]:
        if not values:
            msg = "methods.include must contain at least one method"
            raise ValueError(msg)
        if len(set(values)) != len(values):
            msg = "methods.include must not contain duplicates"
            raise ValueError(msg)
        unknown = sorted(set(values) - KNOWN_METHODS)
        if unknown:
            msg = f"unrecognized methods: {unknown}"
            raise ValueError(msg)
        return values


class EvaluationSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_split: SplitName
    primary_metric: Literal["bits_per_spike"]
    behavior_metric: Literal["mean_r2"]
    evaluate_splits: list[SplitName]

    @model_validator(mode="after")
    def primary_split_is_evaluated(self) -> EvaluationSection:
        if self.primary_split not in self.evaluate_splits:
            msg = "evaluation.primary_split must be listed in evaluation.evaluate_splits"
            raise ValueError(msg)
        if len(set(self.evaluate_splits)) != len(self.evaluate_splits):
            msg = "evaluation.evaluate_splits must not contain duplicates"
            raise ValueError(msg)
        return self


class ReportingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str = Field(min_length=1)


class WindowMatchedComparisonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: DatasetSection
    splits: SplitSection
    window: WindowSection
    references: ReferenceSection
    methods: MethodsSection
    evaluation: EvaluationSection
    reporting: ReportingSection

    @model_validator(mode="after")
    def split_fractions_sum_to_one(self) -> WindowMatchedComparisonConfig:
        total = (
            self.splits.train_fraction + self.splits.validation_fraction + self.splits.test_fraction
        )
        if abs(total - 1.0) > 1e-8:
            msg = "split fractions must sum to 1.0"
            raise ValueError(msg)
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> WindowMatchedComparisonConfig:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            msg = f"malformed window-matched comparison config: {path}"
            raise ValueError(msg) from exc
        if not isinstance(raw, dict):
            msg = f"window-matched comparison config must contain a top-level mapping: {path}"
            raise ValueError(msg)
        return cls.model_validate(raw)


def _parse_args(argv: Sequence[str] | None) -> Path:
    parser = argparse.ArgumentParser(
        description="Run the local MC_Maze Small window-matched comparison."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mc_maze_small_window_matched_comparison.yaml"),
    )
    args = parser.parse_args(argv)
    return args.config


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _base_metadata(
    row: pd.Series,  # type: ignore[type-arg]
    time_bins: int,
    window_seconds: float,
    uses_neural_network: bool,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "time_bins": time_bins,
        "window_seconds": window_seconds,
        "n_trials": int(row.get("n_trials", row.get("n_eval_trials", 0))),
        "n_target_neurons": int(row.get("n_target_neurons", row.get("n_neurons", 0))),
        "uses_neural_network": uses_neural_network,
        "uses_train_only_fit": True,
        "notes": notes,
    }


def _mean_behavior_by_split(behavior_metrics: pd.DataFrame) -> dict[str, dict[str, float]]:
    by_split: dict[str, dict[str, float]] = {}
    if behavior_metrics.empty:
        return by_split
    for split_name, frame in behavior_metrics.groupby("split"):
        by_split[str(split_name)] = {
            "behavior_mean_r2": float(frame["r2"].mean()),
            "behavior_mean_mse": float(frame["mse"].mean()),
            "behavior_mean_mae": float(frame["mae"].mean()),
            "target_count": int(len(frame)),
        }
    return by_split


def _empty_behavior_rows(
    method_name: str,
    prediction_source: str,
    splits: list[str],
) -> list[dict[str, Any]]:
    return [
        {
            "method_name": method_name,
            "split": split,
            "prediction_source": prediction_source,
            "behavior_mean_r2": float("nan"),
            "behavior_mean_mse": float("nan"),
            "behavior_mean_mae": float("nan"),
            "target_count": 0,
        }
        for split in splits
    ]


def _behavior_rows_from_metrics(
    method_name: str,
    prediction_source: str,
    splits: list[str],
    behavior_metrics: pd.DataFrame,
) -> list[dict[str, Any]]:
    by_split = _mean_behavior_by_split(behavior_metrics)
    rows = []
    for split in splits:
        metrics = by_split.get(
            split,
            {
                "behavior_mean_r2": float("nan"),
                "behavior_mean_mse": float("nan"),
                "behavior_mean_mae": float("nan"),
                "target_count": 0,
            },
        )
        rows.append(
            {
                "method_name": method_name,
                "split": split,
                "prediction_source": prediction_source,
                **metrics,
            }
        )
    return rows


def _rows_from_split_metrics(
    method_name: str,
    prediction_source: str,
    split_metrics: pd.DataFrame,
    time_bins: int,
    window_seconds: float,
    uses_neural_network: bool,
    behavior_by_split: dict[str, dict[str, float]] | None = None,
    notes: str = "",
) -> list[dict[str, Any]]:
    rows = []
    behavior_by_split = behavior_by_split or {}
    for _, row in split_metrics.iterrows():
        split_name = str(row["split"])
        metrics = {key: float(row[key]) for key in row.index if key in METRIC_COLUMNS}
        metrics.update(behavior_by_split.get(split_name, {}))
        rows.append(
            build_comparison_row(
                method_name,
                split_name,
                prediction_source,
                metrics,
                _base_metadata(row, time_bins, window_seconds, uses_neural_network, notes),
            )
        )
    return rows


def _unavailable_rows(
    method_name: str,
    prediction_source: str,
    splits: list[str],
    time_bins: int,
    window_seconds: float,
    note: str,
) -> list[dict[str, Any]]:
    return [
        build_comparison_row(
            method_name,
            split,
            prediction_source,
            {},
            {
                "time_bins": time_bins,
                "window_seconds": window_seconds,
                "uses_neural_network": True,
                "uses_train_only_fit": True,
                "notes": note,
            },
        )
        for split in splits
    ]


def _mean_rate_config(config: WindowMatchedComparisonConfig) -> dict[str, Any]:
    return {
        "baseline": {
            "name": "mean_rate",
            "min_rate_hz": 1.0e-4,
            "max_rate_hz": 500.0,
            "use_train_trials_only": True,
            "predict_constant_rate_per_neuron": True,
        },
        "evaluation": {
            "evaluate_splits": config.evaluation.evaluate_splits,
            "evaluate_neuron_groups": ["heldout"],
            "primary_split": config.evaluation.primary_split,
            "primary_neuron_group": "heldout",
        },
    }


def _ridge_config(config: WindowMatchedComparisonConfig) -> dict[str, Any]:
    return {
        "features": {
            "input_neuron_group": "heldin",
            "target_neuron_group": "heldout",
            "smoothing": {"method": "gaussian", "sigma_ms": 200.0, "truncate": 4.0},
            "convert_to_hz": True,
            "standardize_features": False,
        },
        "targets": {"fit_target_type": "rate_hz", "min_rate_hz": 1.0e-4, "max_rate_hz": 500.0},
        "decoder": {"name": "ridge", "alpha": 10000.0, "fit_intercept": True},
        "reference": {"name": "train_mean_rate", "fit_train_trials_only": True},
        "evaluation": {
            "evaluate_splits": config.evaluation.evaluate_splits,
            "primary_split": config.evaluation.primary_split,
            "primary_metric": config.evaluation.primary_metric,
        },
    }


def _factor_config(config: WindowMatchedComparisonConfig) -> dict[str, Any]:
    return {
        "features": {
            "input_neuron_group": "heldin",
            "target_neuron_group": "heldout",
            "smoothing": {"method": "gaussian", "sigma_ms": 200.0, "truncate": 4.0},
            "convert_to_hz": True,
            "standardize_features": True,
        },
        "latent_model": {
            "name": "factor_analysis",
            "latent_dim": 8,
            "random_state": config.splits.seed,
            "max_iter": 1000,
            "tol": 1.0e-4,
            "train_trials_only": True,
        },
        "heldout_decoder": {
            "name": "ridge",
            "alpha": 10000.0,
            "fit_intercept": True,
            "min_rate_hz": 1.0e-4,
            "max_rate_hz": 500.0,
            "train_trials_only": True,
        },
        "behavior_decoder": {
            "enabled": True,
            "alpha": 100.0,
            "fit_intercept": True,
            "target_prefixes": ["hand_pos", "cursor_pos"],
            "derive_velocity": True,
            "velocity_method": "central_difference",
            "standardize_targets": True,
            "train_trials_only": True,
        },
        "evaluation": {
            "evaluate_splits": config.evaluation.evaluate_splits,
            "primary_split": config.evaluation.primary_split,
            "primary_metric": config.evaluation.primary_metric,
        },
    }


def _lfads_config(
    config: WindowMatchedComparisonConfig,
    checkpoint_path: Path,
    cosmoothing: bool,
    direct_model: bool,
    factor_decoder: bool,
) -> dict[str, Any]:
    model: dict[str, Any] = {
        "name": "lfads_gru",
        "checkpoint_path": str(checkpoint_path),
        "encoder_hidden_dim": 64,
        "generator_hidden_dim": 64,
        "latent_dim": 16,
        "factor_dim": 32,
        "dropout": 0.1,
        "min_rate_hz": 1.0e-4,
        "max_rate_hz": 500.0,
    }
    if cosmoothing:
        model["output_dim"] = "all"
    return {
        "splits": {"seed": config.splits.seed},
        "data": {
            "input_neuron_group": "heldin",
            "target_neuron_group": "heldout",
            "max_time_bins": config.window.max_time_bins,
            "batch_size": 4,
            "num_workers": 0,
            "drop_last": False,
        },
        "model": model,
        "evaluation_mode": {
            "use_direct_model_rates_for_heldout": direct_model,
            "also_evaluate_factor_decoder": factor_decoder,
        },
        "heldout_decoder": {
            "name": "ridge",
            "alpha": 1000.0,
            "fit_intercept": True,
            "min_rate_hz": 1.0e-4,
            "max_rate_hz": 500.0,
            "standardize_factors": True,
            "train_trials_only": True,
        },
        "behavior_decoder": {
            "enabled": True,
            "alpha": 100.0,
            "fit_intercept": True,
            "standardize_factors": True,
            "standardize_targets": True,
            "target_prefixes": ["hand_pos", "cursor_pos"],
            "derive_velocity": True,
            "velocity_method": "central_difference",
            "train_trials_only": True,
        },
        "evaluation": {
            "evaluate_splits": config.evaluation.evaluate_splits,
            "primary_split": config.evaluation.primary_split,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = get_repo_root()
    config_arg = _parse_args(argv)
    config_path = config_arg if config_arg.is_absolute() else repo_root / config_arg
    config = WindowMatchedComparisonConfig.from_yaml(config_path)
    processed_path = resolve_configured_path(config.dataset.processed_path, repo_root)
    output_dir = resolve_configured_path(config.reporting.output_dir, repo_root)
    if not processed_path.exists():
        console.print(f"Processed dataset is missing: {_relative(processed_path, repo_root)}")
        console.print(
            "Run: python scripts/prepare_nlb_data.py --config configs/nlb_mc_maze_small.yaml"
        )
        return 2

    dataset = load_neural_dataset(processed_path)
    validate_neural_dataset(dataset)
    dataset_hash = compute_dataset_hash(dataset)
    if dataset_hash != config.dataset.expected_hash:
        console.print(
            f"Dataset hash mismatch: expected {config.dataset.expected_hash}, got {dataset_hash}"
        )
        return 2
    original_time_bins = int(dataset.spikes.shape[1])
    cropped = crop_neural_dataset_time(
        dataset, config.window.max_time_bins, config.window.crop_policy
    )
    validate_neural_dataset(cropped)
    time_window = describe_time_window(
        original_time_bins, cropped.spikes.shape[1], cropped.bin_size_ms
    )
    time_bins = int(time_window["cropped_time_bins"])
    window_seconds = float(time_window["window_seconds"])

    split = create_trial_split(
        cropped.trial_ids,
        config.splits.train_fraction,
        config.splits.validation_fraction,
        config.splits.test_fraction,
        seed=config.splits.seed,
    )
    neuron_mask = create_neuron_mask(
        cropped.spikes.shape[2], config.splits.heldout_neuron_fraction, seed=config.splits.seed
    )
    validate_trial_split(split, cropped.trial_ids)
    validate_neuron_mask(neuron_mask, cropped.spikes.shape[2])

    comparison_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []
    split_names = [str(value) for value in config.evaluation.evaluate_splits]

    if "mean_rate_windowed" in config.methods.include:
        split_metrics, _, _ = evaluate_mean_rate_baseline(
            cropped, split, neuron_mask, _mean_rate_config(config)
        )
        heldout = split_metrics[split_metrics["neuron_group"] == "heldout"].reset_index(drop=True)
        comparison_rows.extend(
            _rows_from_split_metrics(
                "mean_rate_windowed", "constant_rate", heldout, time_bins, window_seconds, False
            )
        )
        behavior_rows.extend(
            _empty_behavior_rows("mean_rate_windowed", "constant_rate", split_names)
        )

    if "ridge_cosmoothing_windowed" in config.methods.include:
        split_metrics, _, _ = run_cosmoothing_baseline(
            cropped, split, neuron_mask, _ridge_config(config)
        )
        comparison_rows.extend(
            _rows_from_split_metrics(
                "ridge_cosmoothing_windowed",
                "ridge",
                split_metrics,
                time_bins,
                window_seconds,
                False,
            )
        )
        behavior_rows.extend(
            _empty_behavior_rows("ridge_cosmoothing_windowed", "ridge", split_names)
        )

    if "factor_latent_windowed" in config.methods.include:
        factor_config = _factor_config(config)
        heldin_count = int(np.count_nonzero(neuron_mask.heldin))
        factor_config["latent_model"]["latent_dim"] = min(
            int(factor_config["latent_model"]["latent_dim"]), max(1, heldin_count - 1)
        )
        split_metrics, _, behavior_metrics, _, _ = run_factor_latent_baseline(
            cropped, split, neuron_mask, factor_config
        )
        behavior_by_split = _mean_behavior_by_split(behavior_metrics)
        comparison_rows.extend(
            _rows_from_split_metrics(
                "factor_latent_windowed",
                "factor_decoder",
                split_metrics,
                time_bins,
                window_seconds,
                False,
                behavior_by_split,
            )
        )
        behavior_rows.extend(
            _behavior_rows_from_metrics(
                "factor_latent_windowed", "factor_decoder", split_names, behavior_metrics
            )
        )

    lfads_jobs = [
        (
            "lfads_gru_factor_decoder",
            "factor_decoder",
            resolve_configured_path(config.references.lfads_heldin_checkpoint_path, repo_root),
            False,
            False,
            True,
        ),
        (
            "lfads_gru_cosmoothing_direct",
            "direct_model",
            resolve_configured_path(config.references.lfads_cosmoothing_checkpoint_path, repo_root),
            True,
            True,
            False,
        ),
        (
            "lfads_gru_cosmoothing_factor_decoder",
            "factor_decoder",
            resolve_configured_path(config.references.lfads_cosmoothing_checkpoint_path, repo_root),
            True,
            False,
            True,
        ),
    ]
    for method_name, prediction_source, checkpoint_path, cosmoothing, direct, factor in lfads_jobs:
        if method_name not in config.methods.include:
            continue
        if not checkpoint_path.exists():
            note = f"checkpoint missing: {_relative(checkpoint_path, repo_root)}"
            comparison_rows.extend(
                _unavailable_rows(
                    method_name, prediction_source, split_names, time_bins, window_seconds, note
                )
            )
            behavior_rows.extend(_empty_behavior_rows(method_name, prediction_source, split_names))
            continue
        try:
            device = resolve_device("cuda")
        except RuntimeError as exc:
            console.print(str(exc))
            return 2
        split_metrics, _, behavior_metrics, _, _ = run_lfads_gru_evaluation(
            cropped,
            split,
            neuron_mask,
            _lfads_config(config, checkpoint_path, cosmoothing, direct, factor),
            device,
        )
        source_metrics = split_metrics[
            split_metrics["prediction_source"] == prediction_source
        ].reset_index(drop=True)
        behavior_by_split = _mean_behavior_by_split(behavior_metrics)
        comparison_rows.extend(
            _rows_from_split_metrics(
                method_name,
                prediction_source,
                source_metrics,
                time_bins,
                window_seconds,
                True,
                behavior_by_split,
                "checkpoint evaluation; no neural training by comparison script",
            )
        )
        behavior_rows.extend(
            _behavior_rows_from_metrics(
                method_name, prediction_source, split_names, behavior_metrics
            )
        )

    comparison_metrics = pd.DataFrame(comparison_rows, columns=COMPARISON_COLUMNS)
    validation_leaderboard = rank_validation_methods(
        comparison_metrics, config.evaluation.primary_split, config.evaluation.primary_metric
    )
    behavior_comparison = pd.DataFrame(behavior_rows, columns=BEHAVIOR_COLUMNS)
    summary = {
        "dataset_name": config.dataset.name,
        "dataset_hash": dataset_hash,
        "original_time_bins": original_time_bins,
        "cropped_time_bins": time_bins,
        "window_seconds": window_seconds,
        **summarize_comparison(
            comparison_metrics, config.evaluation.primary_split, config.evaluation.primary_metric
        ),
        "full_window_mean_rate_bits_per_spike": (
            config.references.full_window_mean_rate_bits_per_spike
        ),
        "full_window_factor_latent_best_bits_per_spike": (
            config.references.full_window_factor_latent_best_bits_per_spike
        ),
        "official_benchmark_claim": False,
        "new_neural_network_trained": False,
    }
    write_window_matched_comparison_outputs(
        output_dir, summary, comparison_metrics, validation_leaderboard, behavior_comparison
    )
    console.print(f"dataset: {config.dataset.name}")
    console.print(f"dataset_hash: {dataset_hash}")
    console.print(f"original_time_bins: {original_time_bins}")
    console.print(f"cropped_time_bins: {time_bins}")
    console.print(f"window_seconds: {window_seconds}")
    console.print(f"best_method_name: {summary.get('best_method_name')}")
    console.print(f"best_prediction_source: {summary.get('best_prediction_source')}")
    console.print(
        f"best_validation_bits_per_spike: {summary.get('best_validation_bits_per_spike')}"
    )
    console.print("official_benchmark_claim: False")
    console.print("new_neural_network_trained: False")
    console.print(f"output_dir: {_relative(output_dir, repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
