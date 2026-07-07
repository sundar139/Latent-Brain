from __future__ import annotations

import argparse
import copy
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rich.console import Console

from latentbrain.data.io import compute_dataset_hash, load_neural_dataset
from latentbrain.data.splits import create_neuron_mask, create_trial_split
from latentbrain.data.validation import (
    validate_neural_dataset,
    validate_neuron_mask,
    validate_trial_split,
)
from latentbrain.eval.calibration import (
    compute_prediction_reference_correlation,
    compute_rate_calibration_table,
    summarize_rate_distribution,
)
from latentbrain.eval.cosmoothing import _broadcast_reference, _reference_rates
from latentbrain.eval.diagnostics import (
    compute_factor_usage,
    compute_loss_scale_diagnostics,
    compute_neuron_prediction_diagnostics,
)
from latentbrain.eval.lfads_eval import (
    extract_lfads_factors,
    load_lfads_gru_from_checkpoint,
    run_lfads_gru_evaluation,
)
from latentbrain.eval.metrics import safe_clip_rates
from latentbrain.eval.reporting import write_lfads_audit_outputs
from latentbrain.eval.windowing import crop_neural_dataset_time, describe_time_window
from latentbrain.paths import get_repo_root, resolve_configured_path
from latentbrain.torch.datasets import create_dataloaders, create_torch_datasets
from latentbrain.torch.device import resolve_device
from latentbrain.train.lfads_diagnostics import loss_drop_fraction, run_tiny_subset_overfit

console = Console(markup=False)


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


class RuntimeSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: Literal["cuda"]
    fail_if_cuda_unavailable: bool


class ReferenceSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_matched_mean_rate_validation_bits_per_spike: float
    window_matched_factor_latent_validation_bits_per_spike: float
    best_tuned_lfads_validation_bits_per_spike: float
    best_tuned_lfads_run_id: str = Field(min_length=1)


class CheckpointSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tuned_lfads_best: str = Field(min_length=1)
    masked_cosmoothing_best: str = Field(min_length=1)


class AuditSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluate_splits: list[Literal["train", "validation", "test"]] = Field(min_length=1)
    primary_split: Literal["validation"]
    rate_bins: int = Field(gt=0)
    tiny_subset_trials: int = Field(gt=0)
    tiny_subset_epochs: int = Field(gt=0)
    tiny_subset_max_time_bins: int = Field(gt=0)
    tiny_subset_learning_rate: float = Field(gt=0.0)
    tiny_subset_expected_loss_drop_fraction: float = Field(ge=0.0)
    save_figures: bool


class ReportingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str = Field(min_length=1)


class LFADSAuditConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: DatasetSection
    splits: SplitSection
    window: WindowSection
    runtime: RuntimeSection
    references: ReferenceSection
    checkpoints: CheckpointSection
    audit: AuditSection
    reporting: ReportingSection

    @model_validator(mode="after")
    def validate_contract(self) -> LFADSAuditConfig:
        total = (
            self.splits.train_fraction + self.splits.validation_fraction + self.splits.test_fraction
        )
        if abs(total - 1.0) > 1.0e-8:
            msg = "split fractions must sum to 1.0"
            raise ValueError(msg)
        if not self.runtime.fail_if_cuda_unavailable:
            msg = "runtime.fail_if_cuda_unavailable must be true"
            raise ValueError(msg)
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> LFADSAuditConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            msg = f"LFADS audit config must contain a top-level mapping: {path}"
            raise ValueError(msg)
        return cls.model_validate(raw)


def _parse_args(argv: Sequence[str] | None) -> Path:
    parser = argparse.ArgumentParser(description="Run local LFADS-style diagnostic audit.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/mc_maze_small_lfads_audit.yaml")
    )
    return parser.parse_args(argv).config


def _cuda_diagnostic() -> dict[str, Any]:
    available = torch.cuda.is_available()
    return {
        "torch_version": torch.__version__,
        "available": available,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if available else "NONE",
    }


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _check_required_paths(config: dict[str, Any]) -> None:
    root = get_repo_root()
    processed = resolve_configured_path(str(config["dataset"]["processed_path"]), root)
    if not processed.exists():
        msg = f"Processed dataset is missing: {_relative(processed, root)}"
        raise FileNotFoundError(msg)
    checkpoint = resolve_configured_path(str(config["checkpoints"]["tuned_lfads_best"]), root)
    if not checkpoint.exists():
        msg = (
            f"LFADS checkpoint is missing: {_relative(checkpoint, root)}\n"
            "Run: python scripts/tune_lfads_gru.py --config "
            "configs/mc_maze_small_lfads_gru_tuning.yaml"
        )
        raise FileNotFoundError(msg)


def _load_dataset(config: dict[str, Any]) -> tuple[Any, str, dict[str, Any]]:
    root = get_repo_root()
    path = resolve_configured_path(str(config["dataset"]["processed_path"]), root)
    dataset = load_neural_dataset(path)
    validate_neural_dataset(dataset)
    dataset_hash = compute_dataset_hash(dataset)
    expected = str(config["dataset"]["expected_hash"])
    if dataset_hash != expected:
        msg = f"Dataset hash mismatch: expected {expected}, got {dataset_hash}"
        raise ValueError(msg)
    original_time_bins = int(dataset.spikes.shape[1])
    cropped = crop_neural_dataset_time(
        dataset, int(config["window"]["max_time_bins"]), str(config["window"]["crop_policy"])
    )
    window = describe_time_window(
        original_time_bins, int(cropped.spikes.shape[1]), cropped.bin_size_ms
    )
    return cropped, dataset_hash, window


def _evaluation_config(config: dict[str, Any], checkpoint_config: dict[str, Any]) -> dict[str, Any]:
    model_config = dict(checkpoint_config.get("model", {}))
    return {
        "data": {
            "max_time_bins": int(config["window"]["max_time_bins"]),
            "batch_size": 4,
            "num_workers": 0,
            "drop_last": False,
        },
        "splits": dict(config["splits"]),
        "model": {**model_config, "checkpoint_path": config["checkpoints"]["tuned_lfads_best"]},
        "evaluation_mode": {
            "use_direct_model_rates_for_heldout": True,
            "also_evaluate_factor_decoder": True,
        },
        "heldout_decoder": {
            "alpha": 1000.0,
            "fit_intercept": True,
            "standardize_factors": True,
            "min_rate_hz": float(model_config.get("min_rate_hz", 1.0e-4)),
            "max_rate_hz": float(model_config.get("max_rate_hz", 500.0)),
        },
        "behavior_decoder": {
            "enabled": bool(
                checkpoint_config.get("evaluation", {}).get("behavior_decoder_enabled", True)
            ),
            "alpha": 100.0,
            "fit_intercept": True,
            "standardize_factors": True,
            "standardize_targets": True,
            "target_prefixes": ["hand_pos", "cursor_pos"],
            "derive_velocity": True,
            "velocity_method": "central_difference",
        },
        "reference": {"name": "train_mean_rate", "fit_train_trials_only": True},
        "evaluation": {"evaluate_splits": list(config["audit"]["evaluate_splits"])},
        "evaluate_splits": list(config["audit"]["evaluate_splits"]),
    }


def _validation_row(split_metrics: pd.DataFrame, source: str) -> pd.Series:
    rows = split_metrics[
        (split_metrics["split"] == "validation") & (split_metrics["prediction_source"] == source)
    ]
    if rows.empty:
        rows = split_metrics[split_metrics["split"] == "validation"]
    return rows.iloc[0]


def _issue_flags(
    summary: dict[str, Any], tiny: pd.DataFrame, factor_usage: pd.DataFrame
) -> list[str]:
    flags: list[str] = []
    reference_bits = float(summary["mean_rate_reference_bits_per_spike"])
    validation_bits = float(summary["validation_bits_per_spike"])
    mean_predicted = float(summary["mean_predicted_rate_hz"])
    observed = float(summary["observed_rate_hz"])
    if validation_bits < reference_bits:
        flags.append("underfitting")
    if observed > 0.0 and mean_predicted < 0.75 * observed:
        flags.append("rate underprediction")
    if observed > 0.0 and mean_predicted > 1.5 * observed:
        flags.append("rate overprediction")
    if int(summary["active_factor_count"]) <= max(1, int(summary["total_factor_count"]) // 8):
        flags.append("KL collapse/posterior underuse")
    if (
        abs(
            float(summary["validation_bits_per_spike"])
            - float(summary["best_tuned_reference_bits_per_spike"])
        )
        > 0.05
    ):
        flags.append("loss/reference mismatch")
    if float(summary["validation_zero_fraction"]) > 0.95:
        flags.append("target sparsity")
    if not tiny.empty and not bool(summary["tiny_overfit_passed"]):
        flags.append("failure to overfit tiny subset")
    if factor_usage.empty:
        flags.append("factor usage unavailable")
    return flags


def _make_figures(
    output_dir: Path,
    summary: dict[str, Any],
    rate_calibration: pd.DataFrame,
    tiny_subset: pd.DataFrame,
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    plt.figure()
    labels = ["mean-rate", "factor-latent", "tuned LFADS", "audited LFADS"]
    values = [
        summary["mean_rate_reference_bits_per_spike"],
        summary["factor_latent_reference_bits_per_spike"],
        summary["best_tuned_reference_bits_per_spike"],
        summary["validation_bits_per_spike"],
    ]
    plt.bar(labels, values)
    plt.ylabel("validation bits/spike")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figures / "validation_bits_by_method.png")
    plt.close()

    plt.figure()
    plt.scatter(
        rate_calibration["mean_reference_rate_hz"],
        rate_calibration["mean_predicted_rate_hz"],
    )
    plt.xlabel("reference rate Hz")
    plt.ylabel("predicted rate Hz")
    plt.tight_layout()
    plt.savefig(figures / "predicted_vs_reference_rate.png")
    plt.close()

    plt.figure()
    plt.plot(rate_calibration["rate_bin"], rate_calibration["observed_rate_hz"], label="observed")
    plt.plot(
        rate_calibration["rate_bin"],
        rate_calibration["mean_predicted_rate_hz"],
        label="predicted",
    )
    plt.xlabel("predicted-rate bin")
    plt.ylabel("rate Hz")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "heldout_rate_calibration.png")
    plt.close()

    plt.figure()
    if not tiny_subset.empty:
        plt.plot(tiny_subset["epoch"], tiny_subset["train_total_loss"], label="train")
        plt.plot(tiny_subset["epoch"], tiny_subset["validation_total_loss"], label="validation")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "train_validation_loss_curve.png")
    plt.close()


def run_lfads_audit(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    root = get_repo_root()
    checkpoint_path = resolve_configured_path(str(config["checkpoints"]["tuned_lfads_best"]), root)
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = copy.deepcopy(checkpoint["config"])
    dataset, dataset_hash, window = _load_dataset(config)
    split = create_trial_split(
        dataset.trial_ids,
        float(config["splits"]["train_fraction"]),
        float(config["splits"]["validation_fraction"]),
        float(config["splits"]["test_fraction"]),
        seed=int(config["splits"]["seed"]),
    )
    neuron_mask = create_neuron_mask(
        dataset.spikes.shape[2],
        float(config["splits"]["heldout_neuron_fraction"]),
        seed=int(config["splits"]["seed"]),
    )
    validate_trial_split(split, dataset.trial_ids)
    validate_neuron_mask(neuron_mask, dataset.spikes.shape[2])
    dataloaders = create_dataloaders(
        create_torch_datasets(dataset, split, neuron_mask, int(config["window"]["max_time_bins"])),
        batch_size=4,
        num_workers=0,
        drop_last=False,
        seed=int(config["splits"]["seed"]),
    )
    device = resolve_device(str(config["runtime"]["device"]))
    input_dim = int(neuron_mask.heldin.sum())
    output_dim = int(dataset.spikes.shape[2])
    eval_config = _evaluation_config(config, checkpoint_config)
    model = load_lfads_gru_from_checkpoint(
        checkpoint_path, input_dim, output_dim, eval_config, device
    )
    extracted = extract_lfads_factors(model, dataloaders, device)
    model_config = dict(eval_config["model"])
    min_rate = float(model_config.get("min_rate_hz", 1.0e-4))
    max_rate = float(model_config.get("max_rate_hz", 500.0))
    target_indices = np.flatnonzero(neuron_mask.heldout)
    reference = _reference_rates(
        extracted["train"]["heldout_spikes"], dataset.bin_size_ms, min_rate, max_rate
    )

    split_metrics, _, _, _, metadata = run_lfads_gru_evaluation(
        dataset, split, neuron_mask, eval_config, device
    )
    loss_rows = []
    neuron_frames = []
    factor_frames = []
    validation_counts = extracted["validation"]["heldout_spikes"]
    validation_reference = _broadcast_reference(reference, validation_counts)
    validation_predicted = safe_clip_rates(
        extracted["validation"]["model_rates_hz"][:, :, target_indices], min_rate, max_rate
    )
    for split_name in config["audit"]["evaluate_splits"]:
        split_key = str(split_name)
        counts = extracted[split_key]["heldout_spikes"]
        reference_rates = _broadcast_reference(reference, counts)
        predicted = safe_clip_rates(
            extracted[split_key]["model_rates_hz"][:, :, target_indices], min_rate, max_rate
        )
        loss_rows.append(
            {"split": split_key, "prediction_source": "direct_model"}
            | compute_loss_scale_diagnostics(
                counts, predicted, reference_rates, dataset.bin_size_ms
            )
        )
        neuron_frames.append(
            compute_neuron_prediction_diagnostics(
                counts, predicted, reference_rates, dataset.bin_size_ms, target_indices, split_key
            )
        )
        factor_frames.append(compute_factor_usage(extracted[split_key]["factors"], split_key))
    loss_scale = pd.DataFrame(loss_rows)
    neuron_diagnostics = pd.concat(neuron_frames, ignore_index=True)
    factor_usage = pd.concat(factor_frames, ignore_index=True)
    rate_calibration = compute_rate_calibration_table(
        validation_counts,
        validation_predicted,
        validation_reference,
        dataset.bin_size_ms,
        int(config["audit"]["rate_bins"]),
    )
    rate_summary = summarize_rate_distribution(
        validation_predicted, validation_reference, validation_counts
    )

    tiny_config = copy.deepcopy(checkpoint_config)
    tiny_config.update(
        {
            key: copy.deepcopy(config[key])
            for key in ("dataset", "splits", "window", "runtime", "audit")
        }
    )
    tiny_subset = run_tiny_subset_overfit(
        tiny_config, resolve_configured_path(str(config["reporting"]["output_dir"]), root)
    )
    tiny_initial = (
        float(tiny_subset.iloc[0]["train_total_loss"]) if not tiny_subset.empty else float("nan")
    )
    tiny_final = (
        float(tiny_subset.iloc[-1]["train_total_loss"]) if not tiny_subset.empty else float("nan")
    )
    tiny_drop = (
        loss_drop_fraction(tiny_initial, tiny_final) if np.isfinite(tiny_initial) else float("nan")
    )
    validation = _validation_row(split_metrics, "direct_model")
    validation_loss = loss_scale[loss_scale["split"] == "validation"].iloc[0]
    validation_factor_usage = factor_usage[factor_usage["split"] == "validation"]
    active_factor_count = int(validation_factor_usage["active"].sum())
    expected_drop = float(config["audit"]["tiny_subset_expected_loss_drop_fraction"])
    summary = {
        "dataset_name": config["dataset"]["name"],
        "dataset_hash": dataset_hash,
        "window_time_bins": int(window["cropped_time_bins"]),
        "window_seconds": float(window["window_seconds"]),
        "cuda_device": torch.cuda.get_device_name(0),
        "checkpoint_audited": _relative(checkpoint_path, root),
        "validation_bits_per_spike": float(validation["bits_per_spike"]),
        "validation_poisson_nll": float(validation["poisson_nll"]),
        "mean_rate_reference_bits_per_spike": float(
            config["references"]["window_matched_mean_rate_validation_bits_per_spike"]
        ),
        "factor_latent_reference_bits_per_spike": float(
            config["references"]["window_matched_factor_latent_validation_bits_per_spike"]
        ),
        "best_tuned_reference_bits_per_spike": float(
            config["references"]["best_tuned_lfads_validation_bits_per_spike"]
        ),
        "best_tuned_lfads_run_id": config["references"]["best_tuned_lfads_run_id"],
        "mean_predicted_rate_hz": float(rate_summary["mean_predicted_rate_hz"]),
        "observed_rate_hz": float(validation_loss["observed_rate_hz"]),
        "mean_reference_rate_hz": float(rate_summary["mean_reference_rate_hz"]),
        "prediction_reference_correlation": compute_prediction_reference_correlation(
            validation_predicted, validation_reference
        ),
        "active_factor_count": active_factor_count,
        "total_factor_count": int(validation_factor_usage.shape[0]),
        "validation_zero_fraction": float(validation_loss["zero_fraction"]),
        "tiny_overfit_initial_loss": tiny_initial,
        "tiny_overfit_final_loss": tiny_final,
        "tiny_overfit_loss_drop_fraction": tiny_drop,
        "tiny_overfit_passed": bool(tiny_drop >= expected_drop),
        "direct_model_available": bool(metadata.get("direct_model_available", False)),
        "factor_decoder_evaluated": bool(metadata.get("factor_decoder_evaluated", False)),
    }
    summary["likely_issue_flags"] = _issue_flags(summary, tiny_subset, factor_usage)
    tables = {
        "split_diagnostics": split_metrics,
        "neuron_diagnostics": neuron_diagnostics,
        "rate_calibration": rate_calibration,
        "loss_scale_diagnostics": loss_scale,
        "tiny_subset_overfit": tiny_subset,
        "factor_usage": factor_usage,
    }
    if bool(config["audit"].get("save_figures", False)):
        _make_figures(
            resolve_configured_path(str(config["reporting"]["output_dir"]), root),
            summary,
            rate_calibration,
            tiny_subset,
        )
    return summary, tables


def main(argv: Sequence[str] | None = None) -> int:
    config_path = _parse_args(argv)
    if not config_path.exists():
        console.print(f"Config not found: {config_path}")
        return 2
    try:
        validated = LFADSAuditConfig.from_yaml(config_path)
        config = validated.model_dump(mode="python")
        diagnostic = _cuda_diagnostic()
        console.print(f"torch: {diagnostic.get('torch_version')}")
        console.print(f"cuda_available: {diagnostic['available']}")
        console.print(f"torch_cuda: {diagnostic.get('torch_cuda')}")
        console.print(f"gpu_name: {diagnostic['gpu']}")
        if not diagnostic["available"]:
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        _check_required_paths(config)
        summary, tables = run_lfads_audit(config)
        output_dir = resolve_configured_path(
            str(config["reporting"]["output_dir"]), get_repo_root()
        )
        write_lfads_audit_outputs(output_dir, summary, tables)
    except Exception as exc:
        console.print(str(exc))
        return 2
    console.print(f"dataset: {summary['dataset_name']}")
    console.print(f"device: {config['runtime']['device']}")
    console.print(f"gpu: {summary['cuda_device']}")
    console.print(f"tuned_checkpoint: {summary['checkpoint_audited']}")
    console.print(f"validation_bits_per_spike: {summary['validation_bits_per_spike']}")
    console.print(f"reference_bits_per_spike: {summary['mean_rate_reference_bits_per_spike']}")
    console.print(f"mean_predicted_rate_hz: {summary['mean_predicted_rate_hz']}")
    console.print(f"observed_heldout_rate_hz: {summary['observed_rate_hz']}")
    console.print(
        f"prediction_reference_correlation: {summary['prediction_reference_correlation']}"
    )
    console.print(f"active_factor_count: {summary['active_factor_count']}")
    console.print(f"tiny_overfit_initial_loss: {summary['tiny_overfit_initial_loss']}")
    console.print(f"tiny_overfit_final_loss: {summary['tiny_overfit_final_loss']}")
    console.print(f"tiny_overfit_loss_drop_fraction: {summary['tiny_overfit_loss_drop_fraction']}")
    console.print(f"likely_issue_flags: {summary['likely_issue_flags']}")
    console.print(f"output_dir: {config['reporting']['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
