from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]
import torch
import yaml
from rich.console import Console

from latentbrain.data.io import compute_dataset_hash, load_neural_dataset
from latentbrain.data.rebinning import rebin_neural_dataset, validate_rebin_factor
from latentbrain.data.schemas import NeuralDataset
from latentbrain.data.splits import create_neuron_mask, create_trial_split
from latentbrain.data.validation import (
    validate_neural_dataset,
    validate_neuron_mask,
    validate_trial_split,
)
from latentbrain.eval.coordinated_dropout import (
    EVALUATION_COLUMNS,
    build_dropout_result_row,
    summarize_dropout_runs,
)
from latentbrain.eval.rebinning import compute_window_bins_for_duration
from latentbrain.eval.reporting import write_lfads_coordinated_dropout_outputs
from latentbrain.eval.windowing import crop_neural_dataset_time
from latentbrain.paths import get_repo_root, resolve_configured_path
from latentbrain.train.lfads_tuning import _train_and_evaluate_run
from latentbrain.train.rebinned_lfads import (
    build_coordinated_dropout_lfads_train_config,
    coordinated_dropout_run_id,
)

console = Console(markup=False)

DIAGNOSTIC_COLUMNS = [
    "run_id",
    "dropout_rate",
    "epoch",
    "configured_input_dropout_rate",
    "realized_input_dropout_fraction",
    "train_total_loss",
    "validation_total_loss",
    "validation_heldout_prediction_loss",
]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local LFADS-style coordinated dropout diagnostics."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mc_maze_small_lfads_coordinated_dropout.yaml"),
    )
    return parser.parse_args(argv)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _cuda_diagnostic() -> dict[str, Any]:
    return {
        "torch": torch.__version__,
        "available": bool(torch.cuda.is_available()),
        "torch_cuda": torch.version.cuda,
        "device_count": int(torch.cuda.device_count()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
    }


def _load_config(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"malformed LFADS coordinated dropout config: {path}"
        raise ValueError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"LFADS coordinated dropout config must contain a mapping: {path}"
        raise ValueError(msg)
    _validate_config(raw)
    return raw


def _validate_config(config: dict[str, Any]) -> None:
    if str(config["runtime"]["device"]) != "cuda":
        msg = "runtime.device must be cuda for LFADS coordinated dropout diagnostics"
        raise ValueError(msg)
    original_bin = int(config["dataset"]["original_bin_size_ms"])
    target_bin = int(config["binning"]["target_bin_size_ms"])
    validate_rebin_factor(original_bin, target_bin)
    compute_window_bins_for_duration(float(config["window"]["duration_seconds"]), target_bin)
    dropout = dict(config["dropout"])
    rates = [float(value) for value in dropout["rates"]]
    if not rates or any(rate < 0.0 or rate >= 1.0 for rate in rates):
        msg = "dropout.rates must contain values in [0, 1)"
        raise ValueError(msg)
    if {str(value) for value in dropout.get("apply_to", [])} != {"train"}:
        msg = "dropout.apply_to must be train only for this real-data workflow"
        raise ValueError(msg)
    if not bool(dropout.get("keep_at_least_one_neuron", False)):
        msg = "dropout.keep_at_least_one_neuron must be true"
        raise ValueError(msg)
    settings = dict(config["lfads_settings"])
    for key in ("encoder_hidden_dim", "generator_hidden_dim", "latent_dim", "factor_dim", "epochs"):
        if int(settings[key]) <= 0:
            msg = f"lfads_settings.{key} must be positive"
            raise ValueError(msg)
    for key in (
        "same_bin_mean_rate_validation_bits_per_spike",
        "same_bin_factor_latent_validation_bits_per_spike",
        "previous_20ms_lfads_validation_bits_per_spike",
    ):
        if key not in config["references"]:
            msg = f"references.{key} is required"
            raise ValueError(msg)


def _prepare_dataset(config: dict[str, Any]) -> tuple[NeuralDataset, str, int]:
    repo_root = get_repo_root()
    processed_path = resolve_configured_path(str(config["dataset"]["processed_path"]), repo_root)
    if not processed_path.exists():
        msg = f"Processed dataset is missing: {_relative(processed_path, repo_root)}"
        raise FileNotFoundError(msg)
    dataset = load_neural_dataset(processed_path)
    validate_neural_dataset(dataset)
    dataset_hash = compute_dataset_hash(dataset)
    expected = str(config["dataset"].get("expected_hash", ""))
    if expected and dataset_hash != expected:
        msg = f"Dataset hash mismatch: expected {expected}, got {dataset_hash}"
        raise ValueError(msg)
    target_bin = int(config["binning"]["target_bin_size_ms"])
    rebinned = rebin_neural_dataset(dataset, target_bin)
    window_bins = compute_window_bins_for_duration(
        float(config["window"]["duration_seconds"]), target_bin
    )
    windowed = crop_neural_dataset_time(rebinned, window_bins, str(config["window"]["crop_policy"]))
    split = create_trial_split(
        windowed.trial_ids,
        float(config["splits"]["train_fraction"]),
        float(config["splits"]["validation_fraction"]),
        float(config["splits"]["test_fraction"]),
        seed=int(config["splits"]["seed"]),
    )
    mask = create_neuron_mask(
        windowed.spikes.shape[2],
        float(config["splits"]["heldout_neuron_fraction"]),
        seed=int(config["splits"]["seed"]),
    )
    validate_trial_split(split, windowed.trial_ids)
    validate_neuron_mask(mask, windowed.spikes.shape[2])
    return windowed, dataset_hash, window_bins


def _factor_decoder_bits(run_dir: Path) -> float:
    path = run_dir / "evaluation" / "split_metrics.csv"
    if not path.exists():
        return float("nan")
    split_metrics = pd.read_csv(path)
    rows = split_metrics[
        (split_metrics["split"] == "validation")
        & (split_metrics["prediction_source"] == "factor_decoder")
    ]
    return float("nan") if rows.empty else float(rows.iloc[0]["bits_per_spike"])


def _history_tables(
    run_dir: Path, run_id: str, dropout_rate: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = pd.read_csv(run_dir / "metrics_history.csv")
    history.insert(0, "dropout_rate", float(dropout_rate))
    history.insert(0, "run_id", run_id)
    diagnostics = history[
        [column for column in DIAGNOSTIC_COLUMNS if column in history.columns]
    ].copy()
    return history, diagnostics


def _write_figures(output_dir: Path, evaluation: pd.DataFrame, training: pd.DataFrame) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.bar(evaluation["run_id"], evaluation["validation_bits_per_spike"])
    plt.ylabel("Validation bits/spike")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(figures / "validation_bits_comparison.png")
    plt.close()

    plt.figure()
    plt.plot(evaluation["dropout_rate"], evaluation["validation_bits_per_spike"], marker="o")
    plt.xlabel("Input dropout rate")
    plt.ylabel("Validation bits/spike")
    plt.tight_layout()
    plt.savefig(figures / "dropout_rate_curve.png")
    plt.close()

    plt.figure()
    for run_id, run_history in training.groupby("run_id"):
        plt.plot(run_history["epoch"], run_history["train_total_loss"], label=f"{run_id} train")
        plt.plot(
            run_history["epoch"],
            run_history["validation_total_loss"],
            linestyle="--",
            label=f"{run_id} validation",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(figures / "train_validation_loss_curve.png")
    plt.close()


def run_lfads_coordinated_dropout(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    _validate_config(config)
    cuda = _cuda_diagnostic()
    if bool(config["runtime"].get("fail_if_cuda_unavailable", True)) and not cuda["available"]:
        msg = "CUDA was requested, but torch.cuda.is_available() is False."
        raise RuntimeError(msg)
    dataset, dataset_hash, window_bins = _prepare_dataset(config)
    repo_root = get_repo_root()
    output_dir = resolve_configured_path(str(config["reporting"]["output_dir"]), repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    histories = []
    diagnostics = []
    references = dict(config["references"])
    for run_index, dropout_rate in enumerate(float(value) for value in config["dropout"]["rates"]):
        run_id = coordinated_dropout_run_id(dropout_rate)
        run_dir = output_dir / "runs" / run_id
        run_config = build_coordinated_dropout_lfads_train_config(
            config, dropout_rate, window_bins, run_dir
        )
        try:
            metrics = _train_and_evaluate_run(run_config, run_index, run_id, dataset)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            metrics = {
                "status": "failed",
                "notes": f"CUDA out of memory: {exc}",
                "output_dir": run_dir,
            }
        else:
            history, dropout_diag = _history_tables(run_dir, run_id, dropout_rate)
            histories.append(history)
            diagnostics.append(dropout_diag)
            metrics.update(
                {
                    "status": "completed",
                    "validation_factor_decoder_bits_per_spike": _factor_decoder_bits(run_dir),
                    "train_total_loss": float(
                        history.iloc[-1].get("train_total_loss", float("nan"))
                    ),
                    "output_dir": run_dir,
                }
            )
        rows.append(build_dropout_result_row(dropout_rate, run_id, metrics, references))

    evaluation = pd.DataFrame(rows, columns=EVALUATION_COLUMNS)
    training = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    dropout_diagnostics = (
        pd.concat(diagnostics, ignore_index=True) if diagnostics else pd.DataFrame()
    )
    summary = summarize_dropout_runs(evaluation, references)
    summary.update(
        {
            "dataset_name": config["dataset"]["name"],
            "dataset_hash": dataset_hash,
            "bin_size_ms": dataset.bin_size_ms,
            "window_seconds": float(config["window"]["duration_seconds"]),
            "cuda_device": cuda["gpu"],
            "dropout_rates_tested": [float(value) for value in config["dropout"]["rates"]],
            "same_bin_mean_rate_reference": float(
                references["same_bin_mean_rate_validation_bits_per_spike"]
            ),
            "same_bin_factor_latent_reference": float(
                references["same_bin_factor_latent_validation_bits_per_spike"]
            ),
            "previous_20ms_lfads_reference": float(
                references["previous_20ms_lfads_validation_bits_per_spike"]
            ),
            "output_dir": str(output_dir),
            "warnings": [
                "This is local coordinated-dropout diagnostic training, not an official NLB "
                "leaderboard result.",
                "The model is LFADS-style only, not full LFADS.",
            ],
        }
    )
    return summary, {
        "training_metrics": training,
        "evaluation_metrics": evaluation,
        "dropout_diagnostics": dropout_diagnostics,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "dataset_name",
        "cuda_device",
        "bin_size_ms",
        "dropout_rates_tested",
        "best_dropout_rate",
        "best_validation_bits_per_spike",
        "best_validation_poisson_nll",
        "best_validation_factor_decoder_bits_per_spike",
        "same_bin_mean_rate_reference",
        "same_bin_factor_latent_reference",
        "previous_20ms_lfads_reference",
        "coordinated_dropout_improves_lfads",
        "beats_same_bin_factor_latent",
        "beats_same_bin_mean_rate",
        "output_dir",
    ):
        console.print(f"{key}: {summary.get(key)}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = _load_config(args.config)
        summary, tables = run_lfads_coordinated_dropout(config)
        output_dir = resolve_configured_path(
            str(config["reporting"]["output_dir"]), get_repo_root()
        )
        write_lfads_coordinated_dropout_outputs(
            output_dir,
            summary,
            tables["training_metrics"],
            tables["evaluation_metrics"],
            tables["dropout_diagnostics"],
        )
        _write_figures(output_dir, tables["evaluation_metrics"], tables["training_metrics"])
        _print_summary(summary)
    except Exception as exc:
        console.print(str(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
