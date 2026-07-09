from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import torch
import yaml
from rich.console import Console

from latentbrain.data.rebinning import validate_rebin_factor
from latentbrain.eval.neural_ode_objectives import rank_neural_ode_objective_results
from latentbrain.eval.rebinning import compute_window_bins_for_duration
from latentbrain.eval.reporting import write_neural_ode_objective_outputs
from latentbrain.paths import get_repo_root, resolve_configured_path
from latentbrain.train.neural_ode_objectives import run_neural_ode_objective_variants

console = Console(markup=False)

REQUIRED_REFERENCES = (
    "train_mean_validation_bits_per_spike",
    "factor_latent_unified_validation_bits_per_spike",
    "previous_neural_ode_refinement_validation_bits_per_spike",
    "switching_ode_validation_bits_per_spike",
    "oracle_validation_bits_per_spike",
)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local deterministic neural-ODE objective diagnostics."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mc_maze_small_neural_ode_objectives.yaml"),
    )
    return parser.parse_args(argv)


def _cuda_diagnostic() -> dict[str, Any]:
    available = bool(torch.cuda.is_available())
    return {
        "torch": torch.__version__,
        "cuda_available": available,
        "torch_cuda": torch.version.cuda,
        "device_count": int(torch.cuda.device_count()),
        "device_name": torch.cuda.get_device_name(0) if available else "NONE",
    }


def _load_config(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"malformed deterministic neural-ODE objective config: {path}"
        raise ValueError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"deterministic neural-ODE objective config must contain a mapping: {path}"
        raise ValueError(msg)
    _validate_config(raw)
    return raw


def _check_processed_path(config: dict[str, Any]) -> None:
    repo_root = get_repo_root()
    processed_path = resolve_configured_path(str(config["dataset"]["processed_path"]), repo_root)
    if not processed_path.exists():
        msg = f"Processed dataset is missing: {processed_path}"
        raise FileNotFoundError(msg)


def _validate_reference_zero(value: float) -> float:
    if abs(float(value)) > 1e-12:
        msg = "train-mean-as-model validation bits/spike must be 0.0"
        raise RuntimeError(msg)
    return float(value)


def _validate_variants(config: dict[str, Any]) -> None:
    variants = list(config["objective_variants"])
    if not variants:
        msg = "objective_variants must not be empty"
        raise ValueError(msg)
    names = [str(variant["name"]) for variant in variants]
    if len(set(names)) != len(names):
        msg = "objective variant names must be unique"
        raise ValueError(msg)
    for variant in variants:
        if float(variant.get("zero_count_weight", 1.0)) <= 0.0:
            msg = "zero_count_weight must be positive"
            raise ValueError(msg)
        if float(variant.get("positive_count_weight", 1.0)) <= 0.0:
            msg = "positive_count_weight must be positive"
            raise ValueError(msg)
        if float(variant.get("rate_calibration_loss_weight", 0.0)) < 0.0:
            msg = "rate_calibration_loss_weight must be non-negative"
            raise ValueError(msg)
        if float(variant.get("drift_regularization", 0.0)) < 0.0:
            msg = "drift_regularization must be non-negative"
            raise ValueError(msg)
        if str(variant.get("scheduler", "cosine")) not in {"none", "cosine"}:
            msg = "scheduler must be none or cosine"
            raise ValueError(msg)


def _validate_config(config: dict[str, Any]) -> None:
    if str(config["runtime"]["device"]) != "cuda":
        msg = "runtime.device must be cuda"
        raise ValueError(msg)
    if not bool(config["runtime"].get("fail_if_cuda_unavailable", False)):
        msg = "runtime.fail_if_cuda_unavailable must be true"
        raise ValueError(msg)
    validate_rebin_factor(
        int(config["dataset"]["original_bin_size_ms"]),
        int(config["binning"]["target_bin_size_ms"]),
    )
    window_bins = compute_window_bins_for_duration(
        float(config["window"]["duration_seconds"]),
        int(config["binning"]["target_bin_size_ms"]),
    )
    if window_bins <= 0:
        msg = "window duration must convert to positive integer bins"
        raise ValueError(msg)
    base_model = dict(config["base_model"])
    target_seconds = float(config["binning"]["target_bin_size_ms"]) / 1000.0
    if abs(float(base_model["dt_seconds"]) - target_seconds) > 1e-12:
        msg = "base_model.dt_seconds must match target bin size"
        raise ValueError(msg)
    if str(base_model.get("name")) != "neural_ode_objectives":
        msg = "base_model.name must be neural_ode_objectives"
        raise ValueError(msg)
    if float(base_model.get("diffusion_scale", 0.0)) != 0.0:
        msg = "base_model.diffusion_scale must be exactly 0.0"
        raise ValueError(msg)
    if str(base_model.get("scheduler", "none")) not in {"none", "cosine"}:
        msg = "base_model.scheduler must be none or cosine"
        raise ValueError(msg)
    if float(base_model.get("drift_regularization", 0.0)) < 0.0:
        msg = "base_model.drift_regularization must be non-negative"
        raise ValueError(msg)
    if str(config["scoring"]["reference_model"]) != "train_heldout_mean_rate":
        msg = "scoring.reference_model must be train_heldout_mean_rate"
        raise ValueError(msg)
    if config["search"]["selection_metric"] != "validation_unified_bits_per_spike":
        msg = "search.selection_metric must be validation_unified_bits_per_spike"
        raise ValueError(msg)
    if int(config["search"]["max_runs"]) <= 0:
        msg = "search.max_runs must be positive"
        raise ValueError(msg)
    _validate_variants(config)
    references = dict(config["references"])
    for key in REQUIRED_REFERENCES:
        if key not in references:
            msg = f"references.{key} is required"
            raise ValueError(msg)
    for key in ("save_unified_checkpoints", "evaluate_checkpoints_by_unified_metric"):
        if not bool(base_model.get(key, False)):
            msg = f"base_model.{key} must be true"
            raise ValueError(msg)


def _write_figures(output_dir: Path, results: pd.DataFrame, leaderboard: pd.DataFrame) -> None:
    import matplotlib  # type: ignore[import-untyped]  # noqa: PLC0415

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]  # noqa: PLC0415

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if not results.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(results["objective_name"], results["validation_unified_bits_per_spike"])
        ax.set_ylabel("Validation unified bits/spike")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(figures / "objective_validation_bits_by_run.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            results["validation_heldout_prediction_loss"],
            results["validation_unified_bits_per_spike"],
        )
        ax.set_xlabel("Validation held-out prediction loss")
        ax.set_ylabel("Validation unified bits/spike")
        fig.tight_layout()
        fig.savefig(figures / "heldout_loss_vs_unified_bits.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(results["zero_count_weight"], results["validation_unified_bits_per_spike"])
        ax.set_xlabel("Zero count weight")
        ax.set_ylabel("Validation unified bits/spike")
        fig.tight_layout()
        fig.savefig(figures / "zero_spike_weight_sensitivity.png", dpi=150)
        plt.close(fig)

    if not leaderboard.empty:
        best_run = str(leaderboard.iloc[0]["run_id"])
        best_row = results[results["run_id"] == best_run].iloc[0]
        components = {
            "heldout_prediction": float(best_row["validation_heldout_prediction_loss"]),
            "z0_kl": float(best_row["z0_kl_loss"]),
            "drift_regularization": float(best_row["drift_regularization_loss"]),
            "rate_calibration": float(best_row["rate_calibration_loss"]),
        }
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(list(components), list(components.values()))
        ax.set_ylabel("Loss component (best run)")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(figures / "objective_loss_components_best.png", dpi=150)
        plt.close(fig)

        history_path = Path(str(best_row["output_dir"])) / "metrics_history.csv"
        if history_path.exists():
            history = pd.read_csv(history_path)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(history["epoch"], history["train_total_loss"], label="train")
            ax.plot(history["epoch"], history["validation_total_loss"], label="validation")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            fig.tight_layout()
            fig.savefig(figures / "train_validation_loss_curve_best.png", dpi=150)
            plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = get_repo_root()
    args = _parse_args(argv)
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    if not config_path.exists():
        console.print(f"Config file is missing: {config_path}")
        return 2
    try:
        config = _load_config(config_path)
        _check_processed_path(config)
    except (OSError, ValueError) as exc:
        console.print(f"Config validation failed: {exc}")
        return 2
    cuda = _cuda_diagnostic()
    if not cuda["cuda_available"]:
        console.print("CUDA was requested, but torch.cuda.is_available() is False.")
        return 2
    try:
        results, summary = run_neural_ode_objective_variants(config)
        _validate_reference_zero(
            float(summary.get("train_mean_validation_bits_per_spike", float("nan")))
        )
    except Exception as exc:
        console.print(str(exc))
        return 2
    output_dir = resolve_configured_path(str(config["reporting"]["output_dir"]), repo_root)
    leaderboard = rank_neural_ode_objective_results(results)
    checkpoint_scores = pd.read_csv(output_dir / "checkpoint_selection.csv")
    diagnostics = pd.read_csv(output_dir / "objective_diagnostics.csv")
    write_neural_ode_objective_outputs(
        output_dir, summary, results, leaderboard, diagnostics, checkpoint_scores
    )
    _write_figures(output_dir, results, leaderboard)
    for key in (
        "dataset_name",
        "cuda_device",
        "bin_size_ms",
        "window_seconds",
        "reference_model",
        "runs_attempted",
        "successful_runs",
        "best_run_id",
        "best_objective_name",
        "best_validation_unified_bits_per_spike",
        "best_validation_poisson_nll",
        "best_factor_decoder_unified_bits_per_spike",
        "best_heldout_loss_weight",
        "best_zero_count_weight",
        "best_positive_count_weight",
        "best_rate_calibration_loss_weight",
        "best_drift_norm",
        "best_drift_regularization_loss",
        "best_checkpoint_source",
        "factor_latent_unified_reference",
        "previous_neural_ode_refinement_reference",
        "beats_factor_latent_unified",
        "beats_previous_neural_ode_refinement",
        "output_dir",
    ):
        console.print(f"{key}: {summary.get(key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
