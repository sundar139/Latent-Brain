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
from latentbrain.eval.neural_ode_tuning import rank_neural_ode_results
from latentbrain.eval.rebinning import compute_window_bins_for_duration
from latentbrain.eval.reporting import write_neural_ode_tuning_outputs
from latentbrain.paths import get_repo_root, resolve_configured_path
from latentbrain.train.neural_ode_tuning import run_neural_ode_tuning

console = Console(markup=False)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local deterministic neural-ODE-style latent dynamics tuning."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mc_maze_small_neural_ode_tuning.yaml"),
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
        msg = f"malformed deterministic neural-ODE-style tuning config: {path}"
        raise ValueError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"deterministic neural-ODE-style tuning config must contain a mapping: {path}"
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
    target_seconds = float(config["binning"]["target_bin_size_ms"]) / 1000.0
    if abs(float(config["model"]["dt_seconds"]) - target_seconds) > 1e-12:
        msg = "model.dt_seconds must match target bin size"
        raise ValueError(msg)
    if str(config["model"].get("name")) != "neural_ode":
        msg = "model.name must be neural_ode"
        raise ValueError(msg)
    scoring = dict(config["scoring"])
    if scoring["reference_model"] != "train_heldout_mean_rate":
        msg = "scoring.reference_model must be train_heldout_mean_rate"
        raise ValueError(msg)
    if config["search"]["selection_metric"] != "validation_unified_bits_per_spike":
        msg = "search.selection_metric must be validation_unified_bits_per_spike"
        raise ValueError(msg)
    if int(config["search"]["max_runs"]) <= 0:
        msg = "search.max_runs must be positive"
        raise ValueError(msg)
    grid = dict(config["grid"])
    for key in (
        "encoder_hidden_dim",
        "drift_hidden_dim",
        "diffusion_hidden_dim",
        "latent_dim",
        "factor_dim",
        "epochs",
    ):
        if any(int(value) <= 0 for value in grid[key]):
            msg = f"grid.{key} values must be positive"
            raise ValueError(msg)
    if any(float(value) < 0.0 or float(value) >= 1.0 for value in grid["input_dropout_rate"]):
        msg = "grid.input_dropout_rate values must be in [0, 1)"
        raise ValueError(msg)
    if any(float(value) < 0.0 for value in grid["kl_scale"]):
        msg = "grid.kl_scale values must be non-negative"
        raise ValueError(msg)
    if any(float(value) != 0.0 for value in grid["diffusion_scale"]):
        msg = "grid.diffusion_scale values must be exactly 0.0"
        raise ValueError(msg)
    references = dict(config["references"])
    for key in (
        "train_mean_validation_bits_per_spike",
        "factor_latent_unified_validation_bits_per_spike",
        "previous_neural_sde_validation_bits_per_spike",
        "previous_best_lfads_family_validation_bits_per_spike",
        "oracle_validation_bits_per_spike",
    ):
        if key not in references:
            msg = f"references.{key} is required"
            raise ValueError(msg)
    for key in ("save_unified_checkpoints", "evaluate_checkpoints_by_unified_metric"):
        if not bool(config["model"].get(key, False)):
            msg = f"model.{key} must be true"
            raise ValueError(msg)


def _write_figures(output_dir: Path, results: pd.DataFrame, leaderboard: pd.DataFrame) -> None:
    import matplotlib  # type: ignore[import-untyped]  # noqa: PLC0415

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]  # noqa: PLC0415

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if not results.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(results["run_id"], results["validation_unified_bits_per_spike"])
        ax.set_ylabel("Validation unified bits/spike")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(figures / "neural_ode_validation_bits_by_run.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(results["run_id"], results["drift_norm"])
        ax.set_ylabel("Drift norm")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(figures / "drift_norm_by_run.png", dpi=150)
        plt.close(fig)

    if not leaderboard.empty:
        best_run = str(leaderboard.iloc[0]["run_id"])
        best_row = results[results["run_id"] == best_run].iloc[0]
        output_value = best_row.get("output_dir")
        run_dir = Path(str(output_value)) if output_value is not None else None
        history_path = run_dir / "metrics_history.csv" if run_dir is not None else None
        if history_path is not None and history_path.exists():
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
        checkpoint_path = run_dir / "checkpoint_scores.csv" if run_dir is not None else None
        if checkpoint_path is not None and checkpoint_path.exists():
            checkpoints = pd.read_csv(checkpoint_path)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(
                checkpoints["epoch"],
                checkpoints["validation_unified_bits_per_spike"],
                marker="o",
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Validation unified bits/spike")
            fig.tight_layout()
            fig.savefig(figures / "validation_bits_vs_epoch_best.png", dpi=150)
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
        results, summary = run_neural_ode_tuning(config)
        _validate_reference_zero(
            float(summary.get("train_mean_validation_bits_per_spike", float("nan")))
        )
    except Exception as exc:
        console.print(str(exc))
        return 2
    output_dir = resolve_configured_path(str(config["reporting"]["output_dir"]), repo_root)
    leaderboard = rank_neural_ode_results(results)
    checkpoint_scores = pd.read_csv(output_dir / "checkpoint_selection.csv")
    write_neural_ode_tuning_outputs(output_dir, summary, results, leaderboard, checkpoint_scores)
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
        "best_validation_unified_bits_per_spike",
        "best_validation_poisson_nll",
        "best_factor_decoder_unified_bits_per_spike",
        "best_drift_norm",
        "best_diffusion_mean",
        "best_checkpoint_source",
        "factor_latent_unified_reference",
        "previous_neural_sde_reference",
        "beats_factor_latent_unified",
        "beats_previous_neural_sde",
        "output_dir",
    ):
        console.print(f"{key}: {summary.get(key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
