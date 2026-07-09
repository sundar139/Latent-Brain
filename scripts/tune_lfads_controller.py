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
from latentbrain.eval.lfads_controller_tuning import rank_controller_results
from latentbrain.eval.rebinning import compute_window_bins_for_duration
from latentbrain.eval.reporting import write_lfads_controller_tuning_outputs
from latentbrain.paths import get_repo_root, resolve_configured_path
from latentbrain.train.lfads_controller_tuning import run_lfads_controller_tuning

console = Console(markup=False)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local controller-style LFADS-family tuning.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mc_maze_small_lfads_controller_tuning.yaml"),
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
        msg = f"malformed LFADS controller tuning config: {path}"
        raise ValueError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"LFADS controller tuning config must contain a mapping: {path}"
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
    compute_window_bins_for_duration(
        float(config["window"]["duration_seconds"]),
        int(config["binning"]["target_bin_size_ms"]),
    )
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
        "controller_hidden_dim",
        "generator_hidden_dim",
        "latent_dim",
        "factor_dim",
        "inferred_input_dim",
        "epochs",
    ):
        if any(int(value) <= 0 for value in grid[key]):
            msg = f"grid.{key} values must be positive"
            raise ValueError(msg)
    if any(float(value) < 0.0 or float(value) >= 1.0 for value in grid["input_dropout_rate"]):
        msg = "grid.input_dropout_rate values must be in [0, 1)"
        raise ValueError(msg)
    for key in ("kl_scale", "inferred_input_kl_scale"):
        if any(float(value) < 0.0 for value in grid[key]):
            msg = f"grid.{key} values must be non-negative"
            raise ValueError(msg)
    references = dict(config["references"])
    for key in (
        "train_mean_validation_bits_per_spike",
        "factor_latent_unified_validation_bits_per_spike",
        "previous_best_lfads_family_validation_bits_per_spike",
        "oracle_validation_bits_per_spike",
    ):
        if key not in references:
            msg = f"references.{key} is required"
            raise ValueError(msg)


def _write_figures(output_dir: Path, results: pd.DataFrame, leaderboard: pd.DataFrame) -> None:
    import matplotlib  # type: ignore[import-untyped]  # noqa: PLC0415

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]  # noqa: PLC0415

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if not results.empty:
        plt.figure()
        plt.bar(results["run_id"], results["validation_unified_bits_per_spike"])
        plt.ylabel("Validation unified bits/spike")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(figures / "controller_validation_bits_by_run.png")
        plt.close()

        plt.figure()
        kl_values = results.get("inferred_input_kl_loss", 0.0)
        plt.bar(results["run_id"], kl_values)
        plt.ylabel("Inferred-input KL")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(figures / "controller_kl_by_run.png")
        plt.close()

    if not leaderboard.empty:
        best_run = str(leaderboard.iloc[0]["run_id"])
        best_row = results[results["run_id"] == best_run].iloc[0]
        history_path = Path(str(best_row["output_dir"])) / "metrics_history.csv"
        if history_path.exists():
            history = pd.read_csv(history_path)
            plt.figure()
            plt.plot(history["epoch"], history["train_total_loss"], label="train")
            plt.plot(history["epoch"], history["validation_total_loss"], label="validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(figures / "train_validation_loss_curve_best.png")
            plt.close()


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
        results, summary = run_lfads_controller_tuning(config)
        _validate_reference_zero(
            float(summary.get("train_mean_validation_bits_per_spike", float("nan")))
        )
    except Exception as exc:
        console.print(str(exc))
        return 2
    output_dir = resolve_configured_path(str(config["reporting"]["output_dir"]), repo_root)
    leaderboard = rank_controller_results(results)
    write_lfads_controller_tuning_outputs(output_dir, summary, results, leaderboard)
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
        "best_inferred_input_kl_loss",
        "factor_latent_unified_reference",
        "previous_best_lfads_family_reference",
        "beats_factor_latent_unified",
        "beats_previous_best_lfads_family",
        "output_dir",
    ):
        console.print(f"{key}: {summary.get(key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
