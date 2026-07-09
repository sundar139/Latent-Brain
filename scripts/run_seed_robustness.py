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
from latentbrain.eval.rebinning import compute_window_bins_for_duration
from latentbrain.eval.reporting import write_seed_robustness_outputs
from latentbrain.eval.seed_robustness import (
    FACTOR_LATENT_METHOD,
    build_seed_robustness_leaderboard,
)
from latentbrain.paths import get_repo_root, resolve_configured_path
from latentbrain.train.seed_robustness import (
    METHOD_TYPES,
    NEURAL_METHOD_TYPES,
    build_carried_forward_config,
    build_seed_effects,
    method_summary_from_results,
    run_seed_robustness,
)

console = Console(markup=False)

REQUIRED_REFERENCES = (
    "train_mean_validation_bits_per_spike",
    "factor_latent_single_seed_reference",
    "neural_ode_refinement_single_seed_reference",
    "neural_ode_objective_single_seed_reference",
    "oracle_validation_bits_per_spike",
)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local multi-seed robustness benchmark.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mc_maze_small_multiseed_robustness.yaml"),
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
        msg = f"malformed seed robustness config: {path}"
        raise ValueError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"seed robustness config must contain a mapping: {path}"
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


def requires_cuda(config: dict[str, Any]) -> bool:
    return any(str(method["type"]) in NEURAL_METHOD_TYPES for method in config["methods"])


def _validate_config(config: dict[str, Any]) -> None:
    if requires_cuda(config):
        if str(config["runtime"]["device"]) != "cuda":
            msg = "runtime.device must be cuda when neural methods are configured"
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
    if str(config["scoring"]["reference_model"]) != "train_heldout_mean_rate":
        msg = "scoring.reference_model must be train_heldout_mean_rate"
        raise ValueError(msg)
    if str(config["splits"]["split_seed_mode"]) not in {"fixed", "varied"}:
        msg = "splits.split_seed_mode must be fixed or varied"
        raise ValueError(msg)
    if str(config["splits"]["initialization_seed_mode"]) != "varied":
        msg = "splits.initialization_seed_mode must be varied"
        raise ValueError(msg)
    seeds = [int(seed) for seed in config["seeds"]]
    if len(seeds) < 3:
        msg = "at least three seeds are required"
        raise ValueError(msg)
    if len(set(seeds)) != len(seeds):
        msg = "seeds must be unique"
        raise ValueError(msg)
    names = [str(method["name"]) for method in config["methods"]]
    if len(set(names)) != len(names):
        msg = "method names must be unique"
        raise ValueError(msg)
    for method in config["methods"]:
        if str(method["type"]) not in METHOD_TYPES:
            msg = f"unknown method type: {method['type']}"
            raise ValueError(msg)
        if str(method["type"]) in NEURAL_METHOD_TYPES:
            fallback = dict(method.get("fallback_config", {}))
            if float(fallback.get("diffusion_scale", 0.0)) != 0.0:
                msg = "neural methods must force diffusion_scale == 0.0"
                raise ValueError(msg)
    if int(config["statistics"]["bootstrap_repeats"]) <= 0:
        msg = "statistics.bootstrap_repeats must be positive"
        raise ValueError(msg)
    references = dict(config["references"])
    for key in REQUIRED_REFERENCES:
        if key not in references:
            msg = f"references.{key} is required"
            raise ValueError(msg)


def _write_figures(
    output_dir: Path,
    results: pd.DataFrame,
    method_summary: pd.DataFrame,
    seed_effects: pd.DataFrame,
) -> None:
    import matplotlib  # type: ignore[import-untyped]  # noqa: PLC0415

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]  # noqa: PLC0415

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    metric = "validation_unified_bits_per_spike"

    if not results.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        for method_name, group in results.groupby("method_name", sort=True):
            ordered = group.sort_values("seed")
            ax.plot(ordered["seed"], ordered[metric], marker="o", label=str(method_name))
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Initialization seed")
        ax.set_ylabel("Validation unified bits/spike")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / "validation_bits_by_method_seed.png", dpi=150)
        plt.close(fig)

    if not method_summary.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        names = [str(value) for value in method_summary["method_name"]]
        means = method_summary["mean_validation_unified_bits_per_spike"].to_numpy(dtype=float)
        low = means - method_summary["ci95_low"].to_numpy(dtype=float)
        high = method_summary["ci95_high"].to_numpy(dtype=float) - means
        ax.errorbar(names, means, yerr=[low, high], fmt="o", capsize=4)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("Mean validation unified bits/spike (95% CI)")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(figures / "method_mean_ci.png", dpi=150)
        plt.close(fig)

    if not seed_effects.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        for method_a, group in seed_effects.groupby("method_a", sort=True):
            ordered = group.sort_values("seed")
            ax.plot(ordered["seed"], ordered["difference"], marker="o", label=str(method_a))
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Initialization seed")
        ax.set_ylabel("Difference vs factor_latent")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / "seed_effects.png", dpi=150)
        plt.close(fig)

    neural = results[results["method_type"] != "factor_latent"] if not results.empty else results
    factor = (
        results[results["method_name"] == FACTOR_LATENT_METHOD] if not results.empty else results
    )
    if not neural.empty and not factor.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        factor_ordered = factor.sort_values("seed")
        ax.plot(
            factor_ordered["seed"],
            factor_ordered[metric],
            marker="s",
            label=FACTOR_LATENT_METHOD,
        )
        for method_name, group in neural.groupby("method_name", sort=True):
            ordered = group.sort_values("seed")
            ax.plot(ordered["seed"], ordered[metric], marker="o", label=str(method_name))
        ax.set_xlabel("Initialization seed")
        ax.set_ylabel("Validation unified bits/spike")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / "neural_ode_vs_factor_latent_by_seed.png", dpi=150)
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
    if requires_cuda(config) and not _cuda_diagnostic()["cuda_available"]:
        console.print("CUDA was requested, but torch.cuda.is_available() is False.")
        return 2
    try:
        results, summary = run_seed_robustness(config)
        _validate_reference_zero(
            float(summary.get("train_mean_validation_bits_per_spike", float("nan")))
        )
    except Exception as exc:
        console.print(str(exc))
        return 2
    output_dir = resolve_configured_path(str(config["reporting"]["output_dir"]), repo_root)
    method_summary = method_summary_from_results(results, config)
    leaderboard = build_seed_robustness_leaderboard(method_summary)
    seed_effects = build_seed_effects(results, FACTOR_LATENT_METHOD)
    carried_forward = build_carried_forward_config(config, summary)
    write_seed_robustness_outputs(
        output_dir, summary, results, method_summary, leaderboard, seed_effects, carried_forward
    )
    _write_figures(output_dir, results, method_summary, seed_effects)
    for key in (
        "dataset_name",
        "cuda_device",
        "bin_size_ms",
        "window_seconds",
        "reference_model",
        "methods_evaluated",
        "seeds_evaluated",
        "total_jobs",
        "successful_jobs",
        "best_mean_method",
        "best_mean_validation_unified_bits_per_spike",
        "best_lower_ci_method",
        "best_lower_ci_validation_unified_bits_per_spike",
        "factor_latent_mean_validation_unified_bits_per_spike",
        "best_neural_method",
        "best_neural_method_mean_validation_unified_bits_per_spike",
        "paired_mean_difference_best_neural_minus_factor_latent",
        "any_neural_beats_factor_latent_mean",
        "any_neural_beats_factor_latent_lower_ci",
        "carried_forward_method",
        "output_dir",
    ):
        console.print(f"{key}: {summary.get(key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
