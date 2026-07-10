from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml
from rich.console import Console

from latentbrain.data.io import compute_dataset_hash, load_neural_dataset
from latentbrain.data.rebinning import rebin_neural_dataset, validate_rebin_factor
from latentbrain.data.schemas import NeuralDataset
from latentbrain.data.validation import validate_neural_dataset
from latentbrain.eval.movement_features import resolve_behavior_source
from latentbrain.eval.reporting import (
    write_movement_window_audit_outputs,
    write_window_audit_outputs,
)
from latentbrain.eval.stratified_cv import FACTOR_LATENT, SPLIT_MEAN_RATE_INVALID
from latentbrain.eval.window_audit import (
    BEHAVIOR_ALIGNED_POLICIES,
    CROP_POLICIES,
    build_window_recommendations,
    crop_to_min_impact,
    evaluate_window_candidate,
    evaluate_window_coverage,
    recommend_movement_window,
    reference_peak_speed,
    speed_profiles,
    summarize_window_candidates,
    trial_movement_table,
    window_entropy_table,
    window_method_summary,
)
from latentbrain.paths import get_repo_root, resolve_configured_path

console = Console(markup=False)

CURRENT_WINDOW_NAME = "from_start_1p28s"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local MC_Maze Small movement-window and alignment audit."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/mc_maze_small_window_audit.yaml")
    )
    return parser.parse_args(argv)


def _load_config(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"malformed window audit config: {path}"
        raise ValueError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"window audit config must contain a mapping: {path}"
        raise ValueError(msg)
    _validate_config(raw)
    return raw


def _is_movement_window_config(config: dict[str, Any]) -> bool:
    """Trial-aware behavior-only audits declare a trial_source and score no models."""
    return "trial_source" in config


def _validate_common_config(config: dict[str, Any]) -> None:
    validate_rebin_factor(
        int(config["dataset"]["original_bin_size_ms"]),
        int(config["binning"]["target_bin_size_ms"]),
    )
    candidates = list(config["window_candidates"])
    if not candidates:
        msg = "window_candidates must not be empty"
        raise ValueError(msg)
    names = [str(candidate["name"]) for candidate in candidates]
    if len(set(names)) != len(names):
        msg = "window candidate names must be unique"
        raise ValueError(msg)
    for candidate in candidates:
        if float(candidate["duration_seconds"]) <= 0.0:
            msg = f"window {candidate['name']!r} duration_seconds must be positive"
            raise ValueError(msg)
        if str(candidate["crop_policy"]) not in CROP_POLICIES:
            msg = f"window {candidate['name']!r} crop_policy must be one of {CROP_POLICIES}"
            raise ValueError(msg)
    if int(config["statistics"]["bootstrap_repeats"]) <= 0:
        msg = "statistics.bootstrap_repeats must be positive"
        raise ValueError(msg)


def _validate_movement_config(config: dict[str, Any]) -> None:
    _validate_common_config(config)
    if not list(config["behavior"]["required_channels"]):
        msg = "behavior.required_channels must not be empty"
        raise ValueError(msg)
    trial_source = config["trial_source"]
    if not bool(trial_source["prefer_trial_aware_raw_extraction"]):
        msg = "trial_source.prefer_trial_aware_raw_extraction must be true for this audit"
        raise ValueError(msg)
    if bool(trial_source["allow_global_crop_to_min_fallback"]):
        msg = (
            "trial_source.allow_global_crop_to_min_fallback must be false; the globally cropped "
            "artifact must never silently supply event-centered windows"
        )
        raise ValueError(msg)
    selection = config["selection"]
    if bool(selection["use_model_scores_for_selection"]):
        msg = "selection.use_model_scores_for_selection must be false"
        raise ValueError(msg)
    for key in ("maximum_clipped_trial_fraction", "minimum_moving_bin_fraction"):
        value = float(selection[key])
        if not 0.0 <= value <= 1.0:
            msg = f"selection.{key} must be in [0, 1]"
            raise ValueError(msg)
    quantiles = {
        float(candidate["speed_threshold_quantile"])
        for candidate in config["window_candidates"]
        if "speed_threshold_quantile" in candidate
    }
    if len(quantiles) > 1:
        msg = f"movement-onset candidates must share one speed_threshold_quantile; got {quantiles}"
        raise ValueError(msg)


def _validate_config(config: dict[str, Any]) -> None:
    if _is_movement_window_config(config):
        _validate_movement_config(config)
        return
    _validate_common_config(config)
    if int(config["statistics"]["bootstrap_repeats"]) <= 0:
        msg = "statistics.bootstrap_repeats must be positive"
        raise ValueError(msg)
    if str(config["scoring"]["reference_model"]) != "train_heldout_mean_rate":
        msg = "scoring.reference_model must be train_heldout_mean_rate"
        raise ValueError(msg)
    methods = {str(method["name"]): dict(method) for method in config["methods"]}
    for name, method in methods.items():
        if not bool(method.get("valid_model", False)) and bool(
            method.get("reportable_as_model_performance", False)
        ):
            msg = f"invalid or reference method must not be reportable as performance: {name}"
            raise ValueError(msg)
    if bool(methods.get("train_mean_rate", {}).get("reportable_as_model_performance", False)):
        msg = "train_mean_rate must not be reportable as model performance"
        raise ValueError(msg)
    if not bool(methods.get(FACTOR_LATENT, {}).get("valid_model", False)):
        msg = "factor_latent must be marked a valid model"
        raise ValueError(msg)
    if bool(methods.get(SPLIT_MEAN_RATE_INVALID, {}).get("valid_model", False)):
        msg = "split_mean_rate_invalid must not be marked a valid model"
        raise ValueError(msg)


def _prepare_dataset(config: dict[str, Any]) -> tuple[NeuralDataset, str]:
    repo_root = get_repo_root()
    processed_path = resolve_configured_path(str(config["dataset"]["processed_path"]), repo_root)
    if not processed_path.exists():
        msg = f"Processed dataset is missing: {processed_path}"
        raise FileNotFoundError(msg)
    dataset = load_neural_dataset(processed_path)
    validate_neural_dataset(dataset)
    dataset_hash = compute_dataset_hash(dataset)
    expected = str(config["dataset"].get("expected_hash", ""))
    if expected and dataset_hash != expected:
        msg = f"Dataset hash mismatch: expected {expected}, got {dataset_hash}"
        raise ValueError(msg)
    rebinned = rebin_neural_dataset(dataset, int(config["binning"]["target_bin_size_ms"]))
    return rebinned, dataset_hash


def _require_behavior_for_aligned_windows(config: dict[str, Any], dataset: NeuralDataset) -> None:
    aligned = [
        str(candidate["name"])
        for candidate in config["window_candidates"]
        if str(candidate["crop_policy"]) in BEHAVIOR_ALIGNED_POLICIES
    ]
    if not aligned:
        return
    names = list(dataset.behavior_names) if dataset.behavior_names is not None else None
    if dataset.behavior is None or resolve_behavior_source(names) is None:
        msg = (
            "behavior-aligned windows require hand_pos or cursor_pos behavior data; "
            f"missing for {aligned}"
        )
        raise ValueError(msg)


def _write_figures(
    output_dir: Path,
    summary: dict[str, Any],
    method_summary: pd.DataFrame,
    window_table: pd.DataFrame,
    profiles: dict[str, np.ndarray],
    bin_size_seconds: float,
) -> None:
    import matplotlib  # type: ignore[import-untyped]  # noqa: PLC0415

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]  # noqa: PLC0415

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    factor = method_summary[method_summary["method_name"] == FACTOR_LATENT]
    if not factor.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        means = factor["mean_unified_bits_per_spike"].to_numpy(dtype=float)
        low = means - factor["ci95_low"].to_numpy(dtype=float)
        high = factor["ci95_high"].to_numpy(dtype=float) - means
        ax.errorbar(factor["window_name"], means, yerr=[low, high], fmt="o", capsize=4)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("Factor-latent unified bits/spike (95% CI)")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(figures / "factor_latent_by_window.png", dpi=150)
        plt.close(fig)

    if not window_table.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(window_table["window_name"], window_table["moving_bin_fraction"])
        ax.set_ylabel("Moving bin fraction")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(figures / "behavior_coverage_by_window.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(window_table["window_name"], window_table["endpoint_direction_entropy"])
        ax.axhline(
            float(np.log(8.0)), color="black", linestyle="--", linewidth=0.8, label="maximum"
        )
        ax.set_ylabel("Endpoint direction entropy (nats)")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / "endpoint_direction_entropy_by_window.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    if profiles:
        for name, profile in profiles.items():
            ax.plot(np.arange(profile.size) * bin_size_seconds, profile, label=name)
        ax.set_xlabel("Time within window (s)")
        ax.set_ylabel("Mean hand speed")
        ax.legend(fontsize=6)
    else:
        ax.text(0.5, 0.5, "behavior unavailable", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(figures / "speed_profile_windows.png", dpi=150)
    plt.close(fig)

    invalid = method_summary[method_summary["method_name"] == SPLIT_MEAN_RATE_INVALID]
    if not invalid.empty and not factor.empty:
        merged = invalid.merge(factor, on="window_name", suffixes=("_invalid", "_factor"))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(
            merged["window_name"],
            merged["mean_unified_bits_per_spike_invalid"]
            - merged["mean_unified_bits_per_spike_factor"],
            color="#E45756",
        )
        ax.set_ylabel("Invalid split-mean advantage over factor-latent")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(figures / "invalid_control_gap_by_window.png", dpi=150)
        plt.close(fig)


def run_window_audit(config: dict[str, Any]) -> dict[str, Any]:
    dataset, dataset_hash = _prepare_dataset(config)
    _require_behavior_for_aligned_windows(config, dataset)
    bin_size_seconds = float(config["binning"]["target_bin_size_ms"]) / 1000.0

    score_frames: list[pd.DataFrame] = []
    behavior_frames: list[pd.DataFrame] = []
    balance_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for candidate in config["window_candidates"]:
        scores, behavior, balance, diagnostic = evaluate_window_candidate(
            config, dict(candidate), dataset
        )
        score_frames.append(scores)
        if not behavior.empty:
            behavior_frames.append(behavior)
        balance_frames.append(balance)
        diagnostics.append(diagnostic)

    scores = pd.concat(score_frames, ignore_index=True)
    behavior_statistics = (
        pd.concat(behavior_frames, ignore_index=True) if behavior_frames else pd.DataFrame()
    )
    balance_statistics = pd.concat(balance_frames, ignore_index=True)
    method_summary = window_method_summary(scores, config)
    window_table = window_entropy_table(diagnostics)

    summary = summarize_window_candidates(
        scores,
        behavior_statistics,
        balance_statistics,
        dict(config["references"]),
        diagnostics,
        method_summary,
        CURRENT_WINDOW_NAME,
    )
    summary.update(
        {
            "dataset_name": config["dataset"]["name"],
            "dataset_hash": dataset_hash,
            "bin_size_ms": int(config["binning"]["target_bin_size_ms"]),
            "reference_model": str(config["scoring"]["reference_model"]),
            "fold_count": int(config["cross_validation"]["fold_count"]),
            "repeats": int(config["cross_validation"]["repeats"]),
            "candidate_windows": [str(row["window_name"]) for row in diagnostics],
            "behavior_source": str(diagnostics[0].get("behavior_source", "unavailable")),
        }
    )
    recommendations = build_window_recommendations(summary)

    output_dir = resolve_configured_path(str(config["reporting"]["output_dir"]), get_repo_root())
    write_window_audit_outputs(
        output_dir,
        summary,
        scores,
        behavior_statistics,
        balance_statistics,
        window_table,
        method_summary,
        recommendations,
    )
    _write_figures(
        output_dir,
        summary,
        method_summary,
        window_table,
        speed_profiles(dataset, [dict(c) for c in config["window_candidates"]], bin_size_seconds),
        bin_size_seconds,
    )
    summary["output_dir"] = str(output_dir)
    return summary


def _trial_sequences(config: dict[str, Any], global_time_bins: int) -> Any:
    """Rebuild the ragged trials from the raw assets, using the ingestion config in provenance."""
    from latentbrain.data.nlb import NLBConfig, load_trial_sequences  # noqa: PLC0415
    from latentbrain.data.validation import validate_trial_sequences  # noqa: PLC0415

    repo_root = get_repo_root()
    raw_dir = resolve_configured_path(str(config["dataset"]["raw_dir"]), repo_root)
    if not raw_dir.exists():
        msg = (
            f"Raw dataset directory is missing: {raw_dir}. Trial-aware extraction requires the "
            "verified raw assets; the globally cropped artifact must not be used instead."
        )
        raise FileNotFoundError(msg)
    provenance_path = resolve_configured_path(str(config["dataset"]["provenance_path"]), repo_root)
    if not provenance_path.exists():
        msg = f"Provenance file is missing: {provenance_path}"
        raise FileNotFoundError(msg)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    nlb_config = NLBConfig.model_validate(provenance["config"])
    sequences = load_trial_sequences(raw_dir, nlb_config)
    validate_trial_sequences(sequences)
    if int(sequences.trial_lengths.min()) != int(global_time_bins):
        msg = (
            f"trial-aware minimum length {int(sequences.trial_lengths.min())} does not match the "
            f"processed artifact's global crop of {global_time_bins} bins"
        )
        raise ValueError(msg)
    return sequences


def _write_movement_figures(
    output_dir: Path,
    movement: pd.DataFrame,
    candidate_statistics: pd.DataFrame,
    profiles: dict[str, np.ndarray],
    target_bin_size_ms: int,
) -> None:
    import matplotlib  # type: ignore[import-untyped]  # noqa: PLC0415

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]  # noqa: PLC0415

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    bin_size_seconds = float(target_bin_size_ms) / 1000.0

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, profile in profiles.items():
        ax.plot(np.arange(profile.size) * bin_size_seconds, profile, label=name)
    ax.set_xlabel("Time within window (s)")
    ax.set_ylabel("Mean speed")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(figures / "movement_speed_profile.png", dpi=150)
    plt.close(fig)

    for column, filename, label in (
        ("peak_speed_time_seconds", "peak_speed_time_distribution.png", "Peak speed time (s)"),
        (
            "movement_onset_time_seconds",
            "movement_onset_time_distribution.png",
            "Movement onset time (s)",
        ),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(movement[column].to_numpy(dtype=float), bins=40)
        ax.set_xlabel(label)
        ax.set_ylabel("Trials")
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=150)
        plt.close(fig)

    for column, filename, label in (
        ("moving_bin_fraction_mean", "movement_coverage_by_window.png", "Moving bin fraction"),
        ("clipped_trial_fraction", "clipping_fraction_by_window.png", "Clipped trial fraction"),
        (
            "endpoint_direction_entropy",
            "endpoint_direction_entropy_by_window.png",
            "Endpoint direction entropy (nats)",
        ),
    ):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(candidate_statistics["window_name"], candidate_statistics[column])
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=150)
        plt.close(fig)


def run_movement_window_audit(config: dict[str, Any]) -> dict[str, Any]:
    """Trial-aware, behavior-only movement-window audit. No model is trained or scored."""
    repo_root = get_repo_root()
    processed_path = resolve_configured_path(str(config["dataset"]["processed_path"]), repo_root)
    if not processed_path.exists():
        msg = f"Processed dataset is missing: {processed_path}"
        raise FileNotFoundError(msg)
    dataset = load_neural_dataset(processed_path)
    validate_neural_dataset(dataset)
    dataset_hash = compute_dataset_hash(dataset)
    expected = str(config["dataset"].get("expected_hash", ""))
    if expected and dataset_hash != expected:
        msg = f"Dataset hash mismatch: expected {expected}, got {dataset_hash}"
        raise ValueError(msg)
    names = list(dataset.behavior_names or [])
    missing = [name for name in config["behavior"]["required_channels"] if name not in names]
    if missing:
        msg = f"processed dataset is missing required behavior channels: {missing}"
        raise ValueError(msg)

    global_time_bins = int(dataset.spikes.shape[1])
    sequences = _trial_sequences(config, global_time_bins)

    quantiles = [
        float(candidate["speed_threshold_quantile"])
        for candidate in config["window_candidates"]
        if "speed_threshold_quantile" in candidate
    ]
    movement = trial_movement_table(sequences, quantiles[0] if quantiles else 0.7)
    crop_impact, crop_summary = crop_to_min_impact(sequences, movement, global_time_bins)

    target_bin_size_ms = int(config["binning"]["target_bin_size_ms"])
    reference_peak = reference_peak_speed(sequences, target_bin_size_ms)
    summaries: list[dict[str, Any]] = []
    coverage_frames: list[pd.DataFrame] = []
    profiles: dict[str, np.ndarray] = {}
    for candidate in config["window_candidates"]:
        coverage, candidate_summary, extras = evaluate_window_coverage(
            sequences, movement, dict(candidate), target_bin_size_ms, reference_peak
        )
        coverage_frames.append(coverage)
        summaries.append(candidate_summary)
        profiles[str(candidate["name"])] = extras["speed_profile"]

    recommendations = recommend_movement_window(
        summaries, dict(config["selection"]), dict(config["references"])
    )
    candidate_statistics = pd.DataFrame(recommendations["candidate_summaries"])
    trial_coverage = pd.concat(coverage_frames, ignore_index=True)

    warnings: list[str] = []
    if not crop_summary["global_crop_suitable_for_movement_window_audit"]:
        warnings.append(
            "global crop_to_min excludes peak speed or movement onset for at least one trial; "
            "event-centered windows were taken from the trial-aware representation"
        )
    if not recommendations["small_window_transfers"]:
        warnings.append("the MC_Maze Small recommended window does not transfer to Large")
    for name, reason in recommendations["rejected_windows"].items():
        warnings.append(f"{name} rejected: {reason}")

    summary: dict[str, Any] = {
        "dataset_name": str(config["dataset"]["name"]),
        "dataset_hash": dataset_hash,
        "trial_source_file": sequences.metadata.get("source_file"),
        "trial_representation": "ragged list[np.ndarray] per trial (no global crop)",
        "trial_count": int(len(sequences.spikes)),
        "neuron_count": int(sequences.spikes[0].shape[1]),
        "trial_length_min": int(sequences.trial_lengths.min()),
        "trial_length_max": int(sequences.trial_lengths.max()),
        "trial_aware_spikes_conserved": bool(sequences.metadata["spikes_conserved"]),
        "source_bin_size_ms": int(sequences.bin_size_ms),
        "target_bin_size_ms": target_bin_size_ms,
        "global_crop_time_bins": global_time_bins,
        "behavior_source": str(movement.iloc[0]["behavior_source"]),
        "behavior_mapping": sequences.metadata.get("behavior_mapping", {}),
        "median_peak_speed_time_seconds": float(movement["peak_speed_time_seconds"].median()),
        "median_movement_onset_time_seconds": float(
            movement["movement_onset_time_seconds"].median()
        ),
        "reference_peak_speed": reference_peak,
        "candidate_windows": [str(row["window_name"]) for row in summaries],
        "moving_bin_fraction_by_window": {
            row["window_name"]: row["moving_bin_fraction_mean"] for row in summaries
        },
        "endpoint_direction_entropy_by_window": {
            row["window_name"]: row["endpoint_direction_entropy"] for row in summaries
        },
        "clipped_trial_fraction_by_window": {
            row["window_name"]: row["clipped_trial_fraction"] for row in summaries
        },
        "small_moving_bin_fraction": float(config["references"]["small_moving_bin_fraction"]),
        "small_endpoint_direction_entropy": float(
            config["references"]["small_endpoint_direction_entropy"]
        ),
        "global_crop_used_for_event_centered_windows": False,
        "models_trained": False,
        "models_scored": False,
        "cross_validation_run": False,
        "official_benchmark_claim": False,
        "warnings": warnings,
        **crop_summary,
        **{key: value for key, value in recommendations.items() if key != "candidate_summaries"},
    }

    output_dir = resolve_configured_path(str(config["reporting"]["output_dir"]), repo_root)
    write_movement_window_audit_outputs(
        output_dir,
        summary,
        candidate_statistics,
        movement,
        trial_coverage,
        crop_impact,
        recommendations,
    )
    _write_movement_figures(
        output_dir, movement, candidate_statistics, profiles, target_bin_size_ms
    )
    summary["output_dir"] = str(output_dir)
    return summary


MOVEMENT_SUMMARY_KEYS = (
    "dataset_name",
    "dataset_hash",
    "trial_count",
    "neuron_count",
    "source_bin_size_ms",
    "target_bin_size_ms",
    "raw_spike_count",
    "global_crop_retained_spike_count",
    "fraction_raw_spikes_excluded",
    "fraction_trials_peak_inside_global_crop",
    "fraction_trials_onset_inside_global_crop",
    "candidate_windows",
    "recommended_window_name",
    "recommended_duration_seconds",
    "recommended_clipped_trial_fraction",
    "recommended_moving_bin_fraction",
    "recommended_peak_speed_coverage",
    "recommended_movement_onset_coverage",
    "recommended_endpoint_direction_entropy",
    "small_window_transfers",
    "global_crop_suitable_for_movement_window_audit",
    "output_dir",
    "warnings",
)


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = get_repo_root()
    args = _parse_args(argv)
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    if not config_path.exists():
        console.print(f"Config file is missing: {config_path}")
        return 2
    try:
        config = _load_config(config_path)
        if _is_movement_window_config(config):
            summary = run_movement_window_audit(config)
            for key in MOVEMENT_SUMMARY_KEYS:
                console.print(f"{key}: {summary.get(key)}")
            return 0
        summary = run_window_audit(config)
    except (OSError, ValueError, FileNotFoundError, ImportError) as exc:
        console.print(f"Window audit failed: {exc}")
        return 2
    for key in (
        "dataset_name",
        "bin_size_ms",
        "candidate_windows",
        "fold_count",
        "repeats",
        "factor_latent_current_window_mean",
        "recommended_window_name",
        "factor_latent_best_window_mean",
        "factor_latent_best_window_ci95_low",
        "factor_latent_best_window_ci95_high",
        "split_mean_invalid_best_window_mean",
        "invalid_control_gap_best_window",
        "endpoint_direction_entropy_current_window",
        "endpoint_direction_entropy_best_window",
        "behavior_coverage_warning",
        "current_window_still_supported",
        "recommended_reporting_mode",
        "invalid_controls_excluded_from_valid_model_selection",
        "output_dir",
    ):
        console.print(f"{key}: {summary.get(key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
