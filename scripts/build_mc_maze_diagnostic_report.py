from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

from latentbrain.data.io import compute_dataset_hash, load_neural_dataset
from latentbrain.paths import get_repo_root, resolve_configured_path
from latentbrain.reporting.mc_maze_diagnostic import write_mc_maze_diagnostic_bundle
from latentbrain.reporting.report_tables import build_method_registry

console = Console(markup=False)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the local MC_Maze Small diagnostic report bundle."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/mc_maze_small_diagnostic_report.yaml")
    )
    return parser.parse_args(argv)


def _load_config(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"malformed diagnostic report config: {path}"
        raise ValueError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"diagnostic report config must contain a mapping: {path}"
        raise ValueError(msg)
    _validate_config(raw)
    return raw


def _validate_config(config: dict[str, Any]) -> None:
    analysis = dict(config["analysis"])
    if str(analysis["canonical_metric"]) != "unified_bits_per_spike":
        msg = "analysis.canonical_metric must be unified_bits_per_spike"
        raise ValueError(msg)
    if bool(analysis.get("official_leaderboard_claim", False)):
        msg = "analysis.official_leaderboard_claim must be false"
        raise ValueError(msg)
    accepted = dict(config["accepted_findings"])
    expected_mode = "recommended_window_stratified_cross_validation"
    if str(accepted["recommended_reporting_mode"]) != expected_mode:
        msg = f"accepted_findings.recommended_reporting_mode must be {expected_mode}"
        raise ValueError(msg)
    expected_window = "behavior_speed_peak_centered_1p28s"
    if str(accepted.get("carried_forward_window", "")) != expected_window:
        msg = f"accepted_findings.carried_forward_window must be {expected_window}"
        raise ValueError(msg)
    if bool(accepted.get("single_split_results_reportable", False)):
        msg = "accepted_findings.single_split_results_reportable must be false"
        raise ValueError(msg)
    carried = str(accepted["carried_forward_valid_method"])
    registry = build_method_registry()
    rows = registry[registry["method_name"] == carried]
    if rows.empty:
        msg = f"carried-forward method is unknown: {carried}"
        raise ValueError(msg)
    if not bool(rows.iloc[0]["valid_model"]):
        msg = f"carried-forward method must be a valid model: {carried}"
        raise ValueError(msg)
    invalid = registry[~registry["valid_model"].astype(bool)]
    if invalid["reportable_as_model_performance"].astype(bool).any():
        msg = "invalid controls must never be reportable as model performance"
        raise ValueError(msg)


def _verify_dataset_hash(config: dict[str, Any]) -> None:
    processed_path = resolve_configured_path(
        str(config["dataset"]["processed_path"]), get_repo_root()
    )
    expected = str(config["dataset"].get("expected_hash", ""))
    if not processed_path.exists() or not expected:
        return
    dataset_hash = compute_dataset_hash(load_neural_dataset(processed_path))
    if dataset_hash != expected:
        msg = f"Dataset hash mismatch: expected {expected}, got {dataset_hash}"
        raise ValueError(msg)


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = get_repo_root()
    args = _parse_args(argv)
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    if not config_path.exists():
        console.print(f"Config file is missing: {config_path}")
        return 2
    try:
        config = _load_config(config_path)
        _verify_dataset_hash(config)
        summary = write_mc_maze_diagnostic_bundle(config)
    except (OSError, ValueError, FileNotFoundError) as exc:
        console.print(f"Diagnostic report build failed: {exc}")
        return 2
    for key in (
        "output_dir",
        "carried_forward_method",
        "carried_forward_window",
        "recommended_reporting_mode",
        "recommended_window_cv_available",
        "factor_latent_recommended_window_mean",
        "factor_latent_recommended_window_ci95_low",
        "factor_latent_recommended_window_ci95_high",
        "split_mean_invalid_recommended_window_mean",
        "factor_latent_minus_split_mean_invalid",
        "leakage_dominance_persists_on_recommended_window",
        "single_split_results_reportable",
        "invalid_rate_controls_present",
        "invalid_controls_excluded_from_model_performance",
        "neural_ode_near_win_seed_specific",
        "split_instability_disclosed",
        "split_mean_advantage_is_target_leakage",
        "official_leaderboard_claim",
        "claim_safety_checklist_passed",
        "missing_inputs",
    ):
        console.print(f"{key}: {summary.get(key)}")
    return 0 if summary.get("claim_safety_checklist_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
