from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rich.console import Console

from latentbrain.data.io import compute_dataset_hash, load_neural_dataset
from latentbrain.data.rebinning import rebin_neural_dataset, validate_rebin_factor
from latentbrain.data.schemas import NeuralDataset, TrialSplit
from latentbrain.data.splits import create_neuron_mask, create_trial_split
from latentbrain.data.validation import (
    validate_neural_dataset,
    validate_neuron_mask,
    validate_trial_split,
)
from latentbrain.eval.rebinning import compute_window_bins_for_duration
from latentbrain.eval.reporting import (
    write_dataset_scoreboard_outputs,
    write_unified_scoreboard_outputs,
)
from latentbrain.eval.scoring import (
    ScoringConfig,
    score_heldout_prediction,
    train_heldout_mean_rate_reference,
)
from latentbrain.eval.unified_scoreboard import (
    build_historical_metric_notes,
    build_unified_score_row,
    load_baseline_suite_scoreboard,
    load_cv_rate_audit_warning,
    load_dataset_cv_scoreboard,
    load_lfads_diagnostics_scoreboard,
    load_lfads_family_candidates,
    load_lfads_pilot_scoreboard,
    load_recommended_window_cv_warning,
    load_seed_robustness_candidates,
    load_split_audit_warning,
    load_stratified_cv_warning,
    load_window_audit_warning,
    rank_unified_validation_scores,
    summarize_unified_scoreboard,
)
from latentbrain.eval.windowing import crop_neural_dataset_time
from latentbrain.paths import get_repo_root, resolve_configured_path

console = Console(markup=False)
SPLITS = {"train", "validation", "test"}
REFERENCE_MODELS = {"train_heldout_mean_rate"}
REQUIRED_KNOWN_VALUES = {
    "train_mean_as_model_validation_bits_per_spike",
    "split_mean_validation_bits_per_spike",
    "factor_latent_unified_validation_bits_per_spike",
    "lfads_unified_validation_bits_per_spike",
    "coordinated_dropout_unified_validation_bits_per_spike",
    "best_oracle_validation_bits_per_spike",
}
REQUIRED_HISTORICAL_VALUES = {
    "old_window_matched_mean_rate_validation_bits_per_spike",
    "old_full_window_mean_rate_validation_bits_per_spike",
}


class DatasetSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    processed_path: str = Field(min_length=1)
    expected_hash: str | None = None
    original_bin_size_ms: int = Field(gt=0)


class SplitSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = Field(ge=0)
    train_fraction: float = Field(gt=0.0, lt=1.0)
    validation_fraction: float = Field(gt=0.0, lt=1.0)
    test_fraction: float = Field(gt=0.0, lt=1.0)
    heldout_neuron_fraction: float = Field(gt=0.0, lt=1.0)


class WindowSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration_seconds: float = Field(gt=0.0)
    crop_policy: str = Field(min_length=1)


class BinningSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_bin_size_ms: int = Field(gt=0)


class ScoringSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_model: str = Field(min_length=1)
    include_poisson_constant: bool
    min_rate_hz: float = Field(gt=0.0)
    max_rate_hz: float = Field(gt=0.0)
    primary_split: str = Field(min_length=1)
    primary_metric: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_scoring(self) -> ScoringSection:
        if self.reference_model not in REFERENCE_MODELS:
            msg = "scoring.reference_model must be train_heldout_mean_rate"
            raise ValueError(msg)
        if self.primary_split not in SPLITS:
            msg = "scoring.primary_split must be a recognized split name"
            raise ValueError(msg)
        if self.primary_metric != "bits_per_spike":
            msg = "scoring.primary_metric must be bits_per_spike"
            raise ValueError(msg)
        if self.max_rate_hz <= self.min_rate_hz:
            msg = "scoring.max_rate_hz must exceed scoring.min_rate_hz"
            raise ValueError(msg)
        return self


class InputsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_audit_dir: str = Field(min_length=1)
    temporal_rebinning_dir: str = Field(min_length=1)
    coordinated_dropout_dir: str = Field(min_length=1)
    rate_calibration_dir: str = Field(min_length=1)
    lfads_unified_tuning_summary_path: str | None = None
    lfads_controller_tuning_summary_path: str | None = None
    neural_sde_tuning_summary_path: str | None = None
    neural_ode_tuning_summary_path: str | None = None
    neural_ode_refinement_summary_path: str | None = None
    neural_ode_objective_summary_path: str | None = None
    switching_ode_tuning_summary_path: str | None = None
    seed_robustness_summary_path: str | None = None
    split_audit_summary_path: str | None = None
    cv_rate_audit_summary_path: str | None = None
    stratified_cv_summary_path: str | None = None
    window_audit_summary_path: str | None = None
    recommended_window_cv_summary_path: str | None = None


class KnownUnifiedValues(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_mean_as_model_validation_bits_per_spike: float
    split_mean_validation_bits_per_spike: float
    factor_latent_unified_validation_bits_per_spike: float
    lfads_unified_validation_bits_per_spike: float
    coordinated_dropout_unified_validation_bits_per_spike: float
    best_oracle_validation_bits_per_spike: float


class HistoricalIncompatibleValues(BaseModel):
    model_config = ConfigDict(extra="forbid")

    old_window_matched_mean_rate_validation_bits_per_spike: float
    old_full_window_mean_rate_validation_bits_per_spike: float


class ReportingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str = Field(min_length=1)


class UnifiedScoreboardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: DatasetSection
    splits: SplitSection
    window: WindowSection
    binning: BinningSection
    scoring: ScoringSection
    inputs: InputsSection
    known_unified_values: KnownUnifiedValues
    historical_incompatible_values: HistoricalIncompatibleValues
    reporting: ReportingSection

    @model_validator(mode="after")
    def validate_contract(self) -> UnifiedScoreboardConfig:
        total = (
            self.splits.train_fraction + self.splits.validation_fraction + self.splits.test_fraction
        )
        if abs(total - 1.0) > 1e-8:
            msg = "split fractions must sum to 1.0"
            raise ValueError(msg)
        validate_rebin_factor(self.dataset.original_bin_size_ms, self.binning.target_bin_size_ms)
        compute_window_bins_for_duration(
            self.window.duration_seconds, self.binning.target_bin_size_ms
        )
        known = set(self.known_unified_values.model_dump())
        if not REQUIRED_KNOWN_VALUES.issubset(known):
            msg = "known_unified_values is missing required unified values"
            raise ValueError(msg)
        historical = set(self.historical_incompatible_values.model_dump())
        if not REQUIRED_HISTORICAL_VALUES.issubset(historical):
            msg = "historical_incompatible_values is missing required historical-only values"
            raise ValueError(msg)
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> UnifiedScoreboardConfig:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            msg = f"malformed unified scoreboard config: {path}"
            raise ValueError(msg) from exc
        if not isinstance(raw, dict):
            msg = f"unified scoreboard config must contain a mapping: {path}"
            raise ValueError(msg)
        return cls.model_validate(raw)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local unified scoring scoreboard.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mc_maze_small_unified_scoreboard.yaml"),
    )
    return parser.parse_args(argv)


def _raw_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        msg = f"scoreboard config must contain a mapping: {path}"
        raise ValueError(msg)
    return dict(loaded)


def _is_dataset_summary_config(config: dict[str, Any]) -> bool:
    """Summary-only configs carry CV evidence, not a model-scoring pipeline."""
    return "known_unified_values" not in config


def _load_config(path: Path) -> dict[str, Any]:
    return UnifiedScoreboardConfig.from_yaml(path).model_dump()


def run_dataset_scoreboard(config: dict[str, Any]) -> dict[str, Any]:
    summary = {
        **load_dataset_cv_scoreboard(config),
        **load_baseline_suite_scoreboard(config),
    }
    if str(config["dataset"]["name"]) == "mc_maze_large":
        summary.update(load_lfads_pilot_scoreboard(config))
        summary.update(load_lfads_diagnostics_scoreboard(config))
    output_dir = resolve_configured_path(str(config["reporting"]["output_dir"]), get_repo_root())
    write_dataset_scoreboard_outputs(output_dir, summary)
    return {"summary": summary, "output_dir": output_dir}


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


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
    return windowed, dataset_hash, window_bins


def _split_ids(split: TrialSplit, name: str) -> np.ndarray:
    if name == "train":
        return split.train
    if name == "validation":
        return split.validation
    if name == "test":
        return split.test
    msg = f"unknown split: {name}"
    raise ValueError(msg)


def _trial_mask(dataset: NeuralDataset, trial_ids: np.ndarray) -> np.ndarray:
    return np.isin(dataset.trial_ids, trial_ids)


def _make_split_and_mask(
    dataset: NeuralDataset, config: dict[str, Any]
) -> tuple[TrialSplit, np.ndarray]:
    split_cfg = config["splits"]
    split = create_trial_split(
        dataset.trial_ids,
        float(split_cfg["train_fraction"]),
        float(split_cfg["validation_fraction"]),
        float(split_cfg["test_fraction"]),
        int(split_cfg["seed"]),
    )
    mask = create_neuron_mask(
        dataset.spikes.shape[2],
        float(split_cfg["heldout_neuron_fraction"]),
        int(split_cfg["seed"]),
    )
    validate_trial_split(split, dataset.trial_ids)
    validate_neuron_mask(mask, dataset.spikes.shape[2])
    return split, mask.heldout


def _build_scoring_config(config: dict[str, Any]) -> ScoringConfig:
    scoring = config["scoring"]
    return ScoringConfig(
        bin_size_ms=int(config["binning"]["target_bin_size_ms"]),
        include_poisson_constant=bool(scoring["include_poisson_constant"]),
        min_rate_hz=float(scoring["min_rate_hz"]),
        max_rate_hz=float(scoring["max_rate_hz"]),
        reference_name=str(scoring["reference_model"]),
    )


def _known_rows(config: dict[str, Any], reference_name: str) -> list[dict[str, Any]]:
    values = config["known_unified_values"]
    return [
        build_unified_score_row(
            "split_mean_diagnostic",
            "split_heldout_mean_rate",
            "validation",
            float(values["split_mean_validation_bits_per_spike"]),
            None,
            False,
            reference_name,
            "Diagnostic control fit on evaluation split; invalid model.",
        ),
        build_unified_score_row(
            "factor_latent",
            "reported_unified_factor_decoder",
            "validation",
            float(values["factor_latent_unified_validation_bits_per_spike"]),
            None,
            True,
            reference_name,
            "Known unified local value from metric audit.",
        ),
        build_unified_score_row(
            "oracle_smoothed_heldout",
            "oracle_control",
            "validation",
            float(values["best_oracle_validation_bits_per_spike"]),
            None,
            False,
            reference_name,
            "Oracle diagnostic uses held-out targets directly; invalid model.",
        ),
    ]


def _write_figures(output_dir: Path, leaderboard: pd.DataFrame, summary: dict[str, Any]) -> None:
    import matplotlib  # type: ignore[import-untyped]  # noqa: PLC0415

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]  # noqa: PLC0415

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(value) for value in leaderboard["method_name"].tolist()]
    bits = [float(value) for value in leaderboard["validation_bits_per_spike"].tolist()]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, bits, color="#4C78A8")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("validation bits/spike")
    ax.set_title("Unified validation leaderboard")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figure_dir / "unified_validation_leaderboard.png", dpi=150)
    plt.close(fig)

    factor = float(summary["factor_latent_validation_bits_per_spike"])
    lfads = float(summary["best_lfads_family_validation_bits_per_spike"])
    oracle = float(summary["oracle_validation_bits_per_spike"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["factor_latent", "best_lfads_family", "oracle_invalid"], [factor, lfads, oracle])
    ax.set_ylabel("validation bits/spike")
    ax.set_title("Oracle diagnostic gap")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figure_dir / "oracle_gap.png", dpi=150)
    plt.close(fig)


def run_unified_scoreboard(config: dict[str, Any]) -> dict[str, Any]:
    UnifiedScoreboardConfig.model_validate(config)
    dataset, dataset_hash, window_bins = _prepare_dataset(config)
    split, heldout_mask = _make_split_and_mask(dataset, config)
    scoring_config = _build_scoring_config(config)
    train_counts = dataset.spikes[_trial_mask(dataset, split.train)][:, :, heldout_mask]
    rows: list[dict[str, Any]] = []
    for split_name in ("train", "validation", "test"):
        counts = dataset.spikes[_trial_mask(dataset, _split_ids(split, split_name))][
            :, :, heldout_mask
        ]
        reference = train_heldout_mean_rate_reference(train_counts, counts.shape, scoring_config)
        scored = score_heldout_prediction(
            counts,
            reference,
            reference,
            scoring_config,
            "train_heldout_mean_rate",
            split_name,
            "train_mean_reference_as_model",
            True,
            "Canonical reference scored as a model; should be 0.0 bits/spike.",
        )
        rows.append(
            build_unified_score_row(
                scored["method_name"],
                scored["prediction_source"],
                scored["split"],
                float(scored["bits_per_spike"]),
                float(scored["poisson_nll"]),
                bool(scored["valid_model"]),
                scored["reference_name"],
                scored["notes"],
            )
        )
    primary_split = str(config["scoring"]["primary_split"])
    train_mean = next(
        row
        for row in rows
        if row["method_name"] == "train_heldout_mean_rate" and row["split"] == primary_split
    )
    if abs(float(train_mean["bits_per_spike"])) > 1e-12:
        msg = "train-heldout mean-rate reference did not score 0.0 bits/spike against itself"
        raise RuntimeError(msg)
    known_train = float(
        config["known_unified_values"]["train_mean_as_model_validation_bits_per_spike"]
    )
    if abs(float(train_mean["bits_per_spike"]) - known_train) > 1e-12:
        msg = "known train-mean unified value does not match canonical scoring"
        raise RuntimeError(msg)
    rows.extend(_known_rows(config, scoring_config.reference_name))
    rows.extend(load_lfads_family_candidates(config))
    rows.extend(load_seed_robustness_candidates(config))
    split_scores = pd.DataFrame(rows)
    leaderboard = rank_unified_validation_scores(split_scores, primary_split)
    historical = build_historical_metric_notes(config["historical_incompatible_values"])
    summary = {
        "dataset_name": config["dataset"]["name"],
        "dataset_hash": dataset_hash,
        "bin_size_ms": scoring_config.bin_size_ms,
        "window_seconds": float(config["window"]["duration_seconds"]),
        "window_bins": window_bins,
        "reference_model": scoring_config.reference_name,
        "primary_split": primary_split,
        **summarize_unified_scoreboard(leaderboard, config["known_unified_values"]),
        **load_split_audit_warning(config),
        **load_cv_rate_audit_warning(config),
        **load_stratified_cv_warning(config),
        **load_window_audit_warning(config),
        **load_recommended_window_cv_warning(config),
        "old_mean_rate_values_historical_only": True,
        "output_dir": config["reporting"]["output_dir"],
    }
    output_dir = resolve_configured_path(str(config["reporting"]["output_dir"]), get_repo_root())
    write_unified_scoreboard_outputs(output_dir, summary, leaderboard, split_scores, historical)
    _write_figures(output_dir, leaderboard, summary)
    return {"summary": summary, "output_dir": output_dir}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    raw = _raw_config(args.config)
    if _is_dataset_summary_config(raw):
        result = run_dataset_scoreboard(raw)
        summary = result["summary"]
        for key in (
            "dataset_name",
            "recommended_window_cv_available",
            "recommended_window_name",
            "recommended_reporting_mode",
            "factor_latent_recommended_window_mean",
            "factor_latent_recommended_window_ci95_low",
            "factor_latent_recommended_window_ci95_high",
            "factor_latent_positive_fraction",
            "factor_latent_beats_invalid_control_mean",
            "leakage_dominance_persists",
            "best_valid_method",
            "baseline_suite_available",
            "baseline_to_beat",
            "baseline_to_beat_mean",
            "baseline_to_beat_ci95_low",
            "baseline_to_beat_ci95_high",
            "baseline_replaced",
            "baseline_replacement_supported",
            "neural_reevaluation_ready",
            "invalid_controls_excluded",
            "lfads_pilot_available",
            "lfads_pilot_complete",
            "lfads_pilot_mean",
            "lfads_pilot_seed_std",
            "lfads_pilot_positive_seed_fraction",
            "lfads_pilot_mean_difference_vs_baseline",
            "lfads_full_evaluation_recommended",
            "lfads_pilot_final_claim_allowed",
            "lfads_diagnostics_available",
            "lfads_integrity_checks_passed",
            "lfads_dominant_failure_mode",
            "lfads_estimated_recoverable_gap",
            "lfads_recommended_next_action",
            "lfads_full_evaluation_allowed",
            "single_split_results_reportable",
            "official_leaderboard_claim",
        ):
            if key in summary:
                console.print(f"{key}: {summary[key]}")
        console.print(f"output_dir: {result['output_dir']}")
        return 0
    config = _load_config(args.config)
    result = run_unified_scoreboard(config)
    summary = result["summary"]
    console.print(f"dataset_name: {summary['dataset_name']}")
    console.print(f"bin_size_ms: {summary['bin_size_ms']}")
    console.print(f"window_seconds: {summary['window_seconds']}")
    console.print(f"reference_model: {summary['reference_model']}")
    console.print(
        f"train_mean_validation_bits_per_spike: {summary['train_mean_validation_bits_per_spike']}"
    )
    console.print(f"best_valid_model: {summary['best_valid_model']}")
    console.print(
        "best_valid_model_validation_bits_per_spike: "
        f"{summary['best_valid_model_validation_bits_per_spike']}"
    )
    console.print(
        "factor_latent_validation_bits_per_spike: "
        f"{summary['factor_latent_validation_bits_per_spike']}"
    )
    console.print(f"best_lfads_family_method: {summary['best_lfads_family_method']}")
    console.print(
        "best_lfads_family_validation_bits_per_spike: "
        f"{summary['best_lfads_family_validation_bits_per_spike']}"
    )
    console.print(
        f"lfads_family_beats_factor_latent: {summary['lfads_family_beats_factor_latent']}"
    )
    console.print(
        f"oracle_validation_bits_per_spike: {summary['oracle_validation_bits_per_spike']}"
    )
    console.print(
        f"old_mean_rate_values_historical_only: {summary['old_mean_rate_values_historical_only']}"
    )
    console.print(f"generalization_risk: {summary['generalization_risk']}")
    console.print(f"single_split_results_reportable: {summary['single_split_results_reportable']}")
    console.print(f"recommended_reporting_mode: {summary['recommended_reporting_mode']}")
    console.print(f"stratified_cv_available: {summary['stratified_cv_available']}")
    console.print(f"window_audit_available: {summary['window_audit_available']}")
    console.print(f"recommended_window_cv_available: {summary['recommended_window_cv_available']}")
    console.print(f"recommended_window_name: {summary['recommended_window_name']}")
    console.print(f"current_window_still_supported: {summary['current_window_still_supported']}")
    console.print(
        f"factor_latent_stratified_cv_mean: {summary['factor_latent_stratified_cv_mean']}"
    )
    console.print(f"invalid_rate_controls_present: {summary['invalid_rate_controls_present']}")
    console.print(
        f"validation_test_instability_detected: {summary['validation_test_instability_detected']}"
    )
    console.print(f"output_dir: {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
