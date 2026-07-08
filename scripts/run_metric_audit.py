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
from latentbrain.data.schemas import NeuralDataset, NeuronMask, TrialSplit
from latentbrain.data.splits import create_neuron_mask, create_trial_split
from latentbrain.data.validation import (
    validate_neural_dataset,
    validate_neuron_mask,
    validate_trial_split,
)
from latentbrain.eval.metric_audit import (
    compute_train_heldout_mean_rates,
    score_prediction_against_reference,
)
from latentbrain.eval.oracle_controls import (
    make_oracle_smoothed_heldout_prediction,
    make_random_rate_prediction,
    make_train_mean_rate_prediction,
    make_trial_shuffled_heldin_prediction,
)
from latentbrain.eval.rebinning import compute_window_bins_for_duration
from latentbrain.eval.reporting import write_metric_audit_outputs
from latentbrain.eval.windowing import crop_neural_dataset_time
from latentbrain.paths import get_repo_root, resolve_configured_path

console = Console(markup=False)
SPLITS = {"train", "validation", "test"}


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


class AuditSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluate_splits: list[str] = Field(min_length=1)
    primary_split: str = Field(min_length=1)
    smoothing_sigma_ms: list[float] = Field(min_length=1)
    shuffle_repeats: int = Field(ge=0)
    shuffle_seed: int = Field(ge=0)
    min_rate_hz: float = Field(gt=0.0)
    max_rate_hz: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_audit(self) -> AuditSection:
        evaluate = {str(value) for value in self.evaluate_splits}
        if not evaluate.issubset(SPLITS):
            msg = "audit.evaluate_splits must contain recognized split names"
            raise ValueError(msg)
        if self.primary_split not in SPLITS:
            msg = "audit.primary_split must be a recognized split name"
            raise ValueError(msg)
        if any(float(value) <= 0.0 for value in self.smoothing_sigma_ms):
            msg = "audit.smoothing_sigma_ms values must be positive"
            raise ValueError(msg)
        if self.max_rate_hz <= self.min_rate_hz:
            msg = "audit.max_rate_hz must exceed audit.min_rate_hz"
            raise ValueError(msg)
        return self


class ReferenceSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_train_heldout_mean_rate_reference: bool
    include_global_mean_reference: bool
    include_split_mean_reference: bool


class ModelOutputsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_existing_outputs: bool
    coordinated_dropout_eval_dir: str = Field(min_length=1)
    temporal_20ms_eval_dir: str = Field(min_length=1)
    factor_latent_reference_bits_per_spike: float
    mean_rate_reference_bits_per_spike: float


class ReportingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str = Field(min_length=1)


class MetricAuditConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: DatasetSection
    splits: SplitSection
    window: WindowSection
    binning: BinningSection
    audit: AuditSection
    references: ReferenceSection
    model_outputs: ModelOutputsSection
    reporting: ReportingSection

    @model_validator(mode="after")
    def validate_contract(self) -> MetricAuditConfig:
        total = (
            self.splits.train_fraction + self.splits.validation_fraction + self.splits.test_fraction
        )
        if abs(total - 1.0) > 1e-8:
            msg = "split fractions must sum to 1.0"
            raise ValueError(msg)
        validate_rebin_factor(
            self.dataset.original_bin_size_ms,
            self.binning.target_bin_size_ms,
        )
        compute_window_bins_for_duration(
            self.window.duration_seconds,
            self.binning.target_bin_size_ms,
        )
        if not self.references.use_train_heldout_mean_rate_reference:
            msg = "references.use_train_heldout_mean_rate_reference must be true"
            raise ValueError(msg)
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> MetricAuditConfig:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            msg = f"malformed metric audit config: {path}"
            raise ValueError(msg) from exc
        if not isinstance(raw, dict):
            msg = f"metric audit config must contain a mapping: {path}"
            raise ValueError(msg)
        return cls.model_validate(raw)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local metric/reference audit diagnostics.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mc_maze_small_metric_audit.yaml"),
    )
    return parser.parse_args(argv)


def _load_config(path: Path) -> dict[str, Any]:
    return MetricAuditConfig.from_yaml(path).model_dump()


def _validate_config(config: dict[str, Any]) -> None:
    MetricAuditConfig.model_validate(config)


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
    windowed = crop_neural_dataset_time(
        rebinned,
        window_bins,
        str(config["window"]["crop_policy"]),
    )
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


def _broadcast(reference_rates: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    return np.broadcast_to(reference_rates, shape).copy()


def _score_with_flags(
    counts: np.ndarray,
    predicted: np.ndarray,
    reference: np.ndarray,
    bin_size_ms: int,
    method: str,
    split: str,
    source: str,
    comparable: bool = True,
    notes: str = "",
) -> dict[str, Any]:
    row = score_prediction_against_reference(
        counts,
        predicted,
        reference,
        bin_size_ms,
        method,
        split,
        source,
    )
    row["directly_comparable"] = comparable
    row["notes"] = notes
    return row


def _metric_subset(metric: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "spike_count",
        "zero_fraction",
        "bits_per_spike",
        "poisson_nll",
        "mean_predicted_rate_hz",
        "mean_reference_rate_hz",
        "observed_rate_hz",
    )
    return {key: metric[key] for key in keys}


def _core_scores(
    dataset: NeuralDataset,
    split: TrialSplit,
    neuron_mask: NeuronMask,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    audit = config["audit"]
    min_rate = float(audit["min_rate_hz"])
    max_rate = float(audit["max_rate_hz"])
    train_mask = _trial_mask(dataset, split.train)
    train_counts = dataset.spikes[train_mask][:, :, neuron_mask.heldout]
    train_heldin = dataset.spikes[train_mask][:, :, neuron_mask.heldin]
    train_reference_rates = compute_train_heldout_mean_rates(
        train_counts,
        dataset.bin_size_ms,
        min_rate,
        max_rate,
    )
    rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    shuffled_rows: list[dict[str, Any]] = []
    unified_ref_ll: dict[str, float] = {}
    seconds = train_counts.size * (dataset.bin_size_ms / 1000.0)
    global_rate = np.clip(train_counts.sum() / seconds, min_rate, max_rate)

    for split_name in audit["evaluate_splits"]:
        split_label = str(split_name)
        mask = _trial_mask(dataset, _split_ids(split, split_label))
        counts = dataset.spikes[mask][:, :, neuron_mask.heldout]
        heldin = dataset.spikes[mask][:, :, neuron_mask.heldin]
        reference = _broadcast(train_reference_rates, counts.shape)
        row = _score_with_flags(
            counts,
            make_train_mean_rate_prediction(
                train_counts,
                counts.shape,
                dataset.bin_size_ms,
                min_rate,
                max_rate,
            ),
            reference,
            dataset.bin_size_ms,
            "train_heldout_mean_rate",
            split_label,
            "constant_rate",
        )
        unified_ref_ll[split_label] = float(row["reference_log_likelihood"])
        rows.append(row)

        if bool(config["references"].get("include_global_mean_reference", True)):
            rows.append(
                _score_with_flags(
                    counts,
                    np.full(counts.shape, float(global_rate)),
                    reference,
                    dataset.bin_size_ms,
                    "global_heldout_mean_rate",
                    split_label,
                    "constant_rate",
                )
            )
        if bool(config["references"].get("include_split_mean_reference", True)):
            split_rates = compute_train_heldout_mean_rates(
                counts,
                dataset.bin_size_ms,
                min_rate,
                max_rate,
            )
            rows.append(
                _score_with_flags(
                    counts,
                    _broadcast(split_rates, counts.shape),
                    reference,
                    dataset.bin_size_ms,
                    "split_heldout_mean_rate",
                    split_label,
                    "leaky_constant_rate",
                    notes="uses evaluation split targets; diagnostic only",
                )
            )
        for sigma in audit["smoothing_sigma_ms"]:
            predicted = make_oracle_smoothed_heldout_prediction(
                counts,
                dataset.bin_size_ms,
                float(sigma),
                min_rate,
                max_rate,
            )
            metric = _score_with_flags(
                counts,
                predicted,
                reference,
                dataset.bin_size_ms,
                "oracle_smoothed_heldout",
                split_label,
                "oracle",
            )
            oracle_rows.append(
                {
                    "control_name": "oracle_smoothed_heldout",
                    "split": split_label,
                    "smoothing_sigma_ms": float(sigma),
                    **_metric_subset(metric),
                    "valid_model": False,
                    "notes": "uses held-out targets directly; upper-bound diagnostic only",
                }
            )

        random_rates = make_random_rate_prediction(
            counts.shape,
            train_reference_rates,
            int(audit["shuffle_seed"]),
            min_rate,
            max_rate,
        )
        random_metric = _score_with_flags(
            counts,
            random_rates,
            reference,
            dataset.bin_size_ms,
            "random_rate",
            split_label,
            "random",
        )
        shuffled_rows.append(
            {
                "control_name": "random_rate",
                "split": split_label,
                "repeat_index": -1,
                "bits_per_spike": random_metric["bits_per_spike"],
                "poisson_nll": random_metric["poisson_nll"],
                "mean_predicted_rate_hz": random_metric["mean_predicted_rate_hz"],
                "notes": "random lognormal perturbation around train held-out mean rates",
            }
        )
        for repeat in range(int(audit["shuffle_repeats"])):
            predicted = make_trial_shuffled_heldin_prediction(
                train_heldin,
                train_counts,
                heldin,
                counts.shape,
                int(audit["shuffle_seed"]) + repeat,
                min_rate,
                max_rate,
            )
            metric = _score_with_flags(
                counts,
                predicted,
                reference,
                dataset.bin_size_ms,
                "trial_shuffled",
                split_label,
                "shuffle",
            )
            shuffled_rows.append(
                {
                    "control_name": "trial_shuffled",
                    "split": split_label,
                    "repeat_index": repeat,
                    "bits_per_spike": metric["bits_per_spike"],
                    "poisson_nll": metric["poisson_nll"],
                    "mean_predicted_rate_hz": metric["mean_predicted_rate_hz"],
                    "notes": "train held-out trials sampled independently of eval held-in activity",
                }
            )
    return (
        pd.DataFrame(rows),
        pd.DataFrame(oracle_rows),
        pd.DataFrame(shuffled_rows),
        unified_ref_ll,
    )


def _numeric(metric: pd.Series, key: str, default: float = float("nan")) -> float:
    value = metric.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _reported_row(
    metric: pd.Series,
    method_name: str,
    source_path: Path,
    unified_ref_ll: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    split = str(metric["split"])
    ref = _numeric(metric, "reference_log_likelihood")
    unified_ref = unified_ref_ll.get(split, float("nan"))
    comparable = bool(
        np.isfinite(ref) and np.isfinite(unified_ref) and abs(ref - unified_ref) < 1e-3
    )
    observations = (
        int(_numeric(metric, "n_trials", 0.0))
        * int(_numeric(metric, "n_time_bins", 0.0))
        * int(_numeric(metric, "n_target_neurons", _numeric(metric, "n_neurons", 0.0)))
    )
    row = {
        "method_name": method_name,
        "split": split,
        "prediction_source": str(metric.get("prediction_source", "reported")),
        "spike_count": _numeric(metric, "spike_count"),
        "total_observations": observations,
        "zero_fraction": _numeric(metric, "zero_fraction"),
        "model_log_likelihood": _numeric(metric, "poisson_log_likelihood"),
        "reference_log_likelihood": ref,
        "log_likelihood_delta": _numeric(metric, "poisson_log_likelihood") - ref,
        "bits_per_spike": _numeric(metric, "bits_per_spike"),
        "poisson_nll": _numeric(metric, "poisson_nll"),
        "mean_predicted_rate_hz": _numeric(metric, "mean_predicted_rate_hz"),
        "mean_reference_rate_hz": _numeric(metric, "mean_reference_rate_hz"),
        "observed_rate_hz": _numeric(metric, "observed_rate_hz"),
        "reference_name": "train_heldout_mean_rate" if comparable else "reported_reference",
        "directly_comparable": comparable,
        "notes": "loaded from existing reported split metrics; predictions not re-scored",
    }
    diag = {
        "method_name": method_name,
        "split": split,
        "prediction_source": row["prediction_source"],
        "reported_bits_per_spike": row["bits_per_spike"],
        "reported_reference_log_likelihood": ref,
        "unified_reference_log_likelihood": unified_ref,
        "reference_delta": ref - unified_ref,
        "directly_comparable": comparable,
        "source_path": str(source_path),
        "notes": row["notes"],
    }
    return row, diag


def _configured_reference_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = config["model_outputs"]
    return [
        {
            "method_name": "configured_previous_mean_rate_reference",
            "split": "validation",
            "reported_bits_per_spike": float(outputs["mean_rate_reference_bits_per_spike"]),
            "directly_comparable": False,
            "notes": "configured previous headline number; no log-likelihood available to audit",
        },
        {
            "method_name": "configured_previous_factor_latent_reference",
            "split": "validation",
            "reported_bits_per_spike": float(outputs["factor_latent_reference_bits_per_spike"]),
            "directly_comparable": False,
            "notes": "configured previous headline number; no predictions available to re-score",
        },
    ]


def _load_existing_outputs(
    config: dict[str, Any], unified_ref_ll: dict[str, float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics: list[dict[str, Any]] = _configured_reference_rows(config)
    if not bool(config["model_outputs"].get("include_existing_outputs", True)):
        return pd.DataFrame(), pd.DataFrame(diagnostics)

    repo_root = get_repo_root()
    rows: list[dict[str, Any]] = []
    eval_specs = [
        (
            "lfads_20ms",
            resolve_configured_path(
                str(config["model_outputs"]["temporal_20ms_eval_dir"]), repo_root
            ),
        ),
        (
            "coordinated_dropout_lfads",
            resolve_configured_path(
                str(config["model_outputs"]["coordinated_dropout_eval_dir"]), repo_root
            ),
        ),
    ]
    for label, eval_dir in eval_specs:
        path = eval_dir / "split_metrics.csv"
        if not path.exists():
            diagnostics.append(
                {
                    "method_name": label,
                    "split": "all",
                    "directly_comparable": False,
                    "notes": f"missing {path}",
                }
            )
            continue
        metrics = pd.read_csv(path)
        for _, metric in metrics.iterrows():
            row, diag = _reported_row(
                metric,
                f"{label}_{metric['prediction_source']}",
                path,
                unified_ref_ll,
            )
            rows.append(row)
            diagnostics.append(diag)

    temporal_eval_dir = resolve_configured_path(
        str(config["model_outputs"]["temporal_20ms_eval_dir"]), repo_root
    )
    baseline_path = temporal_eval_dir.parents[2] / "baseline_metrics_by_bin_size.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        subset = baseline[baseline["bin_size_ms"] == int(config["binning"]["target_bin_size_ms"])]
        for _, metric in subset.iterrows():
            row, diag = _reported_row(
                metric,
                str(metric["method_name"]),
                baseline_path,
                unified_ref_ll,
            )
            rows.append(row)
            diagnostics.append(diag)
    else:
        diagnostics.append(
            {
                "method_name": "temporal_rebinning_baselines",
                "split": "all",
                "directly_comparable": False,
                "notes": f"missing {baseline_path}",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def _summary_value(scores: pd.DataFrame, method: str, split: str = "validation") -> float | None:
    rows = scores[(scores["method_name"] == method) & (scores["split"] == split)]
    if rows.empty:
        return None
    return float(rows.iloc[0]["bits_per_spike"])


def _reported_comparable(
    diagnostics: pd.DataFrame,
    method_name: str,
    split: str,
) -> bool:
    if not {"method_name", "split", "directly_comparable"}.issubset(diagnostics.columns):
        return False
    rows = diagnostics[
        (diagnostics["method_name"] == method_name) & (diagnostics["split"] == split)
    ]
    return bool(not rows.empty and bool(rows.iloc[0]["directly_comparable"]))


def _write_empty_figure(path: Path, message: str) -> None:
    import matplotlib  # type: ignore[import-untyped]

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt  # type: ignore[import-untyped]

    plt.figure(figsize=(7, 4))
    plt.text(0.5, 0.5, message, ha="center", va="center")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _write_figures(
    output_dir: Path,
    unified_scores: pd.DataFrame,
    reference_diagnostics: pd.DataFrame,
    oracle_controls: pd.DataFrame,
) -> None:
    import matplotlib  # type: ignore[import-untyped]

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt  # type: ignore[import-untyped]

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    validation = unified_scores[unified_scores["split"] == "validation"]
    if validation.empty:
        _write_empty_figure(figures / "unified_validation_bits.png", "no validation scores")
    else:
        plt.figure(figsize=(10, 4))
        plt.bar(validation["method_name"], validation["bits_per_spike"])
        plt.ylabel("Validation bits/spike")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(figures / "unified_validation_bits.png")
        plt.close()

    if not reference_diagnostics.empty and "reference_delta" in reference_diagnostics:
        diag = reference_diagnostics[reference_diagnostics["split"] == "validation"]
        diag = diag[np.isfinite(pd.to_numeric(diag["reference_delta"], errors="coerce"))]
        if diag.empty:
            _write_empty_figure(
                figures / "reference_log_likelihoods.png", "no auditable reference LLs"
            )
        else:
            plt.figure(figsize=(8, 4))
            plt.bar(diag["method_name"], diag["reference_delta"])
            plt.ylabel("Reported - unified reference LL")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(figures / "reference_log_likelihoods.png")
            plt.close()
    else:
        _write_empty_figure(figures / "reference_log_likelihoods.png", "no reference diagnostics")

    validation_oracle = oracle_controls[oracle_controls["split"] == "validation"]
    if validation_oracle.empty:
        _write_empty_figure(figures / "oracle_control_comparison.png", "no oracle controls")
    else:
        plt.figure(figsize=(8, 4))
        plt.bar(
            validation_oracle["smoothing_sigma_ms"].astype(str),
            validation_oracle["bits_per_spike"],
        )
        plt.xlabel("Oracle smoothing sigma ms")
        plt.ylabel("Validation bits/spike")
        plt.tight_layout()
        plt.savefig(figures / "oracle_control_comparison.png")
        plt.close()


def run_metric_audit(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    _validate_config(config)
    dataset, dataset_hash, _ = _prepare_dataset(config)
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
    core_scores, oracle_controls, shuffled_controls, unified_ref_ll = _core_scores(
        dataset, split, neuron_mask, config
    )
    reported_scores, reference_diagnostics = _load_existing_outputs(config, unified_ref_ll)
    unified_scores = pd.concat([core_scores, reported_scores], ignore_index=True)
    primary = str(config["audit"]["primary_split"])
    previous_mean_comparable = _reported_comparable(reference_diagnostics, "mean_rate", primary)
    factor = _summary_value(unified_scores, "factor_latent", primary)
    lfads = _summary_value(unified_scores, "lfads_20ms_direct_model", primary)
    dropout = _summary_value(unified_scores, "coordinated_dropout_lfads_direct_model", primary)
    neural_values = [value for value in (lfads, dropout) if value is not None]
    best_neural = max(neural_values) if neural_values else None
    train_mean_bits = _summary_value(unified_scores, "train_heldout_mean_rate", primary)
    if train_mean_bits is None or abs(train_mean_bits) > 1e-12:
        msg = "train held-out mean-rate prediction did not score near zero against itself"
        raise RuntimeError(msg)
    best_oracle = float(
        oracle_controls[oracle_controls["split"] == primary]["bits_per_spike"].max()
    )
    mismatch_found = not previous_mean_comparable
    summary = {
        "dataset_name": config["dataset"]["name"],
        "dataset_hash": dataset_hash,
        "bin_size_ms": int(dataset.bin_size_ms),
        "window_seconds": float(config["window"]["duration_seconds"]),
        "primary_split": primary,
        "reference_name": "train_heldout_mean_rate",
        "train_mean_as_model_validation_bits_per_spike": train_mean_bits,
        "global_mean_validation_bits_per_spike": _summary_value(
            unified_scores, "global_heldout_mean_rate", primary
        ),
        "split_mean_validation_bits_per_spike": _summary_value(
            unified_scores, "split_heldout_mean_rate", primary
        ),
        "factor_latent_unified_validation_bits_per_spike": factor,
        "lfads_unified_validation_bits_per_spike": lfads,
        "coordinated_dropout_unified_validation_bits_per_spike": dropout,
        "best_oracle_validation_bits_per_spike": best_oracle,
        "previous_mean_rate_directly_comparable": previous_mean_comparable,
        "metric_reference_mismatch_found": mismatch_found,
        "mean_rate_inflation_found": mismatch_found,
        "neural_models_trail_under_unified_scoring": bool(
            factor is not None and best_neural is not None and best_neural < factor
        ),
        "likely_conclusion": (
            "previous mean-rate headline bits/spike is not directly comparable unless its "
            "reference log-likelihood matches the unified train-heldout mean-rate reference"
        ),
        "warnings": [
            "This is local metric-audit work, not an official NLB leaderboard result.",
            "Oracle controls are not valid models.",
        ],
    }
    return summary, {
        "unified_scores": unified_scores,
        "reference_diagnostics": reference_diagnostics,
        "oracle_controls": oracle_controls,
        "shuffled_controls": shuffled_controls,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "dataset_name",
        "bin_size_ms",
        "window_seconds",
        "train_mean_as_model_validation_bits_per_spike",
        "global_mean_validation_bits_per_spike",
        "split_mean_validation_bits_per_spike",
        "factor_latent_unified_validation_bits_per_spike",
        "lfads_unified_validation_bits_per_spike",
        "coordinated_dropout_unified_validation_bits_per_spike",
        "best_oracle_validation_bits_per_spike",
        "previous_mean_rate_directly_comparable",
        "metric_reference_mismatch_found",
        "neural_models_trail_under_unified_scoring",
        "likely_conclusion",
        "output_dir",
    ):
        console.print(f"{key}: {summary.get(key)}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = _load_config(args.config)
        summary, tables = run_metric_audit(config)
        output_dir = resolve_configured_path(
            str(config["reporting"]["output_dir"]), get_repo_root()
        )
        summary["output_dir"] = str(output_dir)
        write_metric_audit_outputs(output_dir, summary, **tables)
        _write_figures(
            output_dir,
            tables["unified_scores"],
            tables["reference_diagnostics"],
            tables["oracle_controls"],
        )
        _print_summary(summary)
    except Exception as exc:
        console.print(str(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
