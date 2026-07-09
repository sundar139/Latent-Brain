from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml
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
from latentbrain.eval.generalization import (
    RISK_UNRESOLVED,
    summarize_gap_dictionary,
    summarize_validation_test_gap,
    validation_test_gap_table,
)
from latentbrain.eval.rebinning import compute_window_bins_for_duration
from latentbrain.eval.reporting import write_split_audit_outputs
from latentbrain.eval.scoring import ScoringConfig
from latentbrain.eval.split_audit import (
    compare_split_statistics,
    compute_behavior_split_statistics,
    compute_neuron_split_statistics,
    compute_split_statistics,
    compute_trial_statistics,
    run_repeated_split_baselines,
)
from latentbrain.eval.windowing import crop_neural_dataset_time
from latentbrain.paths import get_repo_root, resolve_configured_path

console = Console(markup=False)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local validation/test split and generalization audit."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/mc_maze_small_split_audit.yaml")
    )
    return parser.parse_args(argv)


def _load_config(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"malformed split audit config: {path}"
        raise ValueError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"split audit config must contain a mapping: {path}"
        raise ValueError(msg)
    _validate_config(raw)
    return raw


def _validate_config(config: dict[str, Any]) -> None:
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
    audit = dict(config["audit"])
    seeds = [int(seed) for seed in audit["repeated_split_seeds"]]
    if len(set(seeds)) != len(seeds):
        msg = "audit.repeated_split_seeds must be unique"
        raise ValueError(msg)
    if len(seeds) < 5:
        msg = "at least five repeated split seeds are required"
        raise ValueError(msg)
    if int(audit["bootstrap_repeats"]) <= 0:
        msg = "audit.bootstrap_repeats must be positive"
        raise ValueError(msg)
    if str(config["scoring"]["reference_model"]) != "train_heldout_mean_rate":
        msg = "scoring.reference_model must be train_heldout_mean_rate"
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
    target_bin = int(config["binning"]["target_bin_size_ms"])
    rebinned = rebin_neural_dataset(dataset, target_bin)
    window_bins = compute_window_bins_for_duration(
        float(config["window"]["duration_seconds"]), target_bin
    )
    windowed = crop_neural_dataset_time(rebinned, window_bins, str(config["window"]["crop_policy"]))
    return windowed, dataset_hash


def _split_and_mask(
    dataset: NeuralDataset, config: dict[str, Any]
) -> tuple[TrialSplit, NeuronMask]:
    splits = config["splits"]
    split = create_trial_split(
        dataset.trial_ids,
        float(splits["train_fraction"]),
        float(splits["validation_fraction"]),
        float(splits["test_fraction"]),
        seed=int(splits["seed"]),
    )
    mask = create_neuron_mask(
        dataset.spikes.shape[2], float(splits["heldout_neuron_fraction"]), seed=int(splits["seed"])
    )
    validate_trial_split(split, dataset.trial_ids)
    validate_neuron_mask(mask, dataset.spikes.shape[2])
    return split, mask


def _split_labels(dataset: NeuralDataset, split: TrialSplit) -> np.ndarray:
    labels = np.empty(dataset.trial_ids.shape[0], dtype=object)
    for name in ("train", "validation", "test"):
        labels[np.isin(dataset.trial_ids, getattr(split, name))] = name
    return labels


def _scoring_config(config: dict[str, Any]) -> ScoringConfig:
    scoring = config["scoring"]
    return ScoringConfig(
        bin_size_ms=int(config["binning"]["target_bin_size_ms"]),
        include_poisson_constant=bool(scoring["include_poisson_constant"]),
        min_rate_hz=float(scoring["min_rate_hz"]),
        max_rate_hz=float(scoring["max_rate_hz"]),
        reference_name=str(scoring["reference_model"]),
    )


def _load_seed_robustness_results(config: dict[str, Any]) -> pd.DataFrame:
    path_value = config["inputs"].get("seed_robustness_results_path")
    if not path_value:
        return pd.DataFrame()
    path = resolve_configured_path(str(path_value), get_repo_root())
    if not path.exists():
        console.print(f"Seed robustness results are missing; model gap diagnostics skipped: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _missing_behavior_variables(config: dict[str, Any], names: list[str] | None) -> list[str]:
    requested = [str(name) for name in config["audit"].get("behavior_variables", [])]
    available = set(names or [])
    return [name for name in requested if name not in available]


def _write_figures(
    output_dir: Path,
    trial_statistics: pd.DataFrame,
    gap_table: pd.DataFrame,
    repeated_split: pd.DataFrame,
    bootstrap_draws: np.ndarray | None,
) -> None:
    import matplotlib  # type: ignore[import-untyped]  # noqa: PLC0415

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]  # noqa: PLC0415

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    if not gap_table.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        for method_name, group in gap_table.groupby("method_name", sort=True):
            ordered = group.sort_values("seed")
            ax.plot(
                ordered["seed"],
                ordered["gap_validation_minus_test"],
                marker="o",
                label=str(method_name),
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Seed")
        ax.set_ylabel("Validation minus test bits/spike")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / "validation_test_gap.png", dpi=150)
        plt.close(fig)

    splits = ["train", "validation", "test"]
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [
        trial_statistics[trial_statistics["split"] == name]["population_rate_hz"]
        .dropna()
        .to_numpy()
        for name in splits
    ]
    ax.boxplot(
        [values for values in data if values.size],
        tick_labels=[n for n, v in zip(splits, data, strict=True) if v.size],
    )
    ax.set_ylabel("Trial population rate (Hz)")
    fig.tight_layout()
    fig.savefig(figures / "split_trial_rate_distributions.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    if trial_statistics["endpoint_distance"].notna().any():
        endpoint = [
            trial_statistics[trial_statistics["split"] == name]["endpoint_distance"]
            .dropna()
            .to_numpy()
            for name in splits
        ]
        ax.boxplot(
            [values for values in endpoint if values.size],
            tick_labels=[n for n, v in zip(splits, endpoint, strict=True) if v.size],
        )
        ax.set_ylabel("Endpoint distance")
    else:
        ax.text(0.5, 0.5, "behavior unavailable", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(figures / "split_behavior_distributions.png", dpi=150)
    plt.close(fig)

    if not repeated_split.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        for method_name, group in repeated_split.groupby("method_name", sort=True):
            ordered = group.sort_values("split_seed")
            ax.plot(
                ordered["split_seed"],
                ordered["test_unified_bits_per_spike"],
                marker="o",
                label=f"{method_name} test",
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Split seed")
        ax.set_ylabel("Test unified bits/spike")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / "repeated_split_factor_latent.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    if bootstrap_draws is not None and bootstrap_draws.size:
        ax.hist(bootstrap_draws, bins=40, color="#4C78A8")
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Bootstrap mean validation-test gap")
        ax.set_ylabel("Count")
    else:
        ax.text(0.5, 0.5, "gap diagnostics unavailable", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(figures / "validation_test_gap_bootstrap.png", dpi=150)
    plt.close(fig)


def _bootstrap_draws(gap_table: pd.DataFrame, audit: dict[str, Any]) -> np.ndarray | None:
    if gap_table.empty:
        return None
    factor = gap_table[gap_table["method_name"] == "factor_latent"]
    source = factor if not factor.empty else gap_table
    differences = source["gap_validation_minus_test"].to_numpy(dtype=np.float64)
    if differences.size < 2:
        return None
    generator = np.random.default_rng(int(audit["bootstrap_seed"]))
    draws = generator.integers(
        0, differences.size, size=(int(audit["bootstrap_repeats"]), differences.size)
    )
    return np.asarray(differences[draws].mean(axis=1))


def run_split_audit(config: dict[str, Any]) -> dict[str, Any]:
    dataset, dataset_hash = _prepare_dataset(config)
    split, mask = _split_and_mask(dataset, config)
    labels = _split_labels(dataset, split)
    heldin_indices = np.flatnonzero(mask.heldin)
    heldout_indices = np.flatnonzero(mask.heldout)
    bin_size_ms = int(config["binning"]["target_bin_size_ms"])
    audit = dict(config["audit"])

    trial_statistics = compute_trial_statistics(
        dataset.spikes,
        dataset.behavior,
        list(dataset.behavior_names) if dataset.behavior_names is not None else None,
        labels,
        bin_size_ms,
        heldin_indices,
        heldout_indices,
    )
    split_statistics = compute_split_statistics(trial_statistics)
    neuron_split_statistics = compute_neuron_split_statistics(
        dataset.spikes, labels, heldin_indices, heldout_indices, bin_size_ms
    )
    behavior_available = dataset.behavior is not None and dataset.behavior_names is not None
    behavior_split_statistics = (
        compute_behavior_split_statistics(dataset.behavior, list(dataset.behavior_names), labels)
        if behavior_available
        and dataset.behavior is not None
        and dataset.behavior_names is not None
        else pd.DataFrame()
    )
    split_comparison = compare_split_statistics(trial_statistics, "validation", "test")

    gap_table = validation_test_gap_table(_load_seed_robustness_results(config))
    gap_summary = summarize_validation_test_gap(
        gap_table,
        int(audit["bootstrap_repeats"]),
        float(audit["confidence_interval"]),
        int(audit["bootstrap_seed"]),
    )
    gap_fields = summarize_gap_dictionary(gap_summary)

    scoring = _scoring_config(config)
    repeated_split = run_repeated_split_baselines(
        dataset,
        [int(seed) for seed in audit["repeated_split_seeds"]],
        [str(method) for method in audit["repeated_split_methods"]],
        float(config["splits"]["train_fraction"]),
        float(config["splits"]["validation_fraction"]),
        float(config["splits"]["test_fraction"]),
        float(config["splits"]["heldout_neuron_fraction"]),
        scoring,
    )
    repeated_factor = repeated_split[repeated_split["method_name"] == "factor_latent"]

    def _split_stat(split_name: str, column: str) -> float:
        rows = split_statistics[split_statistics["split"] == split_name]
        return float("nan") if rows.empty else float(rows.iloc[0][column])

    def _gap_stat(column: str) -> float:
        rows = gap_summary[gap_summary["method_name"] == "factor_latent"]
        return float("nan") if rows.empty else float(rows.iloc[0][column])

    validation_mean = _gap_stat("mean_validation")
    test_mean = _gap_stat("mean_test")
    repeated_validation_mean = (
        float(repeated_factor["validation_unified_bits_per_spike"].mean())
        if not repeated_factor.empty
        else float("nan")
    )
    repeated_test_mean = (
        float(repeated_factor["test_unified_bits_per_spike"].mean())
        if not repeated_factor.empty
        else float("nan")
    )
    repeated_test_positive_fraction = (
        float((repeated_factor["test_unified_bits_per_spike"] > 0.0).mean())
        if not repeated_factor.empty
        else float("nan")
    )
    persists = bool(
        not repeated_factor.empty and repeated_validation_mean > 0.0 and repeated_test_mean < 0.0
    )

    summary: dict[str, Any] = {
        "dataset_name": config["dataset"]["name"],
        "dataset_hash": dataset_hash,
        "bin_size_ms": bin_size_ms,
        "window_seconds": float(config["window"]["duration_seconds"]),
        "accepted_split_seed": int(config["splits"]["seed"]),
        "train_trial_count": int(len(split.train)),
        "validation_trial_count": int(len(split.validation)),
        "test_trial_count": int(len(split.test)),
        "heldin_neuron_count": int(heldin_indices.size),
        "heldout_neuron_count": int(heldout_indices.size),
        "behavior_available": bool(behavior_available),
        "missing_behavior_variables": _missing_behavior_variables(
            config, list(dataset.behavior_names) if dataset.behavior_names is not None else None
        ),
        "validation_heldout_rate_hz": _split_stat("validation", "mean_heldout_rate_hz"),
        "test_heldout_rate_hz": _split_stat("test", "mean_heldout_rate_hz"),
        "factor_latent_validation_mean": validation_mean,
        "factor_latent_test_mean": test_mean,
        "factor_latent_validation_test_gap": _gap_stat("mean_gap"),
        "factor_latent_gap_ci95_low": _gap_stat("gap_ci95_low"),
        "factor_latent_gap_ci95_high": _gap_stat("gap_ci95_high"),
        "repeated_split_validation_mean": repeated_validation_mean,
        "repeated_split_test_mean": repeated_test_mean,
        "repeated_split_test_positive_fraction": repeated_test_positive_fraction,
        "validation_positive_test_negative_persists": persists,
        "repeated_split_seeds": [int(seed) for seed in audit["repeated_split_seeds"]],
        "old_incompatible_mean_rate_values_used_as_targets": False,
        "official_benchmark_claim": False,
        **gap_fields,
    }
    if not gap_fields["model_gap_diagnostics_available"]:
        summary["generalization_risk"] = RISK_UNRESOLVED

    repo_root = get_repo_root()
    output_dir = resolve_configured_path(str(config["reporting"]["output_dir"]), repo_root)
    write_split_audit_outputs(
        output_dir,
        summary,
        trial_statistics,
        split_statistics,
        neuron_split_statistics,
        behavior_split_statistics,
        gap_table,
        gap_summary,
        repeated_split,
        split_comparison,
    )
    _write_figures(
        output_dir,
        trial_statistics,
        gap_table,
        repeated_split,
        _bootstrap_draws(gap_table, audit),
    )
    summary["output_dir"] = str(output_dir)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = get_repo_root()
    args = _parse_args(argv)
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    if not config_path.exists():
        console.print(f"Config file is missing: {config_path}")
        return 2
    try:
        config = _load_config(config_path)
        summary = run_split_audit(config)
    except (OSError, ValueError, FileNotFoundError) as exc:
        console.print(f"Split audit failed: {exc}")
        return 2
    for key in (
        "dataset_name",
        "bin_size_ms",
        "window_seconds",
        "accepted_split_seed",
        "validation_trial_count",
        "test_trial_count",
        "validation_heldout_rate_hz",
        "test_heldout_rate_hz",
        "factor_latent_validation_mean",
        "factor_latent_test_mean",
        "factor_latent_validation_test_gap",
        "generalization_risk",
        "repeated_split_validation_mean",
        "repeated_split_test_mean",
        "repeated_split_test_positive_fraction",
        "validation_positive_test_negative_persists",
        "output_dir",
    ):
        console.print(f"{key}: {summary.get(key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
