from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _format_table(dataframe: pd.DataFrame) -> list[str]:
    columns = [str(column) for column in dataframe.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in dataframe.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def write_baseline_markdown_report(
    output_path: Path,
    dataset_name: str,
    metrics_summary: dict[str, Any],
    split_metrics: pd.DataFrame,
    neuron_metrics: pd.DataFrame,
) -> Path:
    """Write a Markdown report for the local mean-rate baseline."""
    primary_split = metrics_summary.get("primary_split")
    primary_group = metrics_summary.get("primary_neuron_group")
    lines = [
        f"# {dataset_name} mean-rate baseline report",
        "",
        "This is a local sanity baseline, not an official NLB leaderboard result.",
        "No neural network model was trained.",
        "",
        "## Dataset and baseline",
        f"- Dataset hash: {metrics_summary.get('dataset_hash')}",
        f"- Baseline name: {metrics_summary.get('baseline_name')}",
        "- Fit policy: train trials only",
        f"- Primary metric: {primary_split} {primary_group} bits/spike",
        f"- Primary bits/spike: {metrics_summary.get('primary_bits_per_spike')}",
        f"- Primary Poisson NLL: {metrics_summary.get('primary_poisson_nll')}",
        "",
        "## Held-in and held-out neuron groups",
        "Held-in and held-out groups reuse the deterministic local neuron mask. The mean-rate "
        "baseline is fit on train trials only, then evaluated separately on held-in, held-out, "
        "and all neurons for each split.",
        "",
        "## Split metrics",
        *_format_table(split_metrics),
        "",
        "## Neuron metrics preview",
        *_format_table(neuron_metrics.head(20)),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_baseline_outputs(
    output_dir: Path,
    metrics_summary: dict[str, Any],
    split_metrics: pd.DataFrame,
    neuron_metrics: pd.DataFrame,
) -> dict[str, Path]:
    """Write baseline JSON, CSV, and Markdown outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics_summary": output_dir / "metrics_summary.json",
        "split_metrics": output_dir / "split_metrics.csv",
        "neuron_metrics": output_dir / "neuron_metrics.csv",
        "baseline_report": output_dir / "baseline_report.md",
    }
    paths["metrics_summary"].write_text(
        json.dumps(metrics_summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    split_metrics.to_csv(paths["split_metrics"], index=False)
    neuron_metrics.to_csv(paths["neuron_metrics"], index=False)
    write_baseline_markdown_report(
        paths["baseline_report"],
        dataset_name=str(metrics_summary["dataset_name"]),
        metrics_summary=metrics_summary,
        split_metrics=split_metrics,
        neuron_metrics=neuron_metrics,
    )
    return paths


def write_behavior_decoder_markdown_report(
    output_path: Path,
    metrics_summary: dict[str, Any],
    split_metrics: pd.DataFrame,
    target_metrics: pd.DataFrame,
) -> Path:
    """Write a Markdown report for the local behavior decoder baseline."""
    lines = [
        f"# {metrics_summary.get('dataset_name')} behavior decoder report",
        "",
        (
            "This is a local behavior-decoding sanity baseline, "
            "not an official NLB leaderboard result."
        ),
        "No neural network model was trained.",
        "",
        "## Dataset and decoder",
        f"- Dataset hash: {metrics_summary.get('dataset_hash')}",
        f"- Feature neuron group: {metrics_summary.get('feature_neuron_group')}",
        f"- Smoothing: {metrics_summary.get('smoothing')}",
        f"- Behavior targets: {metrics_summary.get('target_names')}",
        f"- Decoder: {metrics_summary.get('decoder_name')}",
        f"- Decoder alpha: {metrics_summary.get('decoder_alpha')}",
        f"- Fit policy: {metrics_summary.get('fit_policy')}",
        f"- Standardization policy: {metrics_summary.get('standardization_policy')}",
        f"- Primary split: {metrics_summary.get('primary_split')}",
        f"- Primary validation R2: {metrics_summary.get('primary_mean_r2')}",
        "",
        "## Split metrics",
        *_format_table(split_metrics),
        "",
        "## Target metrics",
        *_format_table(target_metrics),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_behavior_decoder_outputs(
    output_dir: Path,
    metrics_summary: dict[str, Any],
    split_metrics: pd.DataFrame,
    target_metrics: pd.DataFrame,
    decoder_coefficients: pd.DataFrame,
) -> dict[str, Path]:
    """Write behavior decoder JSON, CSV, and Markdown outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics_summary": output_dir / "metrics_summary.json",
        "split_metrics": output_dir / "split_metrics.csv",
        "target_metrics": output_dir / "target_metrics.csv",
        "decoder_coefficients": output_dir / "decoder_coefficients.csv",
        "report": output_dir / "behavior_decoder_report.md",
    }
    paths["metrics_summary"].write_text(
        json.dumps(metrics_summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    split_metrics.to_csv(paths["split_metrics"], index=False)
    target_metrics.to_csv(paths["target_metrics"], index=False)
    decoder_coefficients.to_csv(paths["decoder_coefficients"], index=False)
    write_behavior_decoder_markdown_report(
        paths["report"], metrics_summary, split_metrics, target_metrics
    )
    return paths


def write_cosmoothing_markdown_report(
    output_path: Path,
    metrics_summary: dict[str, Any],
    split_metrics: pd.DataFrame,
) -> Path:
    """Write a Markdown report for the local co-smoothing ridge baseline."""
    lines = [
        f"# {metrics_summary.get('dataset_name')} co-smoothing ridge report",
        "",
        ("This is a local co-smoothing sanity baseline, not an official NLB leaderboard result."),
        "No neural network model was trained.",
        "",
        "## Dataset and decoder",
        f"- Dataset hash: {metrics_summary.get('dataset_hash')}",
        f"- Input group: {metrics_summary.get('input_neuron_group')} neurons",
        f"- Target group: {metrics_summary.get('target_neuron_group')} neurons",
        f"- Smoothing: {metrics_summary.get('smoothing')}",
        f"- Decoder: {metrics_summary.get('decoder_name')}",
        f"- Decoder alpha: {metrics_summary.get('decoder_alpha')}",
        f"- Fit policy: {metrics_summary.get('fit_policy')}",
        f"- Standardization policy: {metrics_summary.get('standardization_policy')}",
        f"- Reference policy: {metrics_summary.get('reference_policy')}",
        f"- Primary split: {metrics_summary.get('primary_split')}",
        f"- Primary validation bits/spike: {metrics_summary.get('primary_bits_per_spike')}",
        f"- Primary validation Poisson NLL: {metrics_summary.get('primary_poisson_nll')}",
        "",
        "## Split metrics",
        *_format_table(split_metrics),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_cosmoothing_outputs(
    output_dir: Path,
    metrics_summary: dict[str, Any],
    split_metrics: pd.DataFrame,
    neuron_metrics: pd.DataFrame,
    decoder_coefficients: pd.DataFrame,
) -> dict[str, Path]:
    """Write co-smoothing JSON, CSV, and Markdown outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics_summary": output_dir / "metrics_summary.json",
        "split_metrics": output_dir / "split_metrics.csv",
        "neuron_metrics": output_dir / "neuron_metrics.csv",
        "decoder_coefficients": output_dir / "decoder_coefficients.csv",
        "report": output_dir / "cosmoothing_report.md",
    }
    paths["metrics_summary"].write_text(
        json.dumps(metrics_summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    split_metrics.to_csv(paths["split_metrics"], index=False)
    neuron_metrics.to_csv(paths["neuron_metrics"], index=False)
    decoder_coefficients.to_csv(paths["decoder_coefficients"], index=False)
    write_cosmoothing_markdown_report(paths["report"], metrics_summary, split_metrics)
    return paths


def write_cosmoothing_sweep_markdown_report(
    output_path: Path,
    summary: dict[str, Any],
    best_split_metrics: pd.DataFrame,
) -> Path:
    """Write a Markdown report for the local co-smoothing diagnostic sweep."""
    best_config = dict(summary.get("best_config", {}))
    lines = [
        f"# {summary.get('dataset_name')} co-smoothing diagnostic sweep report",
        "",
        "This is a local co-smoothing diagnostic sweep, not an official NLB leaderboard result.",
        "No neural network model was trained.",
        "",
        "## Dataset",
        f"- Dataset hash: {summary.get('dataset_hash')}",
        "",
        "## Sweep grid",
        f"- Grid: {summary.get('sweep_grid')}",
        f"- Configurations tested: {summary.get('n_configurations')}",
        "",
        "## Best validation configuration",
        f"- Best validation bits/spike: {summary.get('best_validation_bits_per_spike')}",
        f"- Best validation Poisson NLL: {summary.get('best_validation_poisson_nll')}",
        f"- Best smoothing sigma: {best_config.get('smoothing_sigma_ms')}",
        f"- Best ridge alpha: {best_config.get('ridge_alpha')}",
        f"- Feature standardization: {best_config.get('standardize_features')}",
        f"- Fit intercept: {best_config.get('fit_intercept')}",
    ]
    if bool(summary.get("all_validation_bits_per_spike_negative", False)):
        lines.extend(
            [
                "",
                "All validation bits/spike values were negative in this local sweep.",
            ]
        )
    lines.extend(["", "## Best train/validation/test metrics", *_format_table(best_split_metrics)])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_cosmoothing_sweep_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    sweep_results: pd.DataFrame,
    best_config: dict[str, Any],
    best_split_metrics: pd.DataFrame,
    best_neuron_metrics: pd.DataFrame,
) -> dict[str, Path]:
    """Write co-smoothing sweep tables, best config, and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "sweep_results": output_dir / "sweep_results.csv",
        "best_config": output_dir / "best_config.json",
        "best_split_metrics": output_dir / "best_split_metrics.csv",
        "best_neuron_metrics": output_dir / "best_neuron_metrics.csv",
        "report": output_dir / "sweep_report.md",
    }
    sweep_results.to_csv(paths["sweep_results"], index=False)
    paths["best_config"].write_text(
        json.dumps(best_config, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    best_split_metrics.to_csv(paths["best_split_metrics"], index=False)
    best_neuron_metrics.to_csv(paths["best_neuron_metrics"], index=False)
    write_cosmoothing_sweep_markdown_report(paths["report"], summary, best_split_metrics)
    return paths


def write_factor_latent_markdown_report(
    output_path: Path,
    metrics_summary: dict[str, Any],
    split_metrics: pd.DataFrame,
    behavior_metrics: pd.DataFrame,
) -> Path:
    """Write a Markdown report for the local factor latent baseline."""
    lines = [
        f"# {metrics_summary.get('dataset_name')} factor latent report",
        "",
        "This is a local latent-variable sanity baseline, not an official NLB leaderboard result.",
        "No neural network model was trained.",
        "This is GPFA-style only; no temporal GP prior is implemented.",
        "",
        "## Dataset and model",
        f"- Dataset hash: {metrics_summary.get('dataset_hash')}",
        f"- Model name: {metrics_summary.get('model_name')}",
        f"- Input group: {metrics_summary.get('input_neuron_group')} neurons",
        f"- Target group: {metrics_summary.get('target_neuron_group')} neurons",
        f"- Smoothing: {metrics_summary.get('smoothing')}",
        f"- Latent dimension: {metrics_summary.get('latent_dim')}",
        f"- Held-out decoder: {metrics_summary.get('heldout_decoder_name')}",
        f"- Held-out decoder alpha: {metrics_summary.get('heldout_decoder_alpha')}",
        f"- Behavior decoder enabled: {metrics_summary.get('behavior_decoder_enabled')}",
        f"- Behavior decoder alpha: {metrics_summary.get('behavior_decoder_alpha')}",
        f"- Fit policy: {metrics_summary.get('fit_policy')}",
        f"- Standardization policy: {metrics_summary.get('standardization_policy')}",
        f"- Reference policy: {metrics_summary.get('reference_policy')}",
        f"- Primary validation bits/spike: {metrics_summary.get('primary_bits_per_spike')}",
        f"- Primary validation Poisson NLL: {metrics_summary.get('primary_poisson_nll')}",
        f"- Primary validation behavior R2: {metrics_summary.get('primary_behavior_mean_r2')}",
        "",
        "## Split metrics",
        *_format_table(split_metrics),
    ]
    if not behavior_metrics.empty:
        lines.extend(["", "## Behavior metrics", *_format_table(behavior_metrics)])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_factor_latent_outputs(
    output_dir: Path,
    metrics_summary: dict[str, Any],
    split_metrics: pd.DataFrame,
    neuron_metrics: pd.DataFrame,
    behavior_metrics: pd.DataFrame,
    latent_summary: pd.DataFrame,
    factor_loadings: pd.DataFrame,
    heldout_decoder_coefficients: pd.DataFrame,
    behavior_decoder_coefficients: pd.DataFrame,
) -> dict[str, Path]:
    """Write factor latent JSON, CSV, and Markdown outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics_summary": output_dir / "metrics_summary.json",
        "split_metrics": output_dir / "split_metrics.csv",
        "neuron_metrics": output_dir / "neuron_metrics.csv",
        "behavior_metrics": output_dir / "behavior_metrics.csv",
        "latent_summary": output_dir / "latent_summary.csv",
        "factor_loadings": output_dir / "factor_loadings.csv",
        "heldout_decoder_coefficients": output_dir / "heldout_decoder_coefficients.csv",
        "behavior_decoder_coefficients": output_dir / "behavior_decoder_coefficients.csv",
        "report": output_dir / "factor_latent_report.md",
    }
    paths["metrics_summary"].write_text(
        json.dumps(metrics_summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    split_metrics.to_csv(paths["split_metrics"], index=False)
    neuron_metrics.to_csv(paths["neuron_metrics"], index=False)
    behavior_metrics.to_csv(paths["behavior_metrics"], index=False)
    latent_summary.to_csv(paths["latent_summary"], index=False)
    factor_loadings.to_csv(paths["factor_loadings"], index=False)
    heldout_decoder_coefficients.to_csv(paths["heldout_decoder_coefficients"], index=False)
    behavior_decoder_coefficients.to_csv(paths["behavior_decoder_coefficients"], index=False)
    write_factor_latent_markdown_report(
        paths["report"], metrics_summary, split_metrics, behavior_metrics
    )
    return paths
