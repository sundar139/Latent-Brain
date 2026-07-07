from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml


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


def write_factor_latent_sweep_markdown_report(
    output_path: Path,
    summary: dict[str, Any],
    best_split_metrics: pd.DataFrame,
) -> Path:
    """Write a Markdown report for the local factor latent diagnostic sweep."""
    best_config = dict(summary.get("best_config", {}))
    lines = [
        f"# {summary.get('dataset_name')} factor latent diagnostic sweep report",
        "",
        "This is a local factor latent diagnostic sweep, not an official NLB leaderboard result.",
        "This is not full GPFA because no temporal GP prior is implemented.",
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
        f"- Best validation behavior mean R2: {summary.get('best_validation_behavior_mean_r2')}",
        f"- Best latent dimension: {best_config.get('latent_dim')}",
        f"- Best smoothing sigma: {best_config.get('smoothing_sigma_ms')}",
        f"- Best held-out decoder alpha: {best_config.get('heldout_decoder_alpha')}",
        f"- Feature standardization: {best_config.get('standardize_features')}",
        "",
        "## Baseline comparisons",
        (
            "- Previous latent_dim 8 validation bits/spike: "
            f"{summary.get('single_factor_latent_validation_bits_per_spike')}"
        ),
        (
            "- Mean-rate validation heldout bits/spike: "
            f"{summary.get('mean_rate_validation_heldout_bits_per_spike')}"
        ),
        "",
        "## Best train/validation/test metrics",
        *_format_table(best_split_metrics),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_factor_latent_sweep_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    sweep_results: pd.DataFrame,
    best_config: dict[str, Any],
    best_split_metrics: pd.DataFrame,
    best_neuron_metrics: pd.DataFrame,
    best_behavior_metrics: pd.DataFrame,
    best_latent_summary: pd.DataFrame,
) -> dict[str, Path]:
    """Write factor latent sweep tables, best config, and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "sweep_results": output_dir / "sweep_results.csv",
        "best_config": output_dir / "best_config.json",
        "best_split_metrics": output_dir / "best_split_metrics.csv",
        "best_neuron_metrics": output_dir / "best_neuron_metrics.csv",
        "best_behavior_metrics": output_dir / "best_behavior_metrics.csv",
        "best_latent_summary": output_dir / "best_latent_summary.csv",
        "report": output_dir / "sweep_report.md",
    }
    sweep_results.to_csv(paths["sweep_results"], index=False)
    paths["best_config"].write_text(
        json.dumps(best_config, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    best_split_metrics.to_csv(paths["best_split_metrics"], index=False)
    best_neuron_metrics.to_csv(paths["best_neuron_metrics"], index=False)
    best_behavior_metrics.to_csv(paths["best_behavior_metrics"], index=False)
    best_latent_summary.to_csv(paths["best_latent_summary"], index=False)
    write_factor_latent_sweep_markdown_report(paths["report"], summary, best_split_metrics)
    return paths


def write_lfads_gru_training_report(
    output_path: Path,
    summary: dict[str, Any],
) -> Path:
    """Write a Markdown report for local LFADS-style GRU training."""
    lines = [
        f"# {summary.get('dataset_name')} LFADS-style GRU training report",
        "",
        "This is an LFADS-style masked co-smoothing training run, not a full LFADS implementation.",
        "This is local validation only, not an official NLB leaderboard result.",
        "",
        "## Dataset and model",
        f"- Dataset hash: {summary.get('dataset_hash')}",
        f"- Model name: {summary.get('model_name')}",
        "- Input neurons: held-in only",
        f"- Training mode: {summary.get('training_mode')}",
        f"- Output dimension policy: {summary.get('output_dim_policy')}",
        f"- Input neuron count: {summary.get('input_dim')}",
        f"- Output neuron count: {summary.get('output_dim')}",
        f"- Encoder hidden dimension: {summary.get('encoder_hidden_dim')}",
        f"- Generator hidden dimension: {summary.get('generator_hidden_dim')}",
        f"- Factor dimension: {summary.get('factor_dim')}",
        f"- Latent dimension: {summary.get('latent_dim')}",
        "",
        "## Training",
        f"- Training epochs: {summary.get('epochs')}",
        f"- KL warmup epochs: {summary.get('kl_warmup_epochs')}",
        f"- Held-in loss weight: {summary.get('heldin_loss_weight')}",
        f"- Held-out loss weight: {summary.get('heldout_loss_weight')}",
        f"- Best validation loss: {summary.get('best_validation_loss')}",
        f"- Best validation total loss: {summary.get('best_validation_total_loss')}",
        f"- Final validation loss: {summary.get('final_validation_loss')}",
        (
            "- Final validation held-out prediction loss: "
            f"{summary.get('final_validation_heldout_prediction_loss')}"
        ),
        "",
        "## Checkpoints",
        f"- Latest checkpoint: {summary.get('latest_checkpoint')}",
        f"- Best validation checkpoint: {summary.get('best_validation_checkpoint')}",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_lfads_gru_evaluation_report(output_path: Path, summary: dict[str, Any]) -> Path:
    """Write a Markdown report for local LFADS-style held-out evaluation."""
    lines = [
        f"# {summary.get('dataset_name')} LFADS-style GRU held-out evaluation report",
        "",
        "This is an LFADS-style sequential VAE foundation, not a full LFADS implementation.",
        "This is a local held-out evaluation, not an official NLB leaderboard result.",
        "No new neural network model was trained by this evaluation script.",
        "",
        "## Dataset and checkpoint",
        f"- Dataset hash: {summary.get('dataset_hash')}",
        f"- Checkpoint path: {summary.get('checkpoint_path')}",
        f"- Model name: {summary.get('model_name')}",
        "- Input group: held-in neurons",
        "- Held-out target group: held-out neurons",
        f"- Factor dimension: {summary.get('factor_dim')}",
        f"- Latent dimension: {summary.get('latent_dim')}",
        "",
        "## Decoders",
        f"- Primary prediction source: {summary.get('primary_prediction_source')}",
        f"- Direct model held-out rates available: {summary.get('direct_model_available')}",
        f"- Factor decoder evaluated: {summary.get('factor_decoder_evaluated')}",
        f"- Held-out decoder alpha: {summary.get('heldout_decoder_alpha')}",
        f"- Behavior decoder enabled: {summary.get('behavior_decoder_enabled')}",
        f"- Behavior decoder alpha: {summary.get('behavior_decoder_alpha')}",
        f"- Fit policy: {summary.get('fit_policy')}",
        "",
        "## Primary metrics",
        f"- Primary split: {summary.get('primary_split')}",
        f"- Primary validation bits/spike: {summary.get('primary_bits_per_spike')}",
        f"- Primary validation Poisson NLL: {summary.get('primary_poisson_nll')}",
        f"- Primary validation behavior mean R²: {summary.get('primary_behavior_mean_r2')}",
        (
            "- Direct model validation bits/spike: "
            f"{summary.get('direct_model_validation_bits_per_spike')}"
        ),
        (
            "- Factor decoder validation bits/spike: "
            f"{summary.get('factor_decoder_validation_bits_per_spike')}"
        ),
        "",
        "## Baseline comparisons",
        (
            "- Mean-rate validation bits/spike: "
            f"{summary.get('mean_rate_validation_bits_per_spike')}"
        ),
        (
            "- Factor latent best validation bits/spike: "
            f"{summary.get('factor_latent_best_validation_bits_per_spike')}"
        ),
        (
            "- Previous LFADS-style held-out bits/spike: "
            f"{summary.get('previous_lfads_eval_validation_bits_per_spike')}"
        ),
        f"- Beats previous LFADS-style evaluation: {summary.get('beats_previous_lfads_eval')}",
        f"- Beats mean-rate reference: {summary.get('beats_mean_rate_reference')}",
        f"- Beats factor-latent reference: {summary.get('beats_factor_latent_reference')}",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _decoder_coefficients_table(
    coefficients: np.ndarray,
    target_names: list[str],
    target_indices: list[int] | None = None,
) -> pd.DataFrame:
    rows = []
    if coefficients.size == 0 or coefficients.shape[1] == 0:
        return pd.DataFrame(
            columns=["factor_index", "target_name", "target_rank", "coefficient"]
            + (["target_neuron_index"] if target_indices is not None else [])
        )
    for factor_index in range(coefficients.shape[0]):
        for target_rank, target_name in enumerate(target_names):
            row = {
                "factor_index": factor_index,
                "target_name": target_name,
                "target_rank": target_rank,
                "coefficient": float(coefficients[factor_index, target_rank]),
            }
            if target_indices is not None:
                row["target_neuron_index"] = int(target_indices[target_rank])
            rows.append(row)
    return pd.DataFrame(rows)


def write_lfads_gru_evaluation_outputs(
    output_dir: Path,
    metrics_summary: dict[str, Any],
    split_metrics: pd.DataFrame,
    neuron_metrics: pd.DataFrame,
    behavior_metrics: pd.DataFrame,
    factor_summary: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Path]:
    """Write local LFADS-style held-out evaluation outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics_summary": output_dir / "metrics_summary.json",
        "split_metrics": output_dir / "split_metrics.csv",
        "neuron_metrics": output_dir / "neuron_metrics.csv",
        "behavior_metrics": output_dir / "behavior_metrics.csv",
        "factor_summary": output_dir / "factor_summary.csv",
        "heldout_decoder_coefficients": output_dir / "heldout_decoder_coefficients.csv",
        "behavior_decoder_coefficients": output_dir / "behavior_decoder_coefficients.csv",
        "report": output_dir / "lfads_gru_eval_report.md",
    }
    paths["metrics_summary"].write_text(
        json.dumps(metrics_summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    split_metrics.to_csv(paths["split_metrics"], index=False)
    neuron_metrics.to_csv(paths["neuron_metrics"], index=False)
    behavior_metrics.to_csv(paths["behavior_metrics"], index=False)
    factor_summary.to_csv(paths["factor_summary"], index=False)
    target_indices = [int(value) for value in metadata.get("target_neuron_indices", [])]
    _decoder_coefficients_table(
        np.asarray(metadata["heldout_decoder_coefficients"], dtype=np.float64),
        [str(index) for index in target_indices],
        target_indices,
    ).to_csv(paths["heldout_decoder_coefficients"], index=False)
    _decoder_coefficients_table(
        np.asarray(metadata["behavior_decoder_coefficients"], dtype=np.float64),
        [str(value) for value in metadata.get("behavior_target_names", [])],
    ).to_csv(paths["behavior_decoder_coefficients"], index=False)
    write_lfads_gru_evaluation_report(paths["report"], metrics_summary)
    return paths


def write_window_matched_comparison_report(
    output_path: Path,
    summary: dict[str, Any],
    validation_leaderboard: pd.DataFrame,
) -> Path:
    """Write a Markdown report for the local window-matched comparison."""
    dataset_name = summary.get("dataset_name")
    lines = [
        f"# {dataset_name} window-matched comparison report",
        "",
        "This is a local window-matched comparison, not an official NLB leaderboard result.",
        "Neural methods are LFADS-style only, not full LFADS.",
        "No new neural network model was trained by this comparison script.",
        "",
        "## Dataset and window",
        f"- Dataset name: {dataset_name}",
        f"- Dataset hash: {summary.get('dataset_hash')}",
        f"- Original time bins: {summary.get('original_time_bins')}",
        f"- Cropped time bins: {summary.get('cropped_time_bins')}",
        f"- Window duration: {summary.get('window_seconds')} seconds",
        "",
        "## Why window matching is required",
        "Full-window baselines and cropped-window neural evaluations are not directly comparable "
        "because they can use different time bins, spike totals, and reference likelihoods. This "
        "comparison recomputes local methods on the same split, held-out mask, 256-bin crop, "
        "Poisson likelihood convention, and bits/spike convention before ranking methods.",
        "",
        "## Validation leaderboard",
        *_format_table(validation_leaderboard),
        "",
        "## Best validation method",
        f"- Method: {summary.get('best_method_name')}",
        f"- Prediction source: {summary.get('best_prediction_source')}",
        f"- Validation bits/spike: {summary.get('best_validation_bits_per_spike')}",
        f"- Validation Poisson NLL: {summary.get('best_validation_poisson_nll')}",
        f"- Behavior mean R2: {summary.get('best_behavior_mean_r2')}",
        "",
        "## Full-window references",
        "Warning: full-window numbers are not directly comparable to these cropped-window metrics.",
        "- Full-window mean-rate bits/spike: "
        f"{summary.get('full_window_mean_rate_bits_per_spike')}",
        "- Full-window factor latent best bits/spike: "
        f"{summary.get('full_window_factor_latent_best_bits_per_spike')}",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_window_matched_comparison_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    comparison_metrics: pd.DataFrame,
    validation_leaderboard: pd.DataFrame,
    behavior_comparison: pd.DataFrame,
) -> dict[str, Path]:
    """Write local window-matched comparison outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "comparison_summary.json",
        "comparison_metrics": output_dir / "comparison_metrics.csv",
        "validation_leaderboard": output_dir / "validation_leaderboard.csv",
        "behavior_comparison": output_dir / "behavior_comparison.csv",
        "report": output_dir / "comparison_report.md",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    comparison_metrics.to_csv(paths["comparison_metrics"], index=False)
    validation_leaderboard.to_csv(paths["validation_leaderboard"], index=False)
    behavior_comparison.to_csv(paths["behavior_comparison"], index=False)
    write_window_matched_comparison_report(paths["report"], summary, validation_leaderboard)
    return paths


def write_lfads_tuning_report(
    output_path: Path,
    summary: dict[str, Any],
    validation_leaderboard: pd.DataFrame,
) -> Path:
    """Write a Markdown report for local LFADS-style CUDA tuning."""
    refs = dict(summary.get("baseline_references", {}))
    best_params = json.loads(json.dumps(summary.get("best_run_params"), default=_json_default))
    lines = [
        f"# {summary.get('dataset_name')} LFADS-style GRU tuning report",
        "",
        "This is local validation tuning only, not an official NLB leaderboard result.",
        "The model is LFADS-style only, not full LFADS.",
        "Generated checkpoints are local and ignored by Git.",
        "",
        "## Dataset and window",
        f"- Dataset name: {summary.get('dataset_name')}",
        f"- Dataset hash: {summary.get('dataset_hash')}",
        f"- Window length: {summary.get('window_time_bins')} bins",
        f"- Window duration: {summary.get('window_seconds')} seconds",
        f"- CUDA device: {summary.get('cuda_device')}",
        "",
        "## Runs",
        f"- Runs attempted: {summary.get('runs_attempted')}",
        f"- Successful runs: {summary.get('successful_runs')}",
        f"- Best run ID: {summary.get('best_run_id')}",
        f"- Best run parameters: {best_params}",
        f"- Best validation bits/spike: {summary.get('best_validation_bits_per_spike')}",
        f"- Best validation Poisson NLL: {summary.get('best_validation_poisson_nll')}",
        f"- Best validation behavior mean R²: {summary.get('best_validation_behavior_mean_r2')}",
        "",
        "## Validation leaderboard",
        *_format_table(validation_leaderboard),
        "",
        "## Baseline comparisons",
        "- Window-matched mean-rate validation bits/spike: "
        f"{refs.get('window_matched_mean_rate_validation_bits_per_spike')}",
        f"- Beats window-matched mean-rate: {summary.get('beats_window_matched_mean_rate')}",
        "- Window-matched factor-latent validation bits/spike: "
        f"{refs.get('window_matched_factor_latent_validation_bits_per_spike')}",
        "- Beats window-matched factor-latent: "
        f"{summary.get('beats_window_matched_factor_latent')}",
        "- Previous LFADS-style masked direct validation bits/spike: "
        f"{refs.get('previous_lfads_masked_direct_validation_bits_per_spike')}",
        "- Beats previous LFADS-style masked direct: "
        f"{summary.get('beats_previous_lfads_masked_direct')}",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_lfads_tuning_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    tuning_results: pd.DataFrame,
    validation_leaderboard: pd.DataFrame,
    best_config: dict[str, Any],
) -> dict[str, Path]:
    """Write local LFADS-style tuning tables and report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "tuning_summary.json",
        "tuning_results": output_dir / "tuning_results.csv",
        "validation_leaderboard": output_dir / "validation_leaderboard.csv",
        "best_config": output_dir / "best_config.yaml",
        "report": output_dir / "tuning_report.md",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    tuning_results.to_csv(paths["tuning_results"], index=False)
    validation_leaderboard.to_csv(paths["validation_leaderboard"], index=False)
    paths["best_config"].write_text(yaml.safe_dump(best_config, sort_keys=False), encoding="utf-8")
    write_lfads_tuning_report(paths["report"], summary, validation_leaderboard)
    return paths


def write_lfads_audit_report(output_path: Path, summary: dict[str, Any]) -> Path:
    """Write a Markdown report for the local LFADS-style diagnostic audit."""
    flags = summary.get("likely_issue_flags", []) or ["none flagged"]
    lines = [
        f"# {summary.get('dataset_name')} LFADS-style diagnostic audit",
        "",
        "This is a local diagnostic audit, not an official NLB leaderboard result.",
        "The model is LFADS-style only, not full LFADS.",
        "Diagnostic overfit runs are local and not benchmark results.",
        "",
        "## Dataset and checkpoint",
        f"- Dataset name: {summary.get('dataset_name')}",
        f"- Dataset hash: {summary.get('dataset_hash')}",
        f"- Window length: {summary.get('window_time_bins')} bins",
        f"- CUDA device: {summary.get('cuda_device')}",
        f"- Checkpoint audited: {summary.get('checkpoint_audited')}",
        "",
        "## Validation summary",
        f"- Validation bits/spike: {summary.get('validation_bits_per_spike')}",
        "- Window-matched mean-rate reference: "
        f"{summary.get('mean_rate_reference_bits_per_spike')}",
        "- Window-matched factor-latent reference: "
        f"{summary.get('factor_latent_reference_bits_per_spike')}",
        "",
        "## Calibration summary",
        f"- Mean predicted held-out rate: {summary.get('mean_predicted_rate_hz')}",
        f"- Observed held-out rate: {summary.get('observed_rate_hz')}",
        f"- Prediction/reference correlation: {summary.get('prediction_reference_correlation')}",
        "",
        "## Factor usage summary",
        f"- Active factor count: {summary.get('active_factor_count')}",
        f"- Total factor count: {summary.get('total_factor_count')}",
        "",
        "## Tiny subset overfit",
        f"- Initial train loss: {summary.get('tiny_overfit_initial_loss')}",
        f"- Final train loss: {summary.get('tiny_overfit_final_loss')}",
        f"- Loss drop fraction: {summary.get('tiny_overfit_loss_drop_fraction')}",
        f"- Meets configured drop criterion: {summary.get('tiny_overfit_passed')}",
        "",
        "## Likely issue flags",
        *[f"- {flag}" for flag in flags],
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_lfads_audit_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    tables: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    """Write local LFADS-style audit JSON/CSV/Markdown artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "audit_summary.json",
        "split_diagnostics": output_dir / "split_diagnostics.csv",
        "neuron_diagnostics": output_dir / "neuron_diagnostics.csv",
        "rate_calibration": output_dir / "rate_calibration.csv",
        "loss_scale_diagnostics": output_dir / "loss_scale_diagnostics.csv",
        "tiny_subset_overfit": output_dir / "tiny_subset_overfit.csv",
        "factor_usage": output_dir / "factor_usage.csv",
        "report": output_dir / "audit_report.md",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    for name, path in paths.items():
        if name in {"summary", "report"}:
            continue
        tables.get(name, pd.DataFrame()).to_csv(path, index=False)
    write_lfads_audit_report(paths["report"], summary)
    return paths
