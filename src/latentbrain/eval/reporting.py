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


def write_temporal_rebinning_report(
    output_path: Path,
    summary: dict[str, Any],
    sparsity_by_bin_size: pd.DataFrame,
    baseline_metrics_by_bin_size: pd.DataFrame,
    lfads_metrics_by_bin_size: pd.DataFrame,
) -> Path:
    """Write a Markdown report for local temporal-binning diagnostics."""
    lines = [
        f"# {summary.get('dataset_name')} temporal rebinning diagnostic report",
        "",
        "This is local temporal-binning diagnostic work, not an official NLB leaderboard result.",
        "The model is LFADS-style only, not full LFADS.",
        (
            "Bits/spike values across different bin sizes are diagnostic and should not be "
            "treated as direct benchmark comparisons."
        ),
        "",
        "## Dataset and bins",
        f"- Dataset name: {summary.get('dataset_name')}",
        f"- Dataset hash: {summary.get('dataset_hash')}",
        f"- Original bin size: {summary.get('original_bin_size_ms')} ms",
        f"- Target bin sizes: {summary.get('target_bin_sizes_ms')} ms",
        f"- Fixed window duration: {summary.get('window_seconds')} seconds",
        "",
        "## Diagnostic conclusions",
        "- Coarser binning reduces zero fraction: "
        f"{summary.get('coarser_bins_reduce_zero_fraction')}",
        f"- LFADS improves at 10 ms or 20 ms: {summary.get('lfads_improves_at_coarser_bins')}",
        f"- Any LFADS run beats same-bin mean-rate: {summary.get('lfads_beat_same_bin_mean_rate')}",
        "",
        "## Sparsity table",
        *_format_table(sparsity_by_bin_size),
        "",
        "## Baseline table by bin size",
        *_format_table(baseline_metrics_by_bin_size),
        "",
        "## LFADS table by bin size",
        *_format_table(lfads_metrics_by_bin_size),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_temporal_rebinning_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    sparsity_by_bin_size: pd.DataFrame,
    baseline_metrics_by_bin_size: pd.DataFrame,
    lfads_metrics_by_bin_size: pd.DataFrame,
) -> dict[str, Path]:
    """Write local temporal-binning diagnostic artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "rebinning_summary.json",
        "sparsity": output_dir / "sparsity_by_bin_size.csv",
        "baseline_metrics": output_dir / "baseline_metrics_by_bin_size.csv",
        "lfads_metrics": output_dir / "lfads_metrics_by_bin_size.csv",
        "report": output_dir / "temporal_rebinning_report.md",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    sparsity_by_bin_size.to_csv(paths["sparsity"], index=False)
    baseline_metrics_by_bin_size.to_csv(paths["baseline_metrics"], index=False)
    lfads_metrics_by_bin_size.to_csv(paths["lfads_metrics"], index=False)
    write_temporal_rebinning_report(
        paths["report"],
        summary,
        sparsity_by_bin_size,
        baseline_metrics_by_bin_size,
        lfads_metrics_by_bin_size,
    )
    return paths


def write_lfads_rate_calibration_report(output_path: Path, summary: dict[str, Any]) -> Path:
    """Write a Markdown report for local LFADS-style rate calibration diagnostics."""
    lines = [
        f"# {summary.get('dataset_name')} LFADS-style rate calibration diagnostic",
        "",
        "This is local rate-calibration diagnostic work, not an official NLB leaderboard result.",
        "The model is LFADS-style only, not full LFADS.",
        "",
        "## Dataset and run",
        f"- Dataset name: {summary.get('dataset_name')}",
        f"- Dataset hash: {summary.get('dataset_hash')}",
        f"- Bin size: {summary.get('bin_size_ms')} ms",
        f"- Window length: {summary.get('window_seconds')} seconds",
        f"- CUDA device: {summary.get('cuda_device')}",
        f"- Existing checkpoint path: {summary.get('existing_checkpoint_path')}",
        "",
        "## Validation bits/spike",
        f"- Raw LFADS: {summary.get('raw_lfads_validation_bits_per_spike')}",
        "- Multiplicative calibrated LFADS: "
        f"{summary.get('multiplicative_calibrated_validation_bits_per_spike')}",
        "- Log-bias calibrated LFADS: "
        f"{summary.get('log_bias_calibrated_validation_bits_per_spike')}",
        f"- Best blend alpha: {summary.get('best_blend_alpha')}",
        f"- Best blend LFADS: {summary.get('best_blend_validation_bits_per_spike')}",
        f"- Initialized LFADS: {summary.get('initialized_lfads_validation_bits_per_spike')}",
        "- Initialized + calibrated LFADS: "
        f"{summary.get('initialized_calibrated_validation_bits_per_spike')}",
        "",
        "## Same-bin references",
        f"- Same-bin mean-rate reference: {summary.get('same_bin_mean_rate_reference')}",
        f"- Same-bin factor-latent reference: {summary.get('same_bin_factor_latent_reference')}",
        "",
        "## Conclusions",
        f"- Calibration improves LFADS: {summary.get('calibration_improves_lfads')}",
        f"- Initialization improves LFADS: {summary.get('initialization_improves_lfads')}",
        "- Any LFADS-family method beats same-bin factor-latent: "
        f"{summary.get('beats_same_bin_factor_latent')}",
        "- Any LFADS-family method beats same-bin mean-rate: "
        f"{summary.get('beats_same_bin_mean_rate')}",
        f"- Best LFADS-family method: {summary.get('best_lfads_family_method')}",
        "",
        "## Interpretation rules",
        "- If alpha near 0 is best, model dynamics are not adding useful held-out information.",
        "- If multiplicative/log-bias helps, rate scale calibration is an issue.",
        "- If initialized readout helps, poor output anchoring is an issue.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_lfads_rate_calibration_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    calibration_metrics: pd.DataFrame,
    blend_metrics: pd.DataFrame,
    initialized_lfads_metrics: pd.DataFrame,
) -> dict[str, Path]:
    """Write local LFADS-style rate calibration JSON/CSV/Markdown artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "rate_calibration_summary.json",
        "calibration_metrics": output_dir / "calibration_metrics.csv",
        "blend_metrics": output_dir / "blend_metrics.csv",
        "initialized_lfads_metrics": output_dir / "initialized_lfads_metrics.csv",
        "report": output_dir / "calibration_report.md",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    calibration_metrics.to_csv(paths["calibration_metrics"], index=False)
    blend_metrics.to_csv(paths["blend_metrics"], index=False)
    initialized_lfads_metrics.to_csv(paths["initialized_lfads_metrics"], index=False)
    write_lfads_rate_calibration_report(paths["report"], summary)
    return paths


def write_lfads_coordinated_dropout_report(
    output_path: Path,
    summary: dict[str, Any],
    evaluation_metrics: pd.DataFrame,
) -> Path:
    """Write a Markdown report for local LFADS-style input dropout diagnostics."""
    leaderboard = evaluation_metrics.sort_values(
        "validation_bits_per_spike", ascending=False, kind="mergesort"
    )
    lines = [
        f"# {summary.get('dataset_name')} LFADS-style coordinated dropout diagnostic",
        "",
        "This is local coordinated-dropout diagnostic training, not an official NLB "
        "leaderboard result.",
        "The model is LFADS-style only, not full LFADS.",
        "",
        "## Dataset and run",
        f"- Dataset name: {summary.get('dataset_name')}",
        f"- Dataset hash: {summary.get('dataset_hash')}",
        f"- Bin size: {summary.get('bin_size_ms')} ms",
        f"- Window length: {summary.get('window_seconds')} seconds",
        f"- CUDA device: {summary.get('cuda_device')}",
        f"- Dropout rates tested: {summary.get('dropout_rates_tested')}",
        "",
        "## Best run",
        f"- Best dropout rate: {summary.get('best_dropout_rate')}",
        f"- Best validation bits/spike: {summary.get('best_validation_bits_per_spike')}",
        f"- Best validation Poisson NLL: {summary.get('best_validation_poisson_nll')}",
        "- Best factor-decoder validation bits/spike: "
        f"{summary.get('best_validation_factor_decoder_bits_per_spike')}",
        "",
        "## Same-bin references",
        f"- Same-bin mean-rate reference: {summary.get('same_bin_mean_rate_reference')}",
        f"- Same-bin factor-latent reference: {summary.get('same_bin_factor_latent_reference')}",
        f"- Previous raw 20 ms LFADS reference: {summary.get('previous_20ms_lfads_reference')}",
        "",
        "## Conclusions",
        "- Coordinated dropout improves LFADS: "
        f"{summary.get('coordinated_dropout_improves_lfads')}",
        f"- Any run beats same-bin factor-latent: {summary.get('beats_same_bin_factor_latent')}",
        f"- Any run beats same-bin mean-rate: {summary.get('beats_same_bin_mean_rate')}",
        "",
        "## Validation leaderboard",
        "| run_id | dropout_rate | validation_bits_per_spike | validation_poisson_nll |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, row in leaderboard.iterrows():
        lines.append(
            f"| {row['run_id']} | {row['dropout_rate']} | "
            f"{row['validation_bits_per_spike']} | {row['validation_poisson_nll']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation rules",
            "- If low dropout helps, model benefits from mild robustness.",
            "- If high dropout hurts, input information is already limited.",
            "- If none help, underfitting/objective may still dominate.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_lfads_coordinated_dropout_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    training_metrics: pd.DataFrame,
    evaluation_metrics: pd.DataFrame,
    dropout_diagnostics: pd.DataFrame,
) -> dict[str, Path]:
    """Write local LFADS-style coordinated dropout JSON/CSV/Markdown artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "coordinated_dropout_summary.json",
        "training_metrics": output_dir / "training_metrics.csv",
        "evaluation_metrics": output_dir / "evaluation_metrics.csv",
        "dropout_diagnostics": output_dir / "dropout_diagnostics.csv",
        "report": output_dir / "coordinated_dropout_report.md",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    training_metrics.to_csv(paths["training_metrics"], index=False)
    evaluation_metrics.to_csv(paths["evaluation_metrics"], index=False)
    dropout_diagnostics.to_csv(paths["dropout_diagnostics"], index=False)
    write_lfads_coordinated_dropout_report(paths["report"], summary, evaluation_metrics)
    return paths


def write_metric_audit_report(
    output_path: Path,
    summary: dict[str, Any],
    unified_scores: pd.DataFrame,
    oracle_controls: pd.DataFrame,
    shuffled_controls: pd.DataFrame,
    reference_diagnostics: pd.DataFrame,
) -> Path:
    """Write a Markdown report for local metric/reference audit diagnostics."""
    formula = "(model_log_likelihood - reference_log_likelihood) / (log(2) * spike_count)"
    lines = [
        f"# {summary.get('dataset_name')} metric audit",
        "",
        "This is local metric-audit work, not an official NLB leaderboard result.",
        "Oracle controls are not valid models.",
        "",
        "## Dataset and scoring",
        f"- Dataset name: {summary.get('dataset_name')}",
        f"- Dataset hash: {summary.get('dataset_hash')}",
        f"- Bin size: {summary.get('bin_size_ms')} ms",
        f"- Window length: {summary.get('window_seconds')} seconds",
        f"- Unified bits/spike formula: {formula}",
        f"- Reference model used in unified scoring: {summary.get('reference_name')}",
        "",
        "## Required checks",
        "- Train-mean-as-model validation bits/spike: "
        f"{summary.get('train_mean_as_model_validation_bits_per_spike')}",
        "- Best oracle validation bits/spike: "
        f"{summary.get('best_oracle_validation_bits_per_spike')}",
        "- Previous mean-rate number directly comparable: "
        f"{summary.get('previous_mean_rate_directly_comparable')}",
        "",
        "## Conclusions",
        f"- Metric/reference mismatch found: {summary.get('metric_reference_mismatch_found')}",
        f"- Mean-rate inflation found: {summary.get('mean_rate_inflation_found')}",
        "- Neural models genuinely trail references under unified scoring: "
        f"{summary.get('neural_models_trail_under_unified_scoring')}",
        f"- Likely conclusion: {summary.get('likely_conclusion')}",
        "",
        "## Unified validation scores",
        "| method | source | bits/spike | reference log-likelihood | comparable |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    validation = unified_scores[
        unified_scores.get("split") == summary.get("primary_split", "validation")
    ]
    for _, row in validation.iterrows():
        lines.append(
            f"| {row.get('method_name')} | {row.get('prediction_source')} | "
            f"{row.get('bits_per_spike')} | {row.get('reference_log_likelihood')} | "
            f"{row.get('directly_comparable', True)} |"
        )
    lines.extend(["", "## Oracle-control scores"])
    if not oracle_controls.empty:
        lines.extend(
            [
                "| control | split | bits/spike | valid model |",
                "| --- | --- | ---: | --- |",
            ]
        )
        for _, row in oracle_controls.iterrows():
            lines.append(
                f"| {row.get('control_name')} | {row.get('split')} | "
                f"{row.get('bits_per_spike')} | {row.get('valid_model')} |"
            )
    lines.extend(["", "## Shuffled/random control scores"])
    if not shuffled_controls.empty:
        lines.extend(["| control | split | bits/spike |", "| --- | --- | ---: |"])
        for _, row in shuffled_controls.iterrows():
            lines.append(
                f"| {row.get('control_name')} | {row.get('split')} | {row.get('bits_per_spike')} |"
            )
    lines.extend(["", "## Existing reported scores"])
    if not reference_diagnostics.empty:
        lines.extend(
            [
                "| method | split | reported bits/spike | directly comparable |",
                "| --- | --- | ---: | --- |",
            ]
        )
        for _, row in reference_diagnostics.iterrows():
            lines.append(
                f"| {row.get('method_name')} | {row.get('split')} | "
                f"{row.get('reported_bits_per_spike')} | {row.get('directly_comparable')} |"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_metric_audit_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    unified_scores: pd.DataFrame,
    reference_diagnostics: pd.DataFrame,
    oracle_controls: pd.DataFrame,
    shuffled_controls: pd.DataFrame,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "metric_audit_summary.json",
        "unified_scores": output_dir / "unified_scores.csv",
        "reference_diagnostics": output_dir / "reference_diagnostics.csv",
        "oracle_controls": output_dir / "oracle_controls.csv",
        "shuffled_controls": output_dir / "shuffled_controls.csv",
        "report": output_dir / "metric_audit_report.md",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    unified_scores.to_csv(paths["unified_scores"], index=False)
    reference_diagnostics.to_csv(paths["reference_diagnostics"], index=False)
    oracle_controls.to_csv(paths["oracle_controls"], index=False)
    shuffled_controls.to_csv(paths["shuffled_controls"], index=False)
    write_metric_audit_report(
        paths["report"],
        summary,
        unified_scores,
        oracle_controls,
        shuffled_controls,
        reference_diagnostics,
    )
    return paths


def write_unified_scoreboard_report(
    output_path: Path,
    summary: dict[str, Any],
    validation_leaderboard: pd.DataFrame,
    historical_metric_notes: pd.DataFrame,
) -> Path:
    """Write a Markdown report for local unified scoring comparisons."""
    formula = "(model_log_likelihood - reference_log_likelihood) / (log(2) * spike_count)"
    lines = [
        f"# {summary.get('dataset_name')} unified scoreboard",
        "",
        "This is local unified scoring, not an official NLB leaderboard result.",
        "Old mean-rate values are historical-only and must not be used as direct targets.",
        *(
            [
                (
                    "Split audit reports high generalization risk. Current results should be "
                    "interpreted as validation-only diagnostics."
                ),
                "No model performance claim should be made until validation/test "
                "instability is resolved.",
            ]
            if str(summary.get("generalization_risk")) == "high"
            else []
        ),
        ("Generated local tuning summaries are ignored by Git and may be absent on a fresh clone."),
        (
            "If local tuning summaries are absent, the scoreboard falls back to configured "
            "known values. Fresh clones may need to rerun local tuning workflows to "
            "reproduce the latest LFADS/dynamics-family entries, including "
            "neural-SDE-style entries."
        ),
        "",
        "## Dataset and scoring",
        f"- Dataset name: {summary.get('dataset_name')}",
        f"- Dataset hash: {summary.get('dataset_hash')}",
        f"- Bin size: {summary.get('bin_size_ms')} ms",
        f"- Window length: {summary.get('window_seconds')} seconds",
        f"- Canonical bits/spike formula: {formula}",
        f"- Reference model: {summary.get('reference_model')}",
        "- Train-mean-as-model scores 0.0 bits/spike under canonical scoring.",
        "",
        "## Required checks",
        "- Train-mean validation bits/spike: "
        f"{summary.get('train_mean_validation_bits_per_spike')}",
        f"- Best valid model: {summary.get('best_valid_model')}",
        "- Best valid model validation bits/spike: "
        f"{summary.get('best_valid_model_validation_bits_per_spike')}",
        f"- Best LFADS-family method: {summary.get('best_lfads_family_method')}",
        "- Best LFADS-family validation bits/spike: "
        f"{summary.get('best_lfads_family_validation_bits_per_spike')}",
        f"- LFADS-family beats factor-latent: {summary.get('lfads_family_beats_factor_latent')}",
        f"- Generalization risk: {summary.get('generalization_risk')}",
        "- Validation/test instability detected: "
        f"{summary.get('validation_test_instability_detected')}",
        f"- Single-split results reportable: {summary.get('single_split_results_reportable')}",
        f"- Recommended reporting mode: {summary.get('recommended_reporting_mode')}",
        f"- Invalid rate controls present: {summary.get('invalid_rate_controls_present')}",
        f"- Rate offset warning: {summary.get('rate_offset_warning')}",
        f"- Stratified CV available: {summary.get('stratified_cv_available')}",
        f"- Factor-latent stratified CV mean: {summary.get('factor_latent_stratified_cv_mean')}",
        "- Factor-latent stratified CV CI95 low: "
        f"{summary.get('factor_latent_stratified_cv_ci95_low')}",
        f"- Window audit available: {summary.get('window_audit_available')}",
        f"- Recommended-window CV available: {summary.get('recommended_window_cv_available')}",
        f"- Recommended window: {summary.get('recommended_window_name')}",
        "- Factor-latent recommended-window mean: "
        f"{summary.get('factor_latent_recommended_window_mean')}",
        "- Factor-latent recommended-window CI95 low: "
        f"{summary.get('factor_latent_recommended_window_ci95_low')}",
        "- Factor-latent beats invalid-control mean: "
        f"{summary.get('factor_latent_beats_invalid_control_mean')}",
        f"- Current window still supported: {summary.get('current_window_still_supported')}",
        "- Best LFADS-family source summary path: "
        f"{summary.get('best_lfads_family_source_summary_path')}",
        "- Oracle diagnostic score: "
        f"{summary.get('oracle_validation_bits_per_spike')} (invalid model)",
        "",
        "## Unified validation leaderboard",
        "| rank | method | source | valid model | bits/spike | reference | oracle control |",
        "| ---: | --- | --- | --- | ---: | --- | --- |",
    ]
    for _, row in validation_leaderboard.iterrows():
        lines.append(
            f"| {row.get('rank')} | {row.get('method_name')} | "
            f"{row.get('prediction_source')} | {row.get('valid_model')} | "
            f"{row.get('validation_bits_per_spike')} | {row.get('reference_name')} | "
            f"{row.get('is_oracle_control')} |"
        )
    lines.extend(
        [
            "",
            "## Historical incompatible values",
            "| metric | value | status | reason |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for _, row in historical_metric_notes.iterrows():
        lines.append(
            f"| {row.get('metric_name')} | {row.get('value')} | "
            f"{row.get('status')} | {row.get('reason')} |"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_unified_scoreboard_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    validation_leaderboard: pd.DataFrame,
    split_scores: pd.DataFrame,
    historical_metric_notes: pd.DataFrame,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "unified_scoreboard_summary.json",
        "validation_leaderboard": output_dir / "unified_validation_leaderboard.csv",
        "split_scores": output_dir / "unified_split_scores.csv",
        "historical_metric_notes": output_dir / "historical_metric_notes.csv",
        "report": output_dir / "unified_scoreboard_report.md",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    validation_leaderboard.to_csv(paths["validation_leaderboard"], index=False)
    split_scores.to_csv(paths["split_scores"], index=False)
    historical_metric_notes.to_csv(paths["historical_metric_notes"], index=False)
    write_unified_scoreboard_report(
        paths["report"], summary, validation_leaderboard, historical_metric_notes
    )
    return paths


def write_lfads_unified_tuning_report(
    output_path: Path,
    summary: dict[str, Any],
    results: pd.DataFrame,
    leaderboard: pd.DataFrame,
) -> Path:
    lines = [
        f"# {summary.get('dataset_name')} LFADS-style unified tuning",
        "",
        "This is local canonical-metric tuning, not an official NLB leaderboard result.",
        "The model is LFADS-style only, not full LFADS.",
        "Old incompatible mean-rate values are not used as tuning targets.",
        "",
        "## Dataset and scoring",
        f"- Dataset name: {summary.get('dataset_name')}",
        f"- Dataset hash: {summary.get('dataset_hash')}",
        f"- Bin size: {summary.get('bin_size_ms')} ms",
        f"- Window length: {summary.get('window_seconds')} seconds",
        f"- CUDA device: {summary.get('cuda_device')}",
        f"- Canonical reference model: {summary.get('reference_model')}",
        "- Train-mean-as-model equals 0.0 bits/spike.",
        "",
        "## Selection",
        f"- Runs attempted: {summary.get('runs_attempted')}",
        f"- Successful runs: {summary.get('successful_runs')}",
        f"- Best run ID: {summary.get('best_run_id')}",
        f"- Best run parameters: {summary.get('best_run_params')}",
        "- Best validation unified bits/spike: "
        f"{summary.get('best_validation_unified_bits_per_spike')}",
        f"- Factor-latent unified reference: {summary.get('factor_latent_unified_reference')}",
        f"- Previous LFADS-family reference: {summary.get('previous_best_lfads_family_reference')}",
        f"- Beats factor-latent: {summary.get('beats_factor_latent_unified')}",
        f"- Beats previous LFADS-family result: {summary.get('beats_previous_best_lfads_family')}",
        "",
        "## Validation leaderboard",
        "| rank | run | bits/spike | poisson NLL | beats factor-latent | notes |",
        "| ---: | --- | ---: | ---: | --- | --- |",
    ]
    for _, row in leaderboard.iterrows():
        lines.append(
            f"| {row.get('rank')} | {row.get('run_id')} | "
            f"{row.get('validation_unified_bits_per_spike')} | "
            f"{row.get('validation_poisson_nll')} | "
            f"{row.get('beats_factor_latent_unified')} | {row.get('notes')} |"
        )
    if results.empty:
        lines.append("| | no successful runs | | | | |")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_lfads_unified_tuning_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    results: pd.DataFrame,
    leaderboard: pd.DataFrame,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "unified_tuning_summary.json",
        "results": output_dir / "unified_tuning_results.csv",
        "leaderboard": output_dir / "unified_validation_leaderboard.csv",
        "report": output_dir / "unified_tuning_report.md",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    results.to_csv(paths["results"], index=False)
    leaderboard.to_csv(paths["leaderboard"], index=False)
    write_lfads_unified_tuning_report(paths["report"], summary, results, leaderboard)
    return paths


def write_lfads_controller_tuning_report(
    output_path: Path,
    summary: dict[str, Any],
    results: pd.DataFrame,
    leaderboard: pd.DataFrame,
) -> Path:
    lines = [
        f"# {summary.get('dataset_name')} controller-style LFADS-family tuning",
        "",
        (
            "This is local controller-style LFADS-family tuning, "
            "not an official NLB leaderboard result."
        ),
        "The model is LFADS-style with inferred inputs, not full LFADS.",
        "Old incompatible mean-rate values are not used as tuning targets.",
        "",
        "## Dataset and scoring",
        f"- Dataset name: {summary.get('dataset_name')}",
        f"- Dataset hash: {summary.get('dataset_hash')}",
        f"- Bin size: {summary.get('bin_size_ms')} ms",
        f"- Window length: {summary.get('window_seconds')} seconds",
        f"- CUDA device: {summary.get('cuda_device')}",
        f"- Canonical reference model: {summary.get('reference_model')}",
        "- Train-mean-as-model equals 0.0 bits/spike.",
        "- Train-mean validation bits/spike: "
        f"{summary.get('train_mean_validation_bits_per_spike')}",
        "",
        "## Selection",
        f"- Runs attempted: {summary.get('runs_attempted')}",
        f"- Successful runs: {summary.get('successful_runs')}",
        f"- Best run ID: {summary.get('best_run_id')}",
        f"- Best run parameters: {summary.get('best_run_params')}",
        "- Best validation unified bits/spike: "
        f"{summary.get('best_validation_unified_bits_per_spike')}",
        f"- Factor-latent unified reference: {summary.get('factor_latent_unified_reference')}",
        f"- Previous LFADS-family reference: {summary.get('previous_best_lfads_family_reference')}",
        f"- Beats factor-latent: {summary.get('beats_factor_latent_unified')}",
        f"- Beats previous LFADS-family result: {summary.get('beats_previous_best_lfads_family')}",
        "",
        "## Inferred-input KL interpretation",
        "- near-zero KL may indicate posterior underuse.",
        "- very large KL may indicate overfitting or weak prior regularization.",
        "",
        "## Validation leaderboard",
        "| rank | run | bits/spike | poisson NLL | beats factor-latent | "
        "beats previous LFADS-family | notes |",
        "| ---: | --- | ---: | ---: | --- | --- | --- |",
    ]
    for _, row in leaderboard.iterrows():
        lines.append(
            f"| {row.get('rank')} | {row.get('run_id')} | "
            f"{row.get('validation_unified_bits_per_spike')} | "
            f"{row.get('validation_poisson_nll')} | "
            f"{row.get('beats_factor_latent_unified')} | "
            f"{row.get('beats_previous_best_lfads_family')} | {row.get('notes')} |"
        )
    if results.empty:
        lines.append("| | no successful runs | | | | | |")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_lfads_controller_tuning_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    results: pd.DataFrame,
    leaderboard: pd.DataFrame,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "controller_tuning_summary.json",
        "results": output_dir / "controller_tuning_results.csv",
        "leaderboard": output_dir / "controller_validation_leaderboard.csv",
        "report": output_dir / "controller_tuning_report.md",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    results.to_csv(paths["results"], index=False)
    leaderboard.to_csv(paths["leaderboard"], index=False)
    write_lfads_controller_tuning_report(paths["report"], summary, results, leaderboard)
    return paths


def write_neural_sde_tuning_report(
    output_path: Path,
    summary: dict[str, Any],
    leaderboard: pd.DataFrame,
) -> Path:
    lines = [
        f"# {summary.get('dataset_name')} neural-SDE-style latent generator tuning",
        "",
        "This is local neural-SDE-style tuning, not an official NLB leaderboard result.",
        (
            "This is a compact Euler/Euler-Maruyama latent generator, "
            "not a full benchmarked neural SDE system."
        ),
        "Old incompatible mean-rate values are not used as tuning targets.",
        "",
        "## Dataset and scoring",
        f"- Dataset name: {summary.get('dataset_name')}",
        f"- Dataset hash: {summary.get('dataset_hash')}",
        f"- Bin size: {summary.get('bin_size_ms')} ms",
        f"- Window length: {summary.get('window_seconds')} seconds",
        f"- CUDA device: {summary.get('cuda_device')}",
        f"- Canonical reference model: {summary.get('reference_model')}",
        "- Train-mean-as-model equals 0.0 bits/spike.",
        "- Train-mean validation bits/spike: "
        f"{summary.get('train_mean_validation_bits_per_spike')}",
        "",
        "## Selection",
        f"- Runs attempted: {summary.get('runs_attempted')}",
        f"- Successful runs: {summary.get('successful_runs')}",
        f"- Best run ID: {summary.get('best_run_id')}",
        f"- Best run parameters: {summary.get('best_run_params')}",
        "- Best validation unified bits/spike: "
        f"{summary.get('best_validation_unified_bits_per_spike')}",
        f"- Best validation Poisson NLL: {summary.get('best_validation_poisson_nll')}",
        "- Best factor-decoder unified bits/spike: "
        f"{summary.get('best_factor_decoder_unified_bits_per_spike')}",
        f"- Factor-latent unified reference: {summary.get('factor_latent_unified_reference')}",
        "- Previous LFADS-family/controller reference: "
        f"{summary.get('previous_best_lfads_family_reference')}",
        f"- Beats factor-latent: {summary.get('beats_factor_latent_unified')}",
        f"- Beats previous LFADS-family result: {summary.get('beats_previous_best_lfads_family')}",
        "",
        "## Drift/diffusion diagnostics",
        f"- Best drift norm: {summary.get('best_drift_norm')}",
        f"- Best diffusion mean: {summary.get('best_diffusion_mean')}",
        "- diffusion scale 0 is deterministic neural ODE-style dynamics.",
        "- nonzero diffusion tests stochastic latent dynamics.",
        "- near-zero diffusion may indicate deterministic dynamics are enough.",
        "- high diffusion with worse validation may indicate noisy latent paths.",
        "",
        "## Validation leaderboard",
        "| rank | run | bits/spike | poisson NLL | diffusion scale | "
        "beats factor-latent | beats previous LFADS-family | notes |",
        "| ---: | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for _, row in leaderboard.iterrows():
        lines.append(
            f"| {row.get('rank')} | {row.get('run_id')} | "
            f"{row.get('validation_unified_bits_per_spike')} | "
            f"{row.get('validation_poisson_nll')} | "
            f"{row.get('diffusion_scale')} | "
            f"{row.get('beats_factor_latent_unified')} | "
            f"{row.get('beats_previous_best_lfads_family')} | {row.get('notes')} |"
        )
    if leaderboard.empty:
        lines.append("| | no successful runs | | | | | | |")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_neural_sde_tuning_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    results: pd.DataFrame,
    leaderboard: pd.DataFrame,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "neural_sde_tuning_summary.json",
        "results": output_dir / "neural_sde_tuning_results.csv",
        "leaderboard": output_dir / "neural_sde_validation_leaderboard.csv",
        "report": output_dir / "neural_sde_tuning_report.md",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    results.to_csv(paths["results"], index=False)
    leaderboard.to_csv(paths["leaderboard"], index=False)
    write_neural_sde_tuning_report(paths["report"], summary, leaderboard)
    return paths


def write_neural_ode_tuning_report(
    output_path: Path,
    summary: dict[str, Any],
    leaderboard: pd.DataFrame,
    checkpoint_scores: pd.DataFrame,
) -> Path:
    lines = [
        f"# {summary.get('dataset_name')} deterministic neural-ODE-style latent dynamics tuning",
        "",
        (
            "This is local deterministic neural-ODE-style tuning, "
            "not an official NLB leaderboard result."
        ),
        ("This is a compact Euler latent generator, not a full benchmarked neural ODE/SDE system."),
        "Old incompatible mean-rate values are not used as tuning targets.",
        "",
        "## Dataset and scoring",
        f"- Dataset name: {summary.get('dataset_name')}",
        f"- Dataset hash: {summary.get('dataset_hash')}",
        f"- Bin size: {summary.get('bin_size_ms')} ms",
        f"- Window length: {summary.get('window_seconds')} seconds",
        f"- CUDA device: {summary.get('cuda_device')}",
        f"- Canonical reference model: {summary.get('reference_model')}",
        "- Train-mean-as-model equals 0.0 bits/spike.",
        "- Train-mean validation bits/spike: "
        f"{summary.get('train_mean_validation_bits_per_spike')}",
        "",
        "## Selection",
        f"- Runs attempted: {summary.get('runs_attempted')}",
        f"- Successful runs: {summary.get('successful_runs')}",
        f"- Best run ID: {summary.get('best_run_id')}",
        f"- Best run parameters: {summary.get('best_run_params')}",
        "- Best validation unified bits/spike: "
        f"{summary.get('best_validation_unified_bits_per_spike')}",
        f"- Best validation Poisson NLL: {summary.get('best_validation_poisson_nll')}",
        "- Best factor-decoder unified bits/spike: "
        f"{summary.get('best_factor_decoder_unified_bits_per_spike')}",
        f"- Best checkpoint source: {summary.get('best_checkpoint_source')}",
        f"- Checkpoint selection: {summary.get('checkpoint_selection_method')}",
        f"- Factor-latent unified reference: {summary.get('factor_latent_unified_reference')}",
        f"- Previous neural-SDE reference: {summary.get('previous_neural_sde_reference')}",
        "- Previous LFADS/controller reference: "
        f"{summary.get('previous_best_lfads_family_reference')}",
        f"- Beats factor-latent: {summary.get('beats_factor_latent_unified')}",
        f"- Beats previous neural-SDE: {summary.get('beats_previous_neural_sde')}",
        "",
        "## Drift diagnostics",
        f"- Best drift norm: {summary.get('best_drift_norm')}",
        f"- Best diffusion mean: {summary.get('best_diffusion_mean')}",
        "- Diffusion scale is forced to 0.0 for deterministic latent dynamics.",
        "",
        "## Interpretation",
        (
            "- Deterministic latent dynamics are tested because diffusion scale zero "
            "won the previous neural-SDE-style tuning."
        ),
        (
            "- If deterministic tuning beats neural-SDE, stochastic diffusion was "
            "unnecessary for this dataset/window."
        ),
        (
            "- If deterministic tuning beats factor-latent, move to robustness/"
            "multiple-seed validation before rSLDS."
        ),
        "",
        "## Validation leaderboard",
        (
            "| rank | run | bits/spike | poisson NLL | checkpoint | "
            "beats factor-latent | beats previous neural-SDE | notes |"
        ),
        "| ---: | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for _, row in leaderboard.iterrows():
        lines.append(
            f"| {row.get('rank')} | {row.get('run_id')} | "
            f"{row.get('validation_unified_bits_per_spike')} | "
            f"{row.get('validation_poisson_nll')} | "
            f"{row.get('best_checkpoint_source')} | "
            f"{row.get('beats_factor_latent_unified')} | "
            f"{row.get('beats_previous_neural_sde')} | {row.get('notes')} |"
        )
    if leaderboard.empty:
        lines.append("| | no successful runs | | | | | | |")
    lines.extend(
        [
            "",
            "## Checkpoint selection",
            (
                "| run | source | epoch | validation loss | validation bits/spike | "
                "selected by loss | selected by unified |"
            ),
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for _, row in checkpoint_scores.iterrows():
        lines.append(
            f"| {row.get('run_id', '')} | {row.get('checkpoint_source')} | "
            f"{row.get('epoch')} | {row.get('validation_total_loss')} | "
            f"{row.get('validation_unified_bits_per_spike')} | "
            f"{row.get('selected_by_loss')} | {row.get('selected_by_unified')} |"
        )
    if checkpoint_scores.empty:
        lines.append("| | no checkpoint scores | | | | | |")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_neural_ode_tuning_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    results: pd.DataFrame,
    leaderboard: pd.DataFrame,
    checkpoint_scores: pd.DataFrame,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "neural_ode_tuning_summary.json",
        "results": output_dir / "neural_ode_tuning_results.csv",
        "leaderboard": output_dir / "neural_ode_validation_leaderboard.csv",
        "checkpoint_selection": output_dir / "checkpoint_selection.csv",
        "report": output_dir / "neural_ode_tuning_report.md",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    results.to_csv(paths["results"], index=False)
    leaderboard.to_csv(paths["leaderboard"], index=False)
    checkpoint_scores.to_csv(paths["checkpoint_selection"], index=False)
    write_neural_ode_tuning_report(paths["report"], summary, leaderboard, checkpoint_scores)
    return paths


def write_switching_ode_tuning_report(
    output_path: Path,
    summary: dict[str, Any],
    leaderboard: pd.DataFrame,
    regime_diagnostics: pd.DataFrame,
) -> Path:
    occupancy = regime_diagnostics[regime_diagnostics["split"] == "validation"]
    one_regime_dominates = float(summary.get("best_max_regime_occupancy") or 0.0) > 0.8
    improves = bool(summary.get("beats_previous_neural_ode"))
    lines = [
        f"# {summary.get('dataset_name')} switching neural-ODE-style latent dynamics tuning",
        "",
        (
            "This is local rSLDS-style switching-dynamics tuning, "
            "not an official NLB leaderboard result."
        ),
        "This is a soft switching neural-ODE-style model, not full Bayesian rSLDS inference.",
        "Old incompatible mean-rate values are not used as tuning targets.",
        "",
        "## Dataset and scoring",
        f"- Dataset name: {summary.get('dataset_name')}",
        f"- Dataset hash: {summary.get('dataset_hash')}",
        f"- Bin size: {summary.get('bin_size_ms')} ms",
        f"- Window length: {summary.get('window_seconds')} seconds",
        f"- CUDA device: {summary.get('cuda_device')}",
        f"- Canonical reference model: {summary.get('reference_model')}",
        "- Train-mean-as-model equals 0.0 bits/spike.",
        "- Train-mean validation bits/spike: "
        f"{summary.get('train_mean_validation_bits_per_spike')}",
        "",
        "## Selection",
        f"- Runs attempted: {summary.get('runs_attempted')}",
        f"- Successful runs: {summary.get('successful_runs')}",
        f"- Best run ID: {summary.get('best_run_id')}",
        f"- Best run parameters: {summary.get('best_run_params')}",
        "- Best validation unified bits/spike: "
        f"{summary.get('best_validation_unified_bits_per_spike')}",
        f"- Best validation Poisson NLL: {summary.get('best_validation_poisson_nll')}",
        "- Best factor-decoder unified bits/spike: "
        f"{summary.get('best_factor_decoder_unified_bits_per_spike')}",
        f"- Best checkpoint source: {summary.get('best_checkpoint_source')}",
        f"- Factor-latent unified reference: {summary.get('factor_latent_unified_reference')}",
        f"- Previous neural-ODE reference: {summary.get('previous_neural_ode_reference')}",
        f"- Previous neural-SDE reference: {summary.get('previous_neural_sde_reference')}",
        f"- Beats factor-latent: {summary.get('beats_factor_latent_unified')}",
        f"- Beats previous neural-ODE: {summary.get('beats_previous_neural_ode')}",
        "",
        "## Regime diagnostics",
        f"- Active regime count: {summary.get('best_active_regime_count')}",
        f"- Mean regime entropy: {summary.get('best_mean_regime_entropy')}",
        f"- Max regime occupancy: {summary.get('best_max_regime_occupancy')}",
        "",
        "### Validation occupancy table",
        *_format_table(occupancy),
        "",
        "## Interpretation",
        "- If one regime dominates, switching did not add meaningful dynamics.",
        "- If multiple regimes are active and score improves, switching dynamics may be useful.",
        "- If switching beats factor-latent, next step is multi-seed robustness before claims.",
        f"- One-regime-dominates diagnostic: {one_regime_dominates}",
        f"- Multiple-regime improvement diagnostic: {not one_regime_dominates and improves}",
        "",
        "## Validation leaderboard",
        (
            "| rank | run | bits/spike | poisson NLL | regimes | entropy | checkpoint | "
            "beats factor-latent | beats previous neural-ODE | notes |"
        ),
        "| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for _, row in leaderboard.iterrows():
        lines.append(
            f"| {row.get('rank')} | {row.get('run_id')} | "
            f"{row.get('validation_unified_bits_per_spike')} | "
            f"{row.get('validation_poisson_nll')} | {row.get('active_regime_count')} | "
            f"{row.get('mean_regime_entropy')} | {row.get('best_checkpoint_source')} | "
            f"{row.get('beats_factor_latent_unified')} | "
            f"{row.get('beats_previous_neural_ode')} | {row.get('notes')} |"
        )
    if leaderboard.empty:
        lines.append("| | no successful runs | | | | | | | | |")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_switching_ode_tuning_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    results: pd.DataFrame,
    leaderboard: pd.DataFrame,
    checkpoint_scores: pd.DataFrame,
    regime_diagnostics: pd.DataFrame,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "switching_ode_tuning_summary.json",
        "results": output_dir / "switching_ode_tuning_results.csv",
        "leaderboard": output_dir / "switching_ode_validation_leaderboard.csv",
        "regime_diagnostics": output_dir / "regime_diagnostics.csv",
        "checkpoint_selection": output_dir / "checkpoint_selection.csv",
        "report": output_dir / "switching_ode_tuning_report.md",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    results.to_csv(paths["results"], index=False)
    leaderboard.to_csv(paths["leaderboard"], index=False)
    checkpoint_scores.to_csv(paths["checkpoint_selection"], index=False)
    regime_diagnostics.to_csv(paths["regime_diagnostics"], index=False)
    write_switching_ode_tuning_report(paths["report"], summary, leaderboard, regime_diagnostics)
    return paths


def write_neural_ode_refinement_report(
    output_path: Path,
    summary: dict[str, Any],
    leaderboard: pd.DataFrame,
    checkpoint_scores: pd.DataFrame,
) -> Path:
    lines = [
        f"# {summary.get('dataset_name')} deterministic neural-ODE refinement",
        "",
        (
            "This is local deterministic neural-ODE refinement, "
            "not an official NLB leaderboard result."
        ),
        "This is a compact Euler latent generator, not a full benchmarked neural ODE/SDE system.",
        "Old incompatible mean-rate values are not used as tuning targets.",
        "",
        "## Dataset and scoring",
        f"- Dataset name: {summary.get('dataset_name')}",
        f"- Dataset hash: {summary.get('dataset_hash')}",
        f"- Bin size: {summary.get('bin_size_ms')} ms",
        f"- Window length: {summary.get('window_seconds')} seconds",
        f"- CUDA device: {summary.get('cuda_device')}",
        f"- Canonical reference model: {summary.get('reference_model')}",
        "- Train-mean-as-model equals 0.0 bits/spike.",
        "- Train-mean validation bits/spike: "
        f"{summary.get('train_mean_validation_bits_per_spike')}",
        "",
        "## Selection",
        f"- Runs attempted: {summary.get('runs_attempted')}",
        f"- Successful runs: {summary.get('successful_runs')}",
        f"- Best run ID: {summary.get('best_run_id')}",
        f"- Best run parameters: {summary.get('best_run_params')}",
        "- Best validation unified bits/spike: "
        f"{summary.get('best_validation_unified_bits_per_spike')}",
        f"- Best validation Poisson NLL: {summary.get('best_validation_poisson_nll')}",
        "- Best factor-decoder unified bits/spike: "
        f"{summary.get('best_factor_decoder_unified_bits_per_spike')}",
        f"- Best checkpoint source: {summary.get('best_checkpoint_source')}",
        f"- Factor-latent unified reference: {summary.get('factor_latent_unified_reference')}",
        f"- Previous neural-ODE reference: {summary.get('previous_neural_ode_reference')}",
        f"- Switching ODE reference: {summary.get('previous_switching_ode_reference')}",
        f"- Beats factor-latent: {summary.get('beats_factor_latent_unified')}",
        f"- Beats previous neural-ODE: {summary.get('beats_previous_neural_ode')}",
        "",
        "## Drift regularization diagnostics",
        f"- Drift norm: {summary.get('best_drift_norm')}",
        f"- Drift regularization loss: {summary.get('best_drift_regularization_loss')}",
        f"- Diffusion mean: {summary.get('best_diffusion_mean')}",
        "",
        "## Scheduler / learning-rate summary",
        f"- Final selected-checkpoint learning rate: {summary.get('best_learning_rate')}",
        f"- Final learning rate: {summary.get('final_learning_rate')}",
        "",
        "## Checkpoint selection",
        *_format_table(checkpoint_scores),
        "",
        "## Validation leaderboard",
        (
            "| rank | run | bits/spike | poisson NLL | factor bits | drift reg | "
            "scheduler | checkpoint | beats factor-latent | beats previous neural-ODE | notes |"
        ),
        "| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for _, row in leaderboard.iterrows():
        lines.append(
            f"| {row.get('rank')} | {row.get('run_id')} | "
            f"{row.get('validation_unified_bits_per_spike')} | "
            f"{row.get('validation_poisson_nll')} | "
            f"{row.get('validation_factor_decoder_unified_bits_per_spike')} | "
            f"{row.get('drift_regularization')} | {row.get('scheduler')} | "
            f"{row.get('best_checkpoint_source')} | "
            f"{row.get('beats_factor_latent_unified')} | "
            f"{row.get('beats_previous_neural_ode')} | {row.get('notes')} |"
        )
    if leaderboard.empty:
        lines.append("| | no successful runs | | | | | | | | | |")
    lines.extend(
        [
            "",
            "## Interpretation",
            (
                "- The switching collapsed to one regime, so this workflow refines "
                "deterministic dynamics rather than adding regimes."
            ),
            (
                "- If refinement beats factor-latent, next step is multi-seed robustness "
                "before claims."
            ),
            (
                "- If refinement does not beat factor-latent, next step is objective redesign "
                "or multiple datasets, not more architecture."
            ),
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_neural_ode_refinement_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    results: pd.DataFrame,
    leaderboard: pd.DataFrame,
    checkpoint_scores: pd.DataFrame,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "neural_ode_refinement_summary.json",
        "results": output_dir / "neural_ode_refinement_results.csv",
        "leaderboard": output_dir / "neural_ode_refinement_leaderboard.csv",
        "checkpoint_selection": output_dir / "checkpoint_selection.csv",
        "report": output_dir / "neural_ode_refinement_report.md",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    results.to_csv(paths["results"], index=False)
    leaderboard.to_csv(paths["leaderboard"], index=False)
    checkpoint_scores.to_csv(paths["checkpoint_selection"], index=False)
    write_neural_ode_refinement_report(paths["report"], summary, leaderboard, checkpoint_scores)
    return paths


def write_neural_ode_objective_report(
    output_path: Path,
    summary: dict[str, Any],
    leaderboard: pd.DataFrame,
    objective_diagnostics: pd.DataFrame,
    checkpoint_scores: pd.DataFrame,
) -> Path:
    """Write a Markdown report for local deterministic neural-ODE objective diagnostics."""
    lines = [
        f"# {summary.get('dataset_name')} deterministic neural-ODE objective diagnostics",
        "",
        (
            "This is local deterministic neural-ODE objective diagnostics, "
            "not an official NLB leaderboard result."
        ),
        (
            "Evaluation uses canonical unweighted unified bits/spike even when training "
            "losses are weighted."
        ),
        "Old incompatible mean-rate values are not used as tuning targets.",
        "",
        "## Dataset and scoring",
        f"- Dataset name: {summary.get('dataset_name')}",
        f"- Dataset hash: {summary.get('dataset_hash')}",
        f"- Bin size: {summary.get('bin_size_ms')} ms",
        f"- Window length: {summary.get('window_seconds')} seconds",
        f"- CUDA device: {summary.get('cuda_device')}",
        f"- Canonical reference model: {summary.get('reference_model')}",
        "- Train-mean-as-model equals 0.0 bits/spike.",
        "- Train-mean validation bits/spike: "
        f"{summary.get('train_mean_validation_bits_per_spike')}",
        f"- Shared seed across variants: {summary.get('shared_seed')}",
        (
            "- All variants share one seed, so score differences are attributable to the "
            "objective and not to initialization."
        ),
        "",
        "## Selection",
        f"- Runs attempted: {summary.get('runs_attempted')}",
        f"- Successful runs: {summary.get('successful_runs')}",
        f"- Best run ID: {summary.get('best_run_id')}",
        f"- Best objective name: {summary.get('best_objective_name')}",
        f"- Best objective parameters: {summary.get('best_run_params')}",
        "- Best validation unified bits/spike: "
        f"{summary.get('best_validation_unified_bits_per_spike')}",
        f"- Best validation Poisson NLL: {summary.get('best_validation_poisson_nll')}",
        "- Best factor-decoder unified bits/spike: "
        f"{summary.get('best_factor_decoder_unified_bits_per_spike')}",
        f"- Best checkpoint source: {summary.get('best_checkpoint_source')}",
        f"- Factor-latent unified reference: {summary.get('factor_latent_unified_reference')}",
        "- Previous neural-ODE refinement reference: "
        f"{summary.get('previous_neural_ode_refinement_reference')}",
        f"- Switching ODE reference: {summary.get('switching_ode_reference')}",
        f"- Beats factor-latent: {summary.get('beats_factor_latent_unified')}",
        "- Beats previous neural-ODE refinement: "
        f"{summary.get('beats_previous_neural_ode_refinement')}",
        "",
        "## Objective diagnostics",
        f"- Best held-out loss weight: {summary.get('best_heldout_loss_weight')}",
        f"- Best zero count weight: {summary.get('best_zero_count_weight')}",
        f"- Best positive count weight: {summary.get('best_positive_count_weight')}",
        f"- Best rate calibration loss weight: {summary.get('best_rate_calibration_loss_weight')}",
        f"- Best rate calibration loss: {summary.get('best_rate_calibration_loss')}",
        f"- Drift norm: {summary.get('best_drift_norm')}",
        f"- Drift regularization loss: {summary.get('best_drift_regularization_loss')}",
        f"- Diffusion mean: {summary.get('best_diffusion_mean')}",
        "",
        *_format_table(objective_diagnostics),
        "",
        "## Checkpoint selection",
        *_format_table(checkpoint_scores),
        "",
        "## Validation leaderboard",
        (
            "| rank | run | objective | bits/spike | poisson NLL | factor bits | heldout weight | "
            "zero weight | positive weight | rate calibration | checkpoint | "
            "beats factor-latent | beats previous refinement | notes |"
        ),
        (
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
            "| --- | --- | --- | --- |"
        ),
    ]
    for _, row in leaderboard.iterrows():
        lines.append(
            f"| {row.get('rank')} | {row.get('run_id')} | {row.get('objective_name')} | "
            f"{row.get('validation_unified_bits_per_spike')} | "
            f"{row.get('validation_poisson_nll')} | "
            f"{row.get('validation_factor_decoder_unified_bits_per_spike')} | "
            f"{row.get('heldout_loss_weight')} | {row.get('zero_count_weight')} | "
            f"{row.get('positive_count_weight')} | "
            f"{row.get('rate_calibration_loss_weight')} | "
            f"{row.get('best_checkpoint_source')} | "
            f"{row.get('beats_factor_latent_unified')} | "
            f"{row.get('beats_previous_neural_ode_refinement')} | {row.get('notes')} |"
        )
    if leaderboard.empty:
        lines.append("| | no successful runs | | | | | | | | | | | | |")
    lines.extend(
        [
            "",
            "## Interpretation",
            ("- If held-out-heavy objectives help, the previous model underweighted co-smoothing."),
            ("- If zero-downweighting helps, sparse-count imbalance was limiting training."),
            "- If rate calibration hurts, the output scale was already adequate.",
            (
                "- If none beat factor-latent, the next step should be multi-seed/local "
                "robustness of the best dynamics model plus expanded baselines/datasets, "
                "not more architecture."
            ),
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_neural_ode_objective_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    results: pd.DataFrame,
    leaderboard: pd.DataFrame,
    objective_diagnostics: pd.DataFrame,
    checkpoint_scores: pd.DataFrame,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "neural_ode_objective_summary.json",
        "results": output_dir / "neural_ode_objective_results.csv",
        "leaderboard": output_dir / "neural_ode_objective_leaderboard.csv",
        "objective_diagnostics": output_dir / "objective_diagnostics.csv",
        "checkpoint_selection": output_dir / "checkpoint_selection.csv",
        "report": output_dir / "neural_ode_objective_report.md",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    results.to_csv(paths["results"], index=False)
    leaderboard.to_csv(paths["leaderboard"], index=False)
    objective_diagnostics.to_csv(paths["objective_diagnostics"], index=False)
    checkpoint_scores.to_csv(paths["checkpoint_selection"], index=False)
    write_neural_ode_objective_report(
        paths["report"], summary, leaderboard, objective_diagnostics, checkpoint_scores
    )
    return paths


def write_seed_robustness_report(
    output_path: Path,
    summary: dict[str, Any],
    method_summary: pd.DataFrame,
    leaderboard: pd.DataFrame,
    seed_effects: pd.DataFrame,
) -> Path:
    """Write a Markdown report for local multi-seed robustness analysis."""
    lines = [
        f"# {summary.get('dataset_name')} multi-seed robustness",
        "",
        "This is local multi-seed robustness analysis, not an official NLB leaderboard result.",
        "Single-seed model leaderboards are not sufficient for claims.",
        "Old incompatible mean-rate values are not used as tuning targets.",
        "",
        "## Dataset and scoring",
        f"- Dataset name: {summary.get('dataset_name')}",
        f"- Dataset hash: {summary.get('dataset_hash')}",
        f"- Bin size: {summary.get('bin_size_ms')} ms",
        f"- Window length: {summary.get('window_seconds')} seconds",
        f"- CUDA device: {summary.get('cuda_device')}",
        f"- Canonical reference model: {summary.get('reference_model')}",
        "- Train-mean-as-model equals 0.0 bits/spike.",
        "- Train-mean validation bits/spike: "
        f"{summary.get('train_mean_validation_bits_per_spike')}",
        "- Evaluation uses canonical unweighted unified bits/spike.",
        "",
        "## Seed policy",
        f"- Split seed mode: {summary.get('split_seed_mode')}",
        f"- Fixed split seed: {summary.get('split_seed')}",
        f"- Initialization seed mode: {summary.get('initialization_seed_mode')}",
        f"- Same seed list across methods: {summary.get('seed_list_shared_across_methods')}",
        (
            "- The trial split and held-in/held-out neuron mask are held fixed across all "
            "methods and seeds, so score spread reflects initialization and training "
            "variance only."
        ),
        (
            "- No seed is derived from a run index. Objective diagnostics previously used "
            "`seed + run_index`, which confounded the method with its initialization."
        ),
        "",
        "## Methods and seeds",
        f"- Methods evaluated: {summary.get('methods_evaluated')}",
        f"- Seeds evaluated: {summary.get('seeds_evaluated')}",
        f"- Total jobs: {summary.get('total_jobs')}",
        f"- Successful jobs: {summary.get('successful_jobs')}",
        f"- Method config hashes: {summary.get('method_config_hashes')}",
        f"- Confidence interval: {summary.get('confidence_interval')}",
        f"- Bootstrap repeats: {summary.get('bootstrap_repeats')}",
        f"- Bootstrap seed: {summary.get('bootstrap_seed')}",
        "",
        "## Method summary (mean, standard deviation, bootstrap CI)",
        *_format_table(method_summary),
        "",
        "## Validation leaderboard",
        *_format_table(leaderboard),
        "",
        "## Paired seed differences against factor-latent",
        *_format_table(seed_effects),
        "",
        "## Outcome",
        f"- Best mean method: {summary.get('best_mean_method')}",
        "- Best mean validation unified bits/spike: "
        f"{summary.get('best_mean_validation_unified_bits_per_spike')}",
        f"- Best lower-CI method: {summary.get('best_lower_ci_method')}",
        "- Best lower-CI validation unified bits/spike: "
        f"{summary.get('best_lower_ci_validation_unified_bits_per_spike')}",
        "- Factor-latent mean validation unified bits/spike: "
        f"{summary.get('factor_latent_mean_validation_unified_bits_per_spike')}",
        f"- Best neural method: {summary.get('best_neural_method')}",
        "- Best neural method mean validation unified bits/spike: "
        f"{summary.get('best_neural_method_mean_validation_unified_bits_per_spike')}",
        "- Paired mean difference (best neural minus factor-latent): "
        f"{summary.get('paired_mean_difference_best_neural_minus_factor_latent')}",
        "- Any neural method beats factor-latent by mean: "
        f"{summary.get('any_neural_beats_factor_latent_mean')}",
        "- Any neural method beats factor-latent by lower CI: "
        f"{summary.get('any_neural_beats_factor_latent_lower_ci')}",
        "",
        "## Carried-forward recommendation",
        f"- Carried-forward method: {summary.get('carried_forward_method')}",
        f"- Reason: {summary.get('carried_forward_reason')}",
        "",
        "## Interpretation",
        (
            "- If neural ODE does not beat factor-latent across seeds, stop adding "
            "architecture on this dataset/window."
        ),
        (
            "- If neural ODE beats factor-latent by mean but not lower CI, run more seeds "
            "before claims."
        ),
        (
            "- If neural ODE beats factor-latent by lower CI, move to held-out test "
            "reporting and additional datasets."
        ),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_seed_robustness_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    results: pd.DataFrame,
    method_summary: pd.DataFrame,
    leaderboard: pd.DataFrame,
    seed_effects: pd.DataFrame,
    carried_forward_config: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "seed_robustness_summary.json",
        "results": output_dir / "seed_robustness_results.csv",
        "leaderboard": output_dir / "seed_robustness_leaderboard.csv",
        "seed_effects": output_dir / "seed_effects.csv",
        "method_summary": output_dir / "method_summary.csv",
        "carried_forward_config": output_dir / "carried_forward_config.yaml",
        "report": output_dir / "seed_robustness_report.md",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    results.to_csv(paths["results"], index=False)
    leaderboard.to_csv(paths["leaderboard"], index=False)
    seed_effects.to_csv(paths["seed_effects"], index=False)
    method_summary.to_csv(paths["method_summary"], index=False)
    paths["carried_forward_config"].write_text(
        yaml.safe_dump(
            json.loads(json.dumps(carried_forward_config, default=_json_default)), sort_keys=False
        ),
        encoding="utf-8",
    )
    write_seed_robustness_report(
        paths["report"], summary, method_summary, leaderboard, seed_effects
    )
    return paths


def write_split_audit_report(
    output_path: Path,
    summary: dict[str, Any],
    split_statistics: pd.DataFrame,
    behavior_split_statistics: pd.DataFrame,
    gap_summary: pd.DataFrame,
    repeated_split: pd.DataFrame,
    split_comparison: pd.DataFrame,
) -> Path:
    """Write a Markdown report for the local validation/test generalization audit."""
    risk = str(summary.get("generalization_risk"))
    lines = [
        f"# {summary.get('dataset_name')} split and generalization audit",
        "",
        "This is local split/generalization audit work, not an official NLB leaderboard result.",
        "No model performance claim should be made until validation/test instability is resolved.",
        "Old incompatible mean-rate values are not used as tuning targets.",
        "",
        "## Dataset and splits",
        f"- Dataset name: {summary.get('dataset_name')}",
        f"- Dataset hash: {summary.get('dataset_hash')}",
        f"- Bin size: {summary.get('bin_size_ms')} ms",
        f"- Window length: {summary.get('window_seconds')} seconds",
        f"- Accepted split seed: {summary.get('accepted_split_seed')}",
        f"- Train trials: {summary.get('train_trial_count')}",
        f"- Validation trials: {summary.get('validation_trial_count')}",
        f"- Test trials: {summary.get('test_trial_count')}",
        f"- Held-in neurons: {summary.get('heldin_neuron_count')}",
        f"- Held-out neurons: {summary.get('heldout_neuron_count')}",
        f"- Behavior available: {summary.get('behavior_available')}",
        f"- Missing behavior variables: {summary.get('missing_behavior_variables')}",
        "",
        "## Split-level spike-rate statistics",
        *_format_table(split_statistics),
        "",
        "## Validation vs test trial-statistic comparison",
        *_format_table(split_comparison),
        "",
        "## Behavior distribution statistics",
        *(
            _format_table(behavior_split_statistics)
            if not behavior_split_statistics.empty
            else ["Behavior statistics are unavailable for this dataset."]
        ),
        "",
        "## Validation/test gap summary",
        *(
            _format_table(gap_summary)
            if not gap_summary.empty
            else ["Model gap diagnostics are unavailable: seed robustness results were not found."]
        ),
        "",
        f"- Validation heldout rate (Hz): {summary.get('validation_heldout_rate_hz')}",
        f"- Test heldout rate (Hz): {summary.get('test_heldout_rate_hz')}",
        f"- Factor-latent validation mean: {summary.get('factor_latent_validation_mean')}",
        f"- Factor-latent test mean: {summary.get('factor_latent_test_mean')}",
        f"- Factor-latent validation-test gap: {summary.get('factor_latent_validation_test_gap')}",
        f"- Generalization risk: {risk}",
        "- Validation/test instability detected: "
        f"{summary.get('validation_test_instability_detected')}",
        "",
        "## Repeated split baselines",
        *_format_table(repeated_split),
        "",
        f"- Repeated split validation mean: {summary.get('repeated_split_validation_mean')}",
        f"- Repeated split test mean: {summary.get('repeated_split_test_mean')}",
        "- Repeated split test-positive fraction: "
        f"{summary.get('repeated_split_test_positive_fraction')}",
        "- Validation-positive/test-negative pattern persists: "
        f"{summary.get('validation_positive_test_negative_persists')}",
        "",
        "## Interpretation",
        (
            "- Validation-positive and test-negative scores across every method undercut "
            "current performance claims."
        ),
        (
            "- If repeated splits show high variance, MC_Maze Small is underpowered for "
            "strong conclusions."
        ),
        (
            "- If the test split has a lower held-out rate or a different behavior "
            "distribution, treat this as distribution-shift risk."
        ),
        (
            "- If repeated-split factor-latent is often test-negative, use cross-validation "
            "or larger data before any claims."
        ),
    ]
    if risk == "high":
        lines.extend(
            [
                "",
                "## Verdict",
                (
                    "Generalization risk is high. The current MC_Maze Small split should be "
                    "reported as unstable rather than conclusive, and every score in this "
                    "repository should be read as a validation-only diagnostic."
                ),
            ]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_split_audit_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    trial_statistics: pd.DataFrame,
    split_statistics: pd.DataFrame,
    neuron_split_statistics: pd.DataFrame,
    behavior_split_statistics: pd.DataFrame,
    gap_table: pd.DataFrame,
    gap_summary: pd.DataFrame,
    repeated_split: pd.DataFrame,
    split_comparison: pd.DataFrame,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "split_audit_summary.json",
        "split_statistics": output_dir / "split_statistics.csv",
        "trial_statistics": output_dir / "trial_statistics.csv",
        "neuron_split_statistics": output_dir / "neuron_split_statistics.csv",
        "behavior_split_statistics": output_dir / "behavior_split_statistics.csv",
        "validation_test_gap": output_dir / "validation_test_gap.csv",
        "repeated_split_factor_latent": output_dir / "repeated_split_factor_latent.csv",
        "report": output_dir / "split_audit_report.md",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    trial_statistics.to_csv(paths["trial_statistics"], index=False)
    split_statistics.to_csv(paths["split_statistics"], index=False)
    neuron_split_statistics.to_csv(paths["neuron_split_statistics"], index=False)
    behavior_split_statistics.to_csv(paths["behavior_split_statistics"], index=False)
    gap_table.to_csv(paths["validation_test_gap"], index=False)
    repeated_split.to_csv(paths["repeated_split_factor_latent"], index=False)
    write_split_audit_report(
        paths["report"],
        summary,
        split_statistics,
        behavior_split_statistics,
        gap_summary,
        repeated_split,
        split_comparison,
    )
    return paths


def write_cv_rate_audit_report(
    output_path: Path,
    summary: dict[str, Any],
    method_summary: pd.DataFrame,
    fa_sensitivity: pd.DataFrame,
    decomposition: pd.DataFrame,
    recommendations: dict[str, Any],
) -> Path:
    """Write a Markdown report for the local cross-validated rate-offset audit."""
    valid = (
        method_summary[method_summary["valid_model"].astype(bool)]
        if not method_summary.empty
        else method_summary
    )
    invalid = (
        method_summary[~method_summary["valid_model"].astype(bool)]
        if not method_summary.empty
        else method_summary
    )
    lines = [
        f"# {summary.get('dataset_name')} cross-validated rate audit",
        "",
        "This is local cross-validated rate-offset audit work, not an official NLB "
        "leaderboard result.",
        "Invalid controls use evaluation split targets and cannot be reported as model "
        "performance.",
        "Old incompatible mean-rate values are not used as tuning targets.",
        "",
        "## Dataset and scoring",
        f"- Dataset name: {summary.get('dataset_name')}",
        f"- Dataset hash: {summary.get('dataset_hash')}",
        f"- Bin size: {summary.get('bin_size_ms')} ms",
        f"- Window length: {summary.get('window_seconds')} seconds",
        f"- Canonical reference model: {summary.get('reference_model')}",
        f"- Split seeds: {summary.get('split_seeds')}",
        f"- FactorAnalysis random states: {summary.get('factor_analysis_random_states')}",
        f"- Accepted split seed: {summary.get('accepted_split_seed')}",
        "",
        "## Repeated-split factor-latent",
        f"- Validation mean: {summary.get('factor_latent_repeated_split_validation_mean')}",
        f"- Validation std: {summary.get('factor_latent_repeated_split_validation_std')}",
        f"- Test mean: {summary.get('factor_latent_repeated_split_test_mean')}",
        f"- Test std: {summary.get('factor_latent_repeated_split_test_std')}",
        f"- Test-positive fraction: {summary.get('factor_latent_test_positive_fraction')}",
        f"- Between-split test variance: {summary.get('between_split_test_variance')}",
        "- Within-split FactorAnalysis random-state test variance: "
        f"{summary.get('within_split_random_state_test_variance')}",
        "- Split variance exceeds random-state variance: "
        f"{summary.get('split_variance_exceeds_random_state_variance')}",
        "",
        "## FactorAnalysis random-state sensitivity",
        f"- Validation range: {summary.get('factor_analysis_random_state_validation_range')}",
        f"- Test range: {summary.get('factor_analysis_random_state_test_range')}",
        "",
        *_format_table(fa_sensitivity),
        "",
        "## Valid rate controls",
        *(_format_table(valid) if not valid.empty else ["No valid rate controls were scored."]),
        "",
        "## Invalid diagnostic controls",
        (
            "These read the evaluation split's own held-out targets. They are diagnostics, "
            "never model performance, and never compete for best valid model."
        ),
        "",
        *(
            _format_table(invalid)
            if not invalid.empty
            else ["No invalid diagnostic controls were scored."]
        ),
        "",
        "## Rate-offset decomposition",
        *_format_table(decomposition),
        "",
        f"- Best valid rate-control method: {summary.get('best_valid_rate_control_method')}",
        f"- Best valid rate-control test mean: {summary.get('best_valid_rate_control_test_mean')}",
        f"- Split-mean invalid test mean: {summary.get('split_mean_rate_invalid_test_mean')}",
        "- Invalid split-mean advantage over factor-latent: "
        f"{summary.get('invalid_split_mean_advantage_over_factor_latent')}",
        f"- Train-only rate calibration helps: {summary.get('train_only_rate_calibration_helps')}",
        "- Train-only rate calibration test gain: "
        f"{summary.get('train_only_rate_calibration_test_gain')}",
        "- Train-only rate calibration gain is negligible: "
        f"{summary.get('train_only_rate_calibration_gain_is_negligible')}",
        "- Rate offset explains the split-mean advantage: "
        f"{summary.get('rate_offset_explains_split_mean_advantage')}",
        "- Invalid controls dominate valid models: "
        f"{summary.get('invalid_controls_dominate_valid_models')}",
        "- Invalid controls excluded from best valid model: "
        f"{summary.get('invalid_controls_excluded_from_best_valid_model')}",
        "",
        "## Reporting recommendation",
        "- Single-split results reportable: "
        f"{recommendations.get('single_split_results_reportable')}",
        f"- Recommended reporting mode: {recommendations.get('recommended_reporting_mode')}",
        f"- Carried forward for reporting: {recommendations.get('carried_forward_for_reporting')}",
        f"- Neural models carried forward: {recommendations.get('neural_models_carried_forward')}",
        f"- Must label invalid: {recommendations.get('must_label_invalid')}",
        f"- Rate-offset warning: {recommendations.get('rate_offset_warning')}",
        "",
        "## Interpretation",
        "- Single-split numbers are not reportable as final performance.",
        "- Factor-latent should be reported as a repeated-split baseline.",
        ("- The invalid split-mean control shows an unmodeled split-level rate offset."),
        "- Invalid controls must not be compared as valid models.",
        ("- If train-only calibration helps, it can be carried forward as a valid baseline."),
        (
            "- If only invalid controls help, the issue is evaluation split mean leakage, not a "
            "deployable model gain."
        ),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_cv_rate_audit_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    repeated_scores: pd.DataFrame,
    fa_sensitivity: pd.DataFrame,
    rate_controls: pd.DataFrame,
    decomposition: pd.DataFrame,
    method_summary: pd.DataFrame,
    recommendations: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "cv_rate_audit_summary.json",
        "repeated_split_scores": output_dir / "repeated_split_scores.csv",
        "fa_sensitivity": output_dir / "factor_analysis_random_state_sensitivity.csv",
        "rate_control_scores": output_dir / "rate_control_scores.csv",
        "rate_offset_decomposition": output_dir / "rate_offset_decomposition.csv",
        "method_summary": output_dir / "method_summary.csv",
        "reporting_recommendations": output_dir / "reporting_recommendations.json",
        "report": output_dir / "cv_rate_audit_report.md",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    paths["reporting_recommendations"].write_text(
        json.dumps(recommendations, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    repeated_scores.to_csv(paths["repeated_split_scores"], index=False)
    fa_sensitivity.to_csv(paths["fa_sensitivity"], index=False)
    rate_controls.to_csv(paths["rate_control_scores"], index=False)
    decomposition.to_csv(paths["rate_offset_decomposition"], index=False)
    method_summary.to_csv(paths["method_summary"], index=False)
    write_cv_rate_audit_report(
        paths["report"], summary, method_summary, fa_sensitivity, decomposition, recommendations
    )
    return paths


def write_stratified_cv_report(
    output_path: Path,
    summary: dict[str, Any],
    method_summary: pd.DataFrame,
    fold_balance: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> Path:
    """Write a Markdown report for local behavior-stratified cross-validation."""
    valid = (
        method_summary[method_summary["reportable_as_model_performance"].astype(bool)]
        if not method_summary.empty
        else method_summary
    )
    invalid = (
        method_summary[~method_summary["valid_model"].astype(bool)]
        if not method_summary.empty
        else method_summary
    )
    lines = [
        f"# {summary.get('dataset_name')} behavior-stratified cross-validation",
        "",
        "This is local stratified cross-validation analysis, not an official NLB leaderboard "
        "result.",
        "Invalid controls use evaluation fold targets and cannot be reported as model performance.",
        "Old incompatible mean-rate values are not used as tuning targets.",
        "",
        "## Dataset and protocol",
        f"- Dataset name: {summary.get('dataset_name')}",
        f"- Dataset hash: {summary.get('dataset_hash')}",
        f"- Bin size: {summary.get('bin_size_ms')} ms",
        f"- Window length: {summary.get('window_seconds')} seconds",
        f"- Canonical reference model: {summary.get('reference_model')}",
        f"- Fold count: {summary.get('fold_count')}",
        f"- Repeats: {summary.get('repeats')}",
        f"- Total folds: {summary.get('total_folds')}",
        f"- Assignment method: {summary.get('assignment_method')}",
        f"- Stratification variables: {summary.get('stratification_variables')}",
        "",
        "## Fold balance",
        f"- Mean population-rate fold range: {summary.get('mean_population_rate_fold_range')}",
        f"- Mean held-out-rate fold range: {summary.get('mean_heldout_rate_fold_range')}",
        f"- Mean endpoint-distance fold range: {summary.get('mean_endpoint_distance_fold_range')}",
        f"- Mean speed fold range: {summary.get('mean_speed_fold_range')}",
        "- Mean endpoint-direction entropy: "
        f"{summary.get('mean_endpoint_direction_entropy')} "
        f"(maximum {summary.get('endpoint_direction_entropy_max')})",
        "- Endpoint directions are concentrated in this dataset and window: "
        f"{summary.get('endpoint_direction_concentrated')}",
        f"- Fold balance warning: {summary.get('fold_balance_warning')}",
        (
            "Low endpoint-direction entropy is a property of the dataset and the cropped window, "
            "not of the fold assignment; where it is low, direction stratification has little "
            "left to balance."
        ),
        "",
        *_format_table(fold_balance),
        "",
        "### Per-repeat fold spread",
        "",
        *_format_table(comparisons),
        "",
        "## Factor-latent stratified cross-validation",
        f"- Mean unified bits/spike: {summary.get('factor_latent_mean_unified_bits_per_spike')}",
        f"- Std unified bits/spike: {summary.get('factor_latent_std_unified_bits_per_spike')}",
        f"- CI95 low: {summary.get('factor_latent_ci95_low')}",
        f"- CI95 high: {summary.get('factor_latent_ci95_high')}",
        f"- Positive fraction: {summary.get('factor_latent_positive_fraction')}",
        "",
        "Reportable valid models:",
        "",
        *(_format_table(valid) if not valid.empty else ["(no reportable valid models)"]),
        "",
        "## Invalid split-mean diagnostic",
        "",
        "Invalid controls use evaluation fold targets and cannot be reported as model performance.",
        "",
        "- Split-mean invalid mean unified bits/spike: "
        f"{summary.get('split_mean_rate_invalid_mean_unified_bits_per_spike')}",
        "- Invalid controls excluded from valid model selection: "
        f"{summary.get('invalid_controls_excluded_from_valid_model_selection')}",
        "",
        *(_format_table(invalid) if not invalid.empty else ["(no invalid controls scored)"]),
        "",
        "## Random versus stratified comparison",
        f"- Stratified factor-latent mean: {summary.get('stratified_factor_latent_mean')}",
        f"- Stratified factor-latent std: {summary.get('stratified_factor_latent_std')}",
        f"- Random-fold factor-latent mean: {summary.get('random_fold_factor_latent_mean')}",
        f"- Random-fold factor-latent std: {summary.get('random_fold_factor_latent_std')}",
        "- Repeated random-split test mean reference: "
        f"{summary.get('random_factor_latent_test_mean_reference')}",
        "- Repeated random-split test-positive fraction reference: "
        f"{summary.get('random_factor_latent_test_positive_fraction_reference')}",
        (
            "The repeated random-split references come from a 70/15/15 protocol with 15 "
            "evaluation trials, whereas cross-validation trains on more trials and evaluates on "
            "larger folds. Their means are therefore not comparable, and a higher "
            "cross-validation mean is a protocol difference rather than a performance gain. "
            "Only the matched random-fold comparison above, which differs from the stratified "
            "run solely in how trials are assigned, supports a variance claim."
        ),
        f"- Stratification reduces variance: {summary.get('stratification_reduces_variance')}",
        f"- Variance reduction fraction: {summary.get('variance_reduction_fraction')}",
        "",
        "## Reporting recommendation",
        f"- Recommended reporting mode: {summary.get('recommended_reporting_mode')}",
        f"- Single-split results reportable: {summary.get('single_split_results_reportable')}",
        f"- Carried-forward method: {summary.get('carried_forward_method')}",
        "",
        "## Interpretation",
        "- Stratified cross-validation is preferred over single-split reporting.",
        "- Invalid controls remain leakage diagnostics only.",
        (
            "- Factor-latent remains the carried-forward valid baseline unless a future valid "
            "method beats it under the same protocol."
        ),
        "- Neural models should not be tuned against the old single split.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_stratified_cv_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    scores: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    fold_balance: pd.DataFrame,
    comparisons: pd.DataFrame,
    method_summary: pd.DataFrame,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "stratified_cv_summary.json",
        "scores": output_dir / "stratified_cv_scores.csv",
        "fold_assignments": output_dir / "stratified_fold_assignments.csv",
        "fold_balance": output_dir / "fold_balance_statistics.csv",
        "comparisons": output_dir / "fold_balance_comparisons.csv",
        "method_summary": output_dir / "stratified_cv_method_summary.csv",
        "report": output_dir / "stratified_cv_report.md",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    scores.to_csv(paths["scores"], index=False)
    fold_assignments.to_csv(paths["fold_assignments"], index=False)
    fold_balance.to_csv(paths["fold_balance"], index=False)
    comparisons.to_csv(paths["comparisons"], index=False)
    method_summary.to_csv(paths["method_summary"], index=False)
    write_stratified_cv_report(paths["report"], summary, method_summary, fold_balance, comparisons)
    return paths


def write_window_audit_report(
    output_path: Path,
    summary: dict[str, Any],
    window_table: pd.DataFrame,
    method_summary: pd.DataFrame,
    recommendations: dict[str, Any],
) -> Path:
    """Write a Markdown report for the local movement-window and alignment audit."""
    valid = (
        method_summary[method_summary["reportable_as_model_performance"].astype(bool)]
        if not method_summary.empty
        else method_summary
    )
    invalid = (
        method_summary[~method_summary["valid_model"].astype(bool)]
        if not method_summary.empty
        else method_summary
    )
    lines = [
        f"# {summary.get('dataset_name')} movement-window and alignment audit",
        "",
        "This is local movement-window audit work, not an official NLB leaderboard result.",
        "Invalid controls use evaluation fold targets and cannot be reported as model performance.",
        "Old incompatible mean-rate values are not used as tuning targets.",
        "",
        "## Dataset and protocol",
        f"- Dataset name: {summary.get('dataset_name')}",
        f"- Dataset hash: {summary.get('dataset_hash')}",
        f"- Bin size: {summary.get('bin_size_ms')} ms",
        f"- Canonical reference model: {summary.get('reference_model')}",
        f"- Fold count: {summary.get('fold_count')}",
        f"- Repeats: {summary.get('repeats')}",
        f"- Reporting mode: {summary.get('recommended_reporting_mode')}",
        f"- Behavior feature source: {summary.get('behavior_source')}",
        "",
        "## Candidate windows",
        "",
        *_format_table(window_table),
        "",
        "## Endpoint direction entropy by window",
        "",
        *[
            f"- {name}: {value}"
            for name, value in dict(summary.get("endpoint_direction_entropy_by_window", {})).items()
        ],
        "",
        "## Movement coverage by window",
        "",
        *[
            f"- {name}: moving_bin_fraction {value}"
            for name, value in dict(summary.get("moving_bin_fraction_by_window", {})).items()
        ],
        f"- Behavior coverage warning: {summary.get('behavior_coverage_warning')}",
        "",
        "## Factor-latent score by window",
        "",
        *(_format_table(valid) if not valid.empty else ["(no reportable valid models)"]),
        "",
        "## Invalid split-mean diagnostic by window",
        "",
        "Invalid controls use evaluation fold targets and cannot be reported as model performance.",
        "",
        *(_format_table(invalid) if not invalid.empty else ["(no invalid controls scored)"]),
        "",
        "- Split-mean invalid mean on the recommended window: "
        f"{summary.get('split_mean_invalid_best_window_mean')}",
        f"- Invalid-control gap: {summary.get('invalid_control_gap_best_window')}",
        "- Invalid controls excluded from window selection: "
        f"{summary.get('invalid_controls_excluded_from_window_selection')}",
        "",
        "## Fold balance warnings",
        "",
        *[f"- {row.window_name}: {row.fold_balance_warning}" for row in window_table.itertuples()],
        "",
        "## Recommendation",
        f"- Recommended window: {summary.get('recommended_window_name')}",
        f"- Current window: {summary.get('current_window_name')}",
        f"- Current window still supported: {summary.get('current_window_still_supported')}",
        "- Current window is an early-window diagnostic: "
        f"{summary.get('current_window_is_early_window_diagnostic')}",
        "- Factor-latent mean on the recommended window: "
        f"{summary.get('factor_latent_best_window_mean')}",
        "- Factor-latent mean on the current window: "
        f"{summary.get('factor_latent_current_window_mean')}",
        "- Factor-latent CI95 on the recommended window: "
        f"[{summary.get('factor_latent_best_window_ci95_low')}, "
        f"{summary.get('factor_latent_best_window_ci95_high')}]",
        f"- Eligible windows: {summary.get('eligible_windows')}",
        f"- Rationale: {summary.get('window_selection_rationale')}",
        "",
        "## Interpretation",
        (
            "Factor-latent means are not comparable across windows as performance: each window "
            "defines a different prediction problem, over different spikes and a different "
            "held-out target distribution. The score is used only as a guard that a candidate "
            "window does not degrade the valid model, never as evidence that one window yields "
            "better performance than another."
        ),
        (
            "Movement coverage is thresholded against the peak hand speed of the whole recording, "
            "not each window's own peak. A per-window threshold is scale-free and would make a "
            "pre-movement window look just as active as a reach window."
        ),
        (
            "- The current `from_start` window may capture mostly early or pre-movement activity "
            "rather than the full reach."
        ),
        (
            "- Behavior-aligned windows test whether reach dynamics are better captured when the "
            "crop follows peak speed or movement onset."
        ),
        "- Invalid controls remain leakage diagnostics only.",
        (
            "- The window recommendation is based only on valid-model performance and behavior "
            "coverage, never on invalid-control gains."
        ),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_window_audit_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    scores: pd.DataFrame,
    behavior_statistics: pd.DataFrame,
    balance_statistics: pd.DataFrame,
    window_table: pd.DataFrame,
    method_summary: pd.DataFrame,
    recommendations: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "window_audit_summary.json",
        "scores": output_dir / "window_candidate_scores.csv",
        "behavior_statistics": output_dir / "window_behavior_statistics.csv",
        "balance_statistics": output_dir / "window_balance_statistics.csv",
        "recommendations": output_dir / "window_recommendations.json",
        "report": output_dir / "window_audit_report.md",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    paths["recommendations"].write_text(
        json.dumps(recommendations, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    scores.to_csv(paths["scores"], index=False)
    behavior_statistics.to_csv(paths["behavior_statistics"], index=False)
    balance_statistics.to_csv(paths["balance_statistics"], index=False)
    write_window_audit_report(
        paths["report"], summary, window_table, method_summary, recommendations
    )
    return paths


def write_recommended_window_cv_report(
    output_path: Path,
    summary: dict[str, Any],
    method_summary: pd.DataFrame,
    fold_balance: pd.DataFrame,
    leakage_diagnostics: pd.DataFrame,
    protocol: dict[str, Any],
) -> Path:
    """Write the claim-safe recommended movement-window CV report."""
    lines = [
        f"# {summary.get('dataset_name')} recommended-window cross-validation",
        "",
        (
            "This is local recommended-window cross-validation analysis, not an official NLB "
            "leaderboard result."
        ),
        "Invalid controls use evaluation fold targets and cannot be reported as model performance.",
        "Old incompatible mean-rate values are not used as tuning targets.",
        "",
        "## Dataset and frozen protocol",
        f"- Dataset name: {summary.get('dataset_name')}",
        f"- Dataset hash: {summary.get('dataset_hash')}",
        f"- Bin size: {summary.get('bin_size_ms')} ms",
        f"- Recommended window: {summary.get('recommended_window_name')}",
        f"- Window crop policy: {summary.get('window_crop_policy')}",
        f"- Fold count: {summary.get('fold_count')}",
        f"- Repeats: {summary.get('repeats')}",
        f"- Total folds: {summary.get('total_folds')}",
        f"- Moving bin fraction mean: {summary.get('moving_bin_fraction_mean')}",
        f"- Endpoint direction entropy mean: {summary.get('endpoint_direction_entropy_mean')}",
        "",
        "## Factor-latent stratified cross-validation",
        f"- Mean unified bits/spike: {summary.get('factor_latent_mean')}",
        f"- Std unified bits/spike: {summary.get('factor_latent_std')}",
        "- CI95: "
        f"[{summary.get('factor_latent_ci95_low')}, {summary.get('factor_latent_ci95_high')}]",
        f"- Positive fraction: {summary.get('factor_latent_positive_fraction')}",
        "",
        *_format_table(method_summary),
        "",
        "## Leakage re-check",
        f"- Invalid split-mean diagnostic mean: {summary.get('split_mean_invalid_mean')}",
        "- Factor-latent minus invalid split-mean: "
        f"{summary.get('factor_latent_minus_split_mean_invalid')}",
        f"- Leakage dominance persists: {summary.get('leakage_dominance_persists')}",
        f"- Conclusion: {summary.get('leakage_dominance_conclusion')}",
        "",
        *_format_table(leakage_diagnostics),
        "",
        "## Fold balance summary",
        f"- Fold balance warning: {summary.get('fold_balance_warning')}",
        "",
        *_format_table(fold_balance),
        "",
        "## Frozen protocol",
        "```yaml",
        yaml.safe_dump(protocol, sort_keys=False).rstrip(),
        "```",
        "",
        "## Interpretation",
        "- Previous `from_start` results were early/pre-movement diagnostics.",
        (
            "- Recommended-window scores are not performance improvements over from-start "
            "scores; they use a different prediction target."
        ),
        (
            "- Recommended-window stratified cross-validation is the carried-forward MC_Maze "
            "Small protocol."
        ),
        "- Invalid controls remain leakage diagnostics only.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_recommended_window_cv_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    scores: pd.DataFrame,
    method_summary: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    behavior_statistics: pd.DataFrame,
    fold_balance: pd.DataFrame,
    leakage_diagnostics: pd.DataFrame,
    protocol: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "recommended_window_cv_summary.json",
        "scores": output_dir / "recommended_window_scores.csv",
        "method_summary": output_dir / "recommended_window_method_summary.csv",
        "fold_assignments": output_dir / "recommended_window_fold_assignments.csv",
        "behavior_statistics": output_dir / "recommended_window_behavior_statistics.csv",
        "fold_balance": output_dir / "recommended_window_fold_balance.csv",
        "leakage_diagnostics": output_dir / "recommended_window_leakage_diagnostics.csv",
        "protocol": output_dir / "recommended_window_protocol.yaml",
        "report": output_dir / "recommended_window_cv_report.md",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    scores.to_csv(paths["scores"], index=False)
    method_summary.to_csv(paths["method_summary"], index=False)
    fold_assignments.to_csv(paths["fold_assignments"], index=False)
    behavior_statistics.to_csv(paths["behavior_statistics"], index=False)
    fold_balance.to_csv(paths["fold_balance"], index=False)
    leakage_diagnostics.to_csv(paths["leakage_diagnostics"], index=False)
    paths["protocol"].write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    write_recommended_window_cv_report(
        paths["report"], summary, method_summary, fold_balance, leakage_diagnostics, protocol
    )
    return paths


def write_movement_window_audit_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    candidate_statistics: pd.DataFrame,
    behavior_statistics: pd.DataFrame,
    trial_coverage: pd.DataFrame,
    crop_impact: pd.DataFrame,
    recommendations: dict[str, Any],
) -> dict[str, Path]:
    """Write the trial-aware movement-window audit outputs (behavior only, no model scores)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "window_audit_summary.json",
        "candidate_statistics": output_dir / "window_candidate_statistics.csv",
        "behavior_statistics": output_dir / "window_behavior_statistics.csv",
        "trial_coverage": output_dir / "window_trial_coverage.csv",
        "crop_impact": output_dir / "crop_to_min_impact.csv",
        "recommendations": output_dir / "window_recommendations.json",
        "report": output_dir / "window_audit_report.md",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    paths["recommendations"].write_text(
        json.dumps(recommendations, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    candidate_statistics.to_csv(paths["candidate_statistics"], index=False)
    behavior_statistics.to_csv(paths["behavior_statistics"], index=False)
    trial_coverage.to_csv(paths["trial_coverage"], index=False)
    crop_impact.to_csv(paths["crop_impact"], index=False)
    write_movement_window_audit_report(
        paths["report"], summary, candidate_statistics, recommendations
    )
    return paths


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a Markdown table without the optional tabulate dependency."""
    if frame.empty:
        return "_no rows_"
    columns = [str(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    rows.extend(
        "| " + " | ".join(str(value) for value in record) + " |"
        for record in frame.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def write_movement_window_audit_report(
    output_path: Path,
    summary: dict[str, Any],
    candidate_statistics: pd.DataFrame,
    recommendations: dict[str, Any],
) -> Path:
    """Markdown report for the trial-aware movement-window audit."""
    crop_removes_events = bool(
        summary.get("crop_to_min_removes_peak_for_any_trial", False)
        or summary.get("crop_to_min_removes_onset_for_any_trial", False)
    )
    crop_verdict = (
        "Global crop_to_min removes behavioral events for at least one trial."
        if crop_removes_events
        else (
            "Global crop_to_min removes only post-event and long-duration tail data; peak speed "
            "and movement onset survive inside the retained interval for every trial."
        )
    )
    lines = [
        f"# {summary.get('dataset_name')} Movement-Window Audit",
        "",
        "This window recommendation is based on behavior and alignment coverage, "
        "not model performance.",
        "The globally crop-to-min processed artifact was not silently used as the source of "
        "event-centered windows.",
        "No official NLB leaderboard result is claimed.",
        "",
        "## Trial-aware source",
        "",
        f"- source file: {summary.get('trial_source_file')}",
        f"- representation: {summary.get('trial_representation')}",
        f"- trials: {summary.get('trial_count')}",
        f"- neurons: {summary.get('neuron_count')}",
        f"- trial length range (source bins): {summary.get('trial_length_min')} "
        f"to {summary.get('trial_length_max')}",
        f"- source bin size: {summary.get('source_bin_size_ms')} ms",
        f"- target bin size: {summary.get('target_bin_size_ms')} ms",
        f"- trial-aware representation conserves raw spikes: "
        f"{summary.get('trial_aware_spikes_conserved')}",
        "",
        "## Crop-to-min impact",
        "",
        f"- raw spike count: {summary.get('raw_spike_count')}",
        f"- global crop retained spike count: {summary.get('global_crop_retained_spike_count')}",
        f"- fraction of raw spikes excluded: {summary.get('fraction_raw_spikes_excluded')}",
        f"- fraction of raw bins excluded: {summary.get('fraction_raw_bins_excluded')}",
        f"- trials with peak speed inside global crop: "
        f"{summary.get('fraction_trials_peak_inside_global_crop')}",
        f"- trials with movement onset inside global crop: "
        f"{summary.get('fraction_trials_onset_inside_global_crop')}",
        f"- global crop suitable for movement-window audit: "
        f"{summary.get('global_crop_suitable_for_movement_window_audit')}",
        "",
        crop_verdict,
        "",
        "Windows were extracted from the trial-aware representation regardless, so this audit does "
        "not depend on the global crop being adequate.",
        "",
        "## Movement timing",
        "",
        f"- behavior source: {summary.get('behavior_source')}",
        f"- median peak-speed time: {summary.get('median_peak_speed_time_seconds')} s",
        f"- median movement-onset time: {summary.get('median_movement_onset_time_seconds')} s",
        f"- reference peak speed: {summary.get('reference_peak_speed')}",
        "",
        "## Candidate windows",
        "",
        _markdown_table(candidate_statistics),
        "",
        "## Behavior coverage",
        "",
        f"- moving-bin fraction by window: {summary.get('moving_bin_fraction_by_window')}",
        f"- endpoint-direction entropy by window: "
        f"{summary.get('endpoint_direction_entropy_by_window')}",
        f"- clipped-trial fraction by window: {summary.get('clipped_trial_fraction_by_window')}",
        "",
        "## Window recommendation",
        "",
        f"- recommended window: {recommendations.get('recommended_window_name')}",
        f"- duration: {recommendations.get('recommended_duration_seconds')} s",
        f"- clipped-trial fraction: {recommendations.get('recommended_clipped_trial_fraction')}",
        f"- moving-bin fraction: {recommendations.get('recommended_moving_bin_fraction')}",
        f"- peak-speed coverage: {recommendations.get('recommended_peak_speed_coverage')}",
        f"- movement-onset coverage: {recommendations.get('recommended_movement_onset_coverage')}",
        f"- endpoint-direction entropy: "
        f"{recommendations.get('recommended_endpoint_direction_entropy')}",
        f"- rationale: {recommendations.get('window_selection_rationale')}",
        f"- rejected windows: {recommendations.get('rejected_windows')}",
        "",
        "## Transfer from MC_Maze Small",
        "",
        f"- Small recommended window: {recommendations.get('small_recommended_window')}",
        f"- transfers to Large: {recommendations.get('small_window_transfers')}",
        f"- Small moving-bin fraction reference: {summary.get('small_moving_bin_fraction')}",
        f"- Small endpoint-direction entropy reference: "
        f"{summary.get('small_endpoint_direction_entropy')}",
        "",
        "## Limitations",
        "",
        "- No model was trained, scored, or cross-validated in this audit.",
        "- Window selection used behavior and alignment coverage only; model scores are excluded "
        "by configuration and by construction.",
        "- The frozen processed artifact is unchanged; trial-aware data is audit-only.",
        "- Padding, when used, contributes zero spikes and holds the last behavior sample, so it "
        "adds no spurious movement.",
        "- These are local ignored artifacts, not official NLB leaderboard results.",
        "",
        "## Next evaluation protocol",
        "",
        f"- carry {recommendations.get('recommended_window_name')} forward into "
        "recommended-window stratified cross-validation on MC_Maze Large.",
        f"- reporting mode: {recommendations.get('recommended_reporting_mode')}",
        "- single-split results remain unreportable and no benchmark claim is made.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_large_recommended_window_cv_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    scores: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    protocol: dict[str, Any],
) -> dict[str, Path]:
    """Write the trial-aware recommended-window CV outputs. No model is trained here."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "recommended_window_cv_summary.json",
        "scores": output_dir / "recommended_window_scores.csv",
        "method_summary": output_dir / "recommended_window_method_summary.csv",
        "fold_assignments": output_dir / "recommended_window_fold_assignments.csv",
        "behavior_statistics": output_dir / "recommended_window_behavior_statistics.csv",
        "fold_balance": output_dir / "recommended_window_fold_balance.csv",
        "leakage_diagnostics": output_dir / "recommended_window_leakage_diagnostics.csv",
        "factor_analysis_sensitivity": output_dir / "factor_analysis_random_state_sensitivity.csv",
        "small_large_comparison": output_dir / "small_large_protocol_comparison.csv",
        "protocol": output_dir / "recommended_window_protocol.yaml",
        "report": output_dir / "recommended_window_cv_report.md",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    scores.to_csv(paths["scores"], index=False)
    for key in (
        "method_summary",
        "fold_assignments",
        "behavior_statistics",
        "fold_balance",
        "leakage_diagnostics",
        "factor_analysis_sensitivity",
        "small_large_comparison",
    ):
        tables[key].to_csv(paths[key], index=False)
    paths["protocol"].write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    write_large_recommended_window_cv_report(
        paths["report"], summary, tables["method_summary"], tables["small_large_comparison"]
    )
    return paths


def write_large_recommended_window_cv_report(
    output_path: Path,
    summary: dict[str, Any],
    method_summary: pd.DataFrame,
    comparison: pd.DataFrame,
) -> Path:
    """Claim-safe Markdown report for the trial-aware recommended-window CV."""
    leakage = (
        "Target leakage remains dominant: the invalid split-mean control still beats factor-latent "
        "on the mean."
        if summary.get("leakage_dominance_persists")
        else "Leakage dominance does not persist: factor-latent beats the invalid control on the "
        "mean."
    )
    lines = [
        f"# {summary.get('dataset_name')} Recommended-Window Cross-Validation",
        "",
        "Event-centered windows were extracted from the trial-aware raw representation, not the "
        "globally crop-to-min processed array.",
        "The split-mean control uses evaluation-fold targets and cannot be reported as model "
        "performance.",
        "MC_Maze Small and MC_Maze Large scores are not interpreted as directly comparable "
        "model-performance measurements.",
        "This is local cross-validation analysis, not an official NLB leaderboard result.",
        "Old incompatible mean-rate values were not used as tuning targets.",
        "",
        "## Evaluation source",
        "",
        f"- dataset: {summary.get('dataset_name')}",
        f"- source dataset hash: {summary.get('dataset_hash')}",
        f"- trial source: {summary.get('trial_source')} ({summary.get('trial_source_file')})",
        f"- trial length range (source bins): {summary.get('trial_length_min')} to "
        f"{summary.get('trial_length_max')}",
        f"- global crop used for event-centered windows: "
        f"{summary.get('global_crop_used_for_event_centered_windows')}",
        f"- trials: {summary.get('trial_count')}",
        f"- neurons: {summary.get('neuron_count')} "
        f"(held-in {summary.get('heldin_neuron_count')}, "
        f"held-out {summary.get('heldout_neuron_count')})",
        "",
        "## Frozen movement window",
        "",
        f"- window: {summary.get('window_name')}",
        f"- crop policy: {summary.get('window_crop_policy')}",
        f"- duration: {summary.get('window_duration_seconds')} s",
        f"- target bin size: {summary.get('target_bin_size_ms')} ms",
        f"- time bins: {summary.get('time_bins')}",
        "- extraction happened at the source bin size, before rebinning.",
        f"- moving-bin fraction: {summary.get('moving_bin_fraction_mean')}",
        f"- endpoint-direction entropy: {summary.get('endpoint_direction_entropy_mean')}",
        "",
        "## Stratified fold protocol",
        "",
        f"- folds: {summary.get('fold_count')} x repeats: {summary.get('repeats')} = "
        f"{summary.get('total_folds')} fold evaluations",
        f"- train trials per fold: {summary.get('train_trials_per_fold')}",
        f"- evaluation trials per fold: {summary.get('eval_trials_per_fold')}",
        f"- held-out neuron mask policy: {summary.get('heldout_mask_policy')}",
        f"- assignment method: {summary.get('assignment_method')}",
        f"- reference model: {summary.get('reference_model')} "
        f"(scores {summary.get('train_mean_rate_mean')} bits/spike against itself)",
        "- the train-heldout mean-rate reference is recomputed from training trials only, on "
        "every fold.",
        "",
        "## Fold balance",
        "",
        f"- population-rate fold range: {summary.get('mean_population_rate_fold_range')}",
        f"- held-out-rate fold range: {summary.get('mean_heldout_rate_fold_range')}",
        f"- endpoint-distance fold range: {summary.get('mean_endpoint_distance_fold_range')}",
        f"- mean-speed fold range: {summary.get('mean_speed_fold_range')}",
        f"- endpoint-direction entropy: {summary.get('mean_endpoint_direction_entropy')} "
        f"(max {summary.get('endpoint_direction_entropy_max')})",
        f"- fold balance warning: {summary.get('fold_balance_warning')}",
        "",
        "## Factor-latent baseline",
        "",
        f"- mean: {summary.get('factor_latent_mean')}",
        f"- std: {summary.get('factor_latent_std')}",
        f"- CI95: [{summary.get('factor_latent_ci95_low')}, "
        f"{summary.get('factor_latent_ci95_high')}]",
        f"- positive fold fraction: {summary.get('factor_latent_positive_fraction')}",
        f"- between-repeat std: {summary.get('factor_latent_between_repeat_std')}",
        f"- within-repeat std: {summary.get('factor_latent_within_repeat_std')}",
        "",
        _markdown_table(method_summary),
        "",
        "## FactorAnalysis random-state sensitivity",
        "",
        "The FactorAnalysis random state is configured explicitly and is never derived from the "
        "fold index or repeat index.",
        f"- random states: {summary.get('factor_analysis_random_states')}",
        f"- range across states: {summary.get('factor_analysis_random_state_range')}",
        f"- std across states: {summary.get('factor_analysis_random_state_std')}",
        f"- warning: {summary.get('factor_analysis_random_state_warning')}",
        "",
        "## Invalid leakage control",
        "",
        f"- split-mean invalid mean: {summary.get('split_mean_invalid_mean')}",
        f"- split-mean invalid std: {summary.get('split_mean_invalid_std')}",
        f"- factor-latent minus invalid: {summary.get('factor_latent_minus_split_mean_invalid')}",
        f"- factor-latent beats invalid on the mean: "
        f"{summary.get('factor_latent_beats_invalid_control_mean')}",
        f"- folds where factor-latent beats invalid: "
        f"{summary.get('factor_latent_beats_invalid_control_fraction')}",
        f"- leakage dominance persists: {summary.get('leakage_dominance_persists')}",
        "",
        leakage,
        "",
        "Beating an invalid control is a leakage diagnostic, never an official benchmark result.",
        "",
        "## Comparison with MC_Maze Small protocol",
        "",
        "Small and Large differ in trials, neurons, firing rates, and target distributions. Only "
        "protocol stability and leakage diagnostics are compared. No cross-dataset "
        "model-performance improvement is claimed.",
        "",
        _markdown_table(comparison),
        "",
        "Conclusions:",
        "",
        *[f"- {line}" for line in summary.get("small_large_comparison_conclusions", [])],
        "",
        "## Reporting recommendation",
        "",
        f"- reporting mode: {summary.get('recommended_reporting_mode')}",
        f"- protocol frozen: {summary.get('protocol_frozen')}",
        f"- single-split results reportable: {summary.get('single_split_results_reportable')}",
        f"- invalid controls excluded from model selection: "
        f"{summary.get('invalid_controls_excluded_from_model_selection')}",
        f"- official leaderboard claim: {summary.get('official_leaderboard_claim')}",
        "",
        "## Limitations",
        "",
        "- factor-latent is a FactorAnalysis-based baseline, not GPFA and not a neural model.",
        "- no LFADS-style, neural-ODE, neural-SDE, or switching model was trained or tuned here.",
        "- the invalid split-mean control reads evaluation-fold targets and is diagnostic only.",
        "- cross-dataset metric differences are not model improvements.",
        "- all outputs are local ignored artifacts.",
        "",
        "## Next research actions",
        "",
        "- expand valid non-neural Large baselines under this frozen protocol.",
        "- reevaluate LFADS-style, neural-ODE, and neural-SDE models under the same protocol.",
        "- run controlled multi-seed and architecture ablations before any comparison claim.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_dataset_scoreboard_outputs(
    output_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Path]:
    """Write the summary-only scoreboard for a dataset whose evidence is CV summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "unified_scoreboard_summary.json",
        "report": output_dir / "unified_scoreboard_report.md",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# {summary.get('dataset_name')} unified scoreboard",
        "",
        "This is a local scoreboard of recommended-window cross-validation evidence, not an "
        "official NLB leaderboard result.",
        "Invalid controls use evaluation-fold targets and can never be the best valid method.",
        "",
        f"- recommended-window CV available: {summary.get('recommended_window_cv_available')}",
        f"- recommended window: {summary.get('recommended_window_name')}",
        f"- reporting mode: {summary.get('recommended_reporting_mode')}",
        f"- factor-latent mean: {summary.get('factor_latent_recommended_window_mean')}",
        f"- factor-latent CI95: "
        f"[{summary.get('factor_latent_recommended_window_ci95_low')}, "
        f"{summary.get('factor_latent_recommended_window_ci95_high')}]",
        f"- factor-latent positive fraction: {summary.get('factor_latent_positive_fraction')}",
        f"- factor-latent beats invalid control: "
        f"{summary.get('factor_latent_beats_invalid_control_mean')}",
        f"- leakage dominance persists: {summary.get('leakage_dominance_persists')}",
        f"- best valid method: {summary.get('best_valid_method')}",
        f"- LFADS pilot available: {summary.get('lfads_pilot_available')}",
        f"- LFADS pilot complete: {summary.get('lfads_pilot_complete')}",
        f"- LFADS pilot mean: {summary.get('lfads_pilot_mean')}",
        f"- LFADS pilot seed std: {summary.get('lfads_pilot_seed_std')}",
        f"- LFADS pilot positive seed fraction: "
        f"{summary.get('lfads_pilot_positive_seed_fraction')}",
        f"- LFADS pilot mean difference vs baseline: "
        f"{summary.get('lfads_pilot_mean_difference_vs_baseline')}",
        f"- full LFADS evaluation recommended: {summary.get('lfads_full_evaluation_recommended')}",
        f"- LFADS pilot final claim allowed: {summary.get('lfads_pilot_final_claim_allowed')}",
        f"- single-split results reportable: {summary.get('single_split_results_reportable')}",
        f"- official leaderboard claim: {summary.get('official_leaderboard_claim')}",
        "",
    ]
    paths["report"].write_text("\n".join(lines), encoding="utf-8")
    return paths


def write_baseline_suite_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    protocol: dict[str, Any],
    readiness: dict[str, Any],
) -> dict[str, Path]:
    """Write the valid baseline suite outputs. No neural model is trained in this workflow."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "baseline_suite_summary.json",
        "outer_fold_scores": output_dir / "outer_fold_scores.csv",
        "inner_selection": output_dir / "inner_selection_results.csv",
        "selected_hyperparameters": output_dir / "selected_hyperparameters.csv",
        "method_summary": output_dir / "method_summary.csv",
        "paired_method_comparisons": output_dir / "paired_method_comparisons.csv",
        "repeat_level_scores": output_dir / "repeat_level_scores.csv",
        "protocol": output_dir / "baseline_protocol.yaml",
        "readiness": output_dir / "neural_reevaluation_readiness.json",
        "report": output_dir / "baseline_suite_report.md",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    paths["readiness"].write_text(
        json.dumps(readiness, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    for key in (
        "outer_fold_scores",
        "inner_selection",
        "selected_hyperparameters",
        "method_summary",
        "paired_method_comparisons",
        "repeat_level_scores",
    ):
        tables[key].to_csv(paths[key], index=False)
    paths["protocol"].write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    write_baseline_suite_report(
        paths["report"],
        summary,
        tables["method_summary"],
        tables["paired_method_comparisons"],
        tables["selected_hyperparameters"],
        readiness,
    )
    return paths


def write_baseline_suite_report(
    output_path: Path,
    summary: dict[str, Any],
    method_summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    selected: pd.DataFrame,
    readiness: dict[str, Any],
) -> Path:
    """Claim-safe Markdown report for the valid baseline suite."""
    valid = method_summary[method_summary["reportable_as_model_performance"].astype(bool)]
    invalid = method_summary[~method_summary["reportable_as_model_performance"].astype(bool)]
    patterns = (
        selected.groupby(["method_name", "selected_hyperparameters_json"])
        .size()
        .reset_index(name="outer_folds")
        .sort_values(["method_name", "outer_folds"], ascending=[True, False])
        if not selected.empty
        else selected
    )
    replaced = bool(summary.get("baseline_replaced"))
    lines = [
        f"# {summary.get('dataset_name')} Valid Baseline Suite",
        "",
        "All event-centered arrays were extracted from the trial-aware source before rebinning.",
        "Hyperparameters were selected using only outer-training data.",
        "Outer folds within a repeat are not treated as statistically independent.",
        "Invalid controls use evaluation targets and cannot be reported as model performance.",
        "MC_Maze Small and MC_Maze Large scores are not interpreted as directly comparable "
        "performance measurements.",
        "This is local cross-validation analysis, not an official NLB leaderboard result.",
        "",
        "## Frozen evaluation protocol",
        "",
        f"- dataset hash: {summary.get('dataset_hash')}",
        f"- window: {summary.get('window_name')}",
        f"- target bin size: {summary.get('target_bin_size_ms')} ms",
        f"- protocol source: {summary.get('protocol_source')}",
        f"- outer fold assignments reused verbatim: {summary.get('outer_assignments_reused')}",
        f"- held-out neuron masks reused verbatim: {summary.get('neuron_masks_reused')}",
        f"- outer folds x repeats: {summary.get('outer_fold_count')} x "
        f"{summary.get('outer_repeats')} = {summary.get('total_outer_evaluations')} evaluations",
        "",
        "## Nested selection design",
        "",
        f"- inner selection enabled: {summary.get('inner_selection_enabled')}",
        f"- inner folds: {summary.get('inner_fold_count')}, built only from outer-training trials",
        "- every selected configuration comes from the declared finite grid.",
        "- smoothing, standardization, dimensionality reduction, and decoders are fit inside the "
        "training partition only.",
        "- outer-evaluation counts never enter selection or calibration.",
        "",
        "## Valid baseline methods",
        "",
        _markdown_table(valid),
        "",
        "## Inner-selection results",
        "",
        "Most frequently selected configuration per method:",
        "",
        _markdown_table(patterns),
        "",
        "## Outer-fold results",
        "",
        f"- factor_latent_fixed mean: {summary.get('factor_latent_fixed_mean')}",
        f"- accepted recommended-window mean: {summary.get('factor_latent_accepted_mean')}",
        f"- reproduction difference: {summary.get('factor_latent_reproduction_difference')}",
        f"- reproduced within tolerance: {summary.get('factor_latent_reproduced')}",
        f"- train_mean_rate mean: {summary.get('train_mean_rate_mean')} (reference, scores zero)",
        "",
        "## Held-out-neuron-mask variability",
        "",
        "Each repeat uses one held-out neuron mask, fixed across that repeat's folds. "
        "Between-repeat standard deviation therefore measures neuron-mask sensitivity, and "
        "within-repeat standard deviation measures trial-fold noise.",
        "",
        "## Paired method comparisons",
        "",
        f"- comparison unit: {summary.get('comparison_unit')}",
        f"- hierarchical paired bootstrap: {summary.get('hierarchical_bootstrap')}",
        f"- naive fold-independent significance test used: "
        f"{summary.get('naive_independent_fold_test_used')}",
        "",
        _markdown_table(comparisons),
        "",
        "## Invalid leakage control",
        "",
        f"- split_mean_rate_invalid mean: {summary.get('split_mean_invalid_mean')}",
        "- the split-mean control reads each evaluation fold's own held-out targets.",
        "- it never enters hyperparameter selection, superiority testing, or baseline ranking.",
        "",
        _markdown_table(invalid),
        "",
        "## Baseline to beat",
        "",
        f"- baseline to beat: {summary.get('baseline_to_beat')}",
        f"- baseline replaced: {replaced}",
        f"- replacement statistically supported: {summary.get('baseline_replacement_supported')}",
        f"- best valid method by mean: {summary.get('best_valid_method')} "
        f"({summary.get('best_valid_method_mean')})",
        f"- paired difference against factor_latent_fixed: "
        f"{summary.get('paired_difference_against_factor_latent')}",
        f"- paired CI95: {summary.get('paired_ci_against_factor_latent')}",
        f"- positive repeat fraction: "
        f"{summary.get('positive_repeat_fraction_against_factor_latent')}",
        "",
        "A method replaces the incumbent only when the paired mean difference is positive, the "
        "paired bootstrap interval excludes zero, and it wins on at least 80 percent of repeats.",
        "",
        "## Neural reevaluation readiness",
        "",
        f"- ready: {readiness.get('ready')}",
        f"- blockers: {readiness.get('blockers')}",
        f"- required neural seeds: {readiness.get('required_neural_seeds')}",
        f"- forbidden old protocols: {readiness.get('forbidden_old_protocols')}",
        f"- neural experiment run during this milestone: "
        f"{readiness.get('neural_experiment_run_during_this_milestone')}",
        "",
        "## Limitations",
        "",
        "- every baseline here is linear or factor-analysis based; none is a neural or dynamical "
        "model, and the reduced-rank method carries no temporal assumptions.",
        "- no LFADS-style, neural-ODE, neural-SDE, switching, controller, or VAE model was trained "
        "or tuned in this milestone.",
        "- 25 outer folds are correlated within repeats; the reported intervals account for that "
        "and are wider than a naive fold-level interval would be.",
        "- results are local ignored artifacts.",
        "",
        "## Next research actions",
        "",
        "- freeze the baseline to beat and the readiness artifact.",
        "- prepare and approve the neural-model reevaluation manifest.",
        "- reevaluate LFADS-style and deterministic neural-ODE models under this exact protocol.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_lfads_pilot_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    protocol: dict[str, Any],
    recommendation: dict[str, Any],
) -> dict[str, Path]:
    """Write claim-safe LFADS feasibility outputs; generated artifacts remain ignored."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "lfads_pilot_summary.json",
        "lfads_pilot_runs": output_dir / "lfads_pilot_runs.csv",
        "fold_seed_scores": output_dir / "fold_seed_scores.csv",
        "seed_summary": output_dir / "seed_summary.csv",
        "fold_summary": output_dir / "fold_summary.csv",
        "paired_baseline_comparison": output_dir / "paired_baseline_comparison.csv",
        "checkpoint_manifest": output_dir / "checkpoint_manifest.csv",
        "training_resource_summary": output_dir / "training_resource_summary.csv",
        "protocol": output_dir / "lfads_pilot_protocol.yaml",
        "recommendation": output_dir / "full_evaluation_recommendation.json",
        "report": output_dir / "lfads_pilot_report.md",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    paths["recommendation"].write_text(
        json.dumps(recommendation, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    paths["protocol"].write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    for key in (
        "lfads_pilot_runs",
        "fold_seed_scores",
        "seed_summary",
        "fold_summary",
        "paired_baseline_comparison",
        "checkpoint_manifest",
        "training_resource_summary",
    ):
        tables[key].to_csv(paths[key], index=False)

    lines = [
        "# MC_Maze Large LFADS Feasibility Pilot",
        "",
        "## Scope",
        "",
        "Controlled repeat-0 feasibility and seed-stability analysis. The pilot evaluates "
        "feasibility and seed stability on one held-out-neuron mask. It is not a final "
        "multi-repeat model comparison.",
        "",
        "## Frozen evaluation protocol",
        "",
        "Event-centered inputs were extracted from the trial-aware source before rebinning.",
        f"Shape: {summary.get('data_shape')}; repeat: {summary.get('repeat_index')}; "
        f"seeds: {summary.get('initialization_seeds')}",
        "",
        "## Model configuration",
        "",
        "Existing LFADS-style GRU; held-in inputs; all-neuron positive-rate output; no controller.",
        "",
        "## Leakage prevention",
        "",
        "Outer-evaluation data were not used for checkpoint selection, early stopping, "
        "normalization, calibration, or hyperparameter selection.",
        f"Leakage checks passed: {summary.get('leakage_checks_passed')}",
        "",
        "## Checkpoint selection",
        "",
        "Every selected checkpoint must maximize unified bits/spike on inner_validation. "
        "Outer-evaluation checkpoint selection is forbidden.",
        f"Checkpoint selection valid: {summary.get('checkpoint_selection_valid')}",
        "",
        "## Pilot execution",
        "",
        f"Completed runs: {summary.get('completed_runs')}; "
        f"failed runs: {summary.get('failed_runs')}",
        f"Mean unified bits/spike: {summary.get('mean_unified_bits_per_spike')}",
        "",
        "## Seed sensitivity",
        "",
        _markdown_table(tables["seed_summary"]),
        "",
        "## Comparison with the valid baseline",
        "",
        "The baseline to beat is factor_latent_train_selected.",
        f"Pilot-repeat baseline mean: {summary.get('pilot_repeat_baseline_mean')}",
        f"Mean paired difference: {summary.get('mean_paired_difference_vs_baseline')}",
        "Paired fold differences are descriptive diagnostics only; no final superiority test is "
        "reported.",
        "The corrected movement window produced stable positive LFADS-style scores, but it did not "
        "resolve the earlier failure mode of trailing the valid factor-latent baseline: no pilot "
        "fold-seed run beat that baseline.",
        "",
        "## Compute and memory",
        "",
        f"Estimated full-evaluation runtime hours: "
        f"{recommendation.get('runtime_estimate_full_evaluation_hours')}",
        f"Estimated peak CUDA memory MB: {recommendation.get('estimated_peak_cuda_memory_mb')}",
        "Runtime is an observed-pilot estimate, not an exact completion-time promise.",
        "",
        "## Full-evaluation gate",
        "",
        f"Proceed: {recommendation.get('proceed')}",
        f"Reasons: {recommendation.get('reasons')}",
        "",
        "## Limitations",
        "",
        "The pilot uses one held-out-neuron mask and is not a final multi-repeat comparison.",
        "MC_Maze Small and MC_Maze Large scores are not treated as directly comparable "
        "model-performance measurements.",
        "This is local research analysis, not an official NLB leaderboard result.",
        "",
        "## Next research action",
        "",
        "Run full five-repeat LFADS evaluation only if the predeclared gate passes.",
        "",
    ]
    paths["report"].write_text("\n".join(lines), encoding="utf-8")
    return paths


def write_neural_ode_pilot_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    protocol: dict[str, Any],
    recommendation: dict[str, Any],
    next_action: dict[str, Any],
) -> dict[str, Path]:
    """Write claim-safe deterministic neural-ODE feasibility outputs; artifacts remain ignored."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "neural_ode_pilot_summary.json",
        "neural_ode_pilot_runs": output_dir / "neural_ode_pilot_runs.csv",
        "fold_seed_scores": output_dir / "fold_seed_scores.csv",
        "seed_summary": output_dir / "seed_summary.csv",
        "fold_summary": output_dir / "fold_summary.csv",
        "paired_baseline_comparison": output_dir / "paired_baseline_comparison.csv",
        "lfads_descriptive_comparison": output_dir / "lfads_descriptive_comparison.csv",
        "checkpoint_manifest": output_dir / "checkpoint_manifest.csv",
        "solver_diagnostics": output_dir / "solver_diagnostics.csv",
        "latent_diagnostics": output_dir / "latent_diagnostics.csv",
        "training_resource_summary": output_dir / "training_resource_summary.csv",
        "protocol": output_dir / "neural_ode_pilot_protocol.yaml",
        "recommendation": output_dir / "full_evaluation_recommendation.json",
        "next_action": output_dir / "next_action_recommendation.json",
        "report": output_dir / "neural_ode_pilot_report.md",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    paths["recommendation"].write_text(
        json.dumps(recommendation, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    paths["next_action"].write_text(
        json.dumps(next_action, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    paths["protocol"].write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    for key in (
        "neural_ode_pilot_runs",
        "fold_seed_scores",
        "seed_summary",
        "fold_summary",
        "paired_baseline_comparison",
        "lfads_descriptive_comparison",
        "checkpoint_manifest",
        "solver_diagnostics",
        "latent_diagnostics",
        "training_resource_summary",
    ):
        tables[key].to_csv(paths[key], index=False)

    lines = [
        "# MC_Maze Large Deterministic Neural-ODE Feasibility Pilot",
        "",
        "## Scope",
        "",
        "This pilot assesses feasibility, stability, and dynamics on one held-out-neuron mask. "
        "It is not a final multi-repeat model comparison.",
        "Diffusion is disabled; this is a deterministic neural-ODE pilot.",
        "",
        "## Frozen evaluation protocol",
        "",
        "Event-centered inputs were extracted from the trial-aware source before rebinning.",
        f"Shape: {summary.get('data_shape')}; repeat: {summary.get('repeat_index')}; "
        f"folds: {summary.get('fold_indices')}; seeds: {summary.get('initialization_seeds')}.",
        f"Input neurons: {summary.get('input_neuron_count')}; output neurons: "
        f"{summary.get('output_neuron_count')}; scored held-out neurons: "
        f"{summary.get('heldout_neuron_count')}.",
        "",
        "## Configuration provenance",
        "",
        "Model dimensions and objective are frozen from the accepted MC_Maze Small deterministic "
        "neural-ODE refinement best run; only input/output dimensions were adapted to the Large "
        "repeat-0 neuron mask.",
        "",
        "## Seed and checkpoint controls",
        "",
        "Each declared seed is applied before model construction, parameter initialization, and "
        "data-loader creation; no seed arithmetic is used.",
        "Outer-evaluation data were not used for checkpoint selection, early stopping, "
        "normalization, calibration, or configuration selection.",
        f"Checkpoint selection split: {summary.get('checkpoint_selection_split')}; "
        f"selection valid: {summary.get('checkpoint_selection_valid')}; leakage checks passed: "
        f"{summary.get('leakage_checks_passed')}.",
        "",
        "## Deterministic dynamics",
        "",
        f"Solver: {summary.get('solver')}; integration step (s): "
        f"{summary.get('integration_step_seconds')}; diffusion enabled: "
        f"{summary.get('diffusion_enabled')}.",
        "Diffusion is unnecessary on this task, consistent with the earlier MC_Maze Small result.",
        "",
        "## Solver stability",
        "",
        f"Solver stability passed: {summary.get('solver_stability_passed')}.",
        _markdown_table(tables["solver_diagnostics"]),
        "",
        "## Pilot scores",
        "",
        f"Completed runs: {summary.get('completed_runs')}; failed runs: "
        f"{summary.get('failed_runs')}.",
        f"Mean unified bits/spike: {summary.get('mean_unified_bits_per_spike')}; run-level std: "
        f"{summary.get('run_level_score_std')}; seed-mean std: {summary.get('seed_mean_std')}.",
        f"Positive run fraction: {summary.get('positive_run_fraction')}; positive seed fraction: "
        f"{summary.get('positive_seed_fraction')}; runs beating baseline: "
        f"{summary.get('runs_beating_baseline')}.",
        _markdown_table(tables["seed_summary"]),
        "",
        "## Valid-baseline comparison",
        "",
        "The baseline to beat is factor_latent_train_selected.",
        f"Pilot-repeat baseline mean: {summary.get('pilot_repeat_baseline_mean')}; mean paired "
        f"difference: {summary.get('mean_paired_difference_vs_baseline')}.",
        "Paired fold differences are descriptive diagnostics only; no final superiority test is "
        "reported.",
        "",
        "## Descriptive comparison with LFADS",
        "",
        "LFADS values are included only as a descriptive reference and do not select the "
        "neural-ODE configuration.",
        f"LFADS pilot mean: {summary.get('lfads_descriptive_reference_mean')}; neural-ODE minus "
        f"LFADS: {summary.get('mean_difference_vs_lfads_reference')}.",
        "",
        "## Near-peak movement behavior",
        "",
        f"Before peak: {summary.get('before_peak_mean_bits_per_spike')}; near peak: "
        f"{summary.get('near_peak_mean_bits_per_spike')}; after peak: "
        f"{summary.get('after_peak_mean_bits_per_spike')}.",
        f"LFADS reference: before 0.03636918, near 0.00264721, after 0.03198509. Near-peak "
        f"failure status: {summary.get('near_peak_failure_status')}.",
        "",
        "## Latent utilization",
        "",
        f"Mean factor effective rank: {summary.get('mean_factor_effective_rank')} "
        f"(fraction {summary.get('mean_factor_effective_rank_fraction')}). LFADS factor effective "
        f"rank {summary.get('lfads_factor_effective_rank')} "
        f"(fraction {summary.get('lfads_factor_effective_rank_fraction')}).",
        "Cross-model effective rank alone is not treated as evidence of superiority.",
        "",
        "## Compute and memory",
        "",
        f"Estimated full-evaluation runtime hours: "
        f"{recommendation.get('runtime_estimate_full_evaluation_hours')}; estimated peak CUDA "
        f"memory MB: {recommendation.get('estimated_peak_cuda_memory_mb')}.",
        "Runtime is an observed-pilot estimate, not an exact completion-time promise.",
        "",
        "## Full-evaluation gate",
        "",
        f"Proceed: {recommendation.get('proceed')}; reasons: {recommendation.get('reasons')}.",
        "",
        "## Recommended next action",
        "",
        f"Recommended next action: {next_action.get('recommended_next_action')}.",
        f"Rationale: {next_action.get('rationale')}.",
        "",
        "## Limitations",
        "",
        "The pilot uses one held-out-neuron mask and cannot support a final project-wide "
        "superiority claim.",
        "This is local research analysis, not an official NLB leaderboard result.",
        "",
    ]
    paths["report"].write_text("\n".join(lines), encoding="utf-8")
    return paths


def write_lfads_diagnostics_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    recommendation: dict[str, Any],
) -> dict[str, Path]:
    """Persist the post-hoc LFADS failure-mode audit without model artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "lfads_diagnostics_summary.json",
        "run_diagnostics": output_dir / "run_diagnostics.csv",
        "checkpoint_diagnostics": output_dir / "checkpoint_diagnostics.csv",
        "neuron_diagnostics": output_dir / "neuron_diagnostics.csv",
        "time_bin_diagnostics": output_dir / "time_bin_diagnostics.csv",
        "latent_diagnostics": output_dir / "latent_diagnostics.csv",
        "rate_diagnostics": output_dir / "rate_diagnostics.csv",
        "objective_diagnostics": output_dir / "objective_diagnostics.csv",
        "baseline_gap_decomposition": output_dir / "baseline_gap_decomposition.csv",
        "recommendation": output_dir / "next_action_recommendation.json",
        "report": output_dir / "lfads_diagnostics_report.md",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    paths["recommendation"].write_text(
        json.dumps(recommendation, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    for key in (
        "run_diagnostics",
        "checkpoint_diagnostics",
        "neuron_diagnostics",
        "time_bin_diagnostics",
        "latent_diagnostics",
        "rate_diagnostics",
        "objective_diagnostics",
        "baseline_gap_decomposition",
    ):
        tables[key].to_csv(paths[key], index=False)

    lines = [
        "# MC_Maze Large LFADS Failure-Mode Audit",
        "",
        "## Scope",
        "",
        "This audit uses already selected checkpoints and does not train or select a new model.",
        "Outer-evaluation diagnostics were not used to change checkpoint selection.",
        "",
        "## Pilot integrity",
        "",
        f"Integrity checks passed: {summary.get('integrity_checks_passed')}",
        f"Accepted checkpoints: {summary.get('accepted_checkpoints')}",
        f"Non-completed manifest/run rows excluded: {summary.get('excluded_preflight_artifacts')}",
        "Terminated preflight processes not represented in the accepted manifest are excluded by "
        "construction.",
        "",
        "## Train, validation, and outer-evaluation behavior",
        "",
        f"Outer-training mean bits/spike: {summary.get('train_mean_unified_bits_per_spike')}",
        f"Inner-validation mean bits/spike: {summary.get('inner_mean_unified_bits_per_spike')}",
        f"Outer-evaluation mean bits/spike: {summary.get('outer_mean_unified_bits_per_spike')}",
        f"Mean train-to-inner gap: {summary.get('mean_train_to_inner_gap')}",
        f"Mean inner-to-outer gap: {summary.get('mean_inner_to_outer_gap')}",
        "",
        "## Per-neuron performance",
        "",
        f"Positive-neuron fraction: {summary.get('positive_neuron_fraction')}",
        f"Negative-neuron fraction: {summary.get('negative_neuron_fraction')}",
        f"Median neuron bits/spike: {summary.get('median_neuron_unified_bits_per_spike')}",
        f"Fraction beating factor-latent: {summary.get('fraction_neurons_beating_factor_latent')}",
        "",
        "## Time-resolved performance",
        "",
        str(summary.get("time_resolved_failure_pattern")),
        f"High-rate-bin mean: {summary.get('high_rate_time_bin_mean_bits_per_spike')}",
        f"Low-rate-bin mean: {summary.get('low_rate_time_bin_mean_bits_per_spike')}",
        "",
        "## Rate calibration",
        "",
        str(summary.get("rate_bias_finding")),
        "Calibrated scores are diagnostic counterfactuals, not model performance.",
        "",
        "## Temporal smoothness",
        "",
        str(summary.get("temporal_smoothness_finding")),
        "",
        "## Latent utilization",
        "",
        f"Mean effective rank: {summary.get('mean_effective_rank')}",
        f"Mean effective-rank fraction: {summary.get('mean_effective_rank_fraction')}",
        f"Posterior collapse detected: {summary.get('posterior_collapse_detected')}",
        str(summary.get("latent_utilization_finding")),
        "",
        "## Objective balance",
        "",
        str(summary.get("objective_balance_finding")),
        "",
        "## Baseline-gap decomposition",
        "",
        _markdown_table(tables["baseline_gap_decomposition"]),
        "",
        "Components may overlap and are not required to sum exactly.",
        "",
        "## Failure-mode classification",
        "",
        "The LFADS pilot was stable and positive but substantially below "
        "factor_latent_train_selected on the pilot neuron mask.",
        f"Dominant failure mode: {summary.get('dominant_failure_mode')}",
        "",
        "## Recommended next action",
        "",
        f"{recommendation.get('recommended_next_action')}",
        f"Rationale: {recommendation.get('rationale')}",
        "Full multi-repeat LFADS evaluation remains disallowed.",
        "",
        "## Limitations",
        "",
        "The pilot covers one held-out-neuron mask and is not a final multi-repeat comparison.",
        "This is local research analysis, not an official NLB leaderboard result.",
        "",
    ]
    paths["report"].write_text("\n".join(lines), encoding="utf-8")
    return paths
