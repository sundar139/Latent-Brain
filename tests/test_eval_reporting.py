from __future__ import annotations

import json
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from latentbrain.eval.reporting import (
    write_baseline_markdown_report,
    write_baseline_outputs,
    write_cosmoothing_markdown_report,
    write_cosmoothing_sweep_markdown_report,
    write_factor_latent_markdown_report,
    write_factor_latent_sweep_markdown_report,
    write_lfads_gru_evaluation_report,
    write_window_matched_comparison_report,
)


def _split_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["validation"],
            "neuron_group": ["heldout"],
            "n_trials": [1],
            "n_neurons": [2],
            "n_time_bins": [3],
            "spike_count": [4.0],
            "poisson_nll": [5.0],
            "poisson_log_likelihood": [-5.0],
            "reference_log_likelihood": [-6.0],
            "bits_per_spike": [0.1],
            "mean_predicted_rate_hz": [7.0],
        }
    )


def _neuron_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "neuron_index": [0],
            "neuron_group": ["heldout"],
            "train_mean_rate_hz": [1.0],
            "total_spikes_all_trials": [2],
            "validation_spikes": [1],
            "test_spikes": [1],
        }
    )


def test_baseline_outputs_write_json_csv_and_markdown(tmp_path: Path) -> None:
    summary = {
        "dataset_name": "mc_maze_small",
        "dataset_hash": "abc",
        "baseline_name": "mean_rate",
        "train_only_fit": True,
        "primary_split": "validation",
        "primary_neuron_group": "heldout",
        "primary_bits_per_spike": 0.1,
        "primary_poisson_nll": 5.0,
    }

    outputs = write_baseline_outputs(tmp_path, summary, _split_metrics(), _neuron_metrics())

    assert json.loads(outputs["metrics_summary"].read_text())["baseline_name"] == "mean_rate"
    assert outputs["split_metrics"].read_text().startswith("split,neuron_group")
    report = outputs["baseline_report"].read_text(encoding="utf-8")
    assert "mc_maze_small" in report
    assert "local sanity baseline, not an official NLB leaderboard result" in report
    assert "No neural network model was trained" in report


def test_markdown_report_mentions_primary_metric_and_fit_policy(tmp_path: Path) -> None:
    report_path = write_baseline_markdown_report(
        tmp_path / "baseline_report.md",
        dataset_name="mc_maze_small",
        metrics_summary={
            "dataset_hash": "abc",
            "baseline_name": "mean_rate",
            "train_only_fit": True,
            "primary_split": "validation",
            "primary_neuron_group": "heldout",
            "primary_bits_per_spike": 0.1,
        },
        split_metrics=_split_metrics(),
        neuron_metrics=_neuron_metrics(),
    )

    report = report_path.read_text(encoding="utf-8")
    assert "Primary metric: validation heldout bits/spike" in report
    assert "Fit policy: train trials only" in report


def test_cosmoothing_report_includes_disclaimers(tmp_path: Path) -> None:
    report_path = write_cosmoothing_markdown_report(
        tmp_path / "cosmoothing_report.md",
        metrics_summary={
            "dataset_name": "mc_maze_small",
            "dataset_hash": "abc",
            "input_neuron_group": "heldin",
            "target_neuron_group": "heldout",
            "smoothing": {"method": "gaussian", "sigma_ms": 50.0},
            "decoder_name": "ridge",
            "decoder_alpha": 100.0,
            "fit_policy": "train trials only",
            "standardization_policy": "train-only held-in features",
            "reference_policy": "train-only held-out mean rates",
            "primary_split": "validation",
            "primary_bits_per_spike": 0.1,
            "primary_poisson_nll": 5.0,
        },
        split_metrics=_split_metrics(),
    )

    report = report_path.read_text(encoding="utf-8")
    assert "local co-smoothing sanity baseline, not an official NLB leaderboard result" in report
    assert "No neural network model was trained" in report


def test_cosmoothing_sweep_report_includes_disclaimers_and_best_config(tmp_path: Path) -> None:
    report_path = write_cosmoothing_sweep_markdown_report(
        tmp_path / "sweep_report.md",
        summary={
            "dataset_name": "mc_maze_small",
            "dataset_hash": "abc",
            "sweep_grid": {"smoothing_sigma_ms": [10.0], "ridge_alpha": [1.0]},
            "n_configurations": 1,
            "best_validation_bits_per_spike": 0.2,
            "best_validation_poisson_nll": 4.0,
            "best_config": {
                "smoothing_sigma_ms": 10.0,
                "ridge_alpha": 1.0,
                "standardize_features": True,
                "fit_intercept": False,
            },
            "all_validation_bits_per_spike_negative": False,
        },
        best_split_metrics=pd.DataFrame(
            {"split": ["validation"], "bits_per_spike": [0.2], "poisson_nll": [4.0]}
        ),
    )

    report = report_path.read_text(encoding="utf-8")
    assert "local co-smoothing diagnostic sweep, not an official NLB leaderboard result" in report
    assert "No neural network model was trained" in report
    assert "Best smoothing sigma: 10.0" in report
    assert "Best ridge alpha: 1.0" in report


def test_factor_latent_report_includes_required_disclaimers(tmp_path: Path) -> None:
    report_path = write_factor_latent_markdown_report(
        tmp_path / "factor_latent_report.md",
        metrics_summary={
            "dataset_name": "mc_maze_small",
            "dataset_hash": "abc",
            "model_name": "factor analysis latent baseline",
            "input_neuron_group": "heldin",
            "target_neuron_group": "heldout",
            "smoothing": {"method": "gaussian", "sigma_ms": 50.0},
            "latent_dim": 2,
            "heldout_decoder_name": "ridge",
            "heldout_decoder_alpha": 1.0,
            "behavior_decoder_enabled": True,
            "behavior_decoder_alpha": 1.0,
            "fit_policy": "train trials only",
            "standardization_policy": "train-only statistics",
            "reference_policy": "train-only held-out mean rates",
            "primary_bits_per_spike": 0.1,
            "primary_poisson_nll": 5.0,
            "primary_behavior_mean_r2": 0.2,
        },
        split_metrics=pd.DataFrame(
            {"split": ["validation"], "bits_per_spike": [0.1], "poisson_nll": [5.0]}
        ),
        behavior_metrics=pd.DataFrame({"split": ["validation"], "target_name": ["x"], "r2": [0.2]}),
    )

    report = report_path.read_text(encoding="utf-8")
    assert "local latent-variable sanity baseline, not an official NLB leaderboard result" in report
    assert "No neural network model was trained" in report
    assert "GPFA-style only; no temporal GP prior is implemented" in report


def test_factor_latent_sweep_report_includes_disclaimers_and_best_config(
    tmp_path: Path,
) -> None:
    report_path = write_factor_latent_sweep_markdown_report(
        tmp_path / "sweep_report.md",
        summary={
            "dataset_name": "mc_maze_small",
            "dataset_hash": "abc",
            "sweep_grid": {"latent_dim": [2], "heldout_decoder_alpha": [1.0]},
            "n_configurations": 1,
            "best_validation_bits_per_spike": 0.2,
            "best_validation_poisson_nll": 4.0,
            "best_validation_behavior_mean_r2": 0.3,
            "best_config": {
                "latent_dim": 2,
                "smoothing_sigma_ms": 50.0,
                "heldout_decoder_alpha": 1.0,
                "standardize_features": True,
            },
            "single_factor_latent_validation_bits_per_spike": 0.04747691544524409,
            "mean_rate_validation_heldout_bits_per_spike": 0.5465273967210786,
        },
        best_split_metrics=pd.DataFrame(
            {"split": ["validation"], "bits_per_spike": [0.2], "poisson_nll": [4.0]}
        ),
    )

    report = report_path.read_text(encoding="utf-8")
    assert "local factor latent diagnostic sweep, not an official NLB leaderboard result" in report
    assert "No neural network model was trained" in report
    assert "not full GPFA because no temporal GP prior is implemented" in report
    assert "Best latent dimension: 2" in report
    assert "Best held-out decoder alpha: 1.0" in report


def test_lfads_evaluation_report_includes_disclaimers_and_references(tmp_path: Path) -> None:
    report_path = write_lfads_gru_evaluation_report(
        tmp_path / "lfads_gru_eval_report.md",
        {
            "dataset_name": "mc_maze_small",
            "dataset_hash": "abc",
            "checkpoint_path": "results/mc_maze_small/lfads_gru/checkpoints/best_validation.pt",
            "model_name": "lfads_gru",
            "factor_dim": 32,
            "latent_dim": 16,
            "heldout_decoder_alpha": 1000.0,
            "behavior_decoder_enabled": True,
            "behavior_decoder_alpha": 100.0,
            "fit_policy": "train trials only",
            "primary_split": "validation",
            "primary_bits_per_spike": 0.2,
            "primary_poisson_nll": 4.0,
            "primary_behavior_mean_r2": 0.1,
            "primary_prediction_source": "direct_model",
            "direct_model_available": True,
            "factor_decoder_evaluated": True,
            "direct_model_validation_bits_per_spike": 0.2,
            "factor_decoder_validation_bits_per_spike": 0.15,
            "previous_lfads_eval_validation_bits_per_spike": -0.01,
            "beats_previous_lfads_eval": True,
            "mean_rate_validation_bits_per_spike": 0.5,
            "factor_latent_best_validation_bits_per_spike": 0.125,
        },
    )

    report = report_path.read_text(encoding="utf-8")
    assert "LFADS-style sequential VAE foundation, not a full LFADS implementation" in report
    assert "local held-out evaluation, not an official NLB leaderboard result" in report
    assert "No new neural network model was trained by this evaluation script" in report
    assert "Primary prediction source: direct_model" in report
    assert "Direct model validation bits/spike: 0.2" in report
    assert "Factor decoder validation bits/spike: 0.15" in report
    assert "Mean-rate validation bits/spike: 0.5" in report
    assert "Factor latent best validation bits/spike: 0.125" in report
    assert "Previous LFADS-style held-out bits/spike: -0.01" in report


def test_window_matched_report_includes_required_caveats(tmp_path: Path) -> None:
    summary = {
        "dataset_name": "mc_maze_small",
        "dataset_hash": "abc",
        "original_time_bins": 2051,
        "cropped_time_bins": 256,
        "window_seconds": 1.28,
        "best_method_name": "mean_rate_windowed",
        "best_prediction_source": "constant_rate",
        "best_validation_bits_per_spike": 0.2,
        "full_window_mean_rate_bits_per_spike": 0.5,
        "full_window_factor_latent_best_bits_per_spike": 0.1,
    }
    leaderboard = pd.DataFrame(
        {
            "method_name": ["mean_rate_windowed"],
            "prediction_source": ["constant_rate"],
            "bits_per_spike": [0.2],
            "poisson_nll": [4.0],
            "behavior_mean_r2": [float("nan")],
        }
    )

    report_path = write_window_matched_comparison_report(
        tmp_path / "comparison_report.md",
        summary,
        leaderboard,
    )

    report = report_path.read_text(encoding="utf-8")
    assert "Window duration: 1.28 seconds" in report
    assert "full-window numbers are not directly comparable" in report
    assert "not an official NLB leaderboard result" in report
    assert "LFADS-style only, not full LFADS" in report
    assert "No new neural network model was trained" in report
