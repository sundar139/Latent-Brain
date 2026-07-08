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
    write_lfads_audit_report,
    write_lfads_coordinated_dropout_report,
    write_lfads_gru_evaluation_report,
    write_lfads_rate_calibration_report,
    write_lfads_tuning_report,
    write_metric_audit_report,
    write_temporal_rebinning_report,
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


def test_lfads_tuning_report_includes_required_statements(tmp_path: Path) -> None:
    report_path = write_lfads_tuning_report(
        tmp_path / "tuning_report.md",
        {
            "dataset_name": "mc_maze_small",
            "dataset_hash": "abc",
            "window_time_bins": 256,
            "window_seconds": 1.28,
            "cuda_device": "Unit GPU",
            "runs_attempted": 2,
            "successful_runs": 1,
            "best_run_id": "run_000",
            "best_run_params": {"latent_dim": 16, "factor_dim": 32},
            "best_validation_bits_per_spike": 0.2,
            "best_validation_poisson_nll": 5.0,
            "best_validation_behavior_mean_r2": 0.1,
            "beats_window_matched_mean_rate": False,
            "beats_window_matched_factor_latent": True,
            "beats_previous_lfads_masked_direct": True,
            "baseline_references": {
                "window_matched_mean_rate_validation_bits_per_spike": 0.7,
                "window_matched_factor_latent_validation_bits_per_spike": 0.03,
                "previous_lfads_masked_direct_validation_bits_per_spike": -0.04,
            },
        },
        pd.DataFrame(
            {
                "rank": [1],
                "run_id": ["run_000"],
                "validation_bits_per_spike": [0.2],
                "validation_poisson_nll": [5.0],
                "validation_behavior_mean_r2": [0.1],
            }
        ),
    )

    report = report_path.read_text(encoding="utf-8")
    assert "CUDA device: Unit GPU" in report
    assert "window-matched mean-rate" in report
    assert "This is local validation tuning only, not an official NLB leaderboard result." in report
    assert "The model is LFADS-style only, not full LFADS." in report
    assert "Generated checkpoints are local and ignored by Git." in report


def test_lfads_audit_report_includes_required_statements(tmp_path: Path) -> None:
    report_path = write_lfads_audit_report(
        tmp_path / "audit_report.md",
        {
            "dataset_name": "mc_maze_small",
            "dataset_hash": "abc",
            "window_time_bins": 256,
            "cuda_device": "Unit GPU",
            "checkpoint_audited": "checkpoint.pt",
            "validation_bits_per_spike": 0.1,
            "mean_rate_reference_bits_per_spike": 0.7,
            "mean_predicted_rate_hz": 2.0,
            "observed_rate_hz": 3.0,
            "prediction_reference_correlation": 0.5,
            "active_factor_count": 1,
            "tiny_overfit_initial_loss": 10.0,
            "tiny_overfit_final_loss": 5.0,
            "tiny_overfit_loss_drop_fraction": 0.5,
            "likely_issue_flags": ["underfitting", "rate underprediction"],
        },
    )

    report = report_path.read_text(encoding="utf-8")
    assert "Calibration summary" in report
    assert "Tiny subset overfit" in report
    assert "The model is LFADS-style only, not full LFADS." in report
    assert "This is a local diagnostic audit, not an official NLB leaderboard result." in report


def test_lfads_rate_calibration_report_includes_references_and_interpretation(
    tmp_path: Path,
) -> None:
    report_path = write_lfads_rate_calibration_report(
        tmp_path / "calibration_report.md",
        {
            "dataset_name": "mc_maze_small",
            "dataset_hash": "abc",
            "bin_size_ms": 20,
            "window_seconds": 1.28,
            "cuda_device": "Unit GPU",
            "existing_checkpoint_path": "checkpoint.pt",
            "raw_lfads_validation_bits_per_spike": 0.01,
            "multiplicative_calibrated_validation_bits_per_spike": 0.02,
            "log_bias_calibrated_validation_bits_per_spike": 0.02,
            "best_blend_alpha": 0.0,
            "best_blend_validation_bits_per_spike": 0.0,
            "initialized_lfads_validation_bits_per_spike": 0.03,
            "same_bin_mean_rate_reference": 0.7,
            "same_bin_factor_latent_reference": 0.03,
            "calibration_improves_lfads": True,
            "initialization_improves_lfads": True,
            "beats_same_bin_factor_latent": False,
            "beats_same_bin_mean_rate": False,
            "best_lfads_family_method": "initialized_lfads",
        },
    )

    report = report_path.read_text(encoding="utf-8")
    assert "Same-bin mean-rate reference: 0.7" in report
    assert "Same-bin factor-latent reference: 0.03" in report
    assert "If alpha near 0 is best" in report
    assert "rate scale calibration is an issue" in report
    assert "poor output anchoring is an issue" in report
    assert "not an official NLB leaderboard result" in report
    assert "LFADS-style only, not full LFADS" in report


def test_lfads_coordinated_dropout_report_includes_references_and_interpretation(
    tmp_path: Path,
) -> None:
    report_path = write_lfads_coordinated_dropout_report(
        tmp_path / "coordinated_dropout_report.md",
        {
            "dataset_name": "mc_maze_small",
            "dataset_hash": "abc",
            "bin_size_ms": 20,
            "window_seconds": 1.28,
            "cuda_device": "Unit GPU",
            "dropout_rates_tested": [0.1, 0.25],
            "best_dropout_rate": 0.1,
            "best_validation_bits_per_spike": 0.02,
            "best_validation_poisson_nll": 1.2,
            "best_validation_factor_decoder_bits_per_spike": 0.01,
            "same_bin_mean_rate_reference": 0.7,
            "same_bin_factor_latent_reference": 0.03,
            "previous_20ms_lfads_reference": 0.01,
            "coordinated_dropout_improves_lfads": True,
            "beats_same_bin_factor_latent": False,
            "beats_same_bin_mean_rate": False,
        },
        pd.DataFrame(
            {
                "run_id": ["dropout_0p10"],
                "dropout_rate": [0.1],
                "validation_bits_per_spike": [0.02],
                "validation_poisson_nll": [1.2],
            }
        ),
    )

    report = report_path.read_text(encoding="utf-8")
    assert "Same-bin mean-rate reference: 0.7" in report
    assert "Same-bin factor-latent reference: 0.03" in report
    assert "If low dropout helps" in report
    assert "If high dropout hurts" in report
    assert "If none help" in report
    assert "not an official NLB leaderboard result" in report
    assert "LFADS-style only, not full LFADS" in report


def test_metric_audit_report_includes_formula_and_conclusions(tmp_path: Path) -> None:
    report_path = write_metric_audit_report(
        tmp_path / "metric_audit_report.md",
        {
            "dataset_name": "mc_maze_small",
            "dataset_hash": "abc",
            "bin_size_ms": 20,
            "window_seconds": 1.28,
            "reference_name": "train_heldout_mean_rate",
            "train_mean_as_model_validation_bits_per_spike": 0.0,
            "best_oracle_validation_bits_per_spike": 0.5,
            "previous_mean_rate_directly_comparable": False,
            "metric_reference_mismatch_found": True,
            "mean_rate_inflation_found": True,
            "neural_models_trail_under_unified_scoring": True,
            "likely_conclusion": "reported mean-rate used a different reference convention",
        },
        pd.DataFrame(
            {
                "method_name": ["train_heldout_mean_rate"],
                "split": ["validation"],
                "bits_per_spike": [0.0],
                "reference_log_likelihood": [-1.0],
                "prediction_source": ["constant_rate"],
            }
        ),
        pd.DataFrame(
            {"control_name": ["oracle"], "split": ["validation"], "bits_per_spike": [0.5]}
        ),
        pd.DataFrame(
            {"control_name": ["random"], "split": ["validation"], "bits_per_spike": [-0.1]}
        ),
        pd.DataFrame({"method_name": ["reported"], "directly_comparable": [False]}),
    )

    report = report_path.read_text(encoding="utf-8")
    assert "(model_log_likelihood - reference_log_likelihood) / (log(2) * spike_count)" in report
    assert "Metric/reference mismatch found: True" in report
    assert "Oracle controls are not valid models" in report
    assert "not an official NLB leaderboard result" in report


def test_temporal_rebinning_report_includes_required_statements(tmp_path: Path) -> None:
    report_path = write_temporal_rebinning_report(
        tmp_path / "temporal_rebinning_report.md",
        {
            "dataset_name": "mc_maze_small",
            "dataset_hash": "abc",
            "original_bin_size_ms": 5,
            "target_bin_sizes_ms": [5, 10, 20],
            "window_seconds": 1.28,
            "coarser_bins_reduce_zero_fraction": True,
            "lfads_improves_at_coarser_bins": True,
            "lfads_beat_same_bin_mean_rate": False,
        },
        pd.DataFrame({"bin_size_ms": [5], "split": ["validation"], "zero_fraction": [0.9]}),
        pd.DataFrame({"bin_size_ms": [5], "method_name": ["mean_rate"], "bits_per_spike": [0.1]}),
        pd.DataFrame({"bin_size_ms": [10], "run_id": ["bin_10ms"], "bits_per_spike": [0.2]}),
    )
    report = report_path.read_text(encoding="utf-8")
    assert "Bits/spike values across different bin sizes are diagnostic" in report
    assert "not an official NLB leaderboard result" in report
    assert "The model is LFADS-style only, not full LFADS." in report
