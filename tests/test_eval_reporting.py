from __future__ import annotations

import json
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from latentbrain.eval.reporting import (
    write_baseline_markdown_report,
    write_baseline_outputs,
    write_cosmoothing_markdown_report,
    write_cosmoothing_sweep_markdown_report,
    write_cv_rate_audit_outputs,
    write_factor_latent_markdown_report,
    write_factor_latent_sweep_markdown_report,
    write_lfads_audit_report,
    write_lfads_controller_tuning_report,
    write_lfads_coordinated_dropout_report,
    write_lfads_gru_evaluation_report,
    write_lfads_rate_calibration_report,
    write_lfads_tuning_report,
    write_lfads_unified_tuning_report,
    write_metric_audit_report,
    write_neural_ode_objective_outputs,
    write_neural_ode_refinement_outputs,
    write_neural_ode_tuning_report,
    write_neural_sde_tuning_report,
    write_seed_robustness_outputs,
    write_split_audit_outputs,
    write_stratified_cv_outputs,
    write_switching_ode_tuning_outputs,
    write_temporal_rebinning_report,
    write_unified_scoreboard_report,
    write_window_audit_outputs,
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


def test_unified_scoreboard_report_includes_formula_and_warnings(tmp_path: Path) -> None:
    report_path = write_unified_scoreboard_report(
        tmp_path / "unified_scoreboard_report.md",
        {
            "dataset_name": "mc_maze_small",
            "dataset_hash": "abc",
            "bin_size_ms": 20,
            "window_seconds": 1.28,
            "reference_model": "train_heldout_mean_rate",
            "train_mean_validation_bits_per_spike": 0.0,
            "best_valid_model": "factor_latent",
            "best_valid_model_validation_bits_per_spike": 0.03,
            "best_lfads_family_method": "coordinated_dropout_lfads",
            "best_lfads_family_validation_bits_per_spike": 0.01,
            "best_lfads_family_source_summary_path": "results/lfads/summary.json",
            "lfads_family_beats_factor_latent": False,
            "oracle_validation_bits_per_spike": 3.0,
        },
        pd.DataFrame(
            {
                "rank": [1, 2],
                "method_name": ["factor_latent", "oracle_smoothed_heldout"],
                "prediction_source": ["factor_decoder", "oracle"],
                "valid_model": [True, False],
                "validation_bits_per_spike": [0.03, 3.0],
                "validation_poisson_nll": [1.0, None],
                "reference_name": ["train_heldout_mean_rate", "train_heldout_mean_rate"],
                "beats_train_mean_reference": [True, True],
                "beats_factor_latent_reference": [False, True],
                "is_oracle_control": [False, True],
                "notes": ["", "Oracle diagnostic; invalid model"],
            }
        ),
        pd.DataFrame(
            {
                "metric_name": ["old_mean"],
                "value": [0.7],
                "status": ["historical_only_not_directly_comparable"],
                "reason": ["not directly comparable"],
            }
        ),
    )

    report = report_path.read_text(encoding="utf-8")
    assert "(model_log_likelihood - reference_log_likelihood) / (log(2) * spike_count)" in report
    assert "Old mean-rate values are historical-only" in report
    assert "Best LFADS-family method: coordinated_dropout_lfads" in report
    assert "LFADS-family beats factor-latent: False" in report
    assert "fresh clone" in report
    assert "results/lfads/summary.json" in report
    assert "Oracle diagnostic score: 3.0 (invalid model)" in report
    assert "not an official NLB leaderboard result" in report


def test_lfads_unified_tuning_report_includes_required_statements(tmp_path: Path) -> None:
    report_path = write_lfads_unified_tuning_report(
        tmp_path / "unified_tuning_report.md",
        {
            "dataset_name": "mc_maze_small",
            "dataset_hash": "abc",
            "bin_size_ms": 20,
            "window_seconds": 1.28,
            "cuda_device": "Unit GPU",
            "reference_model": "train_heldout_mean_rate",
            "runs_attempted": 1,
            "successful_runs": 1,
            "best_run_id": "run_000",
            "best_run_params": {"latent_dim": 16},
            "best_validation_unified_bits_per_spike": 0.02,
            "factor_latent_unified_reference": 0.03,
            "previous_best_lfads_family_reference": 0.01,
            "beats_factor_latent_unified": False,
            "beats_previous_best_lfads_family": True,
        },
        pd.DataFrame({"run_id": ["run_000"]}),
        pd.DataFrame(
            {
                "rank": [1],
                "run_id": ["run_000"],
                "validation_unified_bits_per_spike": [0.02],
                "validation_poisson_nll": [5.0],
                "beats_factor_latent_unified": [False],
                "notes": [""],
            }
        ),
    )

    report = report_path.read_text(encoding="utf-8")
    assert "Canonical reference model: train_heldout_mean_rate" in report
    assert "Factor-latent unified reference: 0.03" in report
    assert "Old incompatible mean-rate values are not used as tuning targets" in report
    assert "not an official NLB leaderboard result" in report
    assert "LFADS-style only, not full LFADS" in report


def test_lfads_controller_tuning_report_includes_required_statements(tmp_path: Path) -> None:
    report_path = write_lfads_controller_tuning_report(
        tmp_path / "controller_tuning_report.md",
        {
            "dataset_name": "mc_maze_small",
            "dataset_hash": "abc",
            "bin_size_ms": 20,
            "window_seconds": 1.28,
            "cuda_device": "Unit GPU",
            "reference_model": "train_heldout_mean_rate",
            "train_mean_validation_bits_per_spike": 0.0,
            "runs_attempted": 1,
            "successful_runs": 1,
            "best_run_id": "run_000",
            "best_run_params": {"latent_dim": 16},
            "best_validation_unified_bits_per_spike": 0.02,
            "factor_latent_unified_reference": 0.03,
            "previous_best_lfads_family_reference": 0.01,
            "beats_factor_latent_unified": False,
            "beats_previous_best_lfads_family": True,
        },
        pd.DataFrame({"run_id": ["run_000"]}),
        pd.DataFrame(
            {
                "rank": [1],
                "run_id": ["run_000"],
                "validation_unified_bits_per_spike": [0.02],
                "validation_poisson_nll": [5.0],
                "beats_factor_latent_unified": [False],
                "beats_previous_best_lfads_family": [True],
                "notes": [""],
            }
        ),
    )

    report = report_path.read_text(encoding="utf-8")
    assert "Canonical reference model: train_heldout_mean_rate" in report
    assert "Factor-latent unified reference: 0.03" in report
    assert "near-zero KL may indicate posterior underuse" in report
    assert "Old incompatible mean-rate values are not used as tuning targets" in report
    assert "not an official NLB leaderboard result" in report
    assert "LFADS-style with inferred inputs, not full LFADS" in report


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


def test_neural_sde_tuning_report_includes_required_statements(tmp_path: Path) -> None:
    report_path = write_neural_sde_tuning_report(
        tmp_path / "neural_sde_tuning_report.md",
        {
            "dataset_name": "mc_maze_small",
            "dataset_hash": "abc",
            "bin_size_ms": 20,
            "window_seconds": 1.28,
            "cuda_device": "Unit GPU",
            "reference_model": "train_heldout_mean_rate",
            "train_mean_validation_bits_per_spike": 0.0,
            "runs_attempted": 1,
            "successful_runs": 1,
            "best_run_id": "run_000",
            "best_run_params": {"diffusion_scale": 0.03},
            "best_validation_unified_bits_per_spike": 0.02,
            "factor_latent_unified_reference": 0.03,
            "previous_best_lfads_family_reference": 0.01,
            "beats_factor_latent_unified": False,
            "beats_previous_best_lfads_family": True,
            "best_drift_norm": 0.4,
            "best_diffusion_mean": 0.01,
        },
        pd.DataFrame(
            {
                "rank": [1],
                "run_id": ["run_000"],
                "validation_unified_bits_per_spike": [0.02],
                "validation_poisson_nll": [2.0],
                "diffusion_scale": [0.03],
                "beats_factor_latent_unified": [False],
                "beats_previous_best_lfads_family": [True],
                "notes": [""],
            }
        ),
    )

    report = report_path.read_text(encoding="utf-8")
    assert "Canonical reference model: train_heldout_mean_rate" in report
    assert "Factor-latent unified reference: 0.03" in report
    assert "nonzero diffusion tests stochastic latent dynamics" in report
    assert "Old incompatible mean-rate values are not used as tuning targets" in report
    assert "not an official NLB leaderboard result" in report


def test_neural_ode_tuning_report_includes_required_statements(tmp_path: Path) -> None:
    report_path = write_neural_ode_tuning_report(
        tmp_path / "neural_ode_tuning_report.md",
        {
            "dataset_name": "mc_maze_small",
            "dataset_hash": "abc",
            "bin_size_ms": 20,
            "window_seconds": 1.28,
            "cuda_device": "Unit GPU",
            "reference_model": "train_heldout_mean_rate",
            "train_mean_validation_bits_per_spike": 0.0,
            "runs_attempted": 1,
            "successful_runs": 1,
            "best_run_id": "run_000",
            "best_run_params": {"latent_dim": 32},
            "best_validation_unified_bits_per_spike": 0.02,
            "best_checkpoint_source": "latest",
            "checkpoint_selection_method": "post_training_unified_rerank",
            "factor_latent_unified_reference": 0.03,
            "previous_neural_sde_reference": 0.025,
            "previous_best_lfads_family_reference": 0.01,
            "beats_factor_latent_unified": False,
            "beats_previous_neural_sde": False,
            "best_drift_norm": 0.4,
            "best_diffusion_mean": 0.0,
        },
        pd.DataFrame(
            {
                "rank": [1],
                "run_id": ["run_000"],
                "validation_unified_bits_per_spike": [0.02],
                "validation_poisson_nll": [2.0],
                "best_checkpoint_source": ["latest"],
                "beats_factor_latent_unified": [False],
                "beats_previous_neural_sde": [False],
                "notes": [""],
            }
        ),
        pd.DataFrame(
            {
                "run_id": ["run_000"],
                "checkpoint_source": ["latest"],
                "epoch": [1],
                "validation_total_loss": [2.0],
                "validation_unified_bits_per_spike": [0.02],
                "selected_by_loss": [False],
                "selected_by_unified": [True],
            }
        ),
    )

    report = report_path.read_text(encoding="utf-8")
    assert "Canonical reference model: train_heldout_mean_rate" in report
    assert "Factor-latent unified reference: 0.03" in report
    assert "Checkpoint selection: post_training_unified_rerank" in report
    assert "Deterministic latent dynamics are tested" in report
    assert "Old incompatible mean-rate values are not used as tuning targets" in report
    assert "not an official NLB leaderboard result" in report


def test_switching_tuning_report_includes_references_and_disclaimers(tmp_path: Path) -> None:
    summary = {
        "dataset_name": "mc_maze_small",
        "dataset_hash": "abc",
        "bin_size_ms": 20,
        "window_seconds": 1.28,
        "cuda_device": "Unit GPU",
        "reference_model": "train_heldout_mean_rate",
        "train_mean_validation_bits_per_spike": 0.0,
        "runs_attempted": 1,
        "successful_runs": 1,
        "best_run_id": "run_000",
        "best_run_params": {"n_regimes": 2},
        "best_validation_unified_bits_per_spike": 0.02,
        "best_validation_poisson_nll": 2.0,
        "best_factor_decoder_unified_bits_per_spike": 0.01,
        "best_checkpoint_source": "latest",
        "factor_latent_unified_reference": 0.03,
        "previous_neural_ode_reference": 0.028,
        "previous_neural_sde_reference": 0.026,
        "beats_factor_latent_unified": False,
        "beats_previous_neural_ode": False,
        "best_active_regime_count": 2,
        "best_mean_regime_entropy": 0.5,
        "best_max_regime_occupancy": 0.6,
    }
    leaderboard = pd.DataFrame(
        [
            {
                "rank": 1,
                "run_id": "run_000",
                "validation_unified_bits_per_spike": 0.02,
                "validation_poisson_nll": 2.0,
                "active_regime_count": 2,
                "mean_regime_entropy": 0.5,
                "best_checkpoint_source": "latest",
                "beats_factor_latent_unified": False,
                "beats_previous_neural_ode": False,
                "notes": "",
            }
        ]
    )
    regime = pd.DataFrame(
        [
            {
                "split": "validation",
                "regime_index": 0,
                "mean_occupancy": 0.6,
                "std_occupancy": 0.1,
                "min_probability": 0.1,
                "max_probability": 0.9,
                "entropy": 0.5,
                "active": True,
            }
        ]
    )

    paths = write_switching_ode_tuning_outputs(
        tmp_path, summary, pd.DataFrame(), leaderboard, pd.DataFrame(), regime
    )
    text = paths["report"].read_text(encoding="utf-8")

    assert "Canonical reference model: train_heldout_mean_rate" in text
    assert "Factor-latent unified reference: 0.03" in text
    assert "If one regime dominates" in text
    assert "Old incompatible mean-rate values are not used as tuning targets" in text
    assert "not an official NLB leaderboard result" in text


def test_neural_ode_refinement_report_includes_requested_statements(tmp_path: Path) -> None:
    summary = {
        "dataset_name": "mc_maze_small",
        "dataset_hash": "abc",
        "bin_size_ms": 20,
        "window_seconds": 1.28,
        "cuda_device": "Unit GPU",
        "reference_model": "train_heldout_mean_rate",
        "train_mean_validation_bits_per_spike": 0.0,
        "runs_attempted": 1,
        "successful_runs": 1,
        "best_run_id": "run_000",
        "best_run_params": {"drift_regularization": 1.0e-5, "scheduler": "cosine"},
        "best_validation_unified_bits_per_spike": 0.02,
        "best_validation_poisson_nll": 2.0,
        "best_factor_decoder_unified_bits_per_spike": 0.01,
        "best_checkpoint_source": "latest",
        "factor_latent_unified_reference": 0.03,
        "previous_neural_ode_reference": 0.028,
        "previous_switching_ode_reference": 0.006,
        "beats_factor_latent_unified": False,
        "beats_previous_neural_ode": False,
        "best_drift_norm": 0.5,
        "best_drift_regularization_loss": 0.001,
        "best_learning_rate": 0.0001,
    }
    leaderboard = pd.DataFrame(
        [
            {
                "rank": 1,
                "run_id": "run_000",
                "validation_unified_bits_per_spike": 0.02,
                "validation_poisson_nll": 2.0,
                "validation_factor_decoder_unified_bits_per_spike": 0.01,
                "input_dropout_rate": 0.25,
                "heldout_loss_weight": 8.0,
                "kl_warmup_epochs": 10,
                "kl_scale": 0.01,
                "drift_regularization": 1.0e-5,
                "scheduler": "cosine",
                "latent_dim": 32,
                "factor_dim": 32,
                "best_checkpoint_source": "latest",
                "beats_factor_latent_unified": False,
                "beats_previous_neural_ode": False,
                "notes": "",
            }
        ]
    )
    checkpoints = pd.DataFrame(
        [{"run_id": "run_000", "checkpoint_source": "latest", "selected_by_unified": True}]
    )

    paths = write_neural_ode_refinement_outputs(
        tmp_path, summary, pd.DataFrame(), leaderboard, checkpoints
    )
    text = paths["report"].read_text(encoding="utf-8")

    assert "Canonical reference model: train_heldout_mean_rate" in text
    assert "Factor-latent unified reference: 0.03" in text
    assert "Drift regularization" in text
    assert "Scheduler / learning-rate" in text
    assert "switching collapsed to one regime" in text
    assert "Old incompatible mean-rate values are not used as tuning targets" in text
    assert "not an official NLB leaderboard result" in text


def test_neural_ode_objective_report_documents_objective_diagnostics(tmp_path: Path) -> None:
    summary = {
        "dataset_name": "mc_maze_small",
        "dataset_hash": "abc",
        "bin_size_ms": 20,
        "window_seconds": 1.28,
        "cuda_device": "Unit GPU",
        "reference_model": "train_heldout_mean_rate",
        "train_mean_validation_bits_per_spike": 0.0,
        "runs_attempted": 2,
        "successful_runs": 2,
        "best_run_id": "run_001_heldout_heavy",
        "best_objective_name": "heldout_heavy",
        "best_run_params": {"heldout_loss_weight": 10.0, "zero_count_weight": 1.0},
        "best_validation_unified_bits_per_spike": 0.029,
        "best_validation_poisson_nll": 2000.0,
        "best_factor_decoder_unified_bits_per_spike": 0.01,
        "best_heldout_loss_weight": 10.0,
        "best_zero_count_weight": 1.0,
        "best_positive_count_weight": 1.0,
        "best_rate_calibration_loss_weight": 0.0,
        "best_rate_calibration_loss": 0.0,
        "best_drift_norm": 0.5,
        "best_drift_regularization_loss": 0.001,
        "best_diffusion_mean": 0.0,
        "best_checkpoint_source": "latest",
        "factor_latent_unified_reference": 0.0316438194429199,
        "previous_neural_ode_refinement_reference": 0.0283514699322505,
        "switching_ode_reference": 0.0065057546390714,
        "beats_factor_latent_unified": False,
        "beats_previous_neural_ode_refinement": True,
        "old_incompatible_mean_rate_values_used_as_targets": False,
    }
    leaderboard = pd.DataFrame(
        [
            {
                "rank": 1,
                "run_id": "run_001_heldout_heavy",
                "objective_name": "heldout_heavy",
                "validation_unified_bits_per_spike": 0.029,
                "validation_poisson_nll": 2000.0,
                "validation_factor_decoder_unified_bits_per_spike": 0.01,
                "heldout_loss_weight": 10.0,
                "zero_count_weight": 1.0,
                "positive_count_weight": 1.0,
                "rate_calibration_loss_weight": 0.0,
                "kl_scale": 0.01,
                "drift_regularization": 1.0e-4,
                "input_dropout_rate": 0.1,
                "best_checkpoint_source": "latest",
                "beats_factor_latent_unified": False,
                "beats_previous_neural_ode_refinement": True,
                "notes": "held-out emphasis",
            }
        ]
    )
    diagnostics = pd.DataFrame(
        [
            {
                "run_id": "run_001_heldout_heavy",
                "objective_name": "heldout_heavy",
                "heldout_loss_weight": 10.0,
                "zero_count_weight": 1.0,
                "positive_count_weight": 1.0,
                "rate_calibration_loss_weight": 0.0,
                "rate_calibration_loss": 0.0,
                "drift_regularization_loss": 0.001,
                "drift_norm": 0.5,
                "mean_predicted_rate": 0.56,
                "mean_observed_rate": 0.58,
            }
        ]
    )
    checkpoints = pd.DataFrame(
        [
            {
                "run_id": "run_001_heldout_heavy",
                "checkpoint_source": "latest",
                "selected_by_unified": True,
            }
        ]
    )

    paths = write_neural_ode_objective_outputs(
        tmp_path, summary, pd.DataFrame(), leaderboard, diagnostics, checkpoints
    )
    text = paths["report"].read_text(encoding="utf-8")

    assert paths["objective_diagnostics"].exists()
    assert "Canonical reference model: train_heldout_mean_rate" in text
    assert "Factor-latent unified reference: 0.0316438194429199" in text
    assert "Previous neural-ODE refinement reference: 0.0283514699322505" in text
    assert "Objective diagnostics" in text
    assert "Best zero count weight" in text
    assert "Best rate calibration loss" in text
    assert "Drift regularization loss" in text
    assert (
        "Evaluation uses canonical unweighted unified bits/spike even when training "
        "losses are weighted." in text
    )
    assert "Old incompatible mean-rate values are not used as tuning targets" in text
    assert "not an official NLB leaderboard result" in text


def _seed_robustness_summary() -> dict:
    return {
        "dataset_name": "mc_maze_small",
        "dataset_hash": "abc",
        "bin_size_ms": 20,
        "window_seconds": 1.28,
        "cuda_device": "Unit GPU",
        "reference_model": "train_heldout_mean_rate",
        "train_mean_validation_bits_per_spike": 0.0,
        "split_seed_mode": "fixed",
        "split_seed": 2027,
        "initialization_seed_mode": "varied",
        "seed_list_shared_across_methods": True,
        "methods_evaluated": ["factor_latent", "neural_ode_refinement"],
        "seeds_evaluated": [2027, 2028, 2029],
        "total_jobs": 6,
        "successful_jobs": 6,
        "method_config_hashes": {"factor_latent": "aaa", "neural_ode_refinement": "bbb"},
        "confidence_interval": 0.95,
        "bootstrap_repeats": 100,
        "bootstrap_seed": 1337,
        "best_mean_method": "factor_latent",
        "best_mean_validation_unified_bits_per_spike": 0.031,
        "best_lower_ci_method": "factor_latent",
        "best_lower_ci_validation_unified_bits_per_spike": 0.0303,
        "factor_latent_mean_validation_unified_bits_per_spike": 0.031,
        "best_neural_method": "neural_ode_refinement",
        "best_neural_method_mean_validation_unified_bits_per_spike": 0.011,
        "paired_mean_difference_best_neural_minus_factor_latent": -0.02,
        "any_neural_beats_factor_latent_mean": False,
        "any_neural_beats_factor_latent_lower_ci": False,
        "carried_forward_method": "factor_latent",
        "carried_forward_reason": "No neural method beats factor-latent across seeds.",
    }


def test_seed_robustness_report_documents_policy_statistics_and_recommendation(
    tmp_path: Path,
) -> None:
    method_summary = pd.DataFrame(
        [
            {
                "method_name": "factor_latent",
                "method_type": "factor_latent",
                "valid_model": True,
                "n_seeds": 3,
                "mean_validation_unified_bits_per_spike": 0.031,
                "std_validation_unified_bits_per_spike": 0.001,
                "median_validation_unified_bits_per_spike": 0.031,
                "min_validation_unified_bits_per_spike": 0.030,
                "max_validation_unified_bits_per_spike": 0.032,
                "ci95_low": 0.0303,
                "ci95_high": 0.0317,
                "mean_validation_poisson_nll": 1900.0,
                "mean_test_unified_bits_per_spike": 0.030,
                "beats_factor_latent_mean": False,
                "beats_factor_latent_lower_ci": False,
                "notes": "baseline",
            }
        ]
    )
    leaderboard = pd.DataFrame(
        [
            {
                "rank": 1,
                "method_name": "factor_latent",
                "method_type": "factor_latent",
                "mean_validation_unified_bits_per_spike": 0.031,
                "std_validation_unified_bits_per_spike": 0.001,
                "ci95_low": 0.0303,
                "ci95_high": 0.0317,
                "mean_test_unified_bits_per_spike": 0.030,
                "beats_factor_latent_mean": False,
                "beats_factor_latent_lower_ci": False,
                "valid_model": True,
                "notes": "baseline",
            }
        ]
    )
    seed_effects = pd.DataFrame(
        [
            {
                "seed": 2027,
                "method_a": "neural_ode_refinement",
                "method_b": "factor_latent",
                "metric": "validation_unified_bits_per_spike",
                "method_a_value": 0.011,
                "method_b_value": 0.031,
                "difference": -0.02,
            }
        ]
    )
    carried_forward = {"carried_forward_method": "factor_latent"}

    paths = write_seed_robustness_outputs(
        tmp_path,
        _seed_robustness_summary(),
        pd.DataFrame(),
        method_summary,
        leaderboard,
        seed_effects,
        carried_forward,
    )
    text = paths["report"].read_text(encoding="utf-8")

    assert paths["method_summary"].exists()
    assert paths["seed_effects"].exists()
    assert paths["carried_forward_config"].exists()

    assert "## Seed policy" in text
    assert "Split seed mode: fixed" in text
    assert "Initialization seed mode: varied" in text
    assert "Same seed list across methods: True" in text
    assert "seed + run_index" in text

    assert "Canonical reference model: train_heldout_mean_rate" in text
    assert "Train-mean-as-model equals 0.0 bits/spike." in text
    assert "ci95_low" in text
    assert "std_validation_unified_bits_per_spike" in text
    assert "Paired seed differences against factor-latent" in text
    assert "Paired mean difference (best neural minus factor-latent): -0.02" in text

    assert "Carried-forward method: factor_latent" in text
    assert "Any neural method beats factor-latent by mean: False" in text
    assert "Any neural method beats factor-latent by lower CI: False" in text

    assert "Old incompatible mean-rate values are not used as tuning targets" in text
    assert "not an official NLB leaderboard result" in text
    assert "Single-seed model leaderboards are not sufficient for claims." in text


def _split_audit_summary(risk: str = "high") -> dict:
    return {
        "dataset_name": "mc_maze_small",
        "dataset_hash": "abc",
        "bin_size_ms": 20,
        "window_seconds": 1.28,
        "accepted_split_seed": 2027,
        "train_trial_count": 70,
        "validation_trial_count": 15,
        "test_trial_count": 15,
        "heldin_neuron_count": 106,
        "heldout_neuron_count": 36,
        "behavior_available": True,
        "missing_behavior_variables": [],
        "validation_heldout_rate_hz": 0.61,
        "test_heldout_rate_hz": 0.55,
        "factor_latent_validation_mean": 0.029,
        "factor_latent_test_mean": -0.0083,
        "factor_latent_validation_test_gap": 0.0373,
        "generalization_risk": risk,
        "validation_test_instability_detected": True,
        "repeated_split_validation_mean": 0.028,
        "repeated_split_test_mean": -0.006,
        "repeated_split_test_positive_fraction": 0.1,
        "validation_positive_test_negative_persists": True,
    }


def test_split_audit_report_includes_gap_repeated_splits_and_risk(tmp_path: Path) -> None:
    split_statistics = pd.DataFrame(
        [{"split": "validation", "n_trials": 15, "mean_heldout_rate_hz": 0.61}]
    )
    behavior = pd.DataFrame([{"split": "test", "behavior_name": "hand_pos_x", "mean": 1.0}])
    gap_summary = pd.DataFrame(
        [
            {
                "method_name": "factor_latent",
                "mean_validation": 0.029,
                "mean_test": -0.0083,
                "mean_gap": 0.0373,
                "gap_ci95_low": 0.031,
                "gap_ci95_high": 0.043,
                "test_positive_fraction": 0.0,
                "generalization_risk": "high",
            }
        ]
    )
    repeated = pd.DataFrame(
        [
            {
                "split_seed": 2027,
                "method_name": "factor_latent",
                "validation_unified_bits_per_spike": 0.031,
                "test_unified_bits_per_spike": -0.008,
                "notes": "Train-only fit.",
            }
        ]
    )
    comparison = pd.DataFrame(
        [
            {
                "metric": "heldout_rate_hz",
                "split_a": "validation",
                "split_b": "test",
                "difference": 0.06,
                "standardized_difference": 0.8,
            }
        ]
    )

    paths = write_split_audit_outputs(
        tmp_path,
        _split_audit_summary(),
        pd.DataFrame([{"trial_index": 0}]),
        split_statistics,
        pd.DataFrame([{"split": "test", "neuron_index": 0}]),
        behavior,
        pd.DataFrame([{"method_name": "factor_latent", "seed": 2027}]),
        gap_summary,
        repeated,
        comparison,
    )
    text = paths["report"].read_text(encoding="utf-8")

    assert paths["repeated_split_factor_latent"].exists()
    assert paths["validation_test_gap"].exists()

    assert "Validation/test gap summary" in text
    assert "Factor-latent validation-test gap: 0.0373" in text
    assert "Repeated split baselines" in text
    assert "Repeated split test-positive fraction: 0.1" in text
    assert "Generalization risk: high" in text
    assert "Accepted split seed: 2027" in text
    assert "Validation trials: 15" in text
    assert "Test trials: 15" in text

    assert (
        "No model performance claim should be made until validation/test instability is resolved."
        in text
    )
    assert "not an official NLB leaderboard result" in text
    assert "Old incompatible mean-rate values are not used as tuning targets" in text
    assert "reported as unstable rather than conclusive" in text


def test_split_audit_report_notes_unavailable_gap_diagnostics(tmp_path: Path) -> None:
    paths = write_split_audit_outputs(
        tmp_path,
        _split_audit_summary("unresolved_missing_data"),
        pd.DataFrame(),
        pd.DataFrame([{"split": "test", "n_trials": 15}]),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame([{"split_seed": 2027, "method_name": "factor_latent"}]),
        pd.DataFrame([{"metric": "heldout_rate_hz"}]),
    )
    text = paths["report"].read_text(encoding="utf-8")

    assert "Model gap diagnostics are unavailable" in text
    assert "Behavior statistics are unavailable" in text
    assert "reported as unstable rather than conclusive" not in text


def test_unified_scoreboard_report_warns_on_high_generalization_risk(tmp_path: Path) -> None:
    summary = {
        "dataset_name": "mc_maze_small",
        "reference_model": "train_heldout_mean_rate",
        "generalization_risk": "high",
        "validation_test_instability_detected": True,
    }

    path = write_unified_scoreboard_report(
        tmp_path / "report.md", summary, pd.DataFrame(), pd.DataFrame()
    )
    text = path.read_text(encoding="utf-8")

    assert "interpreted as validation-only diagnostics" in text
    assert (
        "No model performance claim should be made until validation/test instability is resolved."
        in text
    )
    assert "Generalization risk: high" in text


def _cv_rate_audit_summary() -> dict:
    return {
        "dataset_name": "mc_maze_small",
        "dataset_hash": "abc",
        "bin_size_ms": 20,
        "window_seconds": 1.28,
        "reference_model": "train_heldout_mean_rate",
        "split_seeds": [2027, 2028, 2029],
        "factor_analysis_random_states": [0, 2027],
        "accepted_split_seed": 2027,
        "factor_latent_repeated_split_validation_mean": 0.0267,
        "factor_latent_repeated_split_validation_std": 0.0179,
        "factor_latent_repeated_split_test_mean": 0.0082,
        "factor_latent_repeated_split_test_std": 0.0135,
        "factor_latent_test_positive_fraction": 0.7,
        "between_split_test_variance": 0.00018,
        "within_split_random_state_test_variance": 0.00001,
        "split_variance_exceeds_random_state_variance": True,
        "factor_analysis_random_state_validation_range": 0.005,
        "factor_analysis_random_state_test_range": 0.004,
        "best_valid_rate_control_method": "train_rate_calibrated_factor_latent",
        "best_valid_rate_control_test_mean": 0.011,
        "split_mean_rate_invalid_test_mean": 0.0924,
        "invalid_split_mean_advantage_over_factor_latent": 0.084,
        "train_only_rate_calibration_helps": True,
        "rate_offset_explains_split_mean_advantage": True,
        "invalid_controls_dominate_valid_models": True,
        "invalid_controls_excluded_from_best_valid_model": True,
    }


def _cv_method_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method_name": "factor_latent",
                "valid_model": True,
                "n_scores": 20,
                "mean_test_unified_bits_per_spike": 0.0082,
                "notes": "train-only",
            },
            {
                "method_name": "split_mean_rate_invalid",
                "valid_model": False,
                "n_scores": 20,
                "mean_test_unified_bits_per_spike": 0.0924,
                "notes": "leaks evaluation targets",
            },
        ]
    )


def test_cv_rate_audit_report_includes_repeated_split_sensitivity_and_decomposition(
    tmp_path: Path,
) -> None:
    fa_sensitivity = pd.DataFrame(
        [
            {
                "split_seed": 2027,
                "factor_analysis_random_state": 0,
                "validation_unified_bits_per_spike": 0.0268,
                "test_unified_bits_per_spike": -0.0119,
                "difference_from_random_state_0_validation": 0.0,
                "difference_from_random_state_0_test": 0.0,
                "notes": "randomized SVD",
            }
        ]
    )
    decomposition = pd.DataFrame(
        [
            {
                "split_seed": 2027,
                "split": "test",
                "factor_latent_bits_per_spike": -0.0119,
                "train_rate_calibrated_factor_latent_bits_per_spike": -0.010,
                "split_mean_rate_invalid_bits_per_spike": 0.0924,
                "oracle_split_scaled_factor_latent_invalid_bits_per_spike": 0.05,
                "valid_calibration_gain": 0.0019,
                "invalid_oracle_gain": 0.0619,
                "split_mean_advantage_over_factor_latent": 0.1043,
                "rate_offset_explains_gap": True,
                "notes": "diagnostic only",
            }
        ]
    )
    recommendations = {
        "single_split_results_reportable": False,
        "recommended_reporting_mode": "repeated_split",
        "carried_forward_for_reporting": "factor_latent",
        "neural_models_carried_forward": False,
        "must_label_invalid": ["split_mean_rate_invalid"],
        "rate_offset_warning": "unmodeled split-level rate offset",
    }

    paths = write_cv_rate_audit_outputs(
        tmp_path,
        _cv_rate_audit_summary(),
        pd.DataFrame([{"split_seed": 2027}]),
        fa_sensitivity,
        pd.DataFrame([{"method_name": "factor_latent"}]),
        decomposition,
        _cv_method_summary(),
        recommendations,
    )
    text = paths["report"].read_text(encoding="utf-8")

    assert paths["reporting_recommendations"].exists()
    assert paths["method_summary"].exists()

    assert "Repeated-split factor-latent" in text
    assert "Test mean: 0.0082" in text
    assert "Test-positive fraction: 0.7" in text
    assert "FactorAnalysis random-state sensitivity" in text
    assert "Valid rate controls" in text
    assert "Invalid diagnostic controls" in text
    assert "split_mean_rate_invalid" in text
    assert "Rate-offset decomposition" in text
    assert "Train-only rate calibration helps: True" in text
    assert "Recommended reporting mode: repeated_split" in text

    assert "Single-split numbers are not reportable as final performance." in text
    assert (
        "Invalid controls use evaluation split targets and cannot be reported as model performance."
        in text
    )
    assert "not an official NLB leaderboard result" in text
    assert "Old incompatible mean-rate values are not used as tuning targets" in text


def test_unified_scoreboard_report_includes_repeated_split_reporting_warning(
    tmp_path: Path,
) -> None:
    summary = {
        "dataset_name": "mc_maze_small",
        "reference_model": "train_heldout_mean_rate",
        "generalization_risk": "high",
        "validation_test_instability_detected": True,
        "single_split_results_reportable": False,
        "recommended_reporting_mode": "repeated_split",
        "invalid_rate_controls_present": True,
        "rate_offset_warning": "unmodeled split-level rate offset",
    }

    path = write_unified_scoreboard_report(
        tmp_path / "report.md", summary, pd.DataFrame(), pd.DataFrame()
    )
    text = path.read_text(encoding="utf-8")

    assert "Single-split results reportable: False" in text
    assert "Recommended reporting mode: repeated_split" in text
    assert "Invalid rate controls present: True" in text
    assert "Rate offset warning: unmodeled split-level rate offset" in text


def _stratified_cv_summary() -> dict:
    return {
        "dataset_name": "mc_maze_small",
        "dataset_hash": "abc",
        "bin_size_ms": 20,
        "window_seconds": 1.28,
        "reference_model": "train_heldout_mean_rate",
        "fold_count": 5,
        "repeats": 5,
        "total_folds": 25,
        "assignment_method": "greedy_balanced",
        "stratification_variables": ["use_endpoint_direction", "use_population_rate"],
        "mean_population_rate_fold_range": 0.02,
        "mean_heldout_rate_fold_range": 0.03,
        "mean_endpoint_distance_fold_range": 0.4,
        "mean_speed_fold_range": 0.3,
        "mean_endpoint_direction_entropy": 2.0,
        "endpoint_direction_entropy_max": 2.0794415416798357,
        "endpoint_direction_concentrated": False,
        "fold_balance_warning": "none",
        "factor_latent_mean_unified_bits_per_spike": 0.0143,
        "factor_latent_std_unified_bits_per_spike": 0.0112,
        "factor_latent_ci95_low": 0.0091,
        "factor_latent_ci95_high": 0.0195,
        "factor_latent_positive_fraction": 0.88,
        "split_mean_rate_invalid_mean_unified_bits_per_spike": 0.0924,
        "invalid_controls_excluded_from_valid_model_selection": True,
        "stratified_factor_latent_mean": 0.0143,
        "stratified_factor_latent_std": 0.0112,
        "random_fold_factor_latent_mean": 0.0130,
        "random_fold_factor_latent_std": 0.0160,
        "random_factor_latent_test_mean_reference": 0.008975282435208521,
        "random_factor_latent_test_positive_fraction_reference": 0.76,
        "stratification_reduces_variance": True,
        "variance_reduction_fraction": 0.51,
        "recommended_reporting_mode": "stratified_cross_validation",
        "single_split_results_reportable": False,
        "carried_forward_method": "factor_latent",
    }


def _stratified_method_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method_name": "factor_latent",
                "valid_model": True,
                "reportable_as_model_performance": True,
                "mean_unified_bits_per_spike": 0.0143,
                "notes": "carried forward",
            },
            {
                "method_name": "split_mean_rate_invalid",
                "valid_model": False,
                "reportable_as_model_performance": False,
                "mean_unified_bits_per_spike": 0.0924,
                "notes": "leakage diagnostic",
            },
        ]
    )


def test_stratified_cv_report_includes_balance_comparison_and_warnings(tmp_path: Path) -> None:
    fold_balance = pd.DataFrame(
        [
            {
                "repeat_index": 0,
                "fold_index": 0,
                "n_trials": 20,
                "mean_population_rate_hz": 0.6,
                "mean_heldout_rate_hz": 0.57,
                "mean_endpoint_distance": 4.1,
                "mean_speed": 3.8,
                "endpoint_direction_entropy": 2.0,
                "stratum_count": 12,
            }
        ]
    )
    comparisons = pd.DataFrame(
        [
            {
                "repeat_index": 0,
                "metric": "mean_heldout_rate_hz",
                "min_value": 0.55,
                "max_value": 0.58,
                "range": 0.03,
                "mean_value": 0.57,
                "std_value": 0.01,
                "coefficient_of_variation": 0.018,
            }
        ]
    )

    paths = write_stratified_cv_outputs(
        tmp_path,
        _stratified_cv_summary(),
        pd.DataFrame([{"repeat_index": 0, "fold_index": 0}]),
        pd.DataFrame([{"repeat_index": 0, "trial_index": 0}]),
        fold_balance,
        comparisons,
        _stratified_method_summary(),
    )
    text = paths["report"].read_text(encoding="utf-8")

    assert paths["fold_balance"].exists()
    assert paths["method_summary"].exists()

    assert "## Fold balance" in text
    assert "Fold balance warning: none" in text
    assert "endpoint_direction_entropy" in text
    assert "Mean held-out-rate fold range: 0.03" in text

    assert "## Random versus stratified comparison" in text
    assert "Stratification reduces variance: True" in text
    assert "Variance reduction fraction: 0.51" in text
    assert "Repeated random-split test mean reference: 0.008975282435208521" in text
    # The 70/15/15 reference must never be presented as comparable to a cross-validation mean.
    assert "Their means are therefore not comparable" in text
    assert "protocol difference rather than a performance gain" in text
    assert "Endpoint directions are concentrated in this dataset and window: False" in text

    assert "Recommended reporting mode: stratified_cross_validation" in text
    assert "Single-split results reportable: False" in text
    assert "Carried-forward method: factor_latent" in text

    assert (
        "Invalid controls use evaluation fold targets and cannot be reported as model performance."
        in text
    )
    assert "split_mean_rate_invalid" in text
    assert "Old incompatible mean-rate values are not used as tuning targets" in text
    assert "not an official NLB leaderboard result" in text
    assert "Stratified cross-validation is preferred over single-split reporting." in text


def test_unified_scoreboard_report_exposes_stratified_cv_fields(tmp_path: Path) -> None:
    summary = {
        "dataset_name": "mc_maze_small",
        "reference_model": "train_heldout_mean_rate",
        "generalization_risk": "high",
        "stratified_cv_available": True,
        "factor_latent_stratified_cv_mean": 0.0143,
        "factor_latent_stratified_cv_ci95_low": 0.0091,
    }

    path = write_unified_scoreboard_report(
        tmp_path / "report.md", summary, pd.DataFrame(), pd.DataFrame()
    )
    text = path.read_text(encoding="utf-8")

    assert "Stratified CV available: True" in text
    assert "Factor-latent stratified CV mean: 0.0143" in text
    assert "Factor-latent stratified CV CI95 low: 0.0091" in text


def _window_audit_summary() -> dict:
    return {
        "dataset_name": "mc_maze_small",
        "dataset_hash": "abc",
        "bin_size_ms": 20,
        "reference_model": "train_heldout_mean_rate",
        "fold_count": 5,
        "repeats": 5,
        "behavior_source": "hand_pos",
        "recommended_window_name": "behavior_speed_peak_centered_1p28s",
        "recommended_reporting_mode": "stratified_cross_validation",
        "current_window_name": "from_start_1p28s",
        "current_window_still_supported": False,
        "current_window_is_early_window_diagnostic": False,
        "factor_latent_best_window_mean": 0.0301,
        "factor_latent_current_window_mean": 0.0254,
        "factor_latent_best_window_ci95_low": 0.0240,
        "factor_latent_best_window_ci95_high": 0.0362,
        "split_mean_invalid_best_window_mean": 0.0790,
        "invalid_control_gap_best_window": 0.0489,
        "endpoint_direction_entropy_by_window": {
            "from_start_1p28s": 0.846,
            "behavior_speed_peak_centered_1p28s": 1.72,
        },
        "moving_bin_fraction_by_window": {
            "from_start_1p28s": 0.11,
            "behavior_speed_peak_centered_1p28s": 0.94,
        },
        "behavior_coverage_warning": "none",
        "eligible_windows": ["behavior_speed_peak_centered_1p28s", "from_start_1p28s"],
        "window_selection_rationale": "challenger carries more reach diversity",
        "invalid_controls_excluded_from_window_selection": True,
    }


def _window_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "window_name": "from_start_1p28s",
                "report_label": "Current accepted early window",
                "crop_policy": "from_start",
                "endpoint_direction_entropy": 0.846,
                "moving_bin_fraction": 0.11,
                "fold_balance_warning": "none",
            },
            {
                "window_name": "behavior_speed_peak_centered_1p28s",
                "report_label": "Centered on peak hand speed",
                "crop_policy": "behavior_speed_peak_centered",
                "endpoint_direction_entropy": 1.72,
                "moving_bin_fraction": 0.94,
                "fold_balance_warning": "none",
            },
        ]
    )


def _window_method_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "window_name": "behavior_speed_peak_centered_1p28s",
                "method_name": "factor_latent",
                "valid_model": True,
                "reportable_as_model_performance": True,
                "mean_unified_bits_per_spike": 0.0301,
            },
            {
                "window_name": "behavior_speed_peak_centered_1p28s",
                "method_name": "split_mean_rate_invalid",
                "valid_model": False,
                "reportable_as_model_performance": False,
                "mean_unified_bits_per_spike": 0.0790,
            },
        ]
    )


def test_window_audit_report_includes_candidates_entropy_and_recommendation(tmp_path: Path) -> None:
    recommendations = {
        "recommended_window_name": "behavior_speed_peak_centered_1p28s",
        "recommended_reporting_mode": "stratified_cross_validation",
        "official_benchmark_claim": False,
    }

    paths = write_window_audit_outputs(
        tmp_path,
        _window_audit_summary(),
        pd.DataFrame([{"window_name": "from_start_1p28s"}]),
        pd.DataFrame([{"window_name": "from_start_1p28s", "trial_index": 0}]),
        pd.DataFrame([{"window_name": "from_start_1p28s", "fold_index": 0}]),
        _window_table(),
        _window_method_summary(),
        recommendations,
    )
    text = paths["report"].read_text(encoding="utf-8")

    assert paths["recommendations"].exists()
    assert paths["behavior_statistics"].exists()

    assert "## Candidate windows" in text
    assert "from_start_1p28s" in text
    assert "behavior_speed_peak_centered_1p28s" in text

    assert "## Endpoint direction entropy by window" in text
    assert "0.846" in text
    assert "1.72" in text

    assert "Recommended window: behavior_speed_peak_centered_1p28s" in text
    assert "Current window still supported: False" in text
    assert "Movement coverage by window" in text

    assert (
        "Invalid controls use evaluation fold targets and cannot be reported as model performance."
        in text
    )
    assert "never on invalid-control gains" in text
    # Cross-window scores must never read as a performance comparison.
    assert "not comparable across windows as performance" in text
    assert "peak hand speed of the whole recording" in text
    assert "Old incompatible mean-rate values are not used as tuning targets" in text
    assert "not an official NLB leaderboard result" in text


def test_unified_scoreboard_report_exposes_window_audit_fields(tmp_path: Path) -> None:
    summary = {
        "dataset_name": "mc_maze_small",
        "reference_model": "train_heldout_mean_rate",
        "generalization_risk": "high",
        "window_audit_available": True,
        "recommended_window_name": "behavior_speed_peak_centered_1p28s",
        "current_window_still_supported": False,
    }

    path = write_unified_scoreboard_report(
        tmp_path / "report.md", summary, pd.DataFrame(), pd.DataFrame()
    )
    text = path.read_text(encoding="utf-8")

    assert "Window audit available: True" in text
    assert "Recommended window: behavior_speed_peak_centered_1p28s" in text
    assert "Current window still supported: False" in text
