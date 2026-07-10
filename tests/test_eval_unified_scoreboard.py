from __future__ import annotations

import json
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest

from latentbrain.eval.unified_scoreboard import (
    HISTORICAL_STATUS,
    LEADERBOARD_COLUMNS,
    MULTI_SEED_NOTE,
    SPLIT_SCORE_COLUMNS,
    build_historical_metric_notes,
    build_unified_score_row,
    load_cv_rate_audit_warning,
    load_lfads_family_candidates,
    load_seed_robustness_candidates,
    load_split_audit_warning,
    load_stratified_cv_warning,
    rank_unified_validation_scores,
    summarize_unified_scoreboard,
)


def _lfads_candidate_config(tmp_path: Path) -> dict[str, object]:
    return {
        "scoring": {"reference_model": "train_heldout_mean_rate"},
        "inputs": {
            "coordinated_dropout_dir": str(tmp_path / "dropout"),
            "rate_calibration_dir": str(tmp_path / "calibration"),
            "lfads_unified_tuning_summary_path": str(tmp_path / "unified.json"),
            "lfads_controller_tuning_summary_path": str(tmp_path / "controller.json"),
            "neural_sde_tuning_summary_path": str(tmp_path / "neural_sde.json"),
            "neural_ode_tuning_summary_path": str(tmp_path / "neural_ode.json"),
            "neural_ode_refinement_summary_path": str(tmp_path / "neural_ode_refinement.json"),
            "neural_ode_objective_summary_path": str(tmp_path / "neural_ode_objectives.json"),
            "switching_ode_tuning_summary_path": str(tmp_path / "switching_ode.json"),
            "seed_robustness_summary_path": str(tmp_path / "seed_robustness.json"),
            "split_audit_summary_path": str(tmp_path / "split_audit.json"),
            "cv_rate_audit_summary_path": str(tmp_path / "cv_rate_audit.json"),
            "stratified_cv_summary_path": str(tmp_path / "stratified_cv.json"),
        },
        "known_unified_values": {
            "lfads_unified_validation_bits_per_spike": 0.009,
            "coordinated_dropout_unified_validation_bits_per_spike": 0.01,
        },
    }


def _write_summary(path: Path, bits: float, nll: float = 2.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "best_run_id": path.stem,
                "best_validation_unified_bits_per_spike": bits,
                "best_validation_poisson_nll": nll,
            }
        ),
        encoding="utf-8",
    )


def test_score_row_contains_required_fields() -> None:
    row = build_unified_score_row(
        "factor_latent",
        "factor_decoder",
        "validation",
        0.03,
        10.0,
        True,
        "train_heldout_mean_rate",
        "unified",
    )

    assert set(SPLIT_SCORE_COLUMNS).issubset(row)


def test_loads_lfads_unified_tuning_candidate_from_summary_file(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    _write_summary(tmp_path / "unified.json", 0.012, 1.5)

    rows = load_lfads_family_candidates(config)
    row = next(row for row in rows if row["method_name"] == "lfads_unified_tuning")

    assert row["bits_per_spike"] == 0.012
    assert row["poisson_nll"] == 1.5
    assert row["source_summary_path"] == str((tmp_path / "unified.json").resolve())


def test_loads_controller_tuning_candidate_from_summary_file(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    _write_summary(tmp_path / "controller.json", 0.014, 1.4)

    rows = load_lfads_family_candidates(config)
    row = next(row for row in rows if row["method_name"] == "lfads_controller_tuning")

    assert row["bits_per_spike"] == 0.014
    assert row["source_summary_path"] == str((tmp_path / "controller.json").resolve())


def test_loads_neural_sde_tuning_candidate_from_summary_file(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    _write_summary(tmp_path / "neural_sde.json", 0.016, 1.3)

    rows = load_lfads_family_candidates(config)
    row = next(row for row in rows if row["method_name"] == "neural_sde_tuning")

    assert row["bits_per_spike"] == 0.016
    assert row["poisson_nll"] == 1.3
    assert row["source_summary_path"] == str((tmp_path / "neural_sde.json").resolve())


def test_loads_neural_ode_tuning_candidate_from_summary_file(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    _write_summary(tmp_path / "neural_ode.json", 0.026, 1.2)

    rows = load_lfads_family_candidates(config)
    row = next(row for row in rows if row["method_name"] == "neural_ode_tuning")

    assert row["bits_per_spike"] == 0.026
    assert row["poisson_nll"] == 1.2
    assert row["source_summary_path"] == str((tmp_path / "neural_ode.json").resolve())


def test_switching_summary_is_loaded_when_present(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    _write_summary(tmp_path / "switching_ode.json", 0.027, 1.1)

    rows = load_lfads_family_candidates(config)
    row = next(row for row in rows if row["method_name"] == "switching_ode_tuning")

    assert row["bits_per_spike"] == 0.027
    assert row["poisson_nll"] == 1.1


def test_neural_ode_refinement_summary_is_loaded_when_present(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    _write_summary(tmp_path / "neural_ode_refinement.json", 0.029, 1.0)

    rows = load_lfads_family_candidates(config)
    row = next(row for row in rows if row["method_name"] == "neural_ode_refinement")

    assert row["bits_per_spike"] == 0.029
    assert row["poisson_nll"] == 1.0


def test_neural_ode_refinement_can_become_best_dynamics_family_method(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    _write_summary(tmp_path / "neural_ode.json", 0.026)
    _write_summary(tmp_path / "neural_ode_refinement.json", 0.029)
    rows = [
        build_unified_score_row(
            "factor_latent", "decoder", "validation", 0.03, 1.0, True, "ref", ""
        ),
        *load_lfads_family_candidates(config),
    ]

    summary = summarize_unified_scoreboard(
        rank_unified_validation_scores(pd.DataFrame(rows)),
        {
            "factor_latent_unified_validation_bits_per_spike": 0.03,
            "best_oracle_validation_bits_per_spike": 3.0,
        },
    )

    assert summary["best_lfads_family_method"] == "neural_ode_refinement"
    assert summary["best_lfads_family_validation_bits_per_spike"] == 0.029


def test_neural_ode_objective_summary_is_loaded_when_present(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    _write_summary(tmp_path / "neural_ode_objectives.json", 0.031, 0.9)

    rows = load_lfads_family_candidates(config)
    row = next(row for row in rows if row["method_name"] == "neural_ode_objectives")

    assert row["bits_per_spike"] == 0.031
    assert row["poisson_nll"] == 0.9
    assert row["source_summary_path"] == str((tmp_path / "neural_ode_objectives.json").resolve())


def test_neural_ode_objectives_can_become_best_dynamics_family_method(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    _write_summary(tmp_path / "neural_ode_refinement.json", 0.0283)
    _write_summary(tmp_path / "neural_ode_objectives.json", 0.0299)
    rows = [
        build_unified_score_row(
            "factor_latent", "decoder", "validation", 0.0316, 1.0, True, "ref", ""
        ),
        *load_lfads_family_candidates(config),
    ]

    summary = summarize_unified_scoreboard(
        rank_unified_validation_scores(pd.DataFrame(rows)),
        {
            "factor_latent_unified_validation_bits_per_spike": 0.0316,
            "best_oracle_validation_bits_per_spike": 3.0,
        },
    )

    assert summary["best_valid_model"] == "factor_latent"
    assert summary["best_lfads_family_method"] == "neural_ode_objectives"
    assert summary["best_lfads_family_validation_bits_per_spike"] == 0.0299
    assert summary["lfads_family_beats_factor_latent"] is False


def test_weaker_objective_summary_keeps_refinement_as_best_dynamics_family(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    _write_summary(tmp_path / "neural_ode_refinement.json", 0.0283)
    _write_summary(tmp_path / "neural_ode_objectives.json", 0.0201)
    rows = load_lfads_family_candidates(config)

    summary = summarize_unified_scoreboard(
        rank_unified_validation_scores(pd.DataFrame(rows)),
        {
            "factor_latent_unified_validation_bits_per_spike": 0.0316,
            "best_oracle_validation_bits_per_spike": 3.0,
        },
    )

    assert summary["best_lfads_family_method"] == "neural_ode_refinement"


def test_missing_neural_ode_objective_summary_falls_back_cleanly(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    _write_summary(tmp_path / "neural_ode_refinement.json", 0.0283)

    methods = {row["method_name"] for row in load_lfads_family_candidates(config)}

    assert "neural_ode_objectives" not in methods
    assert "neural_ode_refinement" in methods


def test_malformed_neural_ode_objective_summary_fails_clearly(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    (tmp_path / "neural_ode_objectives.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="malformed LFADS-family summary"):
        load_lfads_family_candidates(config)


def test_switching_can_become_best_dynamics_family_method(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    _write_summary(tmp_path / "neural_ode.json", 0.026)
    _write_summary(tmp_path / "switching_ode.json", 0.028)
    rows = [
        build_unified_score_row(
            "factor_latent", "decoder", "validation", 0.03, 1.0, True, "ref", ""
        ),
        *load_lfads_family_candidates(config),
    ]

    summary = summarize_unified_scoreboard(
        rank_unified_validation_scores(pd.DataFrame(rows)),
        {
            "factor_latent_unified_validation_bits_per_spike": 0.03,
            "best_oracle_validation_bits_per_spike": 3.0,
        },
    )

    assert summary["best_lfads_family_method"] == "switching_ode_tuning"
    assert summary["best_lfads_family_validation_bits_per_spike"] == 0.028


def test_controller_tuning_wins_over_older_lfads_summary(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    _write_summary(tmp_path / "unified.json", 0.010)
    _write_summary(tmp_path / "controller.json", 0.014)
    rows = [
        build_unified_score_row(
            "factor_latent", "decoder", "validation", 0.03, 1.0, True, "ref", ""
        ),
        *load_lfads_family_candidates(config),
    ]

    leaderboard = rank_unified_validation_scores(pd.DataFrame(rows))
    summary = summarize_unified_scoreboard(
        leaderboard,
        {
            "factor_latent_unified_validation_bits_per_spike": 0.03,
            "best_oracle_validation_bits_per_spike": 3.0,
        },
    )

    assert summary["best_valid_model"] == "factor_latent"
    assert summary["best_lfads_family_method"] == "lfads_controller_tuning"
    assert summary["best_lfads_family_validation_bits_per_spike"] == 0.014
    assert summary["lfads_family_beats_factor_latent"] is False


def test_neural_sde_can_become_best_dynamics_family_method(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    _write_summary(tmp_path / "controller.json", 0.014)
    _write_summary(tmp_path / "neural_sde.json", 0.02)
    rows = [
        build_unified_score_row(
            "factor_latent", "decoder", "validation", 0.03, 1.0, True, "ref", ""
        ),
        *load_lfads_family_candidates(config),
    ]

    summary = summarize_unified_scoreboard(
        rank_unified_validation_scores(pd.DataFrame(rows)),
        {
            "factor_latent_unified_validation_bits_per_spike": 0.03,
            "best_oracle_validation_bits_per_spike": 3.0,
        },
    )

    assert summary["best_lfads_family_method"] == "neural_sde_tuning"
    assert summary["best_lfads_family_validation_bits_per_spike"] == 0.02


def test_neural_ode_can_become_best_dynamics_family_method(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    _write_summary(tmp_path / "neural_sde.json", 0.02)
    _write_summary(tmp_path / "neural_ode.json", 0.026)
    rows = [
        build_unified_score_row(
            "factor_latent", "decoder", "validation", 0.03, 1.0, True, "ref", ""
        ),
        *load_lfads_family_candidates(config),
    ]

    summary = summarize_unified_scoreboard(
        rank_unified_validation_scores(pd.DataFrame(rows)),
        {
            "factor_latent_unified_validation_bits_per_spike": 0.03,
            "best_oracle_validation_bits_per_spike": 3.0,
        },
    )

    assert summary["best_lfads_family_method"] == "neural_ode_tuning"
    assert summary["best_lfads_family_validation_bits_per_spike"] == 0.026


def test_missing_lfads_summaries_fall_back_to_static_known_values(tmp_path: Path) -> None:
    rows = load_lfads_family_candidates(_lfads_candidate_config(tmp_path))

    by_method = {row["method_name"]: row for row in rows}
    assert by_method["raw_lfads"]["bits_per_spike"] == 0.009
    assert by_method["coordinated_dropout_lfads"]["bits_per_spike"] == 0.01
    assert by_method["raw_lfads"]["source_summary_path"] is None


def test_malformed_lfads_summary_fails_clearly(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    (tmp_path / "controller.json").write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="malformed LFADS-family summary"):
        load_lfads_family_candidates(config)


def test_validation_ranking_puts_valid_models_in_descending_bits() -> None:
    scores = pd.DataFrame(
        [
            build_unified_score_row("lfads", "direct", "validation", 0.01, 2.0, True, "ref", ""),
            build_unified_score_row(
                "factor_latent", "decoder", "validation", 0.03, 1.0, True, "ref", ""
            ),
            build_unified_score_row("oracle", "oracle", "validation", 3.0, 0.1, False, "ref", ""),
        ]
    )

    ranked = rank_unified_validation_scores(scores)

    assert list(ranked.columns) == LEADERBOARD_COLUMNS
    assert ranked.iloc[0]["method_name"] == "factor_latent"
    assert ranked.iloc[1]["method_name"] == "lfads"
    assert bool(ranked.iloc[2]["is_oracle_control"])
    assert not bool(ranked.iloc[2]["valid_model"])


def test_historical_notes_mark_incompatible_values_as_historical_only() -> None:
    notes = build_historical_metric_notes({"old_mean": 0.7})

    assert notes.iloc[0]["status"] == HISTORICAL_STATUS
    assert "not directly comparable" in notes.iloc[0]["reason"]


def test_summary_identifies_best_valid_and_lfads_family_methods() -> None:
    leaderboard = pd.DataFrame(
        {
            "rank": [1, 2, 3],
            "method_name": ["factor_latent", "coordinated_dropout_lfads", "oracle_smoothed"],
            "prediction_source": ["factor_decoder", "direct_model", "oracle"],
            "valid_model": [True, True, False],
            "validation_bits_per_spike": [0.03, 0.01, 3.0],
            "validation_poisson_nll": [1.0, 2.0, 0.1],
            "reference_name": ["ref", "ref", "ref"],
            "beats_train_mean_reference": [True, True, True],
            "beats_factor_latent_reference": [False, False, True],
            "is_oracle_control": [False, False, True],
            "notes": ["", "", "invalid model"],
        }
    )

    summary = summarize_unified_scoreboard(
        leaderboard,
        {
            "factor_latent_unified_validation_bits_per_spike": 0.03,
            "best_oracle_validation_bits_per_spike": 3.0,
        },
    )

    assert summary["best_valid_model"] == "factor_latent"
    assert summary["best_lfads_family_method"] == "coordinated_dropout_lfads"
    assert summary["lfads_family_beats_factor_latent"] is False
    assert summary["old_mean_rate_values_historical_only"] is True


def _write_seed_robustness_summary(path: Path, best_mean: float, best_lower_ci: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "best_mean_method": "factor_latent",
                "best_mean_validation_unified_bits_per_spike": best_mean,
                "best_lower_ci_method": "factor_latent",
                "best_lower_ci_validation_unified_bits_per_spike": best_lower_ci,
                "seeds_evaluated": [2027, 2028, 2029, 2030, 2031],
            }
        ),
        encoding="utf-8",
    )


def test_seed_robustness_summary_is_loaded_when_present(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    _write_seed_robustness_summary(tmp_path / "seed_robustness.json", 0.0312, 0.0301)

    rows = load_seed_robustness_candidates(config)
    by_method = {row["method_name"]: row for row in rows}

    assert by_method["seed_robustness_best_mean"]["bits_per_spike"] == 0.0312
    assert by_method["seed_robustness_best_lower_ci"]["bits_per_spike"] == 0.0301


def test_seed_robustness_rows_are_marked_as_multi_seed_aggregates(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    _write_seed_robustness_summary(tmp_path / "seed_robustness.json", 0.0312, 0.0301)

    rows = load_seed_robustness_candidates(config)

    assert rows
    for row in rows:
        assert MULTI_SEED_NOTE in row["notes"]
        assert "5 seeds" in row["notes"]
        assert row["source_summary_path"] == str((tmp_path / "seed_robustness.json").resolve())


def test_missing_seed_robustness_summary_falls_back_cleanly(tmp_path: Path) -> None:
    assert load_seed_robustness_candidates(_lfads_candidate_config(tmp_path)) == []


def test_malformed_seed_robustness_summary_fails_clearly(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    (tmp_path / "seed_robustness.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="malformed"):
        load_seed_robustness_candidates(config)


def test_seed_robustness_summary_missing_key_fails_clearly(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    (tmp_path / "seed_robustness.json").write_text(json.dumps({"seeds_evaluated": []}), "utf-8")

    with pytest.raises(ValueError, match="missing best_mean_validation_unified_bits_per_spike"):
        load_seed_robustness_candidates(config)


def test_seed_robustness_aggregates_do_not_win_best_valid_model() -> None:
    scores = pd.DataFrame(
        [
            build_unified_score_row(
                "factor_latent", "decoder", "validation", 0.0316, 1.0, True, "ref", ""
            ),
            build_unified_score_row(
                "neural_ode_refinement", "direct", "validation", 0.0283, 2.0, True, "ref", ""
            ),
            build_unified_score_row(
                "seed_robustness_best_mean",
                "multi_seed_mean",
                "validation",
                0.0320,
                None,
                True,
                "ref",
                MULTI_SEED_NOTE,
            ),
        ]
    )

    summary = summarize_unified_scoreboard(
        rank_unified_validation_scores(scores),
        {
            "factor_latent_unified_validation_bits_per_spike": 0.0316,
            "best_oracle_validation_bits_per_spike": 3.0,
        },
    )

    assert summary["best_valid_model"] == "factor_latent"
    assert summary["best_lfads_family_method"] == "neural_ode_refinement"
    assert summary["seed_robustness_ingested"] is True
    assert summary["seed_robustness_aggregate_methods"] == ["seed_robustness_best_mean"]


def test_scoreboard_reports_no_seed_robustness_when_absent() -> None:
    scores = pd.DataFrame(
        [
            build_unified_score_row(
                "factor_latent", "decoder", "validation", 0.0316, 1.0, True, "ref", ""
            )
        ]
    )

    summary = summarize_unified_scoreboard(
        rank_unified_validation_scores(scores),
        {
            "factor_latent_unified_validation_bits_per_spike": 0.0316,
            "best_oracle_validation_bits_per_spike": 3.0,
        },
    )

    assert summary["seed_robustness_ingested"] is False
    assert summary["seed_robustness_aggregate_methods"] == []


def _write_split_audit(path: Path, risk: str, instability: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "generalization_risk": risk,
                "validation_test_instability_detected": instability,
                "factor_latent_test_mean": -0.0083,
            }
        ),
        encoding="utf-8",
    )


def test_split_audit_summary_is_loaded_when_present(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    _write_split_audit(tmp_path / "split_audit.json", "high")

    warning = load_split_audit_warning(config)

    assert warning["split_audit_available"] is True
    assert warning["generalization_risk"] == "high"
    assert warning["validation_test_instability_detected"] is True
    assert warning["validation_only_diagnostics"] is True
    assert warning["split_audit_summary_path"] == str((tmp_path / "split_audit.json").resolve())


def test_low_risk_split_audit_does_not_flag_validation_only_diagnostics(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    _write_split_audit(tmp_path / "split_audit.json", "low", instability=False)

    warning = load_split_audit_warning(config)

    assert warning["generalization_risk"] == "low"
    assert warning["validation_test_instability_detected"] is False
    assert warning["validation_only_diagnostics"] is False


def test_missing_split_audit_summary_falls_back_cleanly(tmp_path: Path) -> None:
    warning = load_split_audit_warning(_lfads_candidate_config(tmp_path))

    assert warning["split_audit_available"] is False
    assert warning["generalization_risk"] is None
    assert warning["validation_test_instability_detected"] is False


def test_malformed_split_audit_summary_fails_clearly(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    (tmp_path / "split_audit.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="malformed"):
        load_split_audit_warning(config)


def test_split_audit_summary_missing_risk_key_fails_clearly(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    (tmp_path / "split_audit.json").write_text(json.dumps({"other": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing generalization_risk"):
        load_split_audit_warning(config)


def test_non_string_generalization_risk_fails_clearly(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    (tmp_path / "split_audit.json").write_text(
        json.dumps({"generalization_risk": 3}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="must be a string"):
        load_split_audit_warning(config)


def _write_cv_rate_audit(path: Path, dominates: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "single_split_results_reportable": False,
                "recommended_reporting_mode": "repeated_split",
                "invalid_control_methods": [
                    "split_mean_rate_invalid",
                    "oracle_split_scaled_factor_latent_invalid",
                ],
                "invalid_controls_dominate_valid_models": dominates,
                "best_valid_rate_control_method": "factor_latent",
            }
        ),
        encoding="utf-8",
    )


def test_cv_rate_audit_summary_is_loaded_when_present(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    _write_cv_rate_audit(tmp_path / "cv_rate_audit.json")

    warning = load_cv_rate_audit_warning(config)

    assert warning["cv_rate_audit_available"] is True
    assert warning["single_split_results_reportable"] is False
    assert warning["recommended_reporting_mode"] == "repeated_split"
    assert warning["invalid_rate_controls_present"] is True
    assert "unmodeled split-level rate offset" in warning["rate_offset_warning"]
    assert warning["cv_rate_audit_summary_path"] == str((tmp_path / "cv_rate_audit.json").resolve())


def test_cv_rate_audit_without_dominating_invalid_controls_has_no_rate_offset_warning(
    tmp_path: Path,
) -> None:
    config = _lfads_candidate_config(tmp_path)
    _write_cv_rate_audit(tmp_path / "cv_rate_audit.json", dominates=False)

    warning = load_cv_rate_audit_warning(config)

    assert warning["invalid_controls_dominate_valid_models"] is False
    assert warning["rate_offset_warning"] is None


def test_missing_cv_rate_audit_summary_falls_back_cleanly(tmp_path: Path) -> None:
    warning = load_cv_rate_audit_warning(_lfads_candidate_config(tmp_path))

    assert warning["cv_rate_audit_available"] is False
    assert warning["single_split_results_reportable"] is None
    assert warning["recommended_reporting_mode"] is None
    assert warning["invalid_rate_controls_present"] is False


def test_malformed_cv_rate_audit_summary_fails_clearly(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    (tmp_path / "cv_rate_audit.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="malformed"):
        load_cv_rate_audit_warning(config)


def test_cv_rate_audit_summary_missing_key_fails_clearly(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    (tmp_path / "cv_rate_audit.json").write_text(json.dumps({"other": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing single_split_results_reportable"):
        load_cv_rate_audit_warning(config)


def test_invalid_rate_controls_never_become_best_valid_model() -> None:
    scores = pd.DataFrame(
        [
            build_unified_score_row(
                "factor_latent", "decoder", "validation", 0.0316, 1.0, True, "ref", ""
            ),
            build_unified_score_row(
                "split_mean_rate_invalid",
                "invalid_control",
                "validation",
                0.0879,
                None,
                False,
                "ref",
                "leaks evaluation targets",
            ),
        ]
    )

    summary = summarize_unified_scoreboard(
        rank_unified_validation_scores(scores),
        {
            "factor_latent_unified_validation_bits_per_spike": 0.0316,
            "best_oracle_validation_bits_per_spike": 3.0,
        },
    )

    assert summary["best_valid_model"] == "factor_latent"
    assert summary["best_valid_model_validation_bits_per_spike"] == 0.0316


def _write_stratified_cv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "recommended_reporting_mode": "stratified_cross_validation",
                "single_split_results_reportable": False,
                "factor_latent_mean_unified_bits_per_spike": 0.0143,
                "factor_latent_ci95_low": 0.0091,
                "factor_latent_ci95_high": 0.0195,
                "carried_forward_method": "factor_latent",
            }
        ),
        encoding="utf-8",
    )


def test_stratified_cv_summary_is_loaded_when_present(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    _write_stratified_cv(tmp_path / "stratified_cv.json")

    warning = load_stratified_cv_warning(config)

    assert warning["stratified_cv_available"] is True
    assert warning["recommended_reporting_mode"] == "stratified_cross_validation"
    assert warning["single_split_results_reportable"] is False
    assert warning["factor_latent_stratified_cv_mean"] == 0.0143
    assert warning["factor_latent_stratified_cv_ci95_low"] == 0.0091
    assert warning["stratified_cv_summary_path"] == str((tmp_path / "stratified_cv.json").resolve())


def test_missing_stratified_cv_summary_falls_back_cleanly(tmp_path: Path) -> None:
    warning = load_stratified_cv_warning(_lfads_candidate_config(tmp_path))

    assert warning["stratified_cv_available"] is False
    assert warning["factor_latent_stratified_cv_mean"] is None
    assert warning["factor_latent_stratified_cv_ci95_low"] is None


def test_malformed_stratified_cv_summary_fails_clearly(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    (tmp_path / "stratified_cv.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="malformed"):
        load_stratified_cv_warning(config)


def test_stratified_cv_summary_missing_key_fails_clearly(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    (tmp_path / "stratified_cv.json").write_text(json.dumps({"other": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing recommended_reporting_mode"):
        load_stratified_cv_warning(config)


def test_stratified_cv_summary_non_string_mode_fails_clearly(tmp_path: Path) -> None:
    config = _lfads_candidate_config(tmp_path)
    (tmp_path / "stratified_cv.json").write_text(
        json.dumps(
            {
                "recommended_reporting_mode": 3,
                "factor_latent_mean_unified_bits_per_spike": 0.01,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be a string"):
        load_stratified_cv_warning(config)


def test_invalid_fold_controls_never_become_best_valid_model() -> None:
    scores = pd.DataFrame(
        [
            build_unified_score_row(
                "factor_latent", "decoder", "validation", 0.0316, 1.0, True, "ref", ""
            ),
            build_unified_score_row(
                "split_mean_rate_invalid",
                "invalid_control",
                "validation",
                0.0924,
                None,
                False,
                "ref",
                "uses evaluation fold targets",
            ),
        ]
    )

    summary = summarize_unified_scoreboard(
        rank_unified_validation_scores(scores),
        {
            "factor_latent_unified_validation_bits_per_spike": 0.0316,
            "best_oracle_validation_bits_per_spike": 3.0,
        },
    )

    assert summary["best_valid_model"] == "factor_latent"
