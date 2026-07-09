from __future__ import annotations

import json
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest

from latentbrain.eval.unified_scoreboard import (
    HISTORICAL_STATUS,
    LEADERBOARD_COLUMNS,
    SPLIT_SCORE_COLUMNS,
    build_historical_metric_notes,
    build_unified_score_row,
    load_lfads_family_candidates,
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
            "switching_ode_tuning_summary_path": str(tmp_path / "switching_ode.json"),
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
