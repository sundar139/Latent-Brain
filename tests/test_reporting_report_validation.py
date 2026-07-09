from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]

from latentbrain.reporting.report_tables import METHOD_REGISTRY_COLUMNS, build_method_registry
from latentbrain.reporting.report_validation import (
    INVALID_CONTROL_STATEMENT,
    NOT_OFFICIAL_STATEMENT,
    OLD_MEAN_RATE_STATEMENT,
    REQUIRED_REPORT_SECTIONS,
    SEED_CONFOUND_STATEMENT,
    SPLIT_INSTABILITY_STATEMENT,
    validate_claim_safety,
    validate_method_registry,
    validate_report_text,
)


def _findings(**overrides: object) -> dict[str, object]:
    findings = {
        "canonical_metric": "unified_bits_per_spike",
        "official_leaderboard_claim": False,
        "no_official_benchmark_claim": True,
        "carried_forward_valid_method": "factor_latent",
        "single_split_results_reportable": False,
        "recommended_reporting_mode": "repeated_split",
        "invalid_rate_controls_present": True,
        "neural_ode_near_win_seed_specific": True,
        "split_mean_advantage_is_rate_offset": False,
        "split_mean_advantage_is_target_leakage": True,
    }
    return findings | overrides


def _valid_report_text() -> str:
    body = "\n".join(REQUIRED_REPORT_SECTIONS)
    statements = "\n".join(
        (
            NOT_OFFICIAL_STATEMENT,
            INVALID_CONTROL_STATEMENT,
            SEED_CONFOUND_STATEMENT,
            SPLIT_INSTABILITY_STATEMENT,
            OLD_MEAN_RATE_STATEMENT,
        )
    )
    return f"{body}\n{statements}\n"


def test_valid_toy_report_passes_all_validation() -> None:
    registry = build_method_registry()

    assert validate_method_registry(registry) == []
    assert validate_claim_safety(_findings(), registry) == []
    assert validate_report_text(_valid_report_text()) == []


def test_claim_safety_fails_if_official_leaderboard_claim_is_true() -> None:
    failures = validate_claim_safety(
        _findings(official_leaderboard_claim=True), build_method_registry()
    )

    assert any("official leaderboard claim is true" in failure for failure in failures)


def test_claim_safety_fails_if_no_official_benchmark_claim_is_false() -> None:
    failures = validate_claim_safety(
        _findings(no_official_benchmark_claim=False), build_method_registry()
    )

    assert any("no_official_benchmark_claim is false" in failure for failure in failures)


def test_claim_safety_fails_if_invalid_control_is_reportable() -> None:
    registry = build_method_registry()
    registry.loc[
        registry["method_name"] == "split_mean_rate_invalid", "reportable_as_model_performance"
    ] = True

    failures = validate_claim_safety(_findings(), registry)

    assert any("reportable as model performance" in failure for failure in failures)


def test_claim_safety_fails_if_carried_forward_method_is_invalid() -> None:
    failures = validate_claim_safety(
        _findings(carried_forward_valid_method="split_mean_rate_invalid"),
        build_method_registry(),
    )

    assert any("invalid control" in failure for failure in failures)


def test_claim_safety_fails_if_carried_forward_method_is_unknown() -> None:
    failures = validate_claim_safety(
        _findings(carried_forward_valid_method="mystery_model"), build_method_registry()
    )

    assert any("absent from the registry" in failure for failure in failures)


def test_claim_safety_fails_if_single_split_reporting_is_recommended() -> None:
    failures = validate_claim_safety(
        _findings(recommended_reporting_mode="single_split"), build_method_registry()
    )

    assert any("single_split" in failure for failure in failures)


def test_claim_safety_fails_if_single_split_results_are_reportable() -> None:
    failures = validate_claim_safety(
        _findings(single_split_results_reportable=True), build_method_registry()
    )

    assert any("single-split results are marked reportable" in failure for failure in failures)


def test_claim_safety_fails_if_split_mean_advantage_called_rate_offset() -> None:
    failures = validate_claim_safety(
        _findings(split_mean_advantage_is_rate_offset=True), build_method_registry()
    )

    assert any("global rate offset" in failure for failure in failures)


def test_claim_safety_fails_if_leakage_or_seed_confound_not_labelled() -> None:
    leakage = validate_claim_safety(
        _findings(split_mean_advantage_is_target_leakage=False), build_method_registry()
    )
    seed = validate_claim_safety(
        _findings(neural_ode_near_win_seed_specific=False), build_method_registry()
    )

    assert any("target leakage" in failure for failure in leakage)
    assert any("seed-specific" in failure for failure in seed)


def test_claim_safety_fails_if_metric_is_not_canonical() -> None:
    failures = validate_claim_safety(_findings(canonical_metric="raw_nll"), build_method_registry())

    assert any("canonical metric" in failure for failure in failures)


def test_report_text_validation_catches_missing_warnings() -> None:
    text = _valid_report_text().replace(INVALID_CONTROL_STATEMENT, "")

    failures = validate_report_text(text)

    assert any(INVALID_CONTROL_STATEMENT in failure for failure in failures)


def test_report_text_validation_catches_missing_seed_and_split_warnings() -> None:
    without_seed = _valid_report_text().replace(SEED_CONFOUND_STATEMENT, "")
    without_split = _valid_report_text().replace(SPLIT_INSTABILITY_STATEMENT, "")
    without_old_mean = _valid_report_text().replace(OLD_MEAN_RATE_STATEMENT, "")

    assert any(SEED_CONFOUND_STATEMENT in f for f in validate_report_text(without_seed))
    assert any(SPLIT_INSTABILITY_STATEMENT in f for f in validate_report_text(without_split))
    assert any(OLD_MEAN_RATE_STATEMENT in f for f in validate_report_text(without_old_mean))


def test_report_text_validation_catches_missing_sections() -> None:
    text = _valid_report_text().replace("## Negative neural-model findings", "")

    failures = validate_report_text(text)

    assert any("missing section" in failure for failure in failures)


def test_registry_validation_catches_structural_problems() -> None:
    assert validate_method_registry(pd.DataFrame()) == ["method registry is empty"]

    missing_columns = pd.DataFrame([{"method_name": "a"}])
    assert any("missing columns" in f for f in validate_method_registry(missing_columns))

    registry = build_method_registry()
    registry.loc[registry["method_name"] == "factor_latent", "carried_forward"] = False
    assert any(
        "no method is marked carried forward" in f for f in validate_method_registry(registry)
    )

    duplicated = pd.concat([build_method_registry(), build_method_registry()], ignore_index=True)
    assert any("duplicate method names" in f for f in validate_method_registry(duplicated))


def test_registry_validation_catches_invalid_control_marked_carried_forward() -> None:
    registry = build_method_registry()
    registry.loc[registry["method_name"] == "split_mean_rate_invalid", "carried_forward"] = True

    failures = validate_method_registry(registry)

    assert any("invalid control is marked carried forward" in f for f in failures)
    assert set(METHOD_REGISTRY_COLUMNS).issubset(registry.columns)
