from __future__ import annotations

import json
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest

from latentbrain.eval.release_audit import (
    compare_values,
    load_mapping,
    release_readiness,
    validate_claim_safety,
    validate_sources,
)


def test_mapping_sources_and_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "summary.json"
    source.write_text(json.dumps({"dataset_hash": "abc", "complete": True}), encoding="utf-8")
    assert load_mapping(source)["complete"] is True
    with pytest.raises(FileNotFoundError, match="required release source"):
        load_mapping(tmp_path / "missing.json")
    source.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        load_mapping(source)
    csv_path = tmp_path / "scores.csv"
    pd.DataFrame([{"score": 0.5}]).to_csv(csv_path, index=False)
    assert load_mapping(csv_path)["rows"][0]["score"] == 0.5


def test_consistency_uses_numeric_tolerance_and_exact_flags() -> None:
    assert compare_values("score", 0.1, 0.10000000001, 1e-8)["consistent"] is True
    assert compare_values("score", 0.1, 0.2, 1e-8)["consistent"] is False
    assert compare_values("dataset_hash", "abc", "abc", 0.0)["consistent"] is True
    assert compare_values("flag", True, 1, 0.0)["consistent"] is False


def test_claim_safety_rejects_unsafe_wording_and_flags() -> None:
    with pytest.raises(ValueError, match="official leaderboard"):
        validate_claim_safety("official leaderboard performance", {})
    with pytest.raises(ValueError, match="causal"):
        validate_claim_safety("associative", {"causal_claim_allowed": True})
    with pytest.raises(ValueError, match="Small.*Large"):
        validate_claim_safety("Large performance improved over Small", {})


def test_source_validation_rejects_invalid_ranked_control(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"dataset_hash": "abc"}), encoding="utf-8")
    methods = tmp_path / "methods.csv"
    pd.DataFrame([{"method_name": "split_mean_rate_invalid", "valid_model": True}]).to_csv(
        methods, index=False
    )
    config = {
        "sources": {"summary": str(summary)},
        "expected": {"summary.dataset_hash": "abc"},
        "method_summary_path": str(methods),
    }
    with pytest.raises(ValueError, match="invalid control"):
        validate_sources(config, tmp_path)


def test_release_readiness_allows_agents_but_blocks_other_dirty_files() -> None:
    checks = {
        "all_sources_present": True,
        "dataset_hashes_verified": True,
        "metric_consistency_passed": True,
        "claim_consistency_passed": True,
        "quality_gates_passed": True,
        "tests_passed": True,
        "remote_ci_passed": True,
        "generated_outputs_ignored": True,
        "small_complete": True,
        "large_complete": True,
        "neural_search_closed": True,
        "interpretability_complete": True,
        "final_report_complete": True,
        "documentation_complete": True,
    }
    assert release_readiness(checks, ["AGENTS.md"], ["AGENTS.md"])["ready"] is True
    blocked = release_readiness(checks, ["scratch.txt"], ["AGENTS.md"])
    assert blocked["ready"] is False
    assert blocked["blockers"]


def test_final_documents_cover_required_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    report = (root / "docs/latentbrain_research_report.md").read_text(encoding="utf-8")
    required = [
        "## Executive summary",
        "## Research question",
        "## MC_Maze Small findings",
        "## MC_Maze Large findings",
        "## LFADS feasibility and retirement decision",
        "## Deterministic neural-ODE feasibility and retirement decision",
        "## Latent interpretability and behavioral validity",
        "## Conclusions",
    ]
    assert all(section in report for section in required)
    assert "one neuron mask" in report.lower()
    assert "circular temporal-shift control" in report
    assert "1/101" in report
    assert "associative and predictive" in report
    reproducibility = (root / "docs/latentbrain_reproducibility.md").read_text(encoding="utf-8")
    for command in (
        "pytest -q",
        "run_latent_interpretability.py",
        "run_unified_scoreboard.py",
        "run_release_audit.py",
    ):
        assert command in reproducibility
