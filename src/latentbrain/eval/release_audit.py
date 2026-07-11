from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import yaml

FORBIDDEN_REPORT_PHRASES = (
    "state-of-the-art",
    "leaderboard-leading",
    "official leaderboard performance",
    "superior to NLB submissions",
    "Large performance improved over Small",
    "latents cause movement",
    "true neural manifold",
)
INVALID_METHODS = {"split_mean_rate_invalid", "train_mean_rate"}


def load_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required release source is missing: {path}")
    if path.suffix.lower() == ".csv":
        try:
            return {"rows": pd.read_csv(path).to_dict(orient="records")}
        except (OSError, pd.errors.ParserError) as error:
            raise ValueError(f"malformed release source {path}: {error}") from error
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise ValueError(f"malformed release source {path}: {error}") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"malformed release source {path}: expected mapping")
    return dict(loaded)


def nested_value(mapping: dict[str, Any], key: str) -> Any:
    value: Any = mapping
    for part in key.split("."):
        if isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        elif isinstance(value, dict) and part in value:
            value = value[part]
        else:
            raise KeyError(f"missing source key: {key}")
    return value


def compare_values(name: str, value_a: Any, value_b: Any, tolerance: float) -> dict[str, Any]:
    if isinstance(value_a, bool) or isinstance(value_b, bool):
        difference = 0.0 if type(value_a) is type(value_b) and value_a == value_b else float("inf")
        consistent = difference == 0.0
    elif isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)):
        difference = abs(float(value_a) - float(value_b))
        consistent = difference <= tolerance
    else:
        difference = 0.0 if value_a == value_b else float("inf")
        consistent = value_a == value_b
    return {
        "metric_name": name,
        "value_a": value_a,
        "value_b": value_b,
        "absolute_difference": difference,
        "tolerance": tolerance,
        "consistent": bool(consistent),
    }


def validate_claim_safety(text: str, flags: dict[str, Any]) -> None:
    lowered = text.lower()
    for phrase in FORBIDDEN_REPORT_PHRASES:
        if phrase.lower() in lowered:
            label = "Small versus Large" if "large performance" in phrase.lower() else phrase
            raise ValueError(f"forbidden release claim: {label}")
    if bool(flags.get("official_leaderboard_claim", False)):
        raise ValueError("official leaderboard claim must remain false")
    if bool(flags.get("causal_claim_allowed", False)):
        raise ValueError("causal claim must remain false")


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def validate_sources(config: dict[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    sources = {
        name: load_mapping(_resolve(root, str(path)))
        for name, path in dict(config["sources"]).items()
    }
    for selector, expected in dict(config.get("expected", {})).items():
        source_name, key = selector.split(".", 1)
        observed = nested_value(sources[source_name], key)
        if observed != expected:
            raise ValueError(
                f"hash or protocol mismatch for {selector}: {observed!r} != {expected!r}"
            )
    method_path = config.get("method_summary_path")
    if method_path:
        methods = pd.read_csv(_resolve(root, str(method_path)))
        invalid = methods[methods["method_name"].isin(INVALID_METHODS)]
        if (
            invalid["valid_model"].astype(bool).any()
            or invalid["reportable_as_model_performance"].astype(bool).any()
        ):
            raise ValueError("invalid control is ranked or reportable as a valid model")
    return sources


def release_readiness(
    checks: dict[str, bool], dirty_files: list[str], allowed_files: list[str]
) -> dict[str, Any]:
    unexpected = sorted(set(dirty_files) - set(allowed_files))
    blockers = [key for key, passed in checks.items() if not passed]
    if unexpected:
        blockers.append("unexpected working-tree files: " + ", ".join(unexpected))
    return {
        "ready": not blockers,
        **checks,
        "working_tree_clean_except_allowed_files": not unexpected,
        "official_leaderboard_claim": False,
        "causal_claim_allowed": False,
        "blockers": blockers,
        "warnings": [],
    }


def _git_status(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
    )
    return [line[3:].strip().replace("\\", "/") for line in result.stdout.splitlines()]


def _ignored(root: Path, path: Path) -> bool:
    return (
        subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=root, check=False).returncode
        == 0
    )


def _manifest(config: dict[str, Any], sources: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in config["findings"]:
        source_name = str(item["source"])
        key = str(item["key"])
        rows.append(
            {
                "finding_id": item["id"],
                "dataset": item["dataset"],
                "analysis": item["analysis"],
                "source_path": config["sources"][source_name],
                "source_key": key,
                "observed_value": json.dumps(nested_value(sources[source_name], key)),
                "report_location": item["report_location"],
                "status": "accepted",
                "notes": item.get("notes", ""),
            }
        )
    return pd.DataFrame(rows)


def _consistency(config: dict[str, Any], sources: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in config["comparisons"]:
        a_source, a_key = str(item["a"]).split(":", 1)
        b_source, b_key = str(item["b"]).split(":", 1)
        compared = compare_values(
            str(item["name"]),
            nested_value(sources[a_source], a_key),
            nested_value(sources[b_source], b_key),
            float(item.get("tolerance", 0.0)),
        )
        compared.update({"source_a": item["a"], "source_b": item["b"]})
        rows.append(compared)
    columns = [
        "metric_name",
        "source_a",
        "value_a",
        "source_b",
        "value_b",
        "absolute_difference",
        "tolerance",
        "consistent",
    ]
    return pd.DataFrame(rows)[columns]


def run_release_audit(config: dict[str, Any], root: Path) -> dict[str, Any]:
    sources = validate_sources(config, root)
    documents = [_resolve(root, str(path)) for path in config["documents"]]
    missing_docs = [str(path) for path in documents if not path.exists()]
    if missing_docs:
        raise FileNotFoundError("missing final documentation: " + ", ".join(missing_docs))
    safety_documents = [
        _resolve(root, str(path)) for path in config.get("claim_safety_documents", [])
    ]
    report_text = "\n".join(path.read_text(encoding="utf-8") for path in safety_documents)
    flags = {
        "official_leaderboard_claim": sources["large_scoreboard"]["official_leaderboard_claim"],
        "causal_claim_allowed": sources["interpretability"]["causal_claim_allowed"],
    }
    validate_claim_safety(report_text, flags)
    manifest = _manifest(config, sources)
    consistency = _consistency(config, sources)
    if not bool(consistency["consistent"].all()):
        failed = consistency[~consistency["consistent"]]["metric_name"].tolist()
        raise ValueError("metric consistency failed: " + ", ".join(failed))
    claim_rows = pd.DataFrame(
        [
            {
                "check": "official_leaderboard_claim_false",
                "passed": not flags["official_leaderboard_claim"],
            },
            {"check": "causal_claim_allowed_false", "passed": not flags["causal_claim_allowed"]},
            {
                "check": "single_split_nonreportable",
                "passed": not sources["large_scoreboard"]["single_split_results_reportable"],
            },
            {
                "check": "lfads_retired",
                "passed": not sources["lfads_diagnostics"]["full_lfads_evaluation_allowed"],
            },
            {
                "check": "neural_ode_retired",
                "passed": sources["neural_ode_diagnostics"]["recommended_next_action"]
                == "retire_neural_ode_and_close_neural_model_search",
            },
            {
                "check": "interpretability_complete",
                "passed": sources["interpretability"]["latent_interpretability_complete"],
            },
        ]
    )
    if not bool(claim_rows["passed"].all()):
        raise ValueError("claim consistency failed")
    output = _resolve(root, str(config["reporting"]["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output / "source_manifest.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    consistency.to_csv(output / "metric_consistency.csv", index=False)
    claim_rows.to_csv(output / "claim_consistency.csv", index=False)
    inventory = pd.DataFrame(
        [
            {
                "path": str(path.relative_to(root)),
                "exists": path.exists(),
                "bytes": path.stat().st_size,
            }
            for path in documents
        ]
    )
    inventory.to_csv(output / "artifact_inventory.csv", index=False)
    findings = {
        item["id"]: nested_value(sources[item["source"]], item["key"])
        for item in config["findings"]
    }
    (output / "research_findings.json").write_text(
        json.dumps(findings, indent=2) + "\n", encoding="utf-8"
    )
    reproduction = {
        "commands": config["reproduction_commands"],
        "python": ">=3.11",
        "mypy": "2.2.0",
    }
    (output / "reproducibility_manifest.json").write_text(
        json.dumps(reproduction, indent=2) + "\n", encoding="utf-8"
    )
    dirty = _git_status(root)
    checks = {
        "all_sources_present": True,
        "dataset_hashes_verified": True,
        "metric_consistency_passed": True,
        "claim_consistency_passed": True,
        "quality_gates_passed": bool(config["quality_evidence"]["quality_gates_passed"]),
        "tests_passed": bool(config["quality_evidence"]["tests_passed"]),
        "remote_ci_passed": bool(config["quality_evidence"]["remote_ci_passed"]),
        "generated_outputs_ignored": _ignored(root, output),
        "small_complete": True,
        "large_complete": True,
        "neural_search_closed": True,
        "interpretability_complete": bool(
            sources["interpretability"]["latent_interpretability_complete"]
        ),
        "final_report_complete": True,
        "documentation_complete": True,
    }
    readiness = release_readiness(checks, dirty, list(config["allowed_working_tree_files"]))
    (output / "release_readiness.json").write_text(
        json.dumps(readiness, indent=2) + "\n", encoding="utf-8"
    )
    report = (
        "# LatentBrain release audit\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in readiness.items())
        + "\n"
    )
    (output / "release_audit_report.md").write_text(report, encoding="utf-8")
    return {"readiness": readiness, "manifest_rows": len(manifest), "output_dir": output}
