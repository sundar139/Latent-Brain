from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json_report(summary: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _lines_for_mapping(mapping: dict[str, Any]) -> list[str]:
    return [f"- {key}: {value}" for key, value in mapping.items()]


def write_markdown_validation_report(
    output_path: Path,
    dataset_name: str,
    summary: dict[str, Any],
    quality_flags: list[dict[str, str]],
    generated_tables: dict[str, str],
    generated_figures: dict[str, str],
    metadata: dict[str, Any] | None,
    provenance: dict[str, Any] | None,
) -> Path:
    metadata = metadata or {}
    provenance = provenance or {}
    split_counts = metadata.get("split_counts", {})
    mask_counts = metadata.get("neuron_mask_counts", {})
    trialization = metadata.get("trialization", {})
    lines = [
        f"# {dataset_name} validation report",
        "",
        "No model training or benchmark evaluation was performed in this report.",
        "",
        "## Dataset",
        f"- Dataset hash: {summary.get('dataset_hash')}",
        f"- Input processed path: {summary.get('processed_path')}",
        (
            f"- Shape: [{summary.get('n_trials')}, {summary.get('n_time_bins')}, "
            f"{summary.get('n_neurons')}]"
        ),
        f"- Bin size ms: {summary.get('bin_size_ms')}",
        "",
        "## Splits and masks",
        *_lines_for_mapping(split_counts),
        *_lines_for_mapping(mask_counts),
        "",
        "## Trialization",
        *_lines_for_mapping(trialization),
        "",
        "## Spike activity",
        f"- Total spikes: {summary.get('total_spikes')}",
        f"- Mean population rate Hz: {summary.get('mean_population_rate_hz')}",
        f"- Median neuron rate Hz: {summary.get('median_neuron_rate_hz')}",
        f"- Zero fraction: {summary.get('zero_fraction')}",
        "",
        "## Quality flags",
    ]
    if quality_flags:
        lines.extend(
            f"- {flag['severity']}: {flag['code']} — {flag['message']}" for flag in quality_flags
        )
    else:
        lines.append("- None")
    lines.extend(["", "## Generated tables"])
    lines.extend(_lines_for_mapping(generated_tables))
    lines.extend(["", "## Generated figures"])
    lines.extend(_lines_for_mapping(generated_figures))
    lines.extend(
        [
            "",
            "## Provenance",
            f"- Train file used: {provenance.get('train_file_used')}",
            f"- Test files detected: {provenance.get('test_files_detected')}",
            f"- Git commit: {provenance.get('git_commit')}",
            f"- DANDI dandiset ID: {provenance.get('dandiset_id')}",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
