"""Datasheet generation: the four reporting tables from
MD_design_docs/06_task5_data_pipeline_plan.md §4.4, plus the dedup-conflict
count added in data/README.md §3.7.

Emits both datasheet.json (machine-readable, so later phases/CI-style
checks can assert against it) and datasheet.md (human-readable), rendered
from the same underlying numbers — see
MD_design_docs/09_phase1_implementation_design.md §9 on why these must not
be authored independently.
"""

from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from clamp.data.dedup import DedupResult


class DatasheetTables(BaseModel):
    """One field per table in doc 06 §4.4, plus peptide_level_coverage
    (see build_datasheet's docstring on why per_source_coverage alone is
    misleading for the "how many peptides have both labels" question)."""

    per_source_coverage: list[dict]
    peptide_level_coverage: list[dict]
    species_target_cell_frequency: list[dict]
    fidelity_breakdown: list[dict]
    dedup_summary: list[dict]


def build_datasheet(dataset: pd.DataFrame, dedup_result: DedupResult) -> DatasheetTables:
    """per_source_coverage's `pct_with_both` is row-level and, found via a
    real ~90k-row pilot pull, is structurally 0.0% for every source by
    construction of the one-row-per-assay-record schema (doc 06 §4.1) — a
    peptide with both a MIC and an HC50 measurement is two separate rows,
    never one row with both fields populated, so no source can ever show
    a nonzero row-level pct_with_both. That's not a bug in the row-level
    number, but it's a materially misleading answer to the actual
    multitask-modeling-relevant question doc 04 cares about ("how many
    *peptides* have both labels"). peptide_level_coverage answers that
    question instead, grouping by peptide_uid across all rows (including
    ones contributed by other sources sharing that uid) before checking
    label presence — kept alongside per_source_coverage, not replacing it,
    since both numbers are legitimate answers to different questions.
    """
    per_source_coverage = []
    for source, group in dataset.groupby("source"):
        n = len(group)
        per_source_coverage.append(
            {
                "source": source,
                "rows": n,
                "unique_peptide_uids": int(group["peptide_uid"].nunique()),
                "pct_with_mic": round(100 * group["mic_value_uM"].notna().mean(), 1) if n else 0.0,
                "pct_with_hc50": round(100 * group["hc50_value_uM"].notna().mean(), 1) if n else 0.0,
                "pct_with_both": round(
                    100 * (group["mic_value_uM"].notna() & group["hc50_value_uM"].notna()).mean(), 1
                )
                if n
                else 0.0,
            }
        )

    dataset = dataset.copy()
    dataset["_has_mic_uid"] = dataset.groupby("peptide_uid")["mic_value_uM"].transform(lambda s: s.notna().any())
    dataset["_has_hc50_uid"] = dataset.groupby("peptide_uid")["hc50_value_uM"].transform(lambda s: s.notna().any())

    def _peptide_level_row(label: str, uids: pd.DataFrame) -> dict:
        n = len(uids)
        return {
            "source": label,
            "unique_peptide_uids": n,
            "pct_with_mic": round(100 * uids["_has_mic_uid"].mean(), 1) if n else 0.0,
            "pct_with_hc50": round(100 * uids["_has_hc50_uid"].mean(), 1) if n else 0.0,
            "pct_with_both": round(100 * (uids["_has_mic_uid"] & uids["_has_hc50_uid"]).mean(), 1) if n else 0.0,
        }

    peptide_level_coverage = [
        _peptide_level_row(source, group.drop_duplicates("peptide_uid")) for source, group in dataset.groupby("source")
    ]
    peptide_level_coverage.append(_peptide_level_row("ALL", dataset.drop_duplicates("peptide_uid")))

    species_counts = dataset["mic_target_species"].value_counts(dropna=True)
    cell_counts = dataset["hc50_assay_target_cell"].value_counts(dropna=True)
    species_target_cell_frequency = [
        {"kind": "mic_target_species", "name": name, "count": int(count)} for name, count in species_counts.items()
    ] + [
        {"kind": "hc50_assay_target_cell", "name": name, "count": int(count)}
        for name, count in cell_counts.items()
    ]

    fidelity_breakdown = [
        {"source": source, "fidelity_tier": tier, "rows": len(group)}
        for (source, tier), group in dataset.groupby(["source", "chemical_fidelity_tier"], dropna=False)
    ]
    fidelity_breakdown += [
        {"source": "ALL", "fidelity_tier": tier, "rows": len(group)}
        for tier, group in dataset.groupby("chemical_fidelity_tier", dropna=False)
    ]

    dedup_summary = [
        {
            "raw_row_count": dedup_result.raw_row_count,
            "unique_peptide_uid_count": dedup_result.unique_peptide_uid_count,
            "flagged_conflict_count": dedup_result.flagged_conflict_count,
            "overlap_matrix": {f"{a}|{b}": count for (a, b), count in dedup_result.overlap_matrix.items()},
        }
    ]

    return DatasheetTables(
        per_source_coverage=per_source_coverage,
        peptide_level_coverage=peptide_level_coverage,
        species_target_cell_frequency=species_target_cell_frequency,
        fidelity_breakdown=fidelity_breakdown,
        dedup_summary=dedup_summary,
    )


def write_datasheet(tables: DatasheetTables, out_dir: Path) -> None:
    """Writes out_dir/datasheet.json and out_dir/datasheet.md from the same
    DatasheetTables instance."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "datasheet.json").write_text(tables.model_dump_json(indent=2))

    lines = ["# Phase 1 dataset datasheet", ""]
    for title, key in [
        ("Per-source coverage", "per_source_coverage"),
        ("Peptide-level coverage", "peptide_level_coverage"),
        ("Species / target-cell frequency", "species_target_cell_frequency"),
        ("Fidelity tier breakdown", "fidelity_breakdown"),
        ("Dedup summary", "dedup_summary"),
    ]:
        lines.append(f"## {title}")
        rows = getattr(tables, key)
        if not rows:
            lines.append("(no rows)")
            lines.append("")
            continue
        columns = list(rows[0].keys())
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
        lines.append("")

    (out_dir / "datasheet.md").write_text("\n".join(lines))
