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
    """One field per table in doc 06 §4.4."""

    per_source_coverage: list[dict]  # rows contributed, % HC50/MIC/both, % after dedup
    species_target_cell_frequency: list[dict]  # per doc 02 open question #1
    fidelity_breakdown: list[dict]  # doc 06 §5.1 tiers, overall + by source + by label presence
    dedup_summary: list[dict]  # raw rows in, unique peptide_uids out, overlap matrix, conflict count


def build_datasheet(dataset: pd.DataFrame, dedup_result: DedupResult) -> DatasheetTables:
    raise NotImplementedError


def write_datasheet(tables: DatasheetTables, out_dir: Path) -> None:
    """Writes out_dir/datasheet.json and out_dir/datasheet.md from the same
    DatasheetTables instance."""
    raise NotImplementedError
