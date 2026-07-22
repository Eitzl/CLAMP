"""Cross-source deduplication.

MD_design_docs/06_task5_data_pipeline_plan.md §4.3, scaffolded in
MD_design_docs/09_phase1_implementation_design.md §7. This is the
leakage-prevention logic: if the same peptide from two sources lands in
different train/test cluster folds later (Phase 2), the homology split's
whole purpose is defeated.
"""

from pydantic import BaseModel

from clamp.data.schema import LabelQualityFlag, PeptideRecord, Source

# Above this, replicate values for the same (peptide_uid, task,
# species/target_cell) are flagged HIGH_REPLICATE_DISPERSION rather than
# silently trusted — see data/README.md §3.5. A full order of magnitude was
# chosen as a first-pass threshold (more likely a unit/transcription error
# than real assay noise); revisit once the full dataset's real dispersion
# distribution is visible in the datasheet.
REPLICATE_DISPERSION_RATIO_THRESHOLD = 10.0


class DedupResult(BaseModel):
    records: list[PeptideRecord]
    overlap_matrix: dict[tuple[Source, Source], int]
    raw_row_count: int
    unique_peptide_uid_count: int
    flagged_conflict_count: int


def canonicalize_sequence(sequence_raw: str, source: Source) -> str:
    """Strip/uppercase + resolve source-specific non-standard 1-letter
    codes to a common vocabulary. The vocabulary is source-specific and
    grows as real data surfaces gaps (doc 06 §4.3 step 1) — do not build a
    single global mapping table."""
    raise NotImplementedError


def modification_signature(record: PeptideRecord) -> tuple:
    """(nterm_mod, cterm_mod, sorted (position, modification_type) tuples,
    cyclization_type) — the dedup key's non-sequence component."""
    raise NotImplementedError


def compute_peptide_uid(sequence_canonical: str, mod_sig: tuple) -> str:
    """Stable hash of sequence_canonical + mod_sig (doc 06 §4.3 step 3)."""
    raise NotImplementedError


def dedup_records(records: list[PeptideRecord]) -> DedupResult:
    """Group by peptide_uid; on a sequence-only fuzzy match with differing
    metadata richness, the metadata-richer record's smiles/fidelity_tier
    wins (doc 06 §4.3 step 5) — a metadata-poor duplicate must not silently
    downgrade a metadata-rich one. Independent replicate measurements (same
    peptide_uid, same assay type/species/target-cell, different source) are
    kept as separate rows sharing peptide_uid, not merged (doc 06 §4.3 step
    4) — that's what lets cluster-splitting treat them as one entity while
    preserving replicate-noise signal.

    Also runs conflict detection (data/README.md §3.5): within each
    (peptide_uid, task, species/target_cell) group, if replicate values
    span more than REPLICATE_DISPERSION_RATIO_THRESHOLD, tag every row in
    that group with label_quality_flag=HIGH_REPLICATE_DISPERSION. This does
    NOT drop or average the rows — it only flags them for the datasheet
    pass and eventual manual review; auto-resolving a likely-error case
    silently would be worse than leaving it visible.
    """
    raise NotImplementedError


def flag_high_dispersion_replicates(records: list[PeptideRecord]) -> list[PeptideRecord]:
    """Standalone conflict-detection pass, factored out of dedup_records so
    it's independently unit-testable (data/README.md §3.5): group by
    (peptide_uid, task, species/target_cell), compute max/min ratio per
    group, set label_quality_flag=HIGH_REPLICATE_DISPERSION on every row in
    a group exceeding REPLICATE_DISPERSION_RATIO_THRESHOLD."""
    raise NotImplementedError
