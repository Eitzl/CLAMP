"""Cross-source deduplication.

MD_design_docs/06_task5_data_pipeline_plan.md §4.3, scaffolded in
MD_design_docs/09_phase1_implementation_design.md §7. This is the
leakage-prevention logic: if the same peptide from two sources lands in
different train/test cluster folds later (Phase 2), the homology split's
whole purpose is defeated.
"""

import hashlib
from collections import defaultdict

from pydantic import BaseModel

from clamp.data.schema import CyclizationType, FidelityTier, LabelQualityFlag, PeptideRecord, Source

# Above this, replicate values for the same (peptide_uid, task,
# species/target_cell) are flagged HIGH_REPLICATE_DISPERSION rather than
# silently trusted — see data/README.md §3.5. A full order of magnitude was
# chosen as a first-pass threshold (more likely a unit/transcription error
# than real assay noise); revisit once the full dataset's real dispersion
# distribution is visible in the datasheet.
REPLICATE_DISPERSION_RATIO_THRESHOLD = 10.0

# Per-source non-standard 1-letter-code resolution (doc 06 §4.3 step 1).
# Deliberately empty for now — grows per-source as real pulled data
# surfaces gaps (data/README.md §2's "Modification vocabulary" note), not
# a single global table.
_SOURCE_LETTER_OVERRIDES: dict[Source, dict[str, str]] = {}


class DedupResult(BaseModel):
    records: list[PeptideRecord]
    overlap_matrix: dict[tuple[Source, Source], int]
    raw_row_count: int
    unique_peptide_uid_count: int
    flagged_conflict_count: int


def canonicalize_sequence(sequence_raw: str, source: Source) -> str:
    """Strip/uppercase + resolve source-specific non-standard 1-letter
    codes to a common vocabulary."""
    sequence = sequence_raw.strip().upper()
    overrides = _SOURCE_LETTER_OVERRIDES.get(source)
    if overrides:
        sequence = "".join(overrides.get(ch, ch) for ch in sequence)
    return sequence


def modification_signature(record: PeptideRecord) -> tuple:
    return (
        record.nterm_mod,
        record.cterm_mod,
        tuple(sorted((r.position, r.modification_type) for r in record.unusual_residues)),
        record.cyclization_type,
    )


def compute_peptide_uid(sequence_canonical: str, mod_sig: tuple) -> str:
    """Stable hash of sequence_canonical + mod_sig (doc 06 §4.3 step 3)."""
    key = f"{sequence_canonical}|{mod_sig}".encode()
    return hashlib.blake2b(key, digest_size=16).hexdigest()


def _metadata_richness(record: PeptideRecord) -> tuple:
    """Used only to break ties on which record's smiles/fidelity_tier wins
    when several rows collapse onto the same peptide_uid (doc 06 §4.3 step
    5) — never to decide whether rows collapse in the first place, which
    is peptide_uid's job alone."""
    return (
        record.smiles is not None,
        record.chemical_fidelity_tier == FidelityTier.FULL,
        len(record.unusual_residues),
        record.cyclization_type != CyclizationType.NONE,
    )


def flag_high_dispersion_replicates(records: list[PeptideRecord]) -> list[PeptideRecord]:
    """Standalone conflict-detection pass (data/README.md §3.5): group by
    (peptide_uid, task, species/target_cell), compute max/min ratio per
    group, set label_quality_flag=HIGH_REPLICATE_DISPERSION on every row in
    a group exceeding REPLICATE_DISPERSION_RATIO_THRESHOLD. Never drops or
    averages rows — only flags them for the datasheet pass and manual
    review."""
    groups: dict[tuple, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        if record.peptide_uid is None:
            continue
        if record.mic_value is not None:
            groups[(record.peptide_uid, "mic", record.mic_target_species)].append(idx)
        if record.hc50_value is not None:
            groups[(record.peptide_uid, "hc50", record.hc50_assay_target_cell)].append(idx)

    flagged: set[int] = set()
    for (_, task, _), indices in groups.items():
        if len(indices) < 2:
            continue
        values = [records[i].mic_value if task == "mic" else records[i].hc50_value for i in indices]
        values = [v for v in values if v is not None and v > 0]
        if len(values) < 2:
            continue
        if max(values) / min(values) > REPLICATE_DISPERSION_RATIO_THRESHOLD:
            flagged.update(indices)

    if not flagged:
        return records
    return [
        record.model_copy(update={"label_quality_flag": LabelQualityFlag.HIGH_REPLICATE_DISPERSION})
        if idx in flagged
        else record
        for idx, record in enumerate(records)
    ]


def dedup_records(records: list[PeptideRecord]) -> DedupResult:
    """Groups by peptide_uid, applies the metadata-richness tie-break from
    doc 06 §4.3 step 5, and returns both the deduped set and enough
    bookkeeping (overlap matrix, flagged-conflict count) to drive the
    datasheet's dedup summary (doc 06 §4.4). Independent replicate
    measurements are always kept as separate rows sharing peptide_uid —
    this function only ever relabels smiles/fidelity metadata or sets
    label_quality_flag, it never merges or drops a row.
    """
    canonical_seqs = [canonicalize_sequence(r.sequence_raw, r.source) for r in records]
    uids = [compute_peptide_uid(seq, modification_signature(r)) for seq, r in zip(canonical_seqs, records)]

    groups: dict[str, list[int]] = defaultdict(list)
    for idx, uid in enumerate(uids):
        groups[uid].append(idx)

    output: list[PeptideRecord] = []
    overlap_counts: dict[tuple[Source, Source], int] = defaultdict(int)

    for uid, indices in groups.items():
        group_records = [records[i] for i in indices]
        richest = max(group_records, key=_metadata_richness)
        richest_score = _metadata_richness(richest)

        # Rich-vs-rich conflict: doc 06 §4.3 step 5 only specifies the
        # rich-vs-poor case explicitly. When two or more records tie for
        # richest and disagree on smiles, max()'s tie-break (first
        # max-scoring element in list order) would otherwise silently pick
        # a "winner" with no signal that anything was ambiguous. Instead:
        # pick a deterministic winner (lexicographically-smallest source
        # value, so the same input always resolves the same way regardless
        # of incidental list order) and flag every tied-but-disagreeing
        # record — including the chosen winner — so the disagreement is
        # visible rather than silently resolved.
        tied_richest = [r for r in group_records if _metadata_richness(r) == richest_score]
        tied_disagree = len(tied_richest) > 1 and len({r.smiles for r in tied_richest if r.smiles is not None}) > 1
        if tied_disagree:
            richest = min(tied_richest, key=lambda r: r.source)
        tied_richest_ids = {id(r) for r in tied_richest}

        sources_in_group = sorted({r.source for r in group_records})
        for i, source_a in enumerate(sources_in_group):
            for source_b in sources_in_group[i + 1 :]:
                overlap_counts[(source_a, source_b)] += 1

        for seq, record in zip((canonical_seqs[i] for i in indices), group_records):
            updates = {"sequence_canonical": seq, "peptide_uid": uid}
            if record is not richest and _metadata_richness(record) < richest_score and richest.smiles is not None:
                updates.update(
                    smiles=richest.smiles,
                    smiles_source=richest.smiles_source,
                    smiles_validated=richest.smiles_validated,
                    chemical_fidelity_tier=richest.chemical_fidelity_tier,
                    label_quality_flag=LabelQualityFlag.METADATA_SOURCE_CONFLICT,
                )
            elif tied_disagree and id(record) in tied_richest_ids:
                updates["label_quality_flag"] = LabelQualityFlag.METADATA_SOURCE_CONFLICT
            output.append(record.model_copy(update=updates))

    output = flag_high_dispersion_replicates(output)
    flagged_conflict_count = sum(1 for r in output if r.label_quality_flag is not None)

    return DedupResult(
        records=output,
        overlap_matrix=dict(overlap_counts),
        raw_row_count=len(records),
        unique_peptide_uid_count=len(groups),
        flagged_conflict_count=flagged_conflict_count,
    )
