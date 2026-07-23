"""Coverage per MD_design_docs/09_phase1_implementation_design.md §12 and
data/README.md §3.
"""

from clamp.data.dedup import (
    canonicalize_sequence,
    compute_peptide_uid,
    dedup_records,
    flag_high_dispersion_replicates,
    modification_signature,
)
from clamp.data.schema import CyclizationType, FidelityTier, LabelQualityFlag, PeptideRecord, Source, SmilesSource


def _record(source: Source, source_id: str, sequence: str = "KWKLFKKIEK", **kwargs) -> PeptideRecord:
    return PeptideRecord(source=source, source_id=source_id, sequence_raw=sequence, **kwargs)


class TestPeptideUid:
    def test_same_sequence_and_mods_same_uid(self):
        seq = canonicalize_sequence("KWKLFKKIEK", Source.DBAASP)
        sig = modification_signature(_record(Source.DBAASP, "1"))
        uid_a = compute_peptide_uid(seq, sig)
        uid_b = compute_peptide_uid(seq, sig)
        assert uid_a == uid_b

    def test_different_cyclization_type_different_uid(self):
        seq = canonicalize_sequence("KWKLFKKIEK", Source.DBAASP)
        sig_linear = modification_signature(_record(Source.DBAASP, "1"))
        sig_cyclic = modification_signature(
            _record(Source.DBAASP, "1", is_cyclic=True, cyclization_type=CyclizationType.HEAD_TO_TAIL)
        )
        assert compute_peptide_uid(seq, sig_linear) != compute_peptide_uid(seq, sig_cyclic)

    def test_uid_is_stable_across_runs(self):
        seq = canonicalize_sequence("KWKLFKKIEK", Source.DBAASP)
        sig = modification_signature(_record(Source.DBAASP, "1"))
        uids = {compute_peptide_uid(seq, sig) for _ in range(5)}
        assert len(uids) == 1

    def test_canonicalize_sequence_strips_but_preserves_case(self):
        """Case is NOT folded: lowercase denotes a D-amino-acid (QMAP's own
        convention, and p2smi's — see dedup.canonicalize_sequence's
        docstring). Only surrounding whitespace is stripped."""
        assert canonicalize_sequence(" kwklfkkiek ", Source.DBAASP) == "kwklfkkiek"
        assert canonicalize_sequence(" KWKLFKKIEK ", Source.DBAASP) == "KWKLFKKIEK"

    def test_d_and_l_form_sequences_get_different_uids(self):
        """Regression guard: canonicalize_sequence used to uppercase
        everything, so a D-residue-containing peptide (lowercase, per
        QMAP's convention) and its all-L counterpart collapsed onto the
        same peptide_uid — silently merging two chemically distinct
        molecules and defeating the split-safety guarantee this stage
        exists for (data/README.md §3.6)."""
        sig = modification_signature(_record(Source.QMAP, "1"))
        upper_seq = canonicalize_sequence("KWKLFKKIEK", Source.QMAP)
        lower_seq = canonicalize_sequence("kwklfkkiek", Source.QMAP)
        assert compute_peptide_uid(upper_seq, sig) != compute_peptide_uid(lower_seq, sig)


class TestModificationSignatureNormalization:
    """F3: incidental case/whitespace differences in terminal-modification
    codes shouldn't split what should be the same modification across
    sources — see dedup._normalize_mod_code's docstring for why this is a
    normalization pass, not a full name -> abbreviation mapping table."""

    def test_mod_code_case_difference_does_not_change_signature(self):
        a = _record(Source.DBAASP, "1", nterm_mod="ACT")
        b = _record(Source.QMAP, "2", nterm_mod="act")
        assert modification_signature(a) == modification_signature(b)

    def test_mod_code_whitespace_difference_does_not_change_signature(self):
        a = _record(Source.DBAASP, "1", cterm_mod="AMD")
        b = _record(Source.QMAP, "2", cterm_mod=" AMD ")
        assert modification_signature(a) == modification_signature(b)

    def test_genuinely_different_mod_codes_still_differ(self):
        a = _record(Source.DBAASP, "1", nterm_mod="ACT")
        b = _record(Source.QMAP, "2", nterm_mod="SUC")
        assert modification_signature(a) != modification_signature(b)


class TestDedupMerge:
    def test_metadata_rich_record_wins_smiles_on_exact_uid_match(self):
        """data/README.md §3.4: a metadata-rich record (has smiles + FULL
        tier) + a metadata-poor duplicate of the same peptide -> the poor
        record inherits the rich one's smiles/fidelity_tier, not the other
        way around. (Renamed from "...on_fuzzy_match": both records here
        share the byte-identical canonical sequence and modification
        signature, so this is an exact peptide_uid hash collision — there
        is no fuzzy/near-duplicate matching logic anywhere in dedup.py, and
        the old name implied there was.)"""
        rich = _record(
            Source.DBAASP, "1", smiles="CC[C@H](C)N", smiles_source=SmilesSource.DB_CURATED,
            chemical_fidelity_tier=FidelityTier.FULL,
        )
        poor = _record(Source.APD3, "2")  # sequence-only, no smiles at all
        result = dedup_records([rich, poor])
        by_source = {r.source: r for r in result.records}
        assert by_source[Source.DBAASP].smiles == "CC[C@H](C)N"
        assert by_source[Source.APD3].smiles == "CC[C@H](C)N"
        assert by_source[Source.APD3].label_quality_flag == LabelQualityFlag.METADATA_SOURCE_CONFLICT

    def test_rich_vs_rich_disagreement_flags_both_and_picks_deterministic_winner(self):
        """data/README.md §3.4 only specifies rich-vs-poor. When two
        records tie for richest (both FULL tier, both have smiles) but
        disagree on the actual smiles string, the old code silently used
        Python max()'s first-max-in-list-order tie-break with zero signal
        that anything was ambiguous. Now: both get flagged, and the winner
        is deterministic (lexicographically-smallest source value) so
        re-running with the records in a different order can't change the
        result."""
        a = _record(
            Source.DBAASP, "1", smiles="CC[C@H](C)N", smiles_source=SmilesSource.DB_CURATED,
            chemical_fidelity_tier=FidelityTier.FULL,
        )
        b = _record(
            Source.DRAMP, "2", smiles="CC[C@@H](C)N", smiles_source=SmilesSource.DB_CURATED,
            chemical_fidelity_tier=FidelityTier.FULL,
        )
        result_forward = dedup_records([a, b])
        result_reversed = dedup_records([b, a])
        for result in (result_forward, result_reversed):
            by_source = {r.source: r for r in result.records}
            assert by_source[Source.DBAASP].label_quality_flag == LabelQualityFlag.METADATA_SOURCE_CONFLICT
            assert by_source[Source.DRAMP].label_quality_flag == LabelQualityFlag.METADATA_SOURCE_CONFLICT
            # neither rich record's own smiles gets overwritten by the other
            assert by_source[Source.DBAASP].smiles == "CC[C@H](C)N"
            assert by_source[Source.DRAMP].smiles == "CC[C@@H](C)N"
        # order-independence: same result regardless of input list order
        assert {r.source: r.smiles for r in result_forward.records} == {
            r.source: r.smiles for r in result_reversed.records
        }

    def test_rich_vs_rich_agreement_is_not_flagged(self):
        """Two FULL-tier records that happen to agree on smiles are not a
        conflict — only disagreement should be flagged."""
        a = _record(
            Source.DBAASP, "1", smiles="CC[C@H](C)N", smiles_source=SmilesSource.DB_CURATED,
            chemical_fidelity_tier=FidelityTier.FULL,
        )
        b = _record(
            Source.DRAMP, "2", smiles="CC[C@H](C)N", smiles_source=SmilesSource.DB_CURATED,
            chemical_fidelity_tier=FidelityTier.FULL,
        )
        result = dedup_records([a, b])
        assert all(r.label_quality_flag is None for r in result.records)

    def test_replicate_measurements_kept_as_separate_rows(self):
        """data/README.md §3.2: same peptide_uid, same assay conditions,
        different sources, different values -> both rows survive, neither
        averaged nor dropped."""
        a = _record(Source.DBAASP, "1", mic_value=2.0, mic_unit="uM", mic_target_species="E. coli")
        b = _record(Source.DRAMP, "2", mic_value=4.0, mic_unit="uM", mic_target_species="E. coli")
        result = dedup_records([a, b])
        assert len(result.records) == 2
        mic_values = sorted(r.mic_value for r in result.records)
        assert mic_values == [2.0, 4.0]

    def test_complementary_labels_coexist_without_merging(self):
        """data/README.md §3.3: one source has HC50 only, another has MIC
        only, for the same peptide_uid -> both rows present after dedup."""
        a = _record(Source.DBAASP, "1", hc50_value=50.0, hc50_unit="uM")
        b = _record(Source.HEMOLYTIK2, "2", mic_value=10.0, mic_unit="uM", mic_target_species="S. aureus")
        result = dedup_records([a, b])
        assert len(result.records) == 2
        assert result.unique_peptide_uid_count == 1
        assert {r.peptide_uid for r in result.records} == {result.records[0].peptide_uid}

    def test_unrelated_sequences_do_not_collapse(self):
        a = _record(Source.DBAASP, "1", sequence="KWKLFKKIEK")
        b = _record(Source.DBAASP, "2", sequence="ACDEFGHIK")
        result = dedup_records([a, b])
        assert result.unique_peptide_uid_count == 2


class TestConflictDetection:
    """data/README.md §3.5 / dedup.flag_high_dispersion_replicates."""

    def test_order_of_magnitude_spread_flags_high_dispersion(self):
        a = _record(Source.DBAASP, "1", mic_value=2.0, mic_unit="uM", mic_target_species="E. coli")
        b = _record(Source.DRAMP, "2", mic_value=200.0, mic_unit="uM", mic_target_species="E. coli")
        result = dedup_records([a, b])
        assert all(r.label_quality_flag == LabelQualityFlag.HIGH_REPLICATE_DISPERSION for r in result.records)
        # data/README.md §3.7: raw_row_count/flagged_conflict_count are the
        # aggregate numbers the datasheet's dedup summary actually reports
        # by name — previously asserted nowhere in this suite.
        assert result.raw_row_count == 2
        assert result.unique_peptide_uid_count == 1
        assert result.flagged_conflict_count == 2

    def test_flagged_conflict_count_is_zero_when_nothing_is_flagged(self):
        a = _record(Source.DBAASP, "1", mic_value=2.0, mic_unit="uM", mic_target_species="E. coli")
        b = _record(Source.DRAMP, "2", mic_value=3.0, mic_unit="uM", mic_target_species="E. coli")
        result = dedup_records([a, b])
        assert result.flagged_conflict_count == 0
        assert result.raw_row_count == 2

    def test_close_replicate_values_not_flagged(self):
        a = _record(Source.DBAASP, "1", mic_value=2.0, mic_unit="uM", mic_target_species="E. coli")
        b = _record(Source.DRAMP, "2", mic_value=3.0, mic_unit="uM", mic_target_species="E. coli")
        deduped = dedup_records([a, b]).records
        assert all(r.label_quality_flag is None for r in deduped)

    def test_flag_applies_to_all_rows_in_the_conflicting_group(self):
        a = _record(Source.DBAASP, "1", mic_value=2.0, mic_unit="uM", mic_target_species="E. coli")
        b = _record(Source.DRAMP, "2", mic_value=200.0, mic_unit="uM", mic_target_species="E. coli")
        c = _record(Source.QMAP, "3", mic_value=3.0, mic_unit="uM", mic_target_species="E. coli")
        flagged = flag_high_dispersion_replicates([a, b, c])
        # peptide_uid isn't set on these directly-constructed records, so
        # flag_high_dispersion_replicates (called standalone, not via
        # dedup_records) should see them as ungrouped and flag nothing —
        # confirms grouping is keyed on peptide_uid, not just task/species.
        assert all(r.label_quality_flag is None for r in flagged)

    def test_flag_applies_within_dedup_group(self):
        a = _record(Source.DBAASP, "1", mic_value=2.0, mic_unit="uM", mic_target_species="E. coli")
        b = _record(Source.DRAMP, "2", mic_value=200.0, mic_unit="uM", mic_target_species="E. coli")
        c = _record(Source.QMAP, "3", mic_value=3.0, mic_unit="uM", mic_target_species="E. coli")
        deduped = dedup_records([a, b, c]).records
        assert all(r.label_quality_flag == LabelQualityFlag.HIGH_REPLICATE_DISPERSION for r in deduped)

    def test_different_species_not_pooled_into_same_dispersion_group(self):
        a = _record(Source.DBAASP, "1", mic_value=2.0, mic_unit="uM", mic_target_species="E. coli")
        b = _record(Source.DRAMP, "2", mic_value=200.0, mic_unit="uM", mic_target_species="S. aureus")
        deduped = dedup_records([a, b]).records
        assert all(r.label_quality_flag is None for r in deduped)


class TestOverlapMatrix:
    def test_overlap_matrix_counts_shared_peptide_uids_across_sources(self):
        a = _record(Source.DBAASP, "1")
        b = _record(Source.DRAMP, "2")
        c = _record(Source.QMAP, "3", sequence="ACDEFGHIK")
        result = dedup_records([a, b, c])
        assert result.overlap_matrix == {(Source.DBAASP, Source.DRAMP): 1}

    def test_three_way_group_produces_all_three_pairwise_entries(self):
        a = _record(Source.DBAASP, "1")
        b = _record(Source.DRAMP, "2")
        c = _record(Source.QMAP, "3")  # same default sequence -> same group as a, b
        result = dedup_records([a, b, c])
        assert result.overlap_matrix == {
            (Source.DBAASP, Source.DRAMP): 1,
            (Source.DBAASP, Source.QMAP): 1,
            (Source.DRAMP, Source.QMAP): 1,
        }

    def test_counts_accumulate_across_multiple_distinct_groups(self):
        """A defaultdict(int) that got reset per-group instead of shared
        across the whole function would report 1 here instead of 2 — this
        was previously untested since the only prior test used a single
        group."""
        group1_a = _record(Source.DBAASP, "1", sequence="KWKLFKKIEK")
        group1_b = _record(Source.DRAMP, "2", sequence="KWKLFKKIEK")
        group2_a = _record(Source.DBAASP, "3", sequence="ACDEFGHIK")
        group2_b = _record(Source.DRAMP, "4", sequence="ACDEFGHIK")
        result = dedup_records([group1_a, group1_b, group2_a, group2_b])
        assert result.unique_peptide_uid_count == 2
        assert result.overlap_matrix == {(Source.DBAASP, Source.DRAMP): 2}
