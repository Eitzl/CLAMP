"""Intended coverage per MD_design_docs/09_phase1_implementation_design.md
§12 and data/README.md §3. Skipped until src/clamp/data/dedup.py is
implemented.
"""

import pytest

pytestmark = pytest.mark.skip(reason="dedup.py is a NotImplementedError stub — see doc 09 §12 / data/README.md §3")


class TestPeptideUid:
    def test_same_sequence_and_mods_same_uid(self):
        pass

    def test_different_cyclization_type_different_uid(self):
        pass

    def test_uid_is_stable_across_runs(self):
        """Not derived from anything nondeterministic (e.g. dict ordering) —
        rerunning the pipeline must reproduce identical peptide_uids."""


class TestDedupMerge:
    def test_metadata_rich_record_wins_smiles_on_fuzzy_match(self):
        """data/README.md §3.4: DBAASP (full mod annotation) + APD3
        (sequence-only) for the same peptide -> DBAASP's smiles/
        fidelity_tier is kept, not overwritten by the poorer record."""

    def test_replicate_measurements_kept_as_separate_rows(self):
        """data/README.md §3.2: same peptide_uid, same assay conditions,
        different sources, different values -> both rows survive, neither
        averaged nor dropped."""

    def test_complementary_labels_coexist_without_merging(self):
        """data/README.md §3.3: one source has HC50 only, another has MIC
        only, for the same peptide_uid -> both rows present after dedup."""


class TestConflictDetection:
    """data/README.md §3.5 / dedup.flag_high_dispersion_replicates."""

    def test_order_of_magnitude_spread_flags_high_dispersion(self):
        pass

    def test_close_replicate_values_not_flagged(self):
        pass

    def test_flag_applies_to_all_rows_in_the_conflicting_group(self):
        pass


class TestOverlapMatrix:
    def test_overlap_matrix_counts_shared_peptide_uids_across_sources(self):
        pass
