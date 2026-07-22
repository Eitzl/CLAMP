"""Intended coverage per MD_design_docs/09_phase1_implementation_design.md
§12. Uses `responses` to mock DBAASP's /peptides and /peptides/{id}
endpoints against captured fixture JSON — never hits the live API in
tests. Skipped until src/clamp/data/sources/dbaasp.py is implemented and
tests/data/fixtures/dbaasp/ is populated with real captured responses.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="dbaasp.py is a NotImplementedError stub and fixtures/dbaasp/ is empty — see doc 09 §12"
)


class TestPagination:
    def test_iter_ids_pages_through_full_result_set(self):
        pass

    def test_iter_ids_stops_at_total_count(self):
        pass


class TestResume:
    def test_already_cached_id_is_skipped(self):
        pass

    def test_missing_id_is_fetched(self):
        pass


class TestBackoff:
    def test_429_triggers_backoff_not_immediate_retry(self):
        """doc 06 §7.1: no documented rate limit — treat 429 as a signal to
        back off harder, not just retry on a fixed schedule."""

    def test_5xx_is_retried(self):
        pass


class TestToRecords:
    def test_multimer_unnests_monomers(self):
        """doc 06 §1.2: complexity="Multimer" records have an empty
        top-level sequence — real chains are in monomers[]. ~10% of
        records in the doc 06 pilot pull."""

    def test_curated_smiles_extracted_when_present(self):
        pass

    def test_intrachain_bond_maps_to_cyclization_type(self):
        pass

    def test_activity_arrays_explode_into_separate_rows(self):
        """One PeptideRecord per (targetActivity | hemoliticCytotoxicActivity)
        entry, all sharing the same peptide_uid once dedup runs."""
