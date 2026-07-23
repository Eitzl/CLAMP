"""Coverage per MD_design_docs/09_phase1_implementation_design.md §12.
Uses `responses` to mock DBAASP's /peptides and /peptides/{id} endpoints —
never hits the live API in tests. Fixtures under tests/data/fixtures/dbaasp/
are real captured responses (2026-07-22), not hand-invented shapes.
"""

import json

import pytest
import responses

from clamp.data.schema import CyclizationType, Source
from clamp.data.sources.dbaasp import DbaaspPuller, _parse_concentration, load_cached_records


class TestParseConcentration:
    """Regression coverage for a real bug found via a ~90k-row pilot pull
    (not the small hand-picked fixtures elsewhere in this file): DBAASP's
    `concentration` field has (at least) three distinct non-plain-float
    conventions, and an earlier version of this function treated all of
    them as a hyphen-range, crashing normalize.py's parse_range() on ~760
    real rows that were actually ± uncertainty or inequality-bounded
    values, not ranges at all."""

    def test_plain_float_passes_through(self):
        assert _parse_concentration("12.5") == (12.5, False, None)

    def test_hyphen_range_is_tagged_as_range_not_extracted(self):
        assert _parse_concentration("1.58-25.33") == (None, True, "1.58-25.33")

    def test_uncertainty_notation_extracts_point_value_not_flagged_as_range(self):
        assert _parse_concentration("2.2±0.8") == (2.2, False, None)

    @pytest.mark.parametrize("text,expected", [(">400", 400.0), ("<0.5", 0.5), ("≥10", 10.0), ("<<81", 81.0)])
    def test_inequality_bounded_extracts_point_value_not_flagged_as_range(self, text, expected):
        assert _parse_concentration(text) == (expected, False, None)

    def test_none_returns_none_not_a_range(self):
        assert _parse_concentration(None) == (None, False, None)

    def test_genuinely_unparseable_garbage_returns_none_not_a_broken_range(self):
        """Previously, "garbage" (not float, no hyphen) would have been
        tagged is_range=True with range_raw="garbage" — passed downstream
        to normalize.py's parse_range(), which would crash on it. Returning
        (None, False, None) means the row simply has no usable value."""
        assert _parse_concentration("garbage") == (None, False, None)


def _load_fixture(fixtures_dir, name):
    return json.loads((fixtures_dir / "dbaasp" / name).read_text())


class TestPagination:
    @responses.activate
    def test_iter_ids_pages_through_full_result_set(self):
        puller = DbaaspPuller(base_url="https://dbaasp.test", page_size=2)
        responses.add(
            responses.GET,
            "https://dbaasp.test/peptides",
            json={"totalCount": 4, "data": [{"id": 1}, {"id": 2}]},
            match=[responses.matchers.query_param_matcher({"limit": "2", "offset": "0"})],
        )
        responses.add(
            responses.GET,
            "https://dbaasp.test/peptides",
            json={"totalCount": 4, "data": [{"id": 3}, {"id": 4}]},
            match=[responses.matchers.query_param_matcher({"limit": "2", "offset": "2"})],
        )
        assert list(puller.iter_ids()) == ["1", "2", "3", "4"]

    @responses.activate
    def test_iter_ids_stops_at_total_count(self):
        puller = DbaaspPuller(base_url="https://dbaasp.test", page_size=10)
        responses.add(
            responses.GET,
            "https://dbaasp.test/peptides",
            json={"totalCount": 2, "data": [{"id": 1}, {"id": 2}]},
        )
        assert list(puller.iter_ids()) == ["1", "2"]

    @responses.activate
    def test_iter_ids_stops_on_empty_page(self):
        puller = DbaaspPuller(base_url="https://dbaasp.test", page_size=10)
        responses.add(responses.GET, "https://dbaasp.test/peptides", json={"totalCount": 100, "data": []})
        assert list(puller.iter_ids()) == []


class TestResume:
    @responses.activate
    def test_already_cached_id_is_skipped(self, tmp_path):
        cache_dir = tmp_path / "dbaasp"
        cache_dir.mkdir()
        (cache_dir / "1.json").write_text('{"id": 1}')
        puller = DbaaspPuller(base_url="https://dbaasp.test", requests_per_second=1000.0)
        puller.iter_ids = lambda: iter(["1"])
        report = puller.pull(cache_dir)
        assert report.skipped_cached == 1
        assert report.succeeded == 0

    @responses.activate
    def test_missing_id_is_fetched(self, tmp_path):
        cache_dir = tmp_path / "dbaasp"
        responses.add(responses.GET, "https://dbaasp.test/peptides/1", json={"id": 1, "sequence": "ACD"})
        puller = DbaaspPuller(base_url="https://dbaasp.test", requests_per_second=1000.0)
        puller.iter_ids = lambda: iter(["1"])
        report = puller.pull(cache_dir)
        assert report.succeeded == 1
        assert (cache_dir / "1.json").exists()


class TestBackoff:
    @responses.activate
    def test_429_triggers_backoff_not_immediate_retry(self, tmp_path):
        """doc 06 §7.1: no documented rate limit — a 429 gets the same
        exponential-backoff retry path as any other transient error."""
        cache_dir = tmp_path / "dbaasp"
        responses.add(responses.GET, "https://dbaasp.test/peptides/1", status=429)
        responses.add(responses.GET, "https://dbaasp.test/peptides/1", json={"id": 1, "sequence": "ACD"})
        puller = DbaaspPuller(base_url="https://dbaasp.test", requests_per_second=1000.0)
        puller.iter_ids = lambda: iter(["1"])
        report = puller.pull(cache_dir)
        assert report.succeeded == 1
        assert len(responses.calls) == 2

    @responses.activate
    def test_5xx_is_retried(self, tmp_path):
        cache_dir = tmp_path / "dbaasp"
        responses.add(responses.GET, "https://dbaasp.test/peptides/1", status=503)
        responses.add(responses.GET, "https://dbaasp.test/peptides/1", json={"id": 1, "sequence": "ACD"})
        puller = DbaaspPuller(base_url="https://dbaasp.test", requests_per_second=1000.0)
        puller.iter_ids = lambda: iter(["1"])
        report = puller.pull(cache_dir)
        assert report.succeeded == 1

    @responses.activate
    def test_persistent_failure_is_reported_not_raised(self, tmp_path):
        cache_dir = tmp_path / "dbaasp"
        for _ in range(5):
            responses.add(responses.GET, "https://dbaasp.test/peptides/1", status=500)
        puller = DbaaspPuller(base_url="https://dbaasp.test", requests_per_second=1000.0)
        puller.iter_ids = lambda: iter(["1"])
        report = puller.pull(cache_dir)
        assert report.failed == ["1"]
        assert report.succeeded == 0


class TestToRecords:
    def test_multimer_yields_no_records(self, fixtures_dir):
        """doc 06 §1.2: complexity="Multimer" records have an empty
        top-level sequence — their chains are separately, independently
        pullable ids (confirmed live), so unnesting here would double
        count. to_records must not emit a garbage sequence_raw="" row."""
        raw = _load_fixture(fixtures_dir, "1_multimer.json")
        assert DbaaspPuller().to_records(raw) == []

    def test_multi_peptide_complexity_also_yields_no_records(self):
        """A third complexity value found via a real pilot pull:
        "Multi-Peptide" (distinct peptides forming a complex) has the same
        empty-top-level-sequence shape as Multimer and was 7x more common
        in that pilot. Not special-cased in to_records — locks in that the
        generic empty-sequence guard correctly handles it too, rather than
        assuming that without a test."""
        raw = {
            "id": 17,
            "complexity": {"name": "Multi-Peptide"},
            "sequence": "",
            "monomers": [{"id": 1333, "sequence": "SIWGDIGQGVGKAAYWVGKAMGNMSDVNQASRINRKKKH"}],
            "targetActivities": [{"id": 1, "activityMeasureGroup": {"name": "MIC"}}],
            "hemoliticCytotoxicActivities": [],
            "unusualAminoAcids": [],
            "intrachainBonds": [],
            "smiles": [],
        }
        assert DbaaspPuller().to_records(raw) == []

    def test_monomer_with_mic_explodes_into_one_row_per_mic_activity(self, fixtures_dir):
        raw = _load_fixture(fixtures_dir, "16_monomer_with_mic.json")
        records = DbaaspPuller().to_records(raw)
        mic_group_count = sum(1 for a in raw["targetActivities"] if a["activityMeasureGroup"]["name"] == "MIC")
        assert len(records) == mic_group_count
        assert all(r.mic_target_species is not None for r in records)
        assert all(r.sequence_raw == raw["sequence"] for r in records)

    def test_range_valued_concentration_is_preserved_raw(self, fixtures_dir):
        raw = _load_fixture(fixtures_dir, "16_monomer_with_mic.json")
        records = DbaaspPuller().to_records(raw)
        ranged = [r for r in records if r.label_is_range]
        assert ranged
        assert all("-" in r.label_range_raw for r in ranged)
        assert all(r.mic_value is None for r in ranged)

    def test_curated_smiles_extracted_when_present(self, fixtures_dir):
        raw = _load_fixture(fixtures_dir, "16_monomer_with_mic.json")
        records = DbaaspPuller().to_records(raw)
        assert all(r.smiles == raw["smiles"][0]["smiles"] for r in records)

    def test_unusual_amino_acids_mapped_to_unusual_residues(self, fixtures_dir):
        raw = _load_fixture(fixtures_dir, "13_unusual_residues.json")
        records = DbaaspPuller().to_records(raw)
        assert records
        record = records[0]
        assert len(record.unusual_residues) == len(raw["unusualAminoAcids"])
        assert record.unusual_residues[0].modification_type == raw["unusualAminoAcids"][0]["modificationType"]["name"]
        assert record.unusual_residues[0].position == raw["unusualAminoAcids"][0]["position"]

    def test_head_to_tail_bond_maps_to_cyclization_type(self, fixtures_dir):
        raw = _load_fixture(fixtures_dir, "105_head_to_tail_bond.json")
        records = DbaaspPuller().to_records(raw)
        assert records
        assert all(r.is_cyclic for r in records)
        assert all(r.cyclization_type == CyclizationType.HEAD_TO_TAIL for r in records)

    def test_sequence_only_record_with_no_activities_yields_one_labelless_row(self):
        raw = {
            "id": 999,
            "sequence": "ACDEFGHIK",
            "complexity": {"name": "Monomer"},
            "targetActivities": [],
            "hemoliticCytotoxicActivities": [],
            "unusualAminoAcids": [],
            "intrachainBonds": [],
            "smiles": [],
        }
        records = DbaaspPuller().to_records(raw)
        assert len(records) == 1
        assert records[0].mic_value is None
        assert records[0].hc50_value is None

    def test_mbc_activity_measure_group_is_excluded_from_mic(self, fixtures_dir):
        raw = _load_fixture(fixtures_dir, "16_monomer_with_mic.json")
        assert any(a["activityMeasureGroup"]["name"] == "MBC" for a in raw["targetActivities"])
        records = DbaaspPuller().to_records(raw)
        mbc_ids = {a["id"] for a in raw["targetActivities"] if a["activityMeasureGroup"]["name"] == "MBC"}
        emitted_activity_ids = {int(r.source_id.rsplit(":", 1)[-1]) for r in records}
        assert emitted_activity_ids.isdisjoint(mbc_ids)


class TestLoadCachedRecords:
    def test_reads_every_cached_file_in_dir(self, fixtures_dir, tmp_path):
        cache_dir = tmp_path / "dbaasp"
        cache_dir.mkdir()
        for name in ["16_monomer_with_mic.json", "1_multimer.json"]:
            (cache_dir / name.replace("_monomer_with_mic", "").replace("_multimer", "")).write_text(
                (fixtures_dir / "dbaasp" / name).read_text()
            )
        records = load_cached_records(cache_dir)
        # the multimer file contributes zero records, the monomer file
        # contributes one per MIC activity
        assert len(records) > 0
        assert all(r.source == Source.DBAASP for r in records)

    def test_missing_cache_dir_returns_empty(self, tmp_path):
        assert load_cached_records(tmp_path / "does_not_exist") == []
