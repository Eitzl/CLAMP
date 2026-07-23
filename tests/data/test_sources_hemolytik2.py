"""Hemolytik2 parser tests against a real captured API response
(tests/data/fixtures/hemolytik2/sample_api_response.json), confirmed live
2026-07-22 — see sources/hemolytik2.py's module docstring for the real
endpoint/coverage findings.
"""

import json

import pytest

from clamp.data.schema import CyclizationType, Source
from clamp.data.sources.hemolytik2 import Hemolytik2Downloader, _parse_activity, load_cached_records


def _cached_rows_path(fixtures_dir, tmp_path):
    """parse() expects a flat list at a file literally named
    "hemolytik2.json" — that's the post-unwrap shape pull() itself writes
    (payload["data"], not the raw {"status","count","data"} API envelope
    the fixture captures for realism). Reproduce that shape here rather
    than changing parse()'s contract to match the fixture."""
    payload = json.loads((fixtures_dir / "hemolytik2" / "sample_api_response.json").read_text())
    path = tmp_path / "hemolytik2.json"
    path.write_text(json.dumps(payload["data"]))
    return path


class TestParseActivity:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("HC50 =6μM", (6.0, "μM")),
            ("MHC =81μM", (81.0, "μM")),
            ("HD50 = 8.7μM ", (8.7, "μM")),
            ("MHC  <<81μM", (81.0, "μM")),
            ("LC50  >100μM", (100.0, "μM")),
            ("HC50 >200μg/ml", (200.0, "μg/ml")),
            ("LC50 =2.3±0.3μM", (2.3, "μM")),
            ("MHC =31.3μg/ml", (31.3, "μg/ml")),
            # MICRO SIGN (U+00B5), not GREEK SMALL LETTER MU (U+03BC) — the
            # regex documents supporting both, but every other case here
            # (and every real fixture row) happens to use only U+03BC, so
            # this branch was previously dead as far as tests are concerned.
            ("HC50 =6µM", (6.0, "µM")),
            ("MHC =31.3µg/ml", (31.3, "µg/ml")),
        ],
    )
    def test_extracts_point_value_from_labeled_forms(self, text, expected):
        assert _parse_activity(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "77% hemolysis at 100 μM ",
            ">2-3% at ≥20μM",
            "0% hemolysis at 3.13-25 μM",
            "0% hemolysis at 100μM (non-hemolytic)",
            None,
            "",
        ],
    )
    def test_percentage_and_dose_forms_return_none(self, text):
        assert _parse_activity(text) is None

    @pytest.mark.parametrize(
        "text",
        [
            "Reference =6μM",
            "PMID =100μM",
            "Note =6μM",
            "pH =6μM",
            "N =6μM",
        ],
    )
    def test_unrelated_labels_are_rejected_not_silently_accepted(self, text):
        """Regression guard for a real gap the earlier version of this
        regex had: the label group used to be a bare `[A-Za-z]+\\d*` that
        accepted *any* word immediately adjacent to "=value+unit", so
        free-text fields shaped exactly like this (reference tags, pH
        notes) would silently parse as real HC50 measurements. The label
        group is now a whitelist of the actual assay-metric abbreviations
        observed live (HC50/MHC/HD50/LD50/LC50/EC50)."""
        assert _parse_activity(text) is None


class TestParse:
    def test_parses_all_fixture_rows(self, fixtures_dir, tmp_path):
        path = _cached_rows_path(fixtures_dir, tmp_path)
        records = Hemolytik2Downloader().parse([path])
        assert len(records) == 4
        assert all(r.source == Source.HEMOLYTIK2 for r in records)

    def test_cyclic_row_is_flagged_cyclic_with_unknown_type(self, fixtures_dir, tmp_path):
        path = _cached_rows_path(fixtures_dir, tmp_path)
        records = Hemolytik2Downloader().parse([path])
        by_id = {r.source_id: r for r in records}
        assert by_id["1025"].is_cyclic is True
        assert by_id["1025"].cyclization_type == CyclizationType.UNKNOWN
        # "HD50 =11.7μM " on this row should still extract a point value
        assert by_id["1025"].hc50_value == pytest.approx(11.7)

    def test_clean_hc50_row_extracts_value(self, fixtures_dir, tmp_path):
        path = _cached_rows_path(fixtures_dir, tmp_path)
        records = Hemolytik2Downloader().parse([path])
        by_id = {r.source_id: r for r in records}
        assert by_id["1075"].hc50_value == pytest.approx(100.0)
        assert by_id["1075"].hc50_unit == "μM"

    def test_percentage_only_row_has_no_hc50_value(self, fixtures_dir, tmp_path):
        path = _cached_rows_path(fixtures_dir, tmp_path)
        records = Hemolytik2Downloader().parse([path])
        by_id = {r.source_id: r for r in records}
        assert by_id["1022"].hc50_value is None


class TestLoadCachedRecords:
    def test_reads_from_cache_dir(self, fixtures_dir, tmp_path):
        cache_dir = tmp_path / "hemolytik2"
        cache_dir.mkdir()
        payload = json.loads((fixtures_dir / "hemolytik2" / "sample_api_response.json").read_text())
        (cache_dir / "hemolytik2.json").write_text(json.dumps(payload["data"]))
        records = load_cached_records(cache_dir)
        assert len(records) == 4

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_cached_records(tmp_path / "missing") == []
