"""HemoPI2 parser tests against real captured Dataset/ CSVs
(tests/data/fixtures/hemopi2/), confirmed live 2026-07-22. See
sources/hemopi2.py's module docstring for why the pip package isn't used.
"""

from clamp.data.schema import Source
from clamp.data.sources.hemopi2 import HemoPI2Downloader, load_cached_records


class TestParse:
    def test_parses_both_fixture_files(self, fixtures_dir):
        paths = [
            fixtures_dir / "hemopi2" / "cross_val_dataset.csv",
            fixtures_dir / "hemopi2" / "independent_dataset.csv",
        ]
        records = HemoPI2Downloader().parse(paths)
        assert len(records) == 5 + 3
        assert all(r.source == Source.HEMOPI2 for r in records)

    def test_hc50_value_and_unit_populated(self, fixtures_dir):
        path = fixtures_dir / "hemopi2" / "cross_val_dataset.csv"
        records = HemoPI2Downloader().parse([path])
        assert records[0].hc50_value == 596.7
        assert records[0].hc50_unit == "uM"
        assert records[0].sequence_raw == "GLPALISWIKRKRL"

    def test_hc50_assay_target_cell_populated(self, fixtures_dir):
        """Found missing via a real pilot pull: this field was silently
        None for every HemoPI2 record before this fix. Hedged to
        "mammalian" rather than "human" since the repo's own README only
        confirms RBC lysis generically, not the species."""
        path = fixtures_dir / "hemopi2" / "cross_val_dataset.csv"
        records = HemoPI2Downloader().parse([path])
        assert all(r.hc50_assay_target_cell is not None for r in records)
        assert all("erythrocyte" in r.hc50_assay_target_cell for r in records)
        assert all("human" not in r.hc50_assay_target_cell.lower() for r in records)

    def test_greek_mu_column_is_read_correctly(self, fixtures_dir):
        """Regression guard: the real column header uses GREEK SMALL
        LETTER MU (U+03BC), not the MICRO SIGN — a mismatch here would
        silently produce hc50_value=None for every row instead of raising."""
        path = fixtures_dir / "hemopi2" / "cross_val_dataset.csv"
        records = HemoPI2Downloader().parse([path])
        assert all(r.hc50_value is not None for r in records)


class TestLoadCachedRecords:
    def test_reads_all_csvs_in_dir(self, fixtures_dir, tmp_path):
        cache_dir = tmp_path / "hemopi2"
        cache_dir.mkdir()
        for name in ["cross_val_dataset.csv", "independent_dataset.csv"]:
            (cache_dir / name).write_bytes((fixtures_dir / "hemopi2" / name).read_bytes())
        records = load_cached_records(cache_dir)
        assert len(records) == 8

    def test_missing_dir_returns_empty(self, tmp_path):
        assert load_cached_records(tmp_path / "missing") == []
