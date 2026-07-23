"""DRAMP parser tests against real captured samples
(tests/data/fixtures/dramp/), confirmed live 2026-07-22 — see
sources/dramp.py's module docstring for the real download-path finding.
"""

from clamp.data.schema import Source
from clamp.data.sources.dramp import DrampDownloader, load_cached_records


class TestParse:
    def test_parses_rows_with_sequence(self, fixtures_dir):
        amps_path = fixtures_dir / "dramp" / "general_amps_sample.txt"
        records = DrampDownloader().parse([amps_path])
        assert len(records) == 4
        assert all(r.source == Source.DRAMP for r in records)
        assert all(r.sequence_raw for r in records)

    def test_joins_smiles_from_separate_smiles_file(self, fixtures_dir):
        amps_path = fixtures_dir / "dramp" / "general_amps_sample.txt"
        smiles_path = fixtures_dir / "dramp" / "general_smiles_sample.txt"
        records = DrampDownloader().parse([amps_path, smiles_path])
        by_id = {r.source_id: r for r in records}
        # DRAMP00068 is present in both fixture files — its SMILES should
        # come through via the join, even though general_amps_sample.txt
        # is not guaranteed to carry it inline for every row.
        assert by_id["DRAMP00068"].smiles is not None
        assert by_id["DRAMP00068"].smiles.startswith("CC")

    def test_row_without_smiles_in_either_file_has_none(self, fixtures_dir):
        amps_path = fixtures_dir / "dramp" / "general_amps_sample.txt"
        smiles_path = fixtures_dir / "dramp" / "general_smiles_sample.txt"
        records = DrampDownloader().parse([amps_path, smiles_path])
        by_id = {r.source_id: r for r in records}
        assert by_id["DRAMP00017"].smiles is None

    def test_cyclic_free_text_column_sets_is_cyclic(self, fixtures_dir):
        amps_path = fixtures_dir / "dramp" / "general_amps_sample.txt"
        records = DrampDownloader().parse([amps_path])
        by_id = {r.source_id: r for r in records}
        assert by_id["DRAMP00017"].is_cyclic is True
        assert by_id["DRAMP00068"].is_cyclic is False

    def test_branched_topology_is_not_silently_treated_as_linear(self, fixtures_dir):
        """Regression guard for a real bug: `"cyclic" in value.lower()`
        alone would match neither "Branched" nor "Linear", silently
        collapsing a genuinely branched peptide (real fixture row
        DRAMP20865, confirmed live) into is_cyclic=False — indistinguishable
        from an actually-linear peptide, with no signal anything was lost.
        Branched peptides now route the same way cyclic ones do (is_cyclic
        =True, cyclization_type=UNKNOWN), which correctly downgrades their
        fidelity tier downstream instead of falsely claiming FULL."""
        amps_path = fixtures_dir / "dramp" / "general_amps_sample.txt"
        records = DrampDownloader().parse([amps_path])
        by_id = {r.source_id: r for r in records}
        assert by_id["DRAMP20865"].is_cyclic is True
        from clamp.data.schema import CyclizationType

        assert by_id["DRAMP20865"].cyclization_type == CyclizationType.UNKNOWN

    def test_free_terminal_modification_maps_to_none(self, fixtures_dir):
        amps_path = fixtures_dir / "dramp" / "general_amps_sample.txt"
        records = DrampDownloader().parse([amps_path])
        by_id = {r.source_id: r for r in records}
        # DRAMP00017's N-terminal_Modification is "Free" in the real data
        assert by_id["DRAMP00017"].nterm_mod is None


class TestLoadCachedRecords:
    def test_reads_from_cache_dir(self, fixtures_dir, tmp_path):
        cache_dir = tmp_path / "dramp"
        cache_dir.mkdir()
        for name in ["general_amps_sample.txt", "general_smiles_sample.txt"]:
            (cache_dir / name).write_bytes((fixtures_dir / "dramp" / name).read_bytes())
        records = load_cached_records(cache_dir)
        assert len(records) == 4

    def test_missing_dir_returns_empty(self, tmp_path):
        assert load_cached_records(tmp_path / "missing") == []
