"""QMAP parser tests against real captured DBAASPDataset samples
(tests/data/fixtures/qmap/), confirmed live 2026-07-22 against the
qmap-benchmark package's actual API. See sources/qmap.py's module
docstring for why pull()/parse() don't use the plain URL-list shape.
"""

import json


from clamp.data.schema import CyclizationType, Source
from clamp.data.sources.qmap import QmapDownloader, _infer_cyclization, load_cached_records


class _FakeBond:
    def __init__(self, src, dst, bond_type):
        self.src = src
        self.dst = dst
        self.bond_type = bond_type


class TestInferCyclization:
    def test_disulfide_bond_maps_to_disulfide(self):
        bonds = [_FakeBond(3, 34, "DSB")]
        assert _infer_cyclization(bonds, seq_len=44) == CyclizationType.DISULFIDE

    def test_terminal_amd_bond_maps_to_head_to_tail(self):
        """Confirmed against real fixture sample id=105: sequence length 8,
        bond (1, 8) — 1-indexed positions, not 0-indexed (dst=8 would be
        out of range for a 0-indexed length-8 sequence)."""
        bonds = [_FakeBond(1, 8, "AMD")]
        assert _infer_cyclization(bonds, seq_len=8) == CyclizationType.HEAD_TO_TAIL

    def test_non_terminal_amd_bond_is_unknown(self):
        bonds = [_FakeBond(2, 5, "AMD")]
        assert _infer_cyclization(bonds, seq_len=10) == CyclizationType.UNKNOWN

    def test_no_bonds_is_unreachable_here_but_disulfide_takes_priority(self):
        bonds = [_FakeBond(1, 10, "AMD"), _FakeBond(3, 7, "DSB")]
        assert _infer_cyclization(bonds, seq_len=10) == CyclizationType.DISULFIDE


class TestParseFixtures:
    """Exercises the real Sample-shaped dicts via the package's own
    Sample.FromDict, not a hand-rolled parser — parse() itself just reads
    the already-converted PeptideRecord jsonl that pull() writes, so these
    tests go through _sample_to_records via the qmap package classes
    directly to validate the mapping against real captured shapes."""

    def _load_samples(self, fixtures_dir):
        from qmap.benchmark.dataset.sample import Sample

        raw = json.loads((fixtures_dir / "qmap" / "sample_dbaasp_records.json").read_text())
        return [Sample.FromDict(d) for d in raw]

    def test_disulfide_sample_maps_correctly(self, fixtures_dir):
        from clamp.data.sources.qmap import _sample_to_records

        samples = self._load_samples(fixtures_dir)
        disulfide_sample = next(s for s in samples if any(b.bond_type == "DSB" for b in s.bonds))
        records = _sample_to_records(disulfide_sample)
        assert all(r.is_cyclic for r in records)
        assert all(r.cyclization_type == CyclizationType.DISULFIDE for r in records)
        assert all(r.source == Source.QMAP for r in records)

    def test_head_to_tail_sample_maps_correctly(self, fixtures_dir):
        from clamp.data.sources.qmap import _sample_to_records

        samples = self._load_samples(fixtures_dir)
        ht_sample = next(
            s
            for s in samples
            if any(b.bond_type == "AMD" and {b.src, b.dst} == {1, len(s.sequence)} for b in s.bonds)
        )
        records = _sample_to_records(ht_sample)
        assert all(r.cyclization_type == CyclizationType.HEAD_TO_TAIL for r in records)

    def test_plain_sample_with_hc50_yields_hc50_row(self, fixtures_dir):
        from clamp.data.sources.qmap import _sample_to_records

        samples = self._load_samples(fixtures_dir)
        plain_sample = next(s for s in samples if not s.bonds)
        records = _sample_to_records(plain_sample)
        hc50_rows = [r for r in records if r.hc50_value is not None]
        assert hc50_rows
        assert hc50_rows[0].hc50_unit == "uM"
        assert all(not r.is_cyclic for r in records)


class TestParseAndLoadCachedRecords:
    def test_parse_reads_jsonl_written_by_pull_shape(self, fixtures_dir, tmp_path):
        from clamp.data.schema import PeptideRecord

        record = PeptideRecord(source=Source.QMAP, source_id="1:hc50", sequence_raw="ACDEFGHIK", hc50_value=10.0)
        cache_dir = tmp_path / "qmap"
        cache_dir.mkdir()
        (cache_dir / "qmap.jsonl").write_text(record.model_dump_json())
        records = QmapDownloader().parse([cache_dir / "qmap.jsonl"])
        assert len(records) == 1
        assert records[0].hc50_value == 10.0

    def test_load_cached_records_missing_file_returns_empty(self, tmp_path):
        assert load_cached_records(tmp_path / "missing") == []
