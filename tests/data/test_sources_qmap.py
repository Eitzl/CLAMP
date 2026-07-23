"""QMAP parser tests against real captured DBAASPDataset samples
(tests/data/fixtures/qmap/), confirmed live 2026-07-22 against the
qmap-benchmark package's actual API. See sources/qmap.py's module
docstring for why pull()/parse() don't use the plain URL-list shape.
"""

import json
import math


from clamp.data.schema import CyclizationType, Source
from clamp.data.sources.qmap import QmapDownloader, _infer_cyclization, _sample_to_records, load_cached_records


class _FakeBond:
    def __init__(self, src, dst, bond_type):
        self.src = src
        self.dst = dst
        self.bond_type = bond_type


class _FakeConsensus:
    """Minimal stand-in for QMAP's Target / HemolyticActivity objects —
    _sample_to_records only ever reads `.consensus`."""

    def __init__(self, consensus):
        self.consensus = consensus


class _FakeSample:
    def __init__(self, id, sequence, targets, hc50=None, smiles=None, nterminal=None, cterminal=None, bonds=None):
        self.id = id
        self.sequence = sequence
        self.targets = targets
        self.hc50 = hc50
        self.smiles = smiles or []
        self.nterminal = nterminal
        self.cterminal = cterminal
        self.bonds = bonds or []


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


class TestNaNConsensusHandling:
    """Regression coverage: QMAP's aggregated `consensus` can be NaN, and
    the MIC branch must handle it the same way the HC50 branch already does
    (omit the row), rather than emitting a phantom label-less MIC row."""

    def test_nan_target_consensus_yields_no_mic_row(self):
        sample = _FakeSample(
            id=1,
            sequence="KWKLFKKIEK",
            targets={"Escherichia coli": _FakeConsensus(math.nan)},
            hc50=None,
        )
        records = _sample_to_records(sample)
        # No real MIC value and no HC50 -> the only record should be the
        # label-less fallback row, never a MIC row carrying a NaN value.
        assert all(r.mic_value is None for r in records)
        assert all(r.mic_target_species is None for r in records)

    def test_nan_consensus_target_skipped_but_valid_sibling_target_kept(self):
        sample = _FakeSample(
            id=2,
            sequence="KWKLFKKIEK",
            targets={
                "Escherichia coli": _FakeConsensus(math.nan),
                "Staphylococcus aureus": _FakeConsensus(4.0),
            },
        )
        records = _sample_to_records(sample)
        mic_rows = [r for r in records if r.mic_value is not None]
        assert len(mic_rows) == 1
        assert mic_rows[0].mic_target_species == "Staphylococcus aureus"
        assert mic_rows[0].mic_value == 4.0

    def test_valid_consensus_still_produces_mic_row(self):
        sample = _FakeSample(
            id=3,
            sequence="KWKLFKKIEK",
            targets={"Escherichia coli": _FakeConsensus(2.0)},
        )
        records = _sample_to_records(sample)
        mic_rows = [r for r in records if r.mic_value is not None]
        assert len(mic_rows) == 1
        assert mic_rows[0].mic_value == 2.0
        assert mic_rows[0].mic_unit == "uM"


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
