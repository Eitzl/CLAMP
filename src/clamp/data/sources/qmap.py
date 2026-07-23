"""QMAP downloader.

Confirmed live 2026-07-22: `qmap-benchmark`'s real integration point is its
own Python API, not a raw bulk-file URL, exactly as doc 06/09 flagged as
needing confirmation before implementing. `qmap.benchmark.dataset.dataset.
DBAASPDataset()` downloads a cleaned, pre-aggregated DBAASP-derived JSON
from HuggingFace Hub (`anthol42/qmap_benchmark_2025`, file `dbaasp.json`)
via `huggingface_hub.hf_hub_download`. Live-tested: 18,033 samples.

Confirmed shapes from the installed package source:
  Sample: {id, sequence, smiles: list[str], nterminal: None|"ACT",
           cterminal: None|"AMD", bonds: list[Bond],
           targets: dict[species_name, Target], hc50: HemolyticActivity}
  Bond: {src: int, dst: int, bond_type: "DSB"|"AMD"}   (0-indexed positions)
  Target: {name, min_activity, max_activity, consensus}
  HemolyticActivity: {min_hc50, max_hc50, consensus}   (consensus may be NaN)

Important discrepancy to carry into the datasheet: QMAP's `consensus`
values are already-aggregated across whatever replicate measurements
DBAASP had for that (peptide, species) pair — unlike our own
replicate-preserving convention (data/README.md §3.2), so a QMAP row and a
directly-pulled DBAASP row for the "same" measurement will not necessarily
match numerically even though both are DBAASP-derived.

Because `download_urls()`/`parse()` don't fit this access pattern,
`pull()` is overridden directly rather than implementing a URL list.
"""

import math
from pathlib import Path

from clamp.data.schema import CyclizationType, PeptideRecord, Source
from clamp.data.sources.base import BulkDownloader, PullReport


def _infer_cyclization(bonds, seq_len: int) -> CyclizationType:
    """QMAP's own bond taxonomy (bond_type in {DSB, AMD}) is coarser than
    our 5-type enum: DSB maps cleanly to DISULFIDE, but AMD covers both
    head-to-tail and side-chain-to-terminus cyclizations without
    distinguishing them. Heuristic: an AMD bond spanning the first and
    last residue is almost certainly head-to-tail; anything else is
    UNKNOWN rather than guessing which side-chain subtype it is.

    Bond.src/dst are 1-indexed, not 0-indexed — confirmed against a real
    captured sample (id 105, sequence length 8, bond (1, 8)); dst=8 would
    be out of range for a 0-indexed length-8 sequence, so the boundary
    check below is {1, seq_len}, not {0, seq_len - 1}."""
    for bond in bonds:
        if bond.bond_type == "DSB":
            return CyclizationType.DISULFIDE
    for bond in bonds:
        if bond.bond_type == "AMD":
            if {bond.src, bond.dst} == {1, seq_len}:
                return CyclizationType.HEAD_TO_TAIL
            return CyclizationType.UNKNOWN
    return CyclizationType.UNKNOWN


def _sample_to_records(sample) -> list[PeptideRecord]:
    is_cyclic = bool(sample.bonds)
    cyclization_type = _infer_cyclization(sample.bonds, len(sample.sequence)) if is_cyclic else CyclizationType.NONE
    base_kwargs = dict(
        source=Source.QMAP,
        sequence_raw=sample.sequence,
        is_cyclic=is_cyclic,
        cyclization_type=cyclization_type,
        nterm_mod=sample.nterminal,
        cterm_mod=sample.cterminal,
        smiles=sample.smiles[0] if sample.smiles else None,
    )

    records: list[PeptideRecord] = []
    for target_name, target in sample.targets.items():
        records.append(
            PeptideRecord(
                **base_kwargs,
                source_id=f"{sample.id}:mic:{target_name}",
                mic_value=target.consensus,
                mic_unit="uM",
                mic_target_species=target_name,
            )
        )
    if sample.hc50 is not None and not math.isnan(sample.hc50.consensus):
        records.append(
            PeptideRecord(
                **base_kwargs,
                source_id=f"{sample.id}:hc50",
                hc50_value=sample.hc50.consensus,
                hc50_unit="uM",
            )
        )
    if not records:
        records.append(PeptideRecord(**base_kwargs, source_id=str(sample.id)))
    return records


class QmapDownloader(BulkDownloader):
    source = Source.QMAP

    def download_urls(self) -> list[str]:
        return []  # see module docstring — real fetch happens inside pull()

    def pull(self, cache_dir: Path) -> PullReport:
        from qmap.benchmark.dataset.dataset import DBAASPDataset

        cache_dir.mkdir(parents=True, exist_ok=True)
        dataset = DBAASPDataset()
        records = [record for sample in dataset.samples for record in _sample_to_records(sample)]
        (cache_dir / "qmap.jsonl").write_text("\n".join(r.model_dump_json() for r in records))
        return PullReport(
            source=self.source,
            attempted=len(dataset),
            succeeded=len(dataset),
            skipped_cached=0,
            failed=[],
        )

    def parse(self, downloaded_paths: list[Path]) -> list[PeptideRecord]:
        path = next((p for p in downloaded_paths if p.name == "qmap.jsonl"), None)
        if path is None:
            return []
        text = path.read_text().strip()
        if not text:
            return []
        return [PeptideRecord.model_validate_json(line) for line in text.splitlines()]


def load_cached_records(cache_dir: Path) -> list[PeptideRecord]:
    path = cache_dir / "qmap.jsonl"
    if not path.exists():
        return []
    return QmapDownloader().parse([path])
