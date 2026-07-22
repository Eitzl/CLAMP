"""Hemolytik2 bulk downloader (+ REST API).

Verified in MD_design_docs/06_task5_data_pipeline_plan.md §3: 13,215
entries / ~8,700 unique peptides, dedicated hemolysis DB with SMILES,
terminal modifications, D-/L-stereochemistry, and a linear-vs-cyclic flag.
Treated as co-primary (alongside DBAASP) for HC50 given its metadata
richness, not a minor supplement.
"""

from pathlib import Path

from clamp.data.schema import PeptideRecord, Source
from clamp.data.sources.base import BulkDownloader


class Hemolytik2Downloader(BulkDownloader):
    source = Source.HEMOLYTIK2

    def download_urls(self) -> list[str]:
        raise NotImplementedError

    def parse(self, downloaded_paths: list[Path]) -> list[PeptideRecord]:
        raise NotImplementedError
