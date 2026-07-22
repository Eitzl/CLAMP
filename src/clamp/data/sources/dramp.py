"""DRAMP 4.0 bulk downloader.

Bulk files verified in MD_design_docs/06_task5_data_pipeline_plan.md §3:
categorized XLSX/TXT/FASTA files at dramp.cpu-bioinfor.org, including
dedicated `*_smiles.xlsx/.txt/.fasta` files — a direct win over DBAASP for
the subset of peptides DRAMP covers, since curated SMILES ship in the bulk
download rather than requiring per-record conversion.
"""

from pathlib import Path

from clamp.data.schema import PeptideRecord, Source
from clamp.data.sources.base import BulkDownloader


class DrampDownloader(BulkDownloader):
    source = Source.DRAMP

    def download_urls(self) -> list[str]:
        # general_amps.xlsx (or .txt/.fasta) + matching *_smiles file +
        # hemolytic/cytotoxicity-activity-labeled subset files, per doc 06 §3.
        raise NotImplementedError

    def parse(self, downloaded_paths: list[Path]) -> list[PeptideRecord]:
        raise NotImplementedError
