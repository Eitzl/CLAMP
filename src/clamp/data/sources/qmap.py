"""QMAP bulk downloader.

Verified in MD_design_docs/06_task5_data_pipeline_plan.md §3: pre-cleaned,
homology-split-ready MIC+HC50 dataset, `pip install qmap-benchmark`,
github.com/anthol42/QMAP. Not a fresh source of new peptides — pools/cleans
existing DBs — but valuable as (a) an extra pre-cleaned slice and (b) the
external benchmark to report against later (doc 02 §5.4/§5.5).
"""

from pathlib import Path

from clamp.data.schema import PeptideRecord, Source
from clamp.data.sources.base import BulkDownloader


class QmapDownloader(BulkDownloader):
    source = Source.QMAP

    def download_urls(self) -> list[str]:
        # Likely satisfied via the qmap-benchmark package's own data
        # loader rather than a raw URL fetch — confirm against the package
        # API before implementing; may not fit the BulkDownloader url-list
        # shape exactly.
        raise NotImplementedError

    def parse(self, downloaded_paths: list[Path]) -> list[PeptideRecord]:
        raise NotImplementedError
