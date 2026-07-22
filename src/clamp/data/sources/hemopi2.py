"""HemoPI2 bulk downloader.

Verified in MD_design_docs/06_task5_data_pipeline_plan.md §3: 1,926
experimentally validated hemolytic peptides with regression-ready HC50
values, github.com/raghavagps/HemoPI2 / pip package `hemopi2`. Smaller than
Hemolytik2 — used mainly for clean regression framing / benchmark
comparability, not raw volume.
"""

from pathlib import Path

from clamp.data.schema import PeptideRecord, Source
from clamp.data.sources.base import BulkDownloader


class HemoPI2Downloader(BulkDownloader):
    source = Source.HEMOPI2

    def download_urls(self) -> list[str]:
        raise NotImplementedError

    def parse(self, downloaded_paths: list[Path]) -> list[PeptideRecord]:
        raise NotImplementedError
