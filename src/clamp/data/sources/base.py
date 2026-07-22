"""Puller base classes.

See MD_design_docs/09_phase1_implementation_design.md §5. DBAASP is the one
source that needs paginated-id-list-then-per-record-fetch (ApiPuller); the
other four are one-shot bulk file downloads (BulkDownloader).

Logic bodies are intentionally NotImplementedError stubs pending review of
the scaffold — see the design doc for the intended behavior of `pull()` on
each base class before filling these in.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel

from clamp.data.schema import PeptideRecord, Source


class PullReport(BaseModel):
    source: Source
    attempted: int
    succeeded: int
    skipped_cached: int
    failed: list[str]


class ApiPuller(ABC):
    """Paginated-id-list + per-record-fetch pull, with a resumable on-disk
    cache. Currently only DbaaspPuller implements this."""

    source: Source

    @abstractmethod
    def total_count(self) -> int:
        """Total number of records available from the API."""
        ...

    @abstractmethod
    def iter_ids(self) -> Iterator[str]:
        """Enumerate all record ids via the list/search endpoint."""
        ...

    @abstractmethod
    def fetch_record(self, record_id: str) -> dict:
        """Fetch one full record by id."""
        ...

    def pull(self, cache_dir: Path, resume: bool = True) -> PullReport:
        """Rate-limited, resumable cache-to-disk loop.

        For each id from iter_ids(): skip if `{cache_dir}/{id}.json` already
        exists and resume=True; otherwise fetch_record(id), rate-limit per
        settings.dbaasp_requests_per_second, retry with backoff on
        non-200/429 (see clamp.config.Settings and doc 06 §7.1 — no
        documented rate limit, treat 429s as a signal to back off harder,
        not just retry), and write the raw JSON to disk.
        """
        raise NotImplementedError


class BulkDownloader(ABC):
    """One-shot file download(s) + offline parse. DRAMP, Hemolytik2,
    HemoPI2, QMAP all use this shape."""

    source: Source

    @abstractmethod
    def download_urls(self) -> list[str]:
        """URLs of the bulk file(s) to fetch for this source."""
        ...

    @abstractmethod
    def parse(self, downloaded_paths: list[Path]) -> list[PeptideRecord]:
        """Parse downloaded bulk file(s) into PeptideRecords."""
        ...

    def pull(self, cache_dir: Path) -> PullReport:
        """Download each URL to cache_dir if not already present, then
        delegate to parse()."""
        raise NotImplementedError
