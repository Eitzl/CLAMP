"""Puller base classes.

See MD_design_docs/09_phase1_implementation_design.md §5. DBAASP is the one
source that needs paginated-id-list-then-per-record-fetch (ApiPuller); the
other four are one-shot bulk file downloads (BulkDownloader).
"""

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from clamp.config import settings
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
    requests_per_second: float = 1.0

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
        exists and resume=True; otherwise fetch_record(id), paced at
        1/requests_per_second between requests, retried with exponential
        backoff on any requests error (doc 06 §7.1 — no documented rate
        limit; a 429 gets the same exponential-backoff treatment as any
        other transient failure, not an immediate retry), and write the raw
        JSON to disk.
        """
        cache_dir.mkdir(parents=True, exist_ok=True)
        ids = list(self.iter_ids())

        fetch_with_retry = retry(
            retry=retry_if_exception_type(requests.RequestException),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            stop=stop_after_attempt(5),
            reraise=True,
        )(self.fetch_record)

        succeeded = 0
        skipped_cached = 0
        failed: list[str] = []
        for record_id in ids:
            cache_file = cache_dir / f"{record_id}.json"
            if resume and cache_file.exists():
                skipped_cached += 1
                continue
            time.sleep(1.0 / self.requests_per_second)
            try:
                raw = fetch_with_retry(record_id)
            except requests.RequestException:
                failed.append(record_id)
                continue
            cache_file.write_text(json.dumps(raw))
            succeeded += 1

        return PullReport(
            source=self.source,
            attempted=len(ids),
            succeeded=succeeded,
            skipped_cached=skipped_cached,
            failed=failed,
        )


def _filename_for_url(url: str) -> str:
    """Derive a cache filename from a download URL. Handles both plain
    paths (.../general_amps.txt) and query-param-based download endpoints
    (.../download.php?filename=.../general_amps.txt, confirmed live on
    DRAMP) — the latter would otherwise collide on "download.php" for
    every URL sharing that script."""
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if "filename" in query and query["filename"][0]:
        return Path(query["filename"][0]).name
    return Path(parsed.path).name or "download"


class BulkDownloader(ABC):
    """One-shot file download(s) + offline parse. DRAMP, Hemolytik2,
    HemoPI2, QMAP all use this shape (Hemolytik2 and QMAP override pull()
    directly — see their modules — since their real access pattern doesn't
    fit a plain URL-list download)."""

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
        delegate to parse() (used only to validate the downloaded files are
        actually parseable — the parsed records themselves are re-derived
        from disk later by each module's load_cached_records(), not
        threaded through PullReport)."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        urls = self.download_urls()
        paths: list[Path] = []
        skipped_cached = 0
        failed: list[str] = []
        for url in urls:
            dest = cache_dir / _filename_for_url(url)
            if dest.exists():
                skipped_cached += 1
                paths.append(dest)
                continue
            try:
                response = requests.get(url, timeout=settings.http_timeout_s)
                response.raise_for_status()
                dest.write_bytes(response.content)
                paths.append(dest)
            except requests.RequestException:
                failed.append(url)

        succeeded = 0
        if paths:
            try:
                succeeded = len(self.parse(paths))
            except Exception as exc:
                # A parse() bug here (e.g. a source changing its column
                # names) must not look identical to "downloaded fine, zero
                # rows" in the report — that swallowed the distinction
                # between "nothing to parse" and "parsing broke," which
                # would hide a real regression behind a clean-looking
                # succeeded=0. Record it as a failure instead.
                failed.append(f"parse_error: {exc}")

        return PullReport(
            source=self.source,
            attempted=len(urls),
            succeeded=succeeded,
            skipped_cached=skipped_cached,
            failed=failed,
        )
