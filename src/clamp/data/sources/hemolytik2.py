"""Hemolytik2 downloader — REST API only.

No bulk-file download was found on the live site (2026-07-22) despite the
docs mentioning one; a REST API was confirmed live instead:

    GET https://webs.iiitd.edu.in/raghava/hemolytik2/api/api.php
        ?dataType={source|nature|sequence}&dataValue=...

(the bare `.../hemolytik2/api/...` without `/raghava/` 404s — the site is
namespaced under the lab's `/raghava/` path). Live-tested with
`dataType=source&dataValue=Human` -> 9,255 rows. Confirmed response fields:
`id, pmid, year, seq, name, cter, nter, lyn_cyc, ldmix, non_nat, length,
nature, activity, source, origin, exp_str, non_hem` — notably **no SMILES
field** from this endpoint, contrary to the docs' description of the web
UI's peptide cards; Hemolytik2 records go through convert.py like any
other sequence-only source.

There is no "list all" endpoint, so full coverage requires enumerating
facet values query-by-query. `nature` (functional class) is used here
since it's described as a small, closed-ish taxonomy; `source` (organism)
is open-ended (arbitrary species names) and not enumerated. **This
coverage is not confirmed exhaustive** — cross-check the datasheet's
per-source Hemolytik2 row count against the ~13,215-entry figure from the
source paper (bioRxiv 2025.05.12.653624) before trusting it fully; see
data/README.md open items.
"""

import json
import re
from pathlib import Path

import requests

from clamp.config import settings
from clamp.data.schema import CyclizationType, PeptideRecord, Source
from clamp.data.sources.base import BulkDownloader, PullReport

_API_URL = "https://webs.iiitd.edu.in/raghava/hemolytik2/api/api.php"

_NATURE_VALUES = [
    "Antimicrobial",
    "Anticancer",
    "Antifungal",
    "Antiviral",
    "Antiparasitic",
    "Anti-HIV",
    "Antibiofilm",
    # Confirmed live via a real pilot pull: "Antiinflammatory" (no hyphen)
    # 404s against the actual API; the correct value has a hyphen.
    "Anti-inflammatory",
]

_FREE_TEXT_NONE = {"free", "none", ""}

# Best-effort extraction of a point HC50-equivalent value from Hemolytik2's
# free-text `activity` field. Confirmed live to contain wildly inconsistent
# conventions: "HC50 =6μM", "MHC =81μM", "LD50 >400μM", "MHC  <<81μM",
# "LC50 =2.3±0.3μM", "77% hemolysis at 100 μM", ">2-3% at ≥20μM". Only the
# "<LABEL> <op>= <number><unit>" form is extracted; percentage-at-a-single-
# dose reports are skipped entirely (return None) since they can't be
# converted to a point HC50 without curve-fitting across doses, which is
# out of scope here. An inequality-qualified value (">200") is still kept
# as a rough point estimate rather than dropped, since the schema has no
# "censored" representation yet — a documented simplification, not a
# silent one.
#
# The label group is a whitelist of the assay-metric abbreviations
# actually observed live, not a bare `[A-Za-z]+` — an earlier version
# accepted *any* leading word in that slot, so free text like
# "Reference =6μM" or "pH =6μM" (both real-shaped fields in this dataset)
# would silently parse as a 6μM point measurement. Extend this list only
# once a real, currently-unhandled label is confirmed in the live data.
_ACTIVITY_RE = re.compile(
    r"^\s*(?:MHC|HC50|HD50|LD50|LC50|EC50)\s*(?:<<|>>|<=|>=|<|>)?\s*=?\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?:±\s*\d+(?:\.\d+)?\s*)?"
    r"(?P<unit>µg/ml|μg/ml|ug/ml|µM|μM|uM)",
    re.IGNORECASE,
)


def _clean_mod(raw: str | None) -> str | None:
    if not raw:
        return None
    cleaned = raw.strip()
    return None if cleaned.lower() in _FREE_TEXT_NONE else cleaned


def _parse_activity(text: str | None) -> tuple[float, str] | None:
    if not text:
        return None
    match = _ACTIVITY_RE.match(text.strip())
    if not match:
        return None
    return float(match.group("value")), match.group("unit")


class Hemolytik2Downloader(BulkDownloader):
    source = Source.HEMOLYTIK2

    def download_urls(self) -> list[str]:
        return [f"{_API_URL}?dataType=nature&dataValue={value}" for value in _NATURE_VALUES]

    def pull(self, cache_dir: Path) -> PullReport:
        """Overridden: this source's real access pattern is faceted API
        queries merged into one cache file, not a plain URL-list download
        (see module docstring)."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        by_id: dict[str, dict] = {}
        failed: list[str] = []
        for value in _NATURE_VALUES:
            try:
                resp = requests.get(
                    _API_URL,
                    params={"dataType": "nature", "dataValue": value},
                    timeout=settings.http_timeout_s,
                )
                resp.raise_for_status()
                payload = resp.json()
            except (requests.RequestException, ValueError):
                failed.append(value)
                continue
            for row in payload.get("data") or []:
                by_id[row["id"]] = row

        (cache_dir / "hemolytik2.json").write_text(json.dumps(list(by_id.values())))
        return PullReport(
            source=self.source,
            attempted=len(_NATURE_VALUES),
            succeeded=len(_NATURE_VALUES) - len(failed),
            skipped_cached=0,
            failed=failed,
        )

    def parse(self, downloaded_paths: list[Path]) -> list[PeptideRecord]:
        path = next((p for p in downloaded_paths if p.name == "hemolytik2.json"), None)
        if path is None:
            return []
        rows = json.loads(path.read_text())

        records: list[PeptideRecord] = []
        for row in rows:
            sequence = (row.get("seq") or "").strip()
            if not sequence:
                continue
            is_cyclic = (row.get("lyn_cyc") or "").strip().lower() == "cyclic"
            parsed_activity = _parse_activity(row.get("activity"))
            records.append(
                PeptideRecord(
                    source=Source.HEMOLYTIK2,
                    source_id=str(row["id"]),
                    sequence_raw=sequence,
                    is_cyclic=is_cyclic,
                    cyclization_type=CyclizationType.UNKNOWN if is_cyclic else CyclizationType.NONE,
                    nterm_mod=_clean_mod(row.get("nter")),
                    cterm_mod=_clean_mod(row.get("cter")),
                    hc50_value=parsed_activity[0] if parsed_activity else None,
                    hc50_unit=parsed_activity[1] if parsed_activity else None,
                    reference=row.get("pmid") or None,
                )
            )
        return records


def load_cached_records(cache_dir: Path) -> list[PeptideRecord]:
    path = cache_dir / "hemolytik2.json"
    if not path.exists():
        return []
    return Hemolytik2Downloader().parse([path])
