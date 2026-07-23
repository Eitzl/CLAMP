"""DRAMP 4.0 bulk downloader.

Real download path and schema confirmed live on 2026-07-22 (the site's own
Downloads page links to `/download.php?...`, relative to `/downloads/` —
so the actual working URL is `/downloads/download.php?filename=...`; the
bare `/download.php` guess 404s). `general_amps.txt` is tab-separated with
29 columns; only ~44% of rows carry a populated `SMILES` column there, so
`general_smiles.txt` (a separate SMILES-only file, joined by DRAMP_ID) is
fetched too for broader curated-SMILES coverage.

`Linear/Cyclic/Branched`, `N-/C-terminal_Modification`, `Hemolytic_activity`
etc. are free text (e.g. "Cyclic (very possibly)", "Not metioned clearly"
[sic]) — data/README.md §1 rates this source's schema confidence "Medium"
for exactly this reason. This module intentionally does not attempt to
parse structured HC50/MIC values out of `Hemolytic_activity`/`Activity` —
unlike Hemolytik2's activity field (see sources/hemolytik2.py), DRAMP's is
even less consistently formatted and often purely descriptive ("No
hemolysis information..."), so DRAMP is treated here as a SMILES/sequence
source, not a label source. Revisit once a real column-level pass judges
it worth the parsing effort (data/README.md open items).
"""

import csv
from pathlib import Path

from clamp.data.schema import CyclizationType, PeptideRecord, Source
from clamp.data.sources.base import BulkDownloader

_BASE_URL = "https://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/{name}"
_GENERAL_AMPS_URL = _BASE_URL.format(name="general_amps.txt")
_GENERAL_SMILES_URL = _BASE_URL.format(name="general_smiles.txt")

_FREE_TEXT_NONE = {"free", "none", "not metioned clearly", "not mentioned clearly", ""}


def _clean_mod(raw: str | None) -> str | None:
    if not raw:
        return None
    cleaned = raw.strip()
    return None if cleaned.lower() in _FREE_TEXT_NONE else cleaned


class DrampDownloader(BulkDownloader):
    source = Source.DRAMP

    def download_urls(self) -> list[str]:
        return [_GENERAL_AMPS_URL, _GENERAL_SMILES_URL]

    def parse(self, downloaded_paths: list[Path]) -> list[PeptideRecord]:
        amps_path = next((p for p in downloaded_paths if "general_amps" in p.name), None)
        if amps_path is None:
            return []
        smiles_path = next((p for p in downloaded_paths if "general_smiles" in p.name), None)

        smiles_by_id: dict[str, str] = {}
        if smiles_path is not None:
            with smiles_path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    smi = (row.get("SMILES") or "").strip()
                    if smi:
                        smiles_by_id[row["DRAMP_ID"]] = smi

        records: list[PeptideRecord] = []
        with amps_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                sequence = (row.get("Sequence") or "").strip()
                if not sequence:
                    continue
                dramp_id = row["DRAMP_ID"]
                smiles = smiles_by_id.get(dramp_id) or (row.get("SMILES") or "").strip() or None
                # "Branched" is a real third value here (peptides with a
                # branch point, e.g. via a Lys side chain) — treating it
                # the same as "Linear" would silently drop that topology
                # signal. Neither "Cyclic" nor "Branched" tells us which of
                # our 5 p2smi cyclization types applies, so both route
                # through convert.py's p2smi path (via is_cyclic=True) with
                # cyclization_type=UNKNOWN, which correctly downgrades the
                # resulting fidelity tier instead of claiming a clean
                # linear FULL match it can't back up.
                topology = (row.get("Linear/Cyclic/Branched") or "").lower()
                is_cyclic = "cyclic" in topology or "branched" in topology
                records.append(
                    PeptideRecord(
                        source=Source.DRAMP,
                        source_id=dramp_id,
                        sequence_raw=sequence,
                        is_cyclic=is_cyclic,
                        cyclization_type=CyclizationType.UNKNOWN if is_cyclic else CyclizationType.NONE,
                        nterm_mod=_clean_mod(row.get("N-terminal_Modification")),
                        cterm_mod=_clean_mod(row.get("C-terminal_Modification")),
                        smiles=smiles,
                        reference=row.get("Pubmed_ID") or None,
                    )
                )
        return records


def load_cached_records(cache_dir: Path) -> list[PeptideRecord]:
    if not cache_dir.exists():
        return []
    paths = [p for p in cache_dir.glob("*") if p.is_file()]
    if not paths:
        return []
    return DrampDownloader().parse(paths)
