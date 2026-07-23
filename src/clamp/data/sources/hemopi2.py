"""HemoPI2 bulk downloader.

The real labeled dataset lives in the GitHub repo's `Dataset/` folder
(confirmed live 2026-07-22: `cross_val_dataset.csv` +
`independent_dataset.csv`, 1541 + 387 = 1928 rows, matching the
"~1,926 experimentally validated" figure), fetched from
raw.githubusercontent.com. **The pip package `hemopi2` does NOT ship this
data** — installing and inspecting it showed it contains only a
pretrained classifier/regressor model plus feature-encoding lookup tables
(Model/Data/*.csv), no raw sequence+label file — so it is deliberately not
a pyproject dependency; we go straight to the GitHub CSVs instead.

Schema (confirmed live): 3 columns, `SEQUENCE, μM, label` — the μM column
uses GREEK SMALL LETTER MU (U+03BC), not the MICRO SIGN.

`hc50_assay_target_cell` is populated here (found missing via a real
pilot pull — this field was silently None for every HemoPI2 record before
this fix, one of only 3 sources across the pipeline that ever populates
it at all). The repo's own README confirms the assay: "targets for
mammalian red blood cells (RBCs)... the concentration at which 50% of red
blood cells (RBCs) are lysed" — confirmed as RBC lysis, but the README
says "mammalian," not specifically human, so the value below is
deliberately hedged rather than asserting "Human RBC" without source
confirmation of the species.
"""

import csv
from pathlib import Path

from clamp.data.schema import PeptideRecord, Source
from clamp.data.sources.base import BulkDownloader

_BASE_URL = "https://raw.githubusercontent.com/raghavagps/HemoPI2/main/Dataset/{name}"
_FILES = ["cross_val_dataset.csv", "independent_dataset.csv"]
_VALUE_COLUMN = "μM"  # GREEK SMALL LETTER MU + "M", confirmed against the real file
_TARGET_CELL = "mammalian erythrocytes (species not specified in source)"


class HemoPI2Downloader(BulkDownloader):
    source = Source.HEMOPI2

    def download_urls(self) -> list[str]:
        return [_BASE_URL.format(name=name) for name in _FILES]

    def parse(self, downloaded_paths: list[Path]) -> list[PeptideRecord]:
        records: list[PeptideRecord] = []
        for path in downloaded_paths:
            with path.open(newline="", encoding="utf-8") as f:
                for i, row in enumerate(csv.DictReader(f)):
                    sequence = (row.get("SEQUENCE") or "").strip()
                    if not sequence:
                        continue
                    raw_value = row.get(_VALUE_COLUMN)
                    value = float(raw_value) if raw_value not in (None, "") else None
                    records.append(
                        PeptideRecord(
                            source=Source.HEMOPI2,
                            source_id=f"{path.stem}:{i}",
                            sequence_raw=sequence,
                            hc50_value=value,
                            hc50_unit="uM" if value is not None else None,
                            hc50_assay_target_cell=_TARGET_CELL if value is not None else None,
                        )
                    )
        return records


def load_cached_records(cache_dir: Path) -> list[PeptideRecord]:
    if not cache_dir.exists():
        return []
    paths = [p for p in cache_dir.glob("*.csv") if p.is_file()]
    if not paths:
        return []
    return HemoPI2Downloader().parse(paths)
