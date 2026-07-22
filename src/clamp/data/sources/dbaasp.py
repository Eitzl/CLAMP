"""DBAASP puller.

REST API verified in MD_design_docs/06_task5_data_pipeline_plan.md §1:
  GET {base_url}/peptides?limit=&offset=   -> paginated SearchResultItemView list
  GET {base_url}/peptides/{id}             -> full PeptideView

The list endpoint is a thin summary (no smiles/bonds/activities) — do not
try to harvest labels from it, only ids (doc 06 §1.3).

BLOCKING PREREQUISITE (doc 06 §1.3, §7.1): read DBAASP's data-usage-policy
PDF before running pull() at full ~25k-record scale. Not implemented here;
tracked in doc 09 §14/§16 as a manual step, not a code gate.
"""

from collections.abc import Iterator
from pathlib import Path

from clamp.config import settings
from clamp.data.schema import PeptideRecord, Source
from clamp.data.sources.base import ApiPuller


class DbaaspPuller(ApiPuller):
    source = Source.DBAASP

    def __init__(self, base_url: str = settings.dbaasp_base_url, page_size: int = settings.dbaasp_page_size):
        self.base_url = base_url
        self.page_size = page_size

    def total_count(self) -> int:
        # GET {base_url}/peptides?limit=1&offset=0 -> {"totalCount": N, ...}
        raise NotImplementedError

    def iter_ids(self) -> Iterator[str]:
        # Page through /peptides?limit=self.page_size&offset=N until
        # exhausted; yield each result's id.
        raise NotImplementedError

    def fetch_record(self, record_id: str) -> dict:
        # GET {base_url}/peptides/{record_id} -> full PeptideView JSON
        raise NotImplementedError

    def to_records(self, raw: dict) -> list[PeptideRecord]:
        """Map one raw PeptideView JSON blob to PeptideRecord(s).

        One raw DBAASP record can yield multiple PeptideRecords: one row per
        (targetActivity | hemoliticCytotoxicActivity), per doc 06 §4.1's
        "one row per (peptide-construct, assay-record)" convention. Also
        responsible for:
          - un-nesting `monomers[]` for complexity="Multimer" records
            (doc 06 §1.2 — top-level `sequence` is empty for these; a naive
            pull silently drops ~10% of records without this)
          - mapping `intrachainBonds[].cycleType` -> CyclizationType
          - mapping `unusualAminoAcids[]` -> UnusualResidue
          - extracting the curated `smiles[]` entry when present
        """
        raise NotImplementedError


def load_cached_records(cache_dir: Path) -> list[PeptideRecord]:
    """Parse every raw JSON file already on disk under cache_dir into
    PeptideRecords via DbaaspPuller.to_records — the offline half of the
    pull, usable without re-hitting the API."""
    raise NotImplementedError
