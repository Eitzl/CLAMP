"""DBAASP puller.

REST API verified live against `GET https://dbaasp.org/peptides/16` and
several other ids on 2026-07-22 (see MD_design_docs/06_task5_data_pipeline_plan.md
§1 for the original OpenAPI-spec-based description; this module additionally
reflects a handful of things only visible in real responses, noted inline):

  GET {base_url}/peptides?limit=&offset=   -> {"totalCount": N, "data": [...]}
  GET {base_url}/peptides/{id}             -> full PeptideView

The list endpoint is a thin summary (no smiles/bonds/activities) — do not
try to harvest labels from it, only ids.

BLOCKING PREREQUISITE (doc 06 §1.3, §7.1): read DBAASP's data-usage-policy
PDF before running pull() at full ~25k-record scale. Not implemented here;
tracked in doc 09 §14/§16 as a manual step, not a code gate.
"""

import json
import re
import time
from collections.abc import Iterator
from pathlib import Path

import requests

from clamp.config import settings
from clamp.data.schema import CyclizationType, PeptideRecord, Source, UnusualResidue
from clamp.data.sources.base import ApiPuller

# Confirmed live against GET /peptides/105: an AMD-type bond with
# cycleType.name == "NCB" ("N-C-termini bond") is head-to-tail. No live
# example of a disulfide/side-chain bond was found during the same spike,
# so those map by bond `type.name` alone (DSB, confirmed indirectly via
# qmap-benchmark's own bond vocabulary, which is built from this same
# field) and everything else deliberately falls back to UNKNOWN rather
# than guessing — extend these tables once more real examples are seen at
# full-pull scale (see data/README.md open items).
_BOND_TYPE_MAP: dict[str, CyclizationType] = {
    "DSB": CyclizationType.DISULFIDE,
}
_CYCLE_TYPE_MAP: dict[str, CyclizationType] = {
    "NCB": CyclizationType.HEAD_TO_TAIL,
}


def _name(obj: dict | None) -> str | None:
    """DBAASP nests many scalar-ish fields as {"name": ..., "description":
    ...} ("MixedView" in the OpenAPI spec) or leaves them null. Pull the
    name out uniformly; treat an empty string the same as missing."""
    if not obj:
        return None
    if isinstance(obj, dict):
        return obj.get("name") or None
    return obj or None


_UNCERTAINTY_RE = re.compile(r"^\s*([\d.]+)\s*±")
_INEQUALITY_RE = re.compile(r"^\s*[<>≤≥]{1,2}=?\s*([\d.]+)")


def _parse_concentration(raw: str | None) -> tuple[float | None, bool, str | None]:
    """DBAASP's `concentration` field turns out to have (at least) three
    distinct non-plain-float conventions, not just one — found via a real
    ~90k-row pilot pull, not the small hand-picked fixtures used in unit
    tests: a hyphen-separated range ("1.58-25.33"), a ± uncertainty
    notation ("2.2±0.8"), and an inequality-bounded value (">400"). An
    earlier version of this function treated "float() failed" as
    synonymous with "is a range," which incorrectly routed ± and
    inequality forms into normalize.py's hyphen-only parse_range() and
    crashed on ~760 real rows in that pilot. Only a genuine hyphen-range
    is tagged is_range=True now; ± and inequality forms extract their
    point value directly (dropping the uncertainty / treating the bound
    as a rough point — the same documented simplification already used
    for Hemolytik2's censored values elsewhere in this codebase, not a
    new precedent). Anything else unparseable returns (None, False, None)
    rather than propagating a broken "range" downstream."""
    if not raw:
        return None, False, None
    raw = raw.strip()
    try:
        return float(raw), False, None
    except ValueError:
        pass
    if "-" in raw:
        return None, True, raw
    match = _UNCERTAINTY_RE.match(raw) or _INEQUALITY_RE.match(raw)
    if match:
        return float(match.group(1)), False, None
    return None, False, None


def _map_cyclization(bond: dict) -> CyclizationType:
    bond_type = _name(bond.get("type"))
    cycle_type = _name(bond.get("cycleType"))
    if bond_type in _BOND_TYPE_MAP:
        return _BOND_TYPE_MAP[bond_type]
    if cycle_type in _CYCLE_TYPE_MAP:
        return _CYCLE_TYPE_MAP[cycle_type]
    return CyclizationType.UNKNOWN


class DbaaspPuller(ApiPuller):
    source = Source.DBAASP

    def __init__(
        self,
        base_url: str = settings.dbaasp_base_url,
        page_size: int = settings.dbaasp_page_size,
        requests_per_second: float = settings.dbaasp_requests_per_second,
    ):
        self.base_url = base_url.rstrip("/")
        self.page_size = page_size
        self.requests_per_second = requests_per_second

    def total_count(self) -> int:
        resp = requests.get(
            f"{self.base_url}/peptides",
            params={"limit": 1, "offset": 0},
            timeout=settings.http_timeout_s,
        )
        resp.raise_for_status()
        return resp.json()["totalCount"]

    def iter_ids(self) -> Iterator[str]:
        offset = 0
        while True:
            resp = requests.get(
                f"{self.base_url}/peptides",
                params={"limit": self.page_size, "offset": offset},
                timeout=settings.http_timeout_s,
            )
            resp.raise_for_status()
            payload = resp.json()
            items = payload.get("data") or []
            if not items:
                return
            for item in items:
                yield str(item["id"])
            offset += self.page_size
            if offset >= payload.get("totalCount", offset):
                return
            time.sleep(1.0 / self.requests_per_second)

    def fetch_record(self, record_id: str) -> dict:
        resp = requests.get(f"{self.base_url}/peptides/{record_id}", timeout=settings.http_timeout_s)
        resp.raise_for_status()
        return resp.json()

    def to_records(self, raw: dict) -> list[PeptideRecord]:
        """Map one raw PeptideView JSON blob to PeptideRecord(s): one row
        per (targetActivity restricted to MIC | hemoliticCytotoxicActivity),
        per doc 06 §4.1's "one row per assay-record" convention, or a
        single label-less row if a peptide has no activity data at all
        (still useful as an unlabeled SMILES corpus entry).

        Multimer handling (confirmed live against GET /peptides/1,
        "Distinctin"): complexity.name == "Multimer" records have an empty
        top-level `sequence`, and their `monomers[]` are full nested
        PeptideView objects — but those same monomers (e.g. id 16, 36) are
        ALSO independently reachable as their own top-level ids with their
        own full activity data (that's how "Distinctin chain 1", id 16,
        was found in the first place). Unnesting monomers[] here would
        therefore double-count them against the standalone pull. We
        deliberately emit nothing for the complex-level record itself
        (rather than a garbage sequence_raw="" row) and rely on each chain
        already being pulled as its own id — this does mean any label data
        recorded only at the complex level (DBAASP's id=1 record itself
        carried 26 targetActivities of its own) is currently dropped, since
        our single-chain PeptideRecord/SMILES model has no way to
        represent a multi-chain complex. Flagged as an open item, not a
        silent gap.

        A third complexity value, "Multi-Peptide" (distinct peptides
        forming a complex, e.g. two-component bacteriocins), was found via
        a real pilot pull — same empty-top-level-`sequence` shape as
        Multimer, and 7x more common in that pilot (21 vs 3 ids). It isn't
        special-cased above; it's currently handled "correctly" only by
        accident, via the generic `if not sequence_raw: return []` guard
        below — same complex-level-activity-dropping caveat applies, and
        it hasn't been confirmed whether Multi-Peptide's nested monomer
        ids are independently pullable the way Multimer's are (the one
        spot-checked example's monomer id fell outside the pilot's pull
        range). Needs the same verification Multimer already got, at
        full-pull scale.
        """
        complexity = (_name(raw.get("complexity")) or "").lower()
        if complexity == "multimer":
            return []

        sequence_raw = raw.get("sequence") or ""
        if not sequence_raw:
            return []

        source_id = str(raw["id"])
        nterm_mod = _name(raw.get("nTerminus"))
        cterm_mod = _name(raw.get("cTerminus"))

        unusual_residues = [
            UnusualResidue(
                position=u["position"],
                from_residue=u.get("beforeModification") or None,
                modification_type=_name(u.get("modificationType")) or "unknown",
            )
            for u in (raw.get("unusualAminoAcids") or [])
        ]

        bonds = raw.get("intrachainBonds") or []
        is_cyclic = bool(bonds)
        cyclization_type = _map_cyclization(bonds[0]) if bonds else CyclizationType.NONE

        curated_smiles_entries = raw.get("smiles") or []
        # doc 09 §16 item 5: `manuallyEdited` trust level unresolved —
        # pass through whichever entry DBAASP lists first regardless of
        # that flag, for now.
        curated_smiles = curated_smiles_entries[0]["smiles"] if curated_smiles_entries else None

        base_kwargs = dict(
            source=Source.DBAASP,
            sequence_raw=sequence_raw,
            is_cyclic=is_cyclic,
            cyclization_type=cyclization_type,
            nterm_mod=nterm_mod,
            cterm_mod=cterm_mod,
            unusual_residues=unusual_residues,
            smiles=curated_smiles,
        )

        records: list[PeptideRecord] = []

        for activity in raw.get("targetActivities") or []:
            if _name(activity.get("activityMeasureGroup")) != "MIC":
                continue  # MBC/other measure groups are out of scope for mic_value
            value, is_range, range_raw = _parse_concentration(activity.get("concentration"))
            records.append(
                PeptideRecord(
                    **base_kwargs,
                    source_id=f"{source_id}:mic:{activity['id']}",
                    mic_value=value,
                    mic_unit=_name(activity.get("unit")),
                    mic_target_species=_name(activity.get("targetSpecies")),
                    mic_assay_medium=_name(activity.get("medium")),
                    label_is_range=is_range,
                    label_range_raw=range_raw,
                    reference=activity.get("reference") or None,
                )
            )

        for activity in raw.get("hemoliticCytotoxicActivities") or []:
            value, is_range, range_raw = _parse_concentration(activity.get("concentration"))
            records.append(
                PeptideRecord(
                    **base_kwargs,
                    source_id=f"{source_id}:hc50:{activity['id']}",
                    hc50_value=value,
                    hc50_unit=_name(activity.get("unit")),
                    hc50_assay_target_cell=_name(activity.get("targetCell")),
                    label_is_range=is_range,
                    label_range_raw=range_raw,
                    reference=activity.get("reference") or None,
                )
            )

        if not records:
            records.append(PeptideRecord(**base_kwargs, source_id=source_id))

        return records


def load_cached_records(cache_dir: Path) -> list[PeptideRecord]:
    """Parse every raw JSON file already on disk under cache_dir into
    PeptideRecords via DbaaspPuller.to_records — the offline half of the
    pull, usable without re-hitting the API."""
    if not cache_dir.exists():
        return []
    puller = DbaaspPuller()
    records: list[PeptideRecord] = []
    for path in sorted(cache_dir.glob("*.json")):
        raw = json.loads(path.read_text())
        records.extend(puller.to_records(raw))
    return records
