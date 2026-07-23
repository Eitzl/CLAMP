"""Phase 1 pipeline orchestration: manifest-file-based staging, not a
workflow engine — see MD_design_docs/09_phase1_implementation_design.md
§10 for why (linear DAG, one machine, no scheduling/distribution need).

Revisit `dvc` specifically (not Prefect/Luigi) if/when the "versioned
dataset" requirement from MD_design_docs/02_peptideclm_transfer_learning_plan.md
P0 deliverable becomes load-bearing rather than a nice property.
"""

import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from clamp.config import settings
from clamp.data import datasheet as datasheet_module
from clamp.data.convert import convert_record
from clamp.data.dedup import DedupResult, dedup_records
from clamp.data.normalize import log_transform, normalize_concentration
from clamp.data.schema import PeptideRecord
from clamp.data.sources import dbaasp, dramp, hemolytik2, hemopi2, qmap


def _sanitize_row(row: dict) -> dict:
    """A parquet round-trip turns a genuinely-None value in an Optional
    field into a float NaN the moment that same DataFrame column also
    holds a real value in another row — pandas/pyarrow have no other way
    to represent a missing value once a column mixes None with real data.
    This is the normal case here, not an edge case: any row with only MIC
    (no HC50) sits in the same DataFrame as a row with only HC50 (no MIC),
    so hc50_unit/mic_unit/etc. columns are mixed on every realistic pull.
    PeptideRecord's pydantic validation correctly rejects a NaN where a
    str/enum field expects None — found via a real integration-test
    failure, not hypothetically. Every DataFrame->PeptideRecord boundary
    needs this normalization applied first.
    """
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}


@dataclass
class Stage:
    name: str
    run: Callable[[], None]
    inputs: list[str]
    output: str


def _pull_dbaasp() -> None:
    dbaasp.DbaaspPuller().pull(settings.raw_dir / "dbaasp")


def _pull_dramp() -> None:
    dramp.DrampDownloader().pull(settings.raw_dir / "dramp")


def _pull_hemolytik2() -> None:
    hemolytik2.Hemolytik2Downloader().pull(settings.raw_dir / "hemolytik2")


def _pull_hemopi2() -> None:
    hemopi2.HemoPI2Downloader().pull(settings.raw_dir / "hemopi2")


def _pull_qmap() -> None:
    qmap.QmapDownloader().pull(settings.raw_dir / "qmap")


def _load_all_raw_records() -> list[PeptideRecord]:
    return [
        *dbaasp.load_cached_records(settings.raw_dir / "dbaasp"),
        *dramp.load_cached_records(settings.raw_dir / "dramp"),
        *hemolytik2.load_cached_records(settings.raw_dir / "hemolytik2"),
        *hemopi2.load_cached_records(settings.raw_dir / "hemopi2"),
        *qmap.load_cached_records(settings.raw_dir / "qmap"),
    ]


def _convert() -> None:
    converted = []
    for record in _load_all_raw_records():
        result = convert_record(record, curated_smiles=record.smiles)
        converted.append(
            record.model_copy(
                update={
                    "smiles": result.smiles,
                    "smiles_source": result.smiles_source,
                    "smiles_validated": result.validated,
                    "chemical_fidelity_tier": result.fidelity_tier,
                    "dropped_modifications": result.dropped_modifications,
                }
            )
        )
    settings.interim_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.model_dump() for r in converted]).to_parquet(settings.interim_dir / "converted.parquet")


def _dedup() -> None:
    df = pd.read_parquet(settings.interim_dir / "converted.parquet")
    records = [PeptideRecord(**_sanitize_row(row)) for row in df.to_dict(orient="records")]
    result = dedup_records(records)
    pd.DataFrame([r.model_dump() for r in result.records]).to_parquet(settings.interim_dir / "deduped.parquet")
    (settings.interim_dir / "dedup_result.json").write_text(
        json.dumps(
            {
                "raw_row_count": result.raw_row_count,
                "unique_peptide_uid_count": result.unique_peptide_uid_count,
                "flagged_conflict_count": result.flagged_conflict_count,
                "overlap_matrix": {f"{a}|{b}": c for (a, b), c in result.overlap_matrix.items()},
            }
        )
    )


def _molecular_weight(smiles: str | None) -> float | None:
    # Guards against NaN, not just None/"": a genuinely-missing smiles
    # value (any FAILED-tier record where nothing could be generated at
    # all) reaches this function via `df["smiles"].map(...)`, and both
    # pandas' None-in-a-mixed-column behavior and its newer native string
    # dtype can hand this a float('nan') instead of None. `not float("nan")`
    # is False (NaN is truthy), so a bare `if not smiles` guard lets it
    # through to Chem.MolFromSmiles(nan), which crashes with an opaque C++
    # TypeError rather than cleanly returning None — found via a real
    # pilot data pull, not a synthetic test case.
    if not isinstance(smiles, str) or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol) if mol is not None else None


def _normalize() -> None:
    df = pd.read_parquet(settings.interim_dir / "deduped.parquet")
    molecular_weights = df["smiles"].map(_molecular_weight)

    def _normalized_column(value_col: str, unit_col: str) -> list[float | None]:
        out = []
        for raw_row, weight in zip(df.to_dict(orient="records"), molecular_weights):
            row = _sanitize_row(raw_row)
            # weight comes from a pandas Series that mixes real floats with
            # None (from _molecular_weight) — pandas stores that mix as
            # NaN, and `not float("nan")` is False (NaN is truthy), so a
            # plain `not weight` guard silently lets a NaN weight through
            # to normalize_concentration instead of skipping the row.
            weight_missing = weight is None or (isinstance(weight, float) and math.isnan(weight))
            # label_is_range/label_range_raw are single fields shared by
            # whichever ONE of mic_value/hc50_value this row actually
            # populates (schema.py has no separate range-flag per task) —
            # found via a real pilot pull: an MIC-only row with a
            # range-valued concentration was incorrectly treated as
            # "the HC50 column also has range input" when normalizing
            # hc50_value_uM, since the check only looked at the shared
            # is_range/range_raw fields without confirming this column's
            # OWN unit was actually populated — then crashed on
            # normalize_concentration's unit.strip() because hc50_unit is
            # correctly None for an MIC-only row. Gating on row[unit_col]
            # is not None disambiguates which task the range belongs to.
            has_range_input = row["label_is_range"] and row["label_range_raw"] and row[unit_col] is not None
            has_input = row[value_col] is not None or has_range_input
            if not has_input or weight_missing:
                out.append(None)
                continue
            out.append(
                normalize_concentration(
                    row[value_col],
                    row[unit_col],
                    weight,
                    is_range=row["label_is_range"],
                    range_raw=row["label_range_raw"],
                )
            )
        return out

    df["mic_value_uM"] = _normalized_column("mic_value", "mic_unit")
    df["hc50_value_uM"] = _normalized_column("hc50_value", "hc50_unit")
    df["mic_log_uM"] = df["mic_value_uM"].map(log_transform)
    df["hc50_log_uM"] = df["hc50_value_uM"].map(log_transform)

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(settings.processed_dir / "dataset.parquet")


def _datasheet() -> None:
    df = pd.read_parquet(settings.processed_dir / "dataset.parquet")
    dedup_info = json.loads((settings.interim_dir / "dedup_result.json").read_text())
    dedup_result = DedupResult(
        records=[],
        overlap_matrix={tuple(k.split("|")): v for k, v in dedup_info["overlap_matrix"].items()},
        raw_row_count=dedup_info["raw_row_count"],
        unique_peptide_uid_count=dedup_info["unique_peptide_uid_count"],
        flagged_conflict_count=dedup_info["flagged_conflict_count"],
    )
    tables = datasheet_module.build_datasheet(df, dedup_result)
    datasheet_module.write_datasheet(tables, settings.datasheet_dir)


STAGES: list[Stage] = [
    Stage("pull_dbaasp", _pull_dbaasp, inputs=[], output="raw/dbaasp/"),
    Stage("pull_dramp", _pull_dramp, inputs=[], output="raw/dramp/"),
    Stage("pull_hemolytik2", _pull_hemolytik2, inputs=[], output="raw/hemolytik2/"),
    Stage("pull_hemopi2", _pull_hemopi2, inputs=[], output="raw/hemopi2/"),
    Stage("pull_qmap", _pull_qmap, inputs=[], output="raw/qmap/"),
    Stage("convert", _convert, inputs=["raw/*"], output="interim/converted.parquet"),
    Stage("dedup", _dedup, inputs=["interim/converted.parquet"], output="interim/deduped.parquet"),
    Stage("normalize", _normalize, inputs=["interim/deduped.parquet"], output="processed/dataset.parquet"),
    Stage("datasheet", _datasheet, inputs=["processed/dataset.parquet"], output="datasheet/"),
]


def _content_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    if path.is_dir():
        parts = sorted(f"{p.relative_to(path)}:{p.stat().st_mtime_ns}" for p in path.rglob("*") if p.is_file())
        return hashlib.sha256("|".join(parts).encode()).hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest() -> dict:
    if settings.manifest_path.exists():
        return json.loads(settings.manifest_path.read_text())
    return {}


def _save_manifest(manifest: dict) -> None:
    settings.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    settings.manifest_path.write_text(json.dumps(manifest, indent=2))


def run_pipeline(stages: list[Stage] = STAGES, force: bool = False) -> None:
    """Loads/updates settings.manifest_path, skips a stage if its inputs are
    unchanged and force=False, otherwise runs it and updates the manifest.

    Stages with no declared inputs (the five pull_* stages) always run —
    that's intentional, not a manifest gap: each puller has its own
    resume-skip-cached-ids/files logic (sources/base.py), so re-running
    pull_dbaasp on an already-fully-cached raw/dbaasp/ is itself cheap.
    """
    manifest = _load_manifest()
    for stage in stages:
        input_hash = "|".join(
            _content_hash(settings.data_root / p.replace("*", "")) or "" for p in stage.inputs
        )
        cached = manifest.get(stage.name)
        if not force and input_hash and cached and cached.get("inputs_hash") == input_hash:
            continue
        stage.run()
        manifest[stage.name] = {
            "inputs_hash": input_hash,
            "output_path": stage.output,
            "completed_at": time.time(),
        }
        _save_manifest(manifest)
