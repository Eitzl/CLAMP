# Phase 1 Implementation Design: Data Acquisition & Preprocessing Pipeline

*Executable companion to `08_implementation_roadmap.md` Phase 1 ("Data Acquisition & Preprocessing — The Foundation"), translating `06_task5_data_pipeline_plan.md` into an actual package scaffold: module boundaries, data model, CLI, storage layout, and a testing plan. This is a design document, not a finished implementation — code below is illustrative scaffold, and every section that has a real design choice (config system, CLI framework, orchestration strategy, dataframe library) presents options with a recommendation rather than assuming one. Scope decisions locked in for this doc: installable package (not scripts/notebooks), `uv` + `pyproject.toml`, local filesystem storage only (no cloud/object storage — deferred to a later DBTL cycle), lightweight pytest coverage of the pure-logic pieces.*

---

## 0. Scope and non-goals

**In scope** (mirrors doc 06 §6's P0 checklist and doc 08's Phase 1 bullets):
- Pull DBAASP (paginated REST API), DRAMP, Hemolytik2, HemoPI2, and QMAP (bulk files).
- Convert every record to SMILES via the decision table in doc 06 §2.2 (DB-curated → RDKit-from-sequence → `p2smi` → HELM fallback → plain-backbone fallback).
- Validate (round-trip + mass-sanity), classify into fidelity tiers, dedup into `peptide_uid`s, normalize units, log-transform.
- Produce the versioned dataset + datasheet deliverable (doc 06 §4.4).

**Explicitly out of scope for this doc** (later phases per `08_implementation_roadmap.md`):
- Embedding generation, clustering, splitting (Phase 2).
- Model selection pilot (Phase 3).
- Any training code, heads, losses (Phase 4).
- Cloud/object storage, distributed execution — deliberately deferred; the design below should not make that migration hard, but it doesn't build it now.

---

## 1. Package layout

**Recommendation: `src/`-layout installable package**, importable as `clamp`, with a CLI entry point. This is the one scaffold decision already settled by your answer, included here for completeness since everything else hangs off it.

```
CLAMP/
├── pyproject.toml
├── uv.lock
├── .env.example
├── src/
│   └── clamp/
│       ├── __init__.py
│       ├── config.py                # Settings (see §4)
│       ├── cli.py                   # typer app, entry point `clamp-data`
│       └── data/
│           ├── __init__.py
│           ├── schema.py            # PeptideRecord, enums (see §3)
│           ├── sources/
│           │   ├── __init__.py
│           │   ├── base.py          # ApiPuller / BulkDownloader ABCs
│           │   ├── dbaasp.py
│           │   ├── dramp.py
│           │   ├── hemolytik2.py
│           │   ├── hemopi2.py
│           │   └── qmap.py
│           ├── convert/
│           │   ├── __init__.py
│           │   ├── rdkit_backbone.py
│           │   ├── p2smi_adapter.py
│           │   ├── helm.py          # stub; doc 06 §7.3 flags this as an open spike
│           │   └── validate.py      # round-trip + mass-sanity checks
│           ├── dedup.py
│           ├── normalize.py
│           ├── datasheet.py
│           └── pipeline.py          # orchestrator (see §10)
├── tests/
│   ├── conftest.py
│   └── data/
│       ├── fixtures/                # captured sample raw records per source
│       ├── test_schema.py
│       ├── test_convert.py
│       ├── test_dedup.py
│       ├── test_normalize.py
│       └── test_sources_dbaasp.py
└── data/                            # gitignored artifact root (see §13)
    ├── raw/{source}/{id}.json
    ├── interim/
    ├── processed/
    └── datasheet/
```

Why `src/`-layout over a flat `clamp/` package at repo root: prevents accidentally importing an uninstalled local copy during tests (a classic footgun with flat layouts + pytest), and it's the layout `uv init --package` scaffolds by default, so tooling friction is near zero.

---

## 2. Environment & tooling

**`uv` + `pyproject.toml`**, per your call. Sketch of the dependency groups:

```toml
[project]
name = "clamp"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "requests>=2.31",
    "tenacity>=8.2",          # retry/backoff for the DBAASP pull (§5)
    "pydantic>=2.6",
    "pydantic-settings>=2.2",
    "pandas>=2.2",
    "rdkit>=2023.9",          # pip wheels available, no conda needed on 3.11/3.12
    "p2smi>=1.1.1",
    "typer>=0.12",
    "tqdm>=4.66",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "responses>=0.25",        # mock DBAASP HTTP calls in tests (§12)
    "hypothesis>=6.100",      # property-based SMILES round-trip tests (§12)
    "ruff>=0.4",
]

[project.scripts]
clamp-data = "clamp.cli:app"
```

`rdkit` now ships real pip wheels (the `rdkit` PyPI package, not the old `rdkit-pypi` shim), so conda is not required — worth confirming once against your actual Python version before committing, since RDKit wheel availability lags new CPython releases by a few months.

---

## 3. Data model (`schema.py`)

This is the executable form of doc 06 §4.1's unified schema table. **Recommendation: `pydantic.BaseModel`**, not a plain dataclass or a raw dict, because (a) every source hands you heterogeneous, partially-missing JSON/CSV and pydantic's validation-on-construction turns "silently wrong field" bugs into loud ones at ingestion time, and (b) `model_dump()` gives you a clean path to a DataFrame for the tabular stages (§7–9) without hand-rolled serialization.

```python
from enum import StrEnum
from pydantic import BaseModel, Field

class Source(StrEnum):
    DBAASP = "dbaasp"
    DRAMP = "dramp"
    APD3 = "apd3"
    HEMOPI2 = "hemopi2"
    HEMOLYTIK2 = "hemolytik2"
    QMAP = "qmap"

class CyclizationType(StrEnum):
    HEAD_TO_TAIL = "HT"
    SIDE_CHAIN_TO_SIDE_CHAIN = "SCSC"
    SIDE_CHAIN_TO_N_TERM = "SCNT"
    SIDE_CHAIN_TO_C_TERM = "SCCT"
    DISULFIDE = "SS"
    NONE = "none"
    UNKNOWN = "unknown"

class SmilesSource(StrEnum):
    DB_CURATED = "db_curated"
    RDKIT_SEQUENCE = "rdkit_sequence"
    P2SMI = "p2smi"
    HELM_TOOLCHAIN = "helm_toolchain"
    PLAIN_BACKBONE_FALLBACK = "plain_backbone_fallback"

class FidelityTier(StrEnum):
    FULL = "full"
    PARTIAL = "partial"
    BACKBONE_ONLY = "backbone_only"
    FAILED = "failed"

class UnusualResidue(BaseModel):
    position: int
    from_residue: str | None = None
    modification_type: str

class PeptideRecord(BaseModel):
    # identity / provenance
    source: Source
    source_id: str
    sequence_raw: str
    sequence_canonical: str | None = None       # filled by dedup.py
    peptide_uid: str | None = None               # filled by dedup.py

    # structure
    is_multimer: bool = False
    is_cyclic: bool = False
    cyclization_type: CyclizationType = CyclizationType.NONE
    nterm_mod: str | None = None
    cterm_mod: str | None = None
    unusual_residues: list[UnusualResidue] = Field(default_factory=list)

    # chemistry (filled by convert.py)
    smiles: str | None = None
    smiles_source: SmilesSource | None = None
    smiles_validated: bool = False
    chemical_fidelity_tier: FidelityTier | None = None
    dropped_modifications: list[str] = Field(default_factory=list)

    # labels (raw, pre-normalization)
    hc50_value: float | None = None
    hc50_unit: str | None = None
    hc50_assay_target_cell: str | None = None
    mic_value: float | None = None
    mic_unit: str | None = None
    mic_target_species: str | None = None
    mic_assay_medium: str | None = None
    label_is_range: bool = False
    label_range_raw: str | None = None

    # labels (filled by normalize.py)
    hc50_value_uM: float | None = None
    hc50_log_uM: float | None = None
    mic_value_uM: float | None = None
    mic_log_uM: float | None = None

    reference: str | None = None
    duplicate_of: str | None = None
```

One deviation from doc 06's table worth flagging explicitly: doc 06 lists `hc50_value`/`mic_value` as single columns per row, with "one row per (peptide-construct, assay-record)" as the stated convention (doc 06 §4.1). This model matches that — a DBAASP record with two MIC entries against two species becomes two `PeptideRecord`s sharing a `peptide_uid`, not one record with a list field. Keep that convention; it's what makes the masked-multitask loss in doc 04 work cleanly downstream.

---

## 4. Config

**Two real options:**

| | Plain `dataclass` + YAML | `pydantic-settings` |
|---|---|---|
| Env var overrides | manual | built-in (`.env` + env vars, free) |
| Validation | manual | free, same as `schema.py` |
| Extra dependency | none | already a dependency (§2) |
| Familiarity for a solo/small-team research repo | very low ceremony | slightly more structure |

**Recommendation: `pydantic-settings`.** You're already taking the pydantic dependency for `schema.py`; a second validated settings object costs nothing extra and gives free `.env` support for things you'll actually want out of source control (nothing secret here today, but the DBAASP pull's polite request rate, cache directory, and per-source enable/disable flags are exactly the kind of thing you'll want to override per-machine without editing code).

```python
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    data_root: Path = Path("data")
    dbaasp_base_url: str = "https://dbaasp.org"
    dbaasp_requests_per_second: float = 3.0   # doc 06 §1.3: "start conservative"
    dbaasp_page_size: int = 200
    http_timeout_s: float = 30.0

    model_config = {"env_prefix": "CLAMP_", "env_file": ".env"}

settings = Settings()
```

---

## 5. Module: `data/sources/` (pullers)

Doc 06 §3 makes clear the five sources split into two genuinely different access patterns — worth modeling as two base classes rather than forcing one interface on both:

- **`ApiPuller`** (DBAASP only): paginate an id list, then fetch full records one-by-one, resumable via a per-id cache file. This is the ~2–3 hour, ~25k-request job.
- **`BulkDownloader`** (DRAMP, Hemolytik2, HemoPI2, QMAP): fetch one or a handful of files (zip/xlsx/csv), unpack, parse. Minutes, not hours; no per-record rate limiting concern.

```python
# sources/base.py
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

class PullReport(BaseModel):
    source: Source
    attempted: int
    succeeded: int
    skipped_cached: int
    failed: list[str]        # ids/files that errored, for a rerun

class ApiPuller(ABC):
    source: Source

    @abstractmethod
    def total_count(self) -> int: ...

    @abstractmethod
    def iter_ids(self) -> Iterator[str]: ...

    @abstractmethod
    def fetch_record(self, record_id: str) -> dict: ...

    def pull(self, cache_dir: Path, resume: bool = True) -> PullReport:
        """Shared resumable cache-to-disk loop: rate-limited via tenacity,
        skips ids already on disk when resume=True, writes raw JSON keyed by id."""
        ...

class BulkDownloader(ABC):
    source: Source

    @abstractmethod
    def download_urls(self) -> list[str]: ...

    @abstractmethod
    def parse(self, downloaded_paths: list[Path]) -> list[PeptideRecord]: ...

    def pull(self, cache_dir: Path) -> PullReport:
        """Downloads each URL if not already cached, then delegates to parse()."""
        ...
```

`DbaaspPuller(ApiPuller)` implements `iter_ids` via the `/peptides?limit=200&offset=N` pagination doc 06 §1.3 specifies, and `fetch_record` via `/peptides/{id}`. Rate limiting: `tenacity.retry` with exponential backoff on non-200s (doc 06 §7.1 flags that DBAASP's rate limit is undocumented — treat any 429/5xx as a signal to back off, not just retry immediately), wrapped around a simple token-bucket sleep at `settings.dbaasp_requests_per_second`.

Each source module additionally owns **its own** raw-JSON/row → `PeptideRecord` mapping — this is where the "map DBAASP's `intrachainBonds[].cycleType` to your `CyclizationType` enum" type of adapter logic lives (doc 06 §2.1's `p2smi` taxonomy match), kept local to the source that needs it rather than centralized, since each source's raw schema is genuinely different.

**Open item carried from doc 06 §7.1, restated as a Phase-1 blocker, not a nice-to-have:** read DBAASP's data-usage-policy PDF before running `DbaaspPuller.pull()` at full scale. This gates the whole pull, not just a footnote — put it as the literal first task in the Phase 1 runbook (§14).

---

## 6. Module: `data/convert/`

Implements the decision table from doc 06 §2.2 as an explicit dispatch, not a chain of `if` statements buried in one function — makes each path independently testable (§12) and makes the fidelity-tier bookkeeping (doc 06 §5.1) fall out of *which path ran*, rather than being inferred after the fact.

```python
# convert/__init__.py
from clamp.data.schema import PeptideRecord, SmilesSource, FidelityTier

class ConversionResult(BaseModel):
    smiles: str | None
    smiles_source: SmilesSource
    fidelity_tier: FidelityTier
    dropped_modifications: list[str]
    validated: bool

def convert_record(record: PeptideRecord, curated_smiles: str | None = None) -> ConversionResult:
    if curated_smiles is not None:
        return _validate(curated_smiles, SmilesSource.DB_CURATED, record)
    if record.is_cyclic or record.unusual_residues or record.nterm_mod or record.cterm_mod:
        return _convert_via_p2smi(record)          # convert/p2smi_adapter.py
    return _convert_via_rdkit(record)               # convert/rdkit_backbone.py

def _validate(smiles: str, source: SmilesSource, record: PeptideRecord) -> ConversionResult:
    """Round-trip parse + mass-sanity check, doc 06 §2.3."""
    ...
```

`validate.py` implements doc 06 §2.3's two checks as standalone, pure, easily-unit-tested functions:

```python
# convert/validate.py
from rdkit import Chem
from rdkit.Chem import Descriptors

def round_trips(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol))) == Chem.MolToSmiles(mol)

def mass_sanity_check(smiles: str, expected_mass: float, tolerance_frac: float = 0.05) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return abs(Descriptors.MolWt(mol) - expected_mass) / expected_mass <= tolerance_frac
```

`p2smi_adapter.py` wraps the FASTA-header-constraint-string convention `p2smi` expects (doc 06 §2.1: `X`/`C`/`N`/`Z` position tags), building that header from `PeptideRecord.cyclization_type` + `unusual_residues`. `helm.py` is a stub function that raises `NotImplementedError("HELM toolchain not yet selected — see doc 06 §7.3")` — deliberately not built until the "short hands-on spike" doc 06 recommends has actually happened; wiring a real dependency in for a path you haven't chosen yet is exactly the kind of premature work worth avoiding.

---

## 7. Module: `data/dedup.py`

Direct implementation of doc 06 §4.3:

```python
import hashlib

def canonicalize_sequence(sequence_raw: str, source: Source) -> str:
    """Strip/uppercase + resolve source-specific non-standard 1-letter codes
    to a common vocabulary. The vocabulary itself is the ad hoc, source-specific
    part doc 06 §4.3 step 1 warns about — keep the mapping table per-source,
    not global, and grow it as real data surfaces gaps."""
    ...

def modification_signature(record: PeptideRecord) -> tuple:
    return (
        record.nterm_mod,
        record.cterm_mod,
        tuple(sorted((r.position, r.modification_type) for r in record.unusual_residues)),
        record.cyclization_type,
    )

def compute_peptide_uid(sequence_canonical: str, mod_sig: tuple) -> str:
    key = f"{sequence_canonical}|{mod_sig}".encode()
    return hashlib.blake2b(key, digest_size=16).hexdigest()

def dedup_records(records: list[PeptideRecord]) -> "DedupResult":
    """Groups by peptide_uid, applies the metadata-richness tie-break from
    doc 06 §4.3 step 5 (richer record's smiles/fidelity_tier wins on a
    sequence-only fuzzy match), and returns both the deduped set and an
    overlap matrix (source x source counts) for the datasheet."""
    ...
```

Keep `dedup_records` returning **both** the deduped rows and enough bookkeeping (which raw rows collapsed into which `peptide_uid`, per-pair source overlap counts) to drive the datasheet's dedup summary (doc 06 §4.4) — don't discard that provenance once dedup runs, since it's exactly the number needed to sanity-check "did DBAASP's copy of a peptide leak into a different split fold than DRAMP's copy of the same peptide," which is the concrete leakage risk doc 06 §4.3 step 4 calls out.

---

## 8. Module: `data/normalize.py`

```python
import math

def ug_per_ml_to_uM(value_ug_ml: float, molecular_weight: float) -> float:
    return (value_ug_ml / molecular_weight) * 1000

def parse_range(raw: str) -> tuple[float, float]:
    lo, hi = (float(x) for x in raw.replace(" ", "").split("-"))
    return lo, hi

def geometric_mean(lo: float, hi: float) -> float:
    return math.sqrt(lo * hi)

def normalize_concentration(
    value: float | None,
    unit: str,
    molecular_weight: float,
    is_range: bool = False,
    range_raw: str | None = None,
) -> float | None:
    """doc 06 §4.2: convert to uM using SMILES-derived MW (not residue-count
    estimate), collapsing ranges via geometric mean per the documented
    convention. Returns None (not 0 or NaN) for missing input — the log
    transform downstream must be able to distinguish 'no label' from
    'label is exactly zero', which log(0) would silently break anyway."""
    if is_range and range_raw:
        lo, hi = parse_range(range_raw)
        value = geometric_mean(lo, hi)
    if value is None:
        return None
    if unit.lower() in ("ug/ml", "µg/ml", "ug/mL"):
        value = ug_per_ml_to_uM(value, molecular_weight)
    elif unit.lower() not in ("um", "µm", "uM"):
        raise ValueError(f"unrecognized unit: {unit!r}")
    return value

def log_transform(value_uM: float | None) -> float | None:
    return math.log10(value_uM) if value_uM is not None else None
```

The molecular weight input here is deliberately **the RDKit-computed MW of the final generated SMILES** (`Descriptors.MolWt`), not a residue-count estimate — doc 06 §4.2 flags this as the detail that most affects exactly the modified/cyclic peptides the whole SMILES-LM approach exists to handle correctly, so wire `normalize_concentration` to consume `convert_record`'s output, not the raw sequence.

---

## 9. Module: `data/datasheet.py`

A reporting module, not a data-transform one — takes the deduped, normalized DataFrame (or the intermediate `PullReport`/`DedupResult`/fidelity-tier counts) and emits the four tables doc 06 §4.4 specifies: per-source coverage, species/target-cell frequency, fidelity breakdown, dedup/overlap summary. **Recommendation:** emit both a machine-readable `datasheet.json` (so later phases and CI-style checks can assert against it, e.g. "fidelity `full` tier must be ≥ some threshold before Phase 2 starts") and a human-readable `datasheet.md` rendered from the same underlying numbers — don't hand-author the markdown separately from the JSON, or they'll drift.

---

## 10. Orchestration & idempotency (`pipeline.py`)

**Three real options**, roughly in order of how much machinery they bring:

1. **Plain functions + a `manifest.json`** — each pipeline stage is a function `stage(input_paths) -> output_path`, `pipeline.py` calls them in sequence, and a manifest records `{stage: {inputs_hash, output_path, completed_at}}` so a rerun skips a stage whose inputs haven't changed. No new dependency.
2. **`dvc` (Data Version Control) pipelines** — declarative `dvc.yaml` stages with automatic dependency-hash-based caching, plus the side benefit of actual data versioning (doc 02 P0 explicitly wants "a versioned dataset"). Adds a real dependency and a bit of learning curve.
3. **A workflow engine (Prefect/Luigi)** — overkill at this scale (one machine, five sources, a handful of sequential stages); this is built for orchestrating distributed, scheduled, or human-in-the-loop pipelines, none of which apply here.

**Recommendation: start with option 1.** The pipeline is five pull stages → convert → validate → dedup → normalize → datasheet — a linear DAG with no branching, no scheduling, no distribution. A manifest-file approach gets you the actually-needed property (rerun the script tomorrow, don't re-pull 25k DBAASP records you already have) with zero new dependencies. **Revisit `dvc` specifically** (not Prefect/Luigi) the moment "versioned dataset" from doc 02's P0 deliverable becomes a real requirement rather than a nice property — dvc's dependency-hash caching and its data-versioning story solve the same problem in one tool, so it's a natural single upgrade rather than a rewrite, not a "maybe someday, maybe not" open-ended option.

```python
# pipeline.py
from dataclasses import dataclass

@dataclass
class Stage:
    name: str
    run: callable
    inputs: list[str]
    output: str

STAGES = [
    Stage("pull_dbaasp", ..., inputs=[], output="raw/dbaasp/"),
    Stage("pull_dramp", ..., inputs=[], output="raw/dramp/"),
    Stage("pull_hemolytik2", ..., inputs=[], output="raw/hemolytik2/"),
    Stage("pull_hemopi2", ..., inputs=[], output="raw/hemopi2/"),
    Stage("pull_qmap", ..., inputs=[], output="raw/qmap/"),
    Stage("convert", ..., inputs=["raw/*"], output="interim/converted.parquet"),
    Stage("dedup", ..., inputs=["interim/converted.parquet"], output="interim/deduped.parquet"),
    Stage("normalize", ..., inputs=["interim/deduped.parquet"], output="processed/dataset.parquet"),
    Stage("datasheet", ..., inputs=["processed/dataset.parquet"], output="datasheet/"),
]

def run_pipeline(stages: list[Stage] = STAGES, force: bool = False):
    """Loads/updates manifest.json, skips a stage if its inputs are unchanged
    and force=False, otherwise runs it and updates the manifest."""
    ...
```

**Dataframe library for the tabular stages (convert onward):** pandas vs. polars. At this scale (tens of thousands of rows, not millions), either is fine performance-wise — this is an ecosystem-fit question, not a speed one. **Recommendation: pandas**, because RDKit interop examples, `Descriptors.MolWt`-style per-row apply patterns, and the rest of this project's tooling (transformers/HF `datasets`, the multitask training code in doc 04) are all pandas-native in the ecosystem you're already pulling from; introducing polars buys speed you don't need yet at the cost of a second dataframe idiom to context-switch into.

---

## 11. CLI

**Options: `argparse` (stdlib) vs. `click` vs. `typer`.** All three can express "a handful of subcommands with a few flags each" without strain. **Recommendation: `typer`** — it derives the CLI directly from type-hinted function signatures, which pairs naturally with the pydantic models already used throughout (`schema.py`, `config.py`), and its subcommand grouping maps cleanly onto "one subcommand per pipeline stage plus one `run-all`."

```python
# cli.py
import typer
from clamp.data.pipeline import run_pipeline, STAGES

app = typer.Typer()

@app.command()
def pull(source: str, force: bool = False):
    """Run a single source's pull stage, e.g. `clamp-data pull dbaasp`."""
    ...

@app.command()
def convert(force: bool = False): ...

@app.command()
def dedup(force: bool = False): ...

@app.command()
def normalize(force: bool = False): ...

@app.command()
def datasheet(): ...

@app.command(name="run-all")
def run_all(force: bool = False):
    """The full Phase 1 pipeline, doc 08's Phase 1 bullets end to end."""
    run_pipeline(STAGES, force=force)
```

---

## 12. Testing plan

Per your call to include a lightweight testing story: focus on the **pure, high-risk logic**, not integration coverage of live network calls.

| Target | Approach | Why this one |
|---|---|---|
| `convert/validate.py` (`round_trips`, `mass_sanity_check`) | `hypothesis` property-based tests generating/mutating known-valid SMILES | This is precisely the class of bug doc 02 §5.2 flags already happened once (v1's ring-numbering bug) — property-based testing is a good fit because the property ("valid SMILES round-trips") is easy to state and hard to enumerate by hand. |
| `dedup.py` (`compute_peptide_uid`, `modification_signature`, `dedup_records`) | Plain `pytest` unit tests with small synthetic `PeptideRecord` lists, including the doc 06 §4.3 step 5 "metadata-rich record should win" case explicitly | This is the leakage-prevention logic — worth locking down with an explicit regression test for the exact scenario doc 06 warns about (same peptide, different source, different metadata richness). |
| `normalize.py` (`normalize_concentration`, `parse_range`, `log_transform`) | Parametrized `pytest` cases: µM passthrough, µg/mL conversion, range strings, `None` handling, unrecognized-unit error | Small pure functions, cheap to fully cover; unit-conversion bugs are exactly the "quiet, systematic, hard to notice later" class of error doc 06 §4.2 calls out. |
| `sources/dbaasp.py` (`DbaaspPuller`) | `responses` library to mock the `/peptides` and `/peptides/{id}` endpoints against captured fixture JSON (reuse the shape from doc 06's live pilot pull, §5.3) | Never hit the live API in CI/tests — validates pagination, resume-skip-cached logic, and 429/backoff handling without the ~2–3hr real pull or DBAASP ToS concerns. |
| `convert/p2smi_adapter.py` | Unit tests against a handful of hand-picked real DBAASP records spanning the 5 `p2smi` cyclization types (doc 06 §2.1) | Confirms the DBAASP-bond-annotation → `p2smi` FASTA-header-tag mapping doc 06 calls "a small, well-scoped adapter script" actually is one, on real inputs. |
| `schema.py` | A handful of pydantic validation tests (missing required fields raise, enum values are constrained) | Cheap; pydantic does the real work, tests just confirm the model is wired the way you intend. |

**Fixtures:** capture 5–10 real records per source (already-anonymized/public data, so no concern there) into `tests/data/fixtures/{source}/*.json` once, during initial development — this both seeds the `responses` mocks and doubles as the input corpus for the `convert`/`dedup`/`normalize` unit tests, so you're testing against real DBAASP/DRAMP/etc. shapes rather than hand-invented ones that might not match reality.

**Explicitly not covered by this test plan:** the live DBAASP pull's actual 2–3hr behavior, DRAMP/Hemolytik2/HemoPI2/QMAP's real bulk-file parsing end-to-end, and RDKit/`p2smi` correctness on the full real dataset. Those need to be exercised by actually running `clamp-data run-all` once against live sources — no amount of unit testing substitutes for that, and it doesn't need to happen more than once per pipeline-logic change.

---

## 13. Storage layout & data versioning

Local filesystem only, per your call. Concrete layout under `data/` (gitignored):

```
data/
├── raw/
│   ├── dbaasp/{peptide_id}.json          # one file per DBAASP record, resumable cache
│   ├── dramp/*.xlsx                      # bulk files, as downloaded
│   ├── hemolytik2/*
│   ├── hemopi2/*
│   └── qmap/*
├── interim/
│   ├── converted.parquet                 # post-convert, pre-dedup
│   └── deduped.parquet
├── processed/
│   └── dataset.parquet                   # final Phase 1 deliverable
├── datasheet/
│   ├── datasheet.json
│   └── datasheet.md
└── manifest.json                          # pipeline stage cache (§10)
```

Parquet over CSV for the `interim`/`processed` tables: preserves dtypes (in particular the `list[UnusualResidue]`-shaped nested fields) without the string-serialization round-trip CSV would force, and both pandas and polars read/write it natively if you revisit the dataframe choice later.

**Versioning without cloud storage or dvc (yet):** at minimum, stamp `processed/dataset.parquet` and `datasheet/datasheet.json` with a content hash and a `pulled_at` timestamp in the manifest, and treat re-running `run-all` after any source's data has changed as producing a **new** versioned output (`processed/dataset_v{n}.parquet`), not an overwrite — cheap insurance for reproducibility given doc 02 P0 explicitly asks for "a versioned dataset," and it costs nothing extra beyond a naming convention today even without dvc.

---

## 14. Runbook (execution order for Phase 1)

1. **Read DBAASP's data-usage-policy PDF** (doc 06 §1.3, §7.1) — blocking, do this before step 3.
2. `uv sync` to install the environment.
3. `clamp-data pull dbaasp` — expect ~2–3 hours; safe to interrupt and rerun, resumes from `raw/dbaasp/`.
4. `clamp-data pull dramp`, `pull hemolytik2`, `pull hemopi2`, `pull qmap` — bulk downloads, minutes each.
5. `clamp-data convert` — runs the full conversion decision table over everything pulled.
6. `clamp-data dedup`
7. `clamp-data normalize`
8. `clamp-data datasheet` — inspect `datasheet/datasheet.md` by hand before treating Phase 1 as done; this is where doc 06 §5's fidelity-tier and species/target-cell numbers get answered for real, superseding the n=48 pilot estimate.
9. Sanity-check against the doc 06 §6 P0 checklist and the Definition of Done below.

Equivalently, `clamp-data run-all` executes 3–8 in order via `pipeline.py`; running the subcommands individually is preferable the first time through, so a failure in (say) `convert` doesn't obscure whether the DBAASP pull actually finished cleanly.

---

## 15. Definition of done (Phase 1)

Directly from doc 06 §6 and doc 08's Phase 1 bullets:

- [ ] `raw/dbaasp/` contains all ~25k (or current `totalCount`) records, cached to disk.
- [ ] `raw/{dramp,hemolytik2,hemopi2,qmap}/` contain the bulk downloads.
- [ ] Every pulled record has run through `convert_record` and has a `smiles_source` + `chemical_fidelity_tier` recorded (`failed` is an allowed outcome, `null`/unset is not).
- [ ] `processed/dataset.parquet` has one row per (peptide-construct, assay-record), with `peptide_uid` populated and deduped per §7.
- [ ] All `hc50_value_uM`/`mic_value_uM`/`*_log_uM` fields are populated wherever a raw label existed, using SMILES-derived MW.
- [ ] `datasheet/` contains the four tables from doc 06 §4.4, generated from `processed/dataset.parquet`, not hand-written.
- [ ] `pytest` passes for the full suite in §12.
- [ ] The fidelity-tier and species/target-cell numbers from the datasheet have been read and used to actually answer doc 02's open questions #1 (lock HC50 to human RBC?) and #2 (chemical fidelity floor) — a generated datasheet nobody reads doesn't close those questions.

---

## 16. Open risks carried into implementation

Restated from doc 06 §7/§8 as concrete things that can break this specific scaffold, not just abstract caveats:

1. **DBAASP rate limit is undocumented** — `DbaaspPuller` must treat any 429 as "back off harder," not "retry immediately"; don't hardcode a fixed retry count assuming the conservative 3–4 req/s estimate is safe at full scale.
2. **dbAMP 3.0's bulk file contents are unverified** — this scaffold does not include a `sources/dbamp.py` module for that reason; add it only after someone has actually opened the downloaded file and confirmed its schema, per doc 06 §7.2. Don't build a parser against a guessed format.
3. **HELM toolchain is unselected** — `convert/helm.py` is intentionally a stub (§6). Do the short spike doc 06 §7.3 recommends against real pulled data before picking a library, not before.
4. **Licensing/ToS terms for DBAASP (and likely the other sources) are unread** — this is a blocking step in the runbook (§14, step 1), not a footnote.
5. **`manuallyEdited=false` DBAASP SMILES trust level is unresolved** (doc 06 §7.6) — `convert_record`'s `db_curated` path currently accepts any curated SMILES regardless of this flag; revisit once the full-scale fidelity measurement (§9) makes the size of this distinction visible.
