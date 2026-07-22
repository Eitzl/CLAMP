# Data directory: source schemas & deduplication strategy

This documents where each source's raw schema came from, how the Phase 1
pipeline reconciles five structurally different formats into one
`PeptideRecord` (`src/clamp/data/schema.py`), and — the part that isn't
mechanical — what happens at dedup time when two source records for what
looks like the same peptide carry different, partial, or conflicting
labels. See `MD_design_docs/06_task5_data_pipeline_plan.md` for the full
research trace and `MD_design_docs/09_phase1_implementation_design.md` for
the code scaffold this backs.

Directory layout: `raw/{source}/` (as-pulled, source-native format),
`interim/` (post-convert, post-dedup intermediate tables), `processed/`
(final `dataset.parquet`), `datasheet/` (generated coverage/fidelity
reports). See `.gitignore` — everything under `raw/`, `interim/`,
`processed/`, `datasheet/` is local-only and not committed; only this
README and the `.gitkeep` placeholders are tracked.

---

## 1. How each source's schema was determined

None of this was guessed from documentation prose alone — doc 06 checked
each source directly (live API calls, live pilot pulls, primary papers) on
2026-07-22, and flags anything it could *not* independently verify rather
than assuming it. That provenance matters for the pipeline design because
"verified against a live response" and "inferred from a paper's methods
section" carry different risk of the parser being wrong on day one.

| Source | How the schema was obtained | Verification depth |
|---|---|---|
| **DBAASP** | Live OpenAPI 3 spec at `GET https://dbaasp.org/v3/api-docs` (fetched directly, 14,912 bytes, decoded field-by-field), cross-checked against a live full-record pull (`GET /peptides/16`, "Distinctin chain 1") | High — spec + one real record confirmed the two-endpoint shape (`/peptides` list vs `/peptides/{id}` detail) and that the list endpoint is a thin summary lacking `smiles`/bonds/activities |
| **DRAMP 4.0** | Publication (*NAR* 2025, PMC11701585) + the bulk-download page's file listing (categorized XLSX/TXT/FASTA, including dedicated `*_smiles` files) | Medium — file *names* and stated row counts confirmed; exact column-level schema per file not yet parsed from a downloaded sample |
| **Hemolytik2** | Publication (bioRxiv 2025.05.12.653624) + download/API page description | Medium — stated to include SMILES, terminal mods, D-/L-stereochemistry, linear/cyclic flag; exact column names not yet confirmed against a downloaded file |
| **HemoPI2** | GitHub repo (`raghavagps/HemoPI2`) + PyPI package (`hemopi2`) + download page | Medium-high — package existence, size (1,926 peptides), and CSV/FASTA-in-zip format confirmed; not yet parsed |
| **QMAP** | GitHub repo (`anthol42/QMAP`) + PyPI package (`qmap-benchmark`) + preprint (bioRxiv 2026.02.03.703041 / *Sci. Rep.* 2026) | Medium — package is installable and ships ML-ready SMILES + splits per its docs; **the right integration point is likely the package's own data loader, not a raw file URL** (see `src/clamp/data/sources/qmap.py` docstring) — this needs confirming against the package API before `download_urls()`/`parse()` are implemented, since `QmapDownloader` may not fit the `BulkDownloader` url-list shape as cleanly as the other three |

**What this means for the code:** DBAASP's schema (`src/clamp/data/sources/dbaasp.py`) is trusted enough to implement against directly. DRAMP, Hemolytik2, and HemoPI2's parsers should be written *after* a first real bulk-file download, once actual column names are in hand — the docstrings currently describe the expected shape from doc 06, not a confirmed one. QMAP additionally needs a short spike into the package's Python API before its `BulkDownloader` shape is even the right abstraction.

---

## 2. Reconciling different formats into one schema

The five sources differ on every axis: DBAASP and Hemolytik2 expose REST
APIs returning nested JSON; DRAMP and HemoPI2 ship bulk XLSX/CSV/FASTA;
QMAP ships as a Python package with its own loader. Concretely, the
per-source `sources/*.py` modules are where format differences get
absorbed — nothing downstream of `parse()`/`to_records()` should need to
know which source a `PeptideRecord` came from except via the `source`
field itself. The specific reconciliation points:

- **Nesting → flat rows.** DBAASP's `targetActivities[]` and
  `hemoliticCytotoxicActivities[]` arrays each explode into separate
  `PeptideRecord` rows (one row per assay-record, doc 06 §4.1) sharing the
  same `peptide_uid` once computed. Bulk-file sources that are already
  roughly one-row-per-measurement don't need this explosion step.
- **Multimers.** DBAASP records with `complexity="Multimer"` have an
  *empty* top-level `sequence` — the real chains live in `monomers[]`.
  `DbaaspPuller.to_records` is responsible for un-nesting these (doc 06
  §1.2 measured this at ~10% of records in the pilot pull — silently
  reading the top-level `sequence` field would drop them as empty
  strings, not raise an error, which is the dangerous failure mode).
- **Units.** DBAASP's pilot pull found MIC/HC50 values roughly evenly
  split between µM and µg/mL *within the same source*, so unit handling
  can't be a per-source constant — it's read per-record from the
  `*_unit` field and normalized in `normalize.py` using the
  **RDKit-computed molecular weight of the final generated SMILES**
  (not a residue-count estimate — this matters most for exactly the
  modified/cyclic peptides this pipeline exists to handle correctly).
- **Range-valued labels.** DBAASP sometimes reports a concentration as a
  range (e.g. `"1.58-25.33"`) rather than a point value.
  `label_is_range`/`label_range_raw` on `PeptideRecord` preserve the raw
  string; `normalize.py` collapses it to a single value via geometric
  mean (documented convention, doc 06 §4.2) — the raw range stays on the
  row so the choice is auditable, not silently lossy.
- **Modification vocabulary.** Each source names non-canonical residues
  and terminal modifications differently (DBAASP's `unusualAminoAcids[]`
  naming vs. whatever DRAMP/Hemolytik2 use). There is no single global
  mapping table for this — `dedup.canonicalize_sequence` takes `source`
  as an explicit argument specifically so this vocabulary can grow
  per-source as real data surfaces gaps, rather than pretending one
  table covers everything from day one.
- **Curated vs. derived SMILES.** DBAASP and (per its description)
  Hemolytik2 both ship curated SMILES directly (`smiles_source =
  db_curated`, skipping conversion). DRAMP's dedicated `*_smiles` bulk
  files are the same idea. Where no curated SMILES exists, the
  `convert_record` dispatch (`src/clamp/data/convert/__init__.py`) is
  what standardizes generation across sources — RDKit for plain
  backbones, `p2smi` for anything cyclic/modified, regardless of which
  source the record came from.

---

## 3. Deduplication when labels differ across sources

This is the part worth spelling out beyond "hash the sequence" — the
`peptide_uid` in `dedup.py` is a **structural** identity key (canonical
sequence + modification signature), not an assertion that every row
sharing that key agrees on everything else. It won't. Concretely:

### 3.1 Identity key
`peptide_uid = hash(sequence_canonical, modification_signature)` where
`modification_signature = (nterm_mod, cterm_mod, sorted unusual_residues,
cyclization_type)`. Two records collapse to the same `peptide_uid` iff
they represent the same chemical entity — not iff every field matches.

### 3.2 Case: same peptide, same assay conditions, different sources, different values
E.g. DBAASP and DRAMP both report an MIC against *E. coli* for the same
peptide_uid, but the numbers don't match exactly. **These are treated as
independent replicate measurements, not merged or averaged.** Both rows
are kept, both tagged with the same `peptide_uid`. Reasoning: averaging
silently discards real signal about assay/measurement noise (relevant
given doc 06/QMAP's finding that HC50 in particular is noisy and
hard-to-predict); keeping both rows lets a later modeling stage decide
how to weight replicates rather than baking in a premature choice at the
data layer. The cost of this choice is that per-task label counts include
replicate rows, not just unique peptides — the datasheet (§4 below)
reports both raw-row and unique-`peptide_uid` counts so this isn't hidden.

### 3.3 Case: same peptide, different sources, complementary (non-overlapping) labels
E.g. DBAASP has HC50 for a peptide, Hemolytik2 has MIC for the *same*
peptide, neither has both. After dedup these simply coexist as separate
rows sharing `peptide_uid` — no special handling needed, because the
downstream masked-multitask design (`MD_design_docs/04_task5_multitask_architecture_plan.md`
§3) already treats per-task label presence as a per-row mask, not an
assumption that every row has every label. This is in fact the common
case doc 06 §3.5.1 describes ("MIC-labeled ≫ HC50-labeled ≫
both-labeled") and part of why the masked loss design exists.

### 3.4 Case: same peptide, differing metadata richness
E.g. DBAASP has full modification annotation (cyclization type, D-amino
acid positions) for a peptide; APD3's bulk FASTA has the same sequence
with no modification metadata at all (its bulk file is sequence-only, per
doc 06 §3). These collapse on sequence alone with
`metadata_source_conflict=true`-style flagging (tracked via
`dedup.DedupResult`, not a `PeptideRecord` field, since it's a property of
the *merge decision* rather than of either individual record) — the
metadata-richer record's `smiles`/`chemical_fidelity_tier` wins. A
metadata-poor duplicate must never silently downgrade a metadata-rich one
by being processed second.

### 3.5 Case: genuine conflicts (same assay conditions, contradictory values, likely a curation error)
Not yet distinguished from ordinary replicate noise (§3.2) by the current
scaffold — this is an open gap, not a solved case. A large discrepancy
(e.g. one source reports MIC = 2 µM, another reports MIC = 200 µM for the
same peptide against the same species/medium) is more likely a unit or
transcription error upstream than real biological replicate variance, and
averaging or keeping-both-uncritically both mishandle it. **Planned
resolution:** `dedup_records` should compute an intra-`peptide_uid`,
intra-assay-type dispersion statistic (e.g. flag when replicate values for
the same `(peptide_uid, task, species/target_cell)` span more than an
order of magnitude) and surface it as a `label_quality_flag` on the
affected rows for manual review during the datasheet pass (§4), rather
than either auto-resolving it or letting it pass through silently. Not yet
implemented in `dedup.py` — tracked here so it isn't lost.

### 3.6 Why this matters beyond data cleanliness: split safety
Every row sharing a `peptide_uid` — regardless of source, and regardless
of which of §3.2-3.5 applies — **must land in the same train/test cluster
fold** in Phase 2. If DBAASP's copy of a peptide ends up in train and
DRAMP's copy of the same peptide (same `peptide_uid`) ends up in test,
the homology-aware split's entire purpose (preventing leakage) is
defeated by a dedup-stage bug, not a splitting-stage one — this is why
dedup happens *before* clustering/splitting in the pipeline order, and why
`DedupResult.overlap_matrix` exists: it's the number that lets you check
"how much would leak" before it does.

### 3.7 Datasheet reporting
The dedup summary (`MD_design_docs/06_task5_data_pipeline_plan.md` §4.4,
`src/clamp/data/datasheet.py`) reports, per the above: raw row count in,
unique `peptide_uid` count out, the source-by-source overlap matrix, and
(once §3.5 is implemented) a count of flagged label conflicts — so the
scale of every case above is a visible number before Phase 2 starts, not
an assumption.

---

## 4. Open items

- DRAMP/Hemolytik2/HemoPI2 column-level schemas: confirm against a real
  downloaded file before finalizing their `parse()` implementations.
- QMAP integration shape: confirm whether `qmap-benchmark`'s own loader
  API is a better fit than `BulkDownloader` before implementing
  `sources/qmap.py`.
- Conflict-flagging (§3.5) is designed but not implemented in `dedup.py`
  yet.
- dbAMP 3.0 is deliberately **not** in this pipeline (no `sources/dbamp.py`
  module) — its bulk file contents were never confirmed (doc 06 §7.2);
  add it only after someone has opened a real downloaded file.
