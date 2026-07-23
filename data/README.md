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
| **DBAASP** | Live OpenAPI 3 spec at `GET https://dbaasp.org/v3/api-docs`, cross-checked against several live full-record pulls (`GET /peptides/1` [multimer], `/16` [monomer w/ MIC], `/13` [unusual residues], `/105` [head-to-tail bond]), captured as fixtures under `tests/data/fixtures/dbaasp/` | High — spec + 4 real records of different shapes; implemented in `sources/dbaasp.py` and covered by `tests/data/test_sources_dbaasp.py`. One residual gap: no live example of a disulfide/side-chain bond was found in the spike, so `_CYCLE_TYPE_MAP`/`_BOND_TYPE_MAP` there only confirm the head-to-tail (`cycleType.name == "NCB"`) and disulfide (`type.name == "DSB"`, cross-confirmed via QMAP's own bond vocabulary) mappings — everything else falls back to `UNKNOWN` rather than guessing. |
| **DRAMP 4.0** | Real bulk file downloaded and parsed live (2026-07-22): `general_amps.txt` (29 tab-separated columns, 11,687 rows) + `general_smiles.txt` (SMILES-only, joined by `DRAMP_ID`) | High for the schema itself — confirmed columns, confirmed only ~44% of `general_amps.txt` rows carry inline SMILES (the rest need the join). **Still Medium for label extraction**: `Hemolytic_activity`/`Activity`/`Linear-Cyclic-Branched`/terminal-mod columns are free text ("Cyclic (very possibly)", "Not metioned clearly" [sic]) — `sources/dramp.py` deliberately does not attempt to parse HC50/MIC values out of them, treating DRAMP as a SMILES/sequence source only. The real download path is also non-obvious: the site's own Downloads page links to `/download.php?filename=...` relative to `/downloads/`, so the actual working URL is `/downloads/download.php?...` — the bare `/download.php` guess 404s. |
| **Hemolytik2** | Real REST API called live (2026-07-22): `GET https://webs.iiitd.edu.in/raghava/hemolytik2/api/api.php?dataType=nature&dataValue=...` (note the `/raghava/` path segment — `.../hemolytik2/api/...` without it 404s) | High for the fields that exist — confirmed `id, pmid, year, seq, name, cter, nter, lyn_cyc, ldmix, non_nat, length, nature, activity, source, origin, exp_str, non_hem`, captured as a fixture. **No SMILES field from this endpoint** (contrary to the docs' description of the web UI) — Hemolytik2 records go through `convert.py` like any sequence-only source. **Coverage is not confirmed exhaustive**: there is no "list all" endpoint, so `sources/hemolytik2.py` enumerates a documented-but-possibly-incomplete list of `nature` facet values and unions the results — cross-check the datasheet's Hemolytik2 row count against the ~13,215-entry figure from the source paper before trusting full coverage. |
| **HemoPI2** | Real GitHub repo files downloaded and parsed live (2026-07-22): `raghavagps/HemoPI2/Dataset/{cross_val,independent}_dataset.csv`, 1541 + 387 = 1928 rows | High — confirmed exact 3-column schema (`SEQUENCE, μM, label`; the μM column uses GREEK SMALL LETTER MU, U+03BC, not the MICRO SIGN — a real gotcha, caught by testing against the actual file bytes). **Important correction from the original plan**: the pip package `hemopi2` does **not** ship this data — it contains only a pretrained classifier/regressor + feature-encoding tables. It is deliberately not a pyproject dependency; `sources/hemopi2.py` fetches the raw GitHub CSVs directly instead. |
| **QMAP** | Real package API called live (2026-07-22): `qmap.benchmark.dataset.dataset.DBAASPDataset()`, which pulls a cleaned, pre-aggregated DBAASP-derived JSON from HuggingFace Hub (`anthol42/qmap_benchmark_2025/dbaasp.json`) | High — confirmed exactly as doc 06/09 speculated: "the right integration point is the package's own data loader, not a raw file URL." Live-tested: 18,033 samples. Confirmed `Sample`/`Bond`/`Target`/`HemolyticActivity` shapes from the installed package source. **One real bug this caught**: `Bond.src`/`Bond.dst` positions are 1-indexed, not 0-indexed as first assumed — found because a real captured sample had `dst == sequence_length`, which is out of range for 0-indexed positions. **Discrepancy to keep in mind**: QMAP's `Target.consensus`/`HemolyticActivity.consensus` values are already aggregated across whatever DBAASP replicates existed for that (peptide, species) pair — unlike our own replicate-preserving convention (§3.2 below), so a QMAP row for the "same" measurement won't necessarily match a directly-pulled DBAASP row numerically even though both are DBAASP-derived. `qmap-benchmark` is a real pyproject dependency (it's genuinely needed at runtime, unlike `hemopi2`). |

**What this means for the code:** all five sources are now implemented in `src/clamp/data/sources/` against confirmed real schemas (see `MD_design_docs/09_phase1_implementation_design.md` and the per-module docstrings for exact detail), backed by real fixtures under `tests/data/fixtures/{source}/` and real tests under `tests/data/test_sources_*.py`. The residual open items are narrower than "schema unconfirmed" now — see §4.

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
  *empty* top-level `sequence` — the real chains live in `monomers[]`
  (doc 06 §1.2 measured this at ~10% of records in the pilot pull —
  silently reading the top-level `sequence` field would drop them as
  empty strings, not raise an error, which is the dangerous failure
  mode). The original plan here was for `DbaaspPuller.to_records` to
  un-nest `monomers[]` itself; a live check of a real Multimer record
  (id=1, "Distinctin") changed that plan: each nested monomer (e.g. id
  16, 36) is **also** independently reachable as its own top-level id
  with its own full activity data — that's how "Distinctin chain 1" (id
  16) was found as a plain monomer example in the first place. Unnesting
  in `to_records` would therefore double-count those chains against the
  standalone pull. `to_records` now emits nothing for the complex-level
  record itself (see §4's open item on what that costs — the complex's
  own label data, when it has any, is currently dropped, not
  duplicated).
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
averaging or keeping-both-uncritically both mishandle it. **Implemented**
in `dedup.flag_high_dispersion_replicates` (called from `dedup_records`):
groups rows by `(peptide_uid, task, species/target_cell)`, and when
replicate values within a group span more than
`REPLICATE_DISPERSION_RATIO_THRESHOLD` (10x, a first-pass threshold — see
that constant's docstring), tags every row in the group with
`label_quality_flag=HIGH_REPLICATE_DISPERSION`. Rows are never dropped or
averaged — only flagged for the datasheet pass and manual review. Covered
by `tests/data/test_dedup.py::TestConflictDetection`, including the case
where two different species' values must *not* be pooled into the same
dispersion check.

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

All five sources are now implemented against real, live-confirmed
schemas (§1), and dedup's conflict-flagging (§3.5) is implemented. What's
still genuinely open, carried forward rather than hidden:

- **DBAASP cyclization-type mapping is incomplete.** Only head-to-tail
  (`cycleType.name == "NCB"`) and disulfide (`type.name == "DSB"`) are
  confirmed against live examples; `SIDE_CHAIN_TO_SIDE_CHAIN`/`_N_TERM`/
  `_C_TERM` records will map to `CyclizationType.UNKNOWN` until a real
  example of each is seen at full-pull scale and the mapping tables in
  `sources/dbaasp.py` are extended.
- **DBAASP multimer complex-level labels are dropped.** A Multimer-type
  record's own top-level `targetActivities` (confirmed live: DBAASP id=1,
  "Distinctin", carries 26 of its own) describe the multi-chain complex,
  which our single-chain `PeptideRecord`/SMILES model can't represent —
  `to_records` emits nothing for the complex-level record itself, relying
  on each chain being independently pullable as its own id. This may
  undercount labels for peptides whose primary characterization is at the
  complex level; revisit if the datasheet shows this is a meaningful
  fraction.
- **Hemolytik2 coverage is enumeration-based, not confirmed exhaustive** —
  no "list all" endpoint was found; cross-check the datasheet's row count
  against the ~13,215-entry figure from the source paper.
- **Hemolytik2's `activity` free-text parser** (`_parse_activity`) only
  extracts clean `<LABEL> <op>= <value><unit>` forms; percentage-at-a-dose
  reports ("77% hemolysis at 100 μM") are skipped entirely rather than
  curve-fit into a point HC50, and inequality-qualified values (">200")
  are kept as a rough point rather than modeled as censored — the schema
  has no censored-value representation yet.
- **DRAMP is treated as SMILES/sequence-only** — its `Hemolytic_activity`/
  `Activity` columns are free text and not currently parsed into
  structured labels; revisit if a real column-level pass judges the
  parsing effort worthwhile.
- **QMAP's `AMD` bond type is ambiguous** between our `HEAD_TO_TAIL` and
  side-chain cyclization types beyond the terminal-position heuristic in
  `sources/qmap.py::_infer_cyclization`; non-terminal `AMD` bonds map to
  `UNKNOWN`.
- **HELM toolchain** (`convert/helm.py`) remains an intentional stub —
  unchanged by this session's work, still pending the short spike doc 06
  §7.3 recommends.
- **`manuallyEdited=false` DBAASP SMILES trust level** (doc 09 §16 item 5)
  remains unresolved — `convert_record`'s `db_curated` path accepts any
  curated SMILES DBAASP lists first, regardless of this flag.
- dbAMP 3.0 is deliberately **not** in this pipeline (no `sources/dbamp.py`
  module) — its bulk file contents were never confirmed (doc 06 §7.2);
  add it only after someone has opened a real downloaded file.
- **DBAASP has a third `complexity` value, `"Multi-Peptide"`**, found via
  a real pilot pull (21 of 176 ids in that pilot — 7x more common than
  `"Multimer"`, which had only 3). It's handled safely today only by
  accident, via `to_records`'s generic empty-sequence guard, not because
  it's special-cased — and it's unconfirmed whether Multi-Peptide's
  nested `monomers[]` chains are independently pullable the way
  Multimer's are (the one example checked had its monomer id outside the
  pilot's pull range). Needs the same live verification Multimer already
  got, at full-pull scale.
- **Hemolytik2 raw sequences sometimes embed non-residue characters
  directly in the sequence string** — found via a real pilot pull: Greek
  `Δ` for dehydro modifications and `*` for bond/cyclization markers
  (e.g. `KWKL-ΔF-KKIGAV-ΔF-KVL`, `GW*LR**K**AAK**SVGK**FY*Y*K**HK*Y*Y*IK*A`).
  `sources/hemolytik2.py` passes these straight through as `sequence_raw`
  with no stripping or structuring into `unusual_residues`/
  `cyclization_type`, so both RDKit and p2smi choke on the literal
  punctuation — a real, uninvestigated contributor to that source's
  above-average FAILED rate. Not fixed this session; needs the modified-
  sequence annotation convention actually decoded first (what exactly
  does `*` vs `**` denote structurally?), not a blind strip.
- **Fixed this session, from the same pilot pull**: the mass-sanity check
  was systematically mis-flagging correctly-curated, terminally-modified
  peptides (palmitoylation, etc.) as `FAILED` — `_estimate_peptide_mass`
  has no way to account for a modification it can't see, so `_validate`
  now skips the mass-sanity gate specifically for `DB_CURATED` records
  that declare a modification the estimator can't estimate (round_trips
  still applies). DBAASP's `concentration` field turned out to have three
  non-plain-float conventions (range/±uncertainty/inequality), not one;
  and `label_is_range`/`label_range_raw` being single fields shared
  between the MIC and HC50 slots on a row caused a real cross-
  contamination crash in `pipeline.py::_normalize()` that's now fixed.
  `sources/hemolytik2.py`'s `_NATURE_VALUES` had a real spelling bug
  (`"Antiinflammatory"` vs. the API's actual `"Anti-inflammatory"`) now
  corrected. `datasheet.py` now also reports peptide-level (not just
  row-level) label coverage, since the row-level `pct_with_both` is
  structurally 0.0% for every source by construction of the schema.
  `sources/hemopi2.py` now populates `hc50_assay_target_cell` (was
  silently `None` before — one of three sources that never set it).
