# Phase 1 Pilot Data Report

*A real pilot run of the implemented Phase 1 pipeline (`src/clamp/data/`) against live sources, executed 2026-07-22. This is an empirical report on what actually came back over the wire and out of `convert`/`dedup`/`normalize`, not a code review. All numbers below are from one consistent, reproducible run against the final (as of this session) state of the pipeline code; the driver script and raw/interim/processed artifacts live under `/tmp/clamp_pilot/` (not committed — local scratch only) and are described in §7.*

**Caveat on scope:** DBAASP was pulled at pilot scale only (ids 1–210, ~176 records), per instructions — not the full ~25k-record corpus. Every number below that involves DBAASP is a *small-sample estimate*, not a Phase 1-final figure. DRAMP, Hemolytik2, HemoPI2, and QMAP were pulled in full via their real `Downloader().pull()` paths.

---

## 1. What was actually pulled

| Source | Method | Result |
|---|---|---|
| **DBAASP** | `DbaaspPuller.fetch_record` called directly, ids 1–210, rate-limited at the puller's own `requests_per_second` (3/s) | **176 records cached** (of 210 attempted). **34 ids returned HTTP 400** ("Unexpected error occurred. Please try again later.") rather than 404 — these are gap ids in DBAASP's numbering (verified: ids 2, 40, 100, 155, 196 all reproduce a consistent 400, not a transient error). **Zero true 404s.** Of the 176 cached records: **152 Monomer**, **3 Multimer**, **21 "Multi-Peptide"** (see §6 — a third complexity type not previously documented). `to_records()` explodes the 152 usable monomers into **1,093 (peptide, assay-record) rows** (one row per MIC/HC50 activity, or one label-less row if a peptide has no activity data). |
| **DRAMP** | `DrampDownloader().pull()`, real download of `general_amps.txt` (11,687 rows) + `general_smiles.txt` | **Succeeded in full.** 2 rows had an empty `Sequence` field and were correctly skipped → **11,685 PeptideRecords**. |
| **Hemolytik2** | `Hemolytik2Downloader().pull()`, 8 `nature`-facet queries | **7/8 facets succeeded on the first try; 1 failed.** The `"Antiinflammatory"` facet value in `sources/hemolytik2.py`'s `_NATURE_VALUES` list 404s against the live API — the correct spelling is `"Anti-inflammatory"` (with a hyphen); confirmed live (`Anti-inflammatory` → 200, 9 rows; `Antiinflammatory` → 404). This is a real, fixable bug in the source module, not a transient failure. Worked around manually for this pilot by re-querying with the correct spelling and merging (6,459 → **6,468 records**). **Cross-check against the ~13,215-entry figure the design docs flag from the source paper: 6,468 is only ~49% of that** — the facet-enumeration approach (no "list all" endpoint exists) is confirmed *not* exhaustive, exactly the risk data/README.md's open items called out. |
| **HemoPI2** | `HemoPI2Downloader().pull()`, real GitHub CSVs | **Succeeded in full.** `cross_val_dataset.csv` (1,540 data rows) + `independent_dataset.csv` (386 data rows) = **1,926 records**, none with an empty sequence. Note: data/README.md cites "1541 + 387 = 1928" — that count almost certainly includes each file's header row; the real data-row count is 1,926, matching what was loaded. |
| **QMAP** | `QmapDownloader().pull()`, real `qmap-benchmark` → HuggingFace Hub fetch | **Succeeded in full, no errors.** 18,033 samples exploded into **69,525 PeptideRecords** (one row per per-species MIC target + one row per non-NaN HC50 consensus). |

**Total raw rows loaded into the pilot pipeline: 90,697** (1,093 + 11,685 + 6,468 + 1,926 + 69,525).

No source pull was silently skipped; the one real failure (Hemolytik2's misspelled facet) is a small, fixable, one-line bug, not a structural problem, and was worked around for this pilot without modifying the module.

---

## 2. Fidelity tier breakdown after `convert_record`

| Source | full | partial | backbone_only | failed | rows |
|---|---|---|---|---|---|
| dbaasp | 954 (87.3%) | 0 | 0 | 139 (12.7%) | 1,093 |
| dramp | 5,352 (45.8%) | 5,381 (46.1%) | 5 (0.0%) | 947 (8.1%) | 11,685 |
| hemolytik2 | 2,765 (42.8%) | 2,889 (44.7%) | 12 (0.2%) | 802 (12.4%) | 6,468 |
| hemopi2 | 1,926 (100%) | 0 | 0 | 0 | 1,926 |
| qmap | 67,098 (96.5%) | 304 (0.4%) | 0 | 2,123 (3.1%) | 69,525 |
| **ALL** | **78,095 (86.1%)** | **8,574 (9.5%)** | **17 (0.0%)** | **4,011 (4.4%)** | **90,697** |

No conversion call raised an exception (0 of 90,697 rows) — the dispatch logic itself is robust on real data. But 4,011 rows (4.4%) landed in `FAILED`, concentrated in DBAASP (12.7%) and Hemolytik2 (12.4%), and it is **not** evenly distributed noise. Breaking down the 4,011 `FAILED` rows by why they failed:

- **1,307 rows (32.6% of all FAILED) have a `db_curated` SMILES that round-trips fine and is almost certainly correct — it just fails the mass-sanity check.** Root cause, confirmed against real DBAASP JSON (id 51, sequence `"KLlK"`): `convert/__init__.py`'s `_estimate_peptide_mass()` — the expected-mass estimate used to gate the mass-sanity check — is computed purely from the plain amino-acid letters (`sequence_canonical`/`sequence_raw`) and **never looks at `nterm_mod`/`cterm_mod`/`unusual_residues` at all**, even on the `db_curated` path where a curated SMILES is trusted outright. DBAASP id 51 has `nTerminus = {"name": "C16", ...Palmitic acid...}` — an N-terminal palmitoyl (C16 fatty-acid) modification — so the real, correct SMILES weighs 738 Da against the naive linear-peptide estimate of ~501 Da (47% off, way outside the 5% tolerance), and the record is marked `FAILED` despite the SMILES being right. Of the 1,307: **697 (53.3%) have a non-null `nterm_mod`/`cterm_mod`** (mostly acylation — `C16`, `C12`, `C7`, palmitic/myristic/decanoic/lauric acid, plus `ACT`/acetylation and `AMD`/amidation) that fully explains the mismatch; **607 (46.4%) have neither a term-mod nor an unusual residue set in our schema, yet still miss by 7.5–9.0%** (e.g. QMAP's all-lowercase `rrvsrrfmrr`-style sequences) — this second group is unexplained by anything visible in the schema and is flagged for human follow-up in §6, not resolved here.
- **2,008 rows (p2smi path) and 635 rows (plain rdkit path) produced no SMILES at all** (smiles=None), overwhelmingly because the raw sequence contains a character neither RDKit's `Chem.MolFromSequence` nor p2smi's residue table recognizes: `X` (used across sources as a catch-all unusual-residue marker but still literally present in the sequence string), `Z`, `O` (not a standard 1-letter code, seen in QMAP sequences like `CFQWORNMRKVR`), and — a distinct, currently-unhandled case — **Hemolytik2 raw sequences that embed modification annotations directly in the letter string**, e.g. `KWKL-ΔF-KKIGAV-ΔF-KVL` (Greek delta, dehydro-Phe marker) and `GW*LR**K**AAK**SVGK**...` (asterisks, presumably bond/cyclization markers). `sources/hemolytik2.py`'s parser passes these straight through as `sequence_raw` with no stripping/structuring, so of course both conversion paths choke on the literal `Δ`/`*` characters. This is a source-module parsing gap, not a convert.py bug.
- **56 `plain_backbone_fallback` and 5 `rdkit_sequence` rows** produced a SMILES that still failed round-trip/mass-sanity even after falling back — small counts, not investigated further.

**Bottom line for doc 06 §5.1 / doc 09 §12's stated goal ("test whether the mass-sanity tolerance is sane on real data, not just fixtures"): the answer from this pilot is *not fully* — roughly a third of everything currently in `FAILED` is very likely a false negative caused by the expected-mass estimator ignoring terminal modifications on the exact `db_curated` path that's supposed to be the most-trusted one.** This is the single most actionable finding in this report; see §6.

---

## 3. Label coverage

Two different numbers matter here and they tell different stories — **row-level** (what fraction of raw assay-records carry a label) and **peptide-level** (what fraction of *unique* peptides have a label, once dedup pools rows sharing a `peptide_uid` across sources). The schema's one-row-per-assay-record convention (§0 of this doc; doc 06 §4.1) means **`pct_with_both` at the row level is structurally 0.0% for every source, by construction** — a DBAASP peptide with both a MIC and an HC50 measurement becomes two separate rows, never one row with both fields populated. `datasheet.py`'s `build_datasheet()` reports this row-level metric and it is correct but not very informative on its own; see §6 for why this is worth fixing.

**Row-level:**

| Source | rows | % with MIC | % with HC50 | % with both (row-level) |
|---|---|---|---|---|
| dbaasp | 1,093 | 86.6% | 12.7% | 0.0% |
| dramp | 11,685 | 0.0% | 0.0% | 0.0% |
| hemolytik2 | 6,468 | 0.0% | 8.1% | 0.0% |
| hemopi2 | 1,926 | 0.0% | 100.0% | 0.0% |
| qmap | 69,525 | 90.2% | 4.0% | 0.0% |
| **ALL** | **90,697** | **69.9%** | **5.9%** | **0.0%** |

(DRAMP is deliberately a SMILES/sequence-only source per data/README.md — 0% label coverage there is expected, not a bug.)

**Peptide-level** (per unique `peptide_uid`, computed for this report since the datasheet doesn't currently do this — see §6):

| Source (peptide_uids that appear in this source) | unique peptide_uids | % with ≥1 MIC row | % with ≥1 HC50 row | % with both |
|---|---|---|---|---|
| dbaasp | 136 | 86.0% | 39.7% | 29.4% |
| dramp | 11,108 | 5.0% | 2.1% | 0.8% |
| hemolytik2 | 3,323 | 13.8% | 17.1% | 4.9% |
| hemopi2 | 1,926 | 22.4% | 100.0% | 22.4% |
| qmap | 17,203 | 76.1% | 16.1% | 14.4% |
| **ALL (30,880 unique peptide_uids)** | 30,880 | 42.5% | 14.5% | **8.0% (2,478 peptides)** |

The "5.0%/2.1%/0.8%" for DRAMP and "13.8%/17.1%/4.9%" for Hemolytik2 are entirely borrowed via dedup from *other* sources sharing the same `peptide_uid` — DRAMP and Hemolytik2 themselves contribute zero MIC/HC50 values, but ~5–17% of their peptides also show up in DBAASP/QMAP/HemoPI2, which do carry labels. This is dedup doing its job (§4), and it's the clearest evidence in this pilot that cross-source identity matching is working as intended.

---

## 4. Dedup results

- **Raw row count: 90,697 → unique `peptide_uid` count: 30,880** (34.1% of rows are the "first" occurrence of their peptide_uid; the rest are either genuine assay-record siblings of the same peptide or cross-source duplicates).
- **2,486 of 30,880 unique peptide_uids (8.1%) appear in more than one source.** Those 2,486 shared peptides account for **13,332 of the 90,697 raw rows (14.7%)** — i.e. roughly 1 in 7 rows in this pilot would silently leak across a train/test split boundary if dedup didn't run before clustering, which is exactly the risk data/README.md §3.6 flags.

**Overlap matrix** (source pairs, count of shared `peptide_uid`s):

| Pair | Shared peptides |
|---|---|
| dramp ↔ qmap | 1,240 |
| hemolytik2 ↔ qmap | 546 |
| hemopi2 ↔ qmap | 498 |
| dramp ↔ hemolytik2 | 357 |
| hemolytik2 ↔ hemopi2 | 241 |
| dramp ↔ hemopi2 | 161 |
| dbaasp ↔ qmap | 110 |
| dbaasp ↔ dramp | 15 |
| dbaasp ↔ hemopi2 | 8 |
| dbaasp ↔ hemolytik2 | 6 |

QMAP (DBAASP-derived, 18k peptides) unsurprisingly overlaps with everything the most. The DBAASP↔QMAP overlap (110) is low mostly because this pilot only pulled ~150 real DBAASP peptides, not the full ~25k corpus — expect this number to scale up substantially once DBAASP is pulled at full scale, since QMAP is itself built from DBAASP.

**Quality flags:**

| `label_quality_flag` | rows |
|---|---|
| (none) | 73,406 (81.0%) |
| `metadata_source_conflict` | 16,155 (17.8%) |
| `high_replicate_dispersion` | 1,136 (1.3%) |

**16,155 rows (17.8%) got their smiles/fidelity metadata overridden by a richer same-`peptide_uid` record from another source** — a large fraction, but expected given the overlap counts above: any time a DRAMP or Hemolytik2 sequence-only row shares a `peptide_uid` with a DBAASP/QMAP curated-SMILES row, every row in that group *except* the richest gets flagged. This is dedup step §3.4 (data/README.md) working as designed, not a defect — but it is worth knowing that ~1 in 6 rows in the pilot dataset has had its chemistry fields silently swapped out for another source's, which matters if a downstream consumer ever wants to trace "whose SMILES is this, actually."

**1,136 rows (1.3%) were flagged `HIGH_REPLICATE_DISPERSION`** — i.e. belong to a `(peptide_uid, task, species/target_cell)` group where replicate values span more than the 10× threshold. At 1.3% of rows this looks like a plausible, non-alarming base rate for real-world assay noise / occasional unit-transcription error, but it hasn't been eyeballed row-by-row in this pilot — worth a manual spot-check once the fixed normalize-stage bug in §6 stops generating false negatives that could otherwise mask real dispersion cases.

---

## 5. Species / target-cell frequency

**MIC target species: 691 distinct organisms** recorded across the pilot. Top 10 (all from DBAASP+QMAP, since those are the only MIC-labeled sources here):

| Species | Count |
|---|---|
| Escherichia coli | 11,084 |
| Staphylococcus aureus | 10,279 |
| Pseudomonas aeruginosa | 7,529 |
| Bacillus subtilis | 3,405 |
| Klebsiella pneumoniae | 3,047 |
| Staphylococcus epidermidis | 2,721 |
| Acinetobacter baumannii | 2,408 |
| Enterococcus faecalis | 2,126 |
| Salmonella enterica | 2,019 |
| Micrococcus luteus | 1,274 |

Unsurprising — classic ESKAPE-adjacent Gram+/Gram− lab strains dominate, as expected for AMP databases.

**HC50 target cell: only 15 distinct values, and — this is the important part — only 139 of the 5,383 HC50-labeled rows (2.6%) carry a target-cell annotation at all**, and every single one of those 139 comes from DBAASP. The other 5,244 HC50-labeled rows (97.4%: 2,797 from QMAP, 1,926 from HemoPI2, 521 from Hemolytik2) have an HC50 *value* but **no target-cell field populated whatsoever** in the current schema mapping — `sources/hemopi2.py`, `sources/hemolytik2.py`, and `sources/qmap.py` simply never set `hc50_assay_target_cell` when constructing an HC50 record.

Within the 139 DBAASP rows that *do* have a target cell: **92 (66.2%) are "Human erythrocytes"**, and 106 (76.3%) mention "erythrocyte" in some species (mouse, sheep, horse). This directly bears on doc 02's open question #1 ("should HC50 be locked to human RBC?") but **the honest answer from this pilot is that the question can't be fully answered from the data as currently ingested** — the visible 66%/human-RBC-dominant signal comes from a 2.6% slice of the labeled data; the other 97.4% (which includes HemoPI2, whose entire premise per its own paper is human-RBC hemolysis, and Hemolytik2, likewise primarily human-RBC-focused) is currently *not distinguishable by cell type at all* in this pipeline's output, purely because the field isn't populated at the source-module level, not because the underlying source data lacks it. This is a real, fixable gap worth prioritizing before Phase 2's split design leans on this field. See §6.

---

## 6. Things that look wrong or surprising — recommended human follow-up

Ranked roughly by how much modeling/data-quality impact each one has:

1. **Mass-sanity check systematically mis-flags acylated/lipidated peptides as `FAILED` even when the curated SMILES is correct (§2).** `_estimate_peptide_mass()` in `convert/__init__.py` ignores `nterm_mod`/`cterm_mod`/`unusual_residues` entirely, so any `db_curated`-path record with a real terminal modification (palmitoylation, myristoylation, PEGylation, amidation on a very short peptide, etc.) gets an expected mass computed for the *bare, unmodified* peptide and then compared against the *actual* (correct, modified) SMILES's mass — guaranteed mismatch for a nontrivial fraction of exactly the class of AMP the whole "handle modified peptides with a SMILES-LM" thesis exists to serve. Quantified impact in this pilot: 1,307 of 4,011 FAILED rows (32.6%) — likely mostly recoverable if `_estimate_peptide_mass` (or the mass-sanity gate generally) accounted for terminal modifications before comparing. **Recommend fixing this before treating any FAILED-tier count as load-bearing for a Phase 1 go/no-go decision** — the real full-scale FAILED rate is probably meaningfully lower than 4.4% once this is fixed.
   - A residual, unexplained 607-row sub-case (46.4% of the 1,307) has *no* term-mod/unusual-residue set in our schema yet still misses the mass-sanity tolerance by 7.5–9.0%, concentrated in QMAP's all-lowercase sequences (e.g. `rrvsrrfmrr`). Worth a domain-expert look — possibly QMAP's lowercase convention denotes something with a genuinely different formula than the naive D-isomer assumption, or the schema is missing a modification field for these.

2. **DBAASP's `concentration` free-text field turned out to have three distinct non-plain-float conventions (hyphen range, `±` uncertainty, inequality-bounded), not one — this crashed the normalize stage on ~760 real rows the first time this pilot ran, and has since been narrowed to a smaller, still-real 107-row residual bug.** The current `_parse_concentration()` in `sources/dbaasp.py` now correctly distinguishes `"1.58-25.33"` (genuine range) from `"2.2±0.8"` and `">400"` (point-value extraction) — but the residual 107 errors (all DBAASP, all in the `hc50_value` normalize column, all `AttributeError: 'NoneType' object has no attribute 'strip'`) are a **distinct, still-unfixed bug**: `PeptideRecord.label_is_range`/`label_range_raw` are single fields shared between the MIC and HC50 slots on a row, but a DBAASP row only ever populates *one* of `mic_value`/`hc50_value` at a time. `normalize.py`'s per-column `has_input` check (reproduced identically in `pipeline.py::_normalize()`) is `row[value_col] is not None or (row["label_is_range"] and row["label_range_raw"])` — for an MIC-only row whose MIC concentration happened to be range-valued, this check *also* evaluates true for the **HC50** column (since it reads the same shared `label_is_range`/`label_range_raw`), triggers a spurious attempt to normalize a nonexistent HC50 value, and crashes on `unit.strip()` because `hc50_unit` is correctly `None` for that row. **Recommend gating the range-based `has_input` check on `row[unit_col] is not None` too** (cheap, low-risk fix — verified empirically that this discriminates correctly: the row's *own* unit column is always populated when its own value is range-valued).

3. **A third DBAASP `complexity` type, `"Multi-Peptide"`, behaves like `"Multimer"` (empty top-level `sequence`, real chains in `monomers[]`) but isn't mentioned anywhere in `sources/dbaasp.py`'s comments or data/README.md's open items — and in this 176-id pilot it's 7× more common than `"Multimer"`.** Counts: 152 Monomer, 3 Multimer, **21 "Multi-Peptide"**. `to_records()` happens to handle it "correctly" today only by accident, via the generic `if not sequence_raw: return []` guard (not because anyone special-cased it) — meaning its nested monomer chains are silently dropped exactly like Multimer's, *and* it's unclear (not checked in this pilot) whether Multi-Peptide's nested monomer ids are independently pullable the way Distinctin's chains were confirmed to be for Multimer. Spot-checked id 17: nested monomer id 1333 is far outside this pilot's 1–210 pull range, so its label data (visible in the raw JSON) simply isn't recoverable here. Across the 24 complex-type ids (3 Multimer + 21 Multi-Peptide) in this pilot, **112 activity records are dropped at the complex level** — worth checking at full-pull scale whether Multi-Peptide's monomers are actually independently reachable, the same verification already done for Multimer.

4. **`datasheet.py`'s `pct_with_both` column is structurally 0.0% for every source and will remain so forever, by construction of the one-row-per-assay-record schema (§3).** Not a bug exactly — it's doing exactly what it's coded to do — but it's a materially misleading number to hand a reader who's trying to answer "how many peptides have both labels," which is the actual multitask-modeling-relevant question doc 04 cares about. The datasheet should compute this at the `peptide_uid` level (group by uid, check `.any()` per task) the way this report had to do manually to get a real answer (8.0% peptide-level, §3) — **recommend adding a peptide-level coverage table to `datasheet.py` alongside the existing row-level one**, not replacing it (both numbers are legitimate, they just answer different questions).

5. **HC50 target-cell/species is only populated for 2.6% of HC50-labeled rows (§5), entirely from DBAASP.** `sources/hemopi2.py`, `sources/hemolytik2.py`, and `sources/qmap.py` never set `hc50_assay_target_cell` when constructing HC50 records — this isn't necessarily a missing-data problem in the underlying sources (HemoPI2 in particular is *specifically* a human-RBC hemolysis benchmark by construction, per its own paper) so much as a schema-population gap in three of the five source modules. **This blocks a confident answer to doc 02's open question #1** until it's fixed — recommend populating a constant/known cell type for HemoPI2 (human RBC, per its documented experimental design) and checking whether Hemolytik2's raw API response and QMAP's `Sample` objects carry a species/cell-type field that's currently being read but not mapped through.

6. **`pandas>=3.0`'s new default string dtype represents missing values as `float('nan')` instead of `None`**, which breaks two real things encountered in this pilot, both worth checking against the installed environment before treating `pipeline.py`'s `run-all` as trustworthy:
   - `pipeline.py::_normalize()`'s `df["smiles"].map(_molecular_weight)` crashes with a confusing RDKit/Boost C++ `TypeError` (not a Python-level, easy-to-diagnose error) the moment any row has a missing SMILES, because `Chem.MolFromSmiles(float('nan'))` isn't a string. Worked around in this pilot's driver via an explicit `.astype(object).where(pd.notnull(df), None)` cast before any RDKit call.
   - `pipeline.py::_dedup()`'s `[PeptideRecord(**row) for row in df.to_dict(orient="records")]` (reconstructing records from a `read_parquet()`'d DataFrame) **reproducibly raises a pydantic `ValidationError`** the moment any optional `str` field (e.g. `nterm_mod`, `hc50_unit`) is missing for a row, because pydantic correctly rejects a bare `float` where `str | None` is expected. **This means `pipeline.py`'s actual orchestrated `run-all`/`dedup` stage, as committed, cannot get past the dedup stage today under the installed pandas version (3.0.5) without this same cast.** Confirmed by direct reproduction, not inferred. Recommend adding the same object-dtype-with-None-sentinel cast (or an explicit `df.model_validate`-friendly serialization step, e.g. write/read via `model_dump_json`/`model_validate_json` instead of a raw parquet round-trip) directly in `pipeline.py`, not just in this pilot's throwaway driver.

7. **Hemolytik2's `_NATURE_VALUES` list contains a typo** (`"Antiinflammatory"` instead of the API's actual `"Anti-inflammatory"`), silently losing that facet's ~9 records on every real pull until fixed (§1). Small in isolation, but it's exactly the kind of silent-partial-coverage bug that a one-line fix + a live smoke-test would have caught.

8. **Hemolytik2 raw sequences sometimes embed non-residue characters directly in the sequence string** (Greek `Δ` for dehydro modifications, `*` for bond/cyclization markers — e.g. `KWKL-ΔF-KKIGAV-ΔF-KVL`, `GW*LR**K**AAK**SVGK**FY*Y*K**HK*Y*Y*IK*A`) that `sources/hemolytik2.py` doesn't strip or structure into `unusual_residues`/`cyclization_type` at all, guaranteeing both RDKit and p2smi choke on the literal punctuation. This is a real parsing gap in the Hemolytik2 source module, contributing to that source's above-average FAILED rate (12.4%).

None of the above required speculation — every number in this section came from actually inspecting failed real records (`.parquet`/`.json` under `/tmp/clamp_pilot/`), not from reading the code and guessing.

---

## 7. Artifacts

- **Driver script** (not checked in, scratch only): `/tmp/claude-1000/.../scratchpad/run_pilot_pipeline.py` — imports `clamp.data`'s real `convert_record`/`dedup_records`/`normalize_concentration`/`build_datasheet` functions directly, points them at `/tmp/clamp_pilot/` instead of `settings.raw_dir`, and does not modify `pipeline.py`.
- **Pilot raw cache**: `/tmp/clamp_pilot/raw/{dbaasp,dramp,hemolytik2,hemopi2,qmap}/` (~88 MB total).
- **Interim**: `/tmp/clamp_pilot/interim/{converted,deduped}.parquet`, `dedup_result.json`, `normalize_errors.json`, `convert_errors.json`.
- **Final processed pilot dataset**: `/tmp/clamp_pilot/processed/dataset.parquet` (90,697 rows, all columns).
- **Datasheet**: `/tmp/clamp_pilot/datasheet/{datasheet.json,datasheet.md}` — generated by the real `datasheet.build_datasheet`/`write_datasheet`, unmodified.
- **This report**: `MD_design_docs/phase1_pilot_data_report.md`.
- **Compact CSV export for visualization**: `MD_design_docs/phase1_pilot_dataset_sample.csv` — 6,776 rows, columns `source, sequence_raw, smiles, chemical_fidelity_tier, mic_value_uM, mic_target_species, hc50_value_uM, hc50_assay_target_cell, peptide_uid, label_quality_flag`. Sampling strategy: **all** 5,383 HC50-labeled rows (kept in full — the rarer, higher-value task), up to 500 MIC-only rows per source, and up to 200 unlabeled rows per source (for fidelity-tier/sequence diversity), deduplicated, fixed seed (42). Verified to load cleanly with a plain `pandas.read_csv(...)`, no special parsing needed. The full 90,697-row dataset (43.7 MB as CSV) is available at `/tmp/clamp_pilot/processed/dataset.parquet` if a fuller analysis is needed beyond this sample.
