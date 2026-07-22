# Task 5.3 Deep-Dive: Data Sourcing & SMILES Conversion — Concrete, Executable Pipeline

*Expands §5.3 (data), the P0 line of §5.5, and open question #2 of §6.4 in `02_peptideclm_transfer_learning_plan.md` into something a script can actually implement. Everything source-related below was checked directly against the live DBAASP REST API (queried today, 2026-07-22), the live GitHub/PyPI repos, and the primary database papers — not re-derived from the source doc's prose. One material correction to the source doc is flagged up front. Anything I could not verify is called out explicitly in §8.*

---

## 0. Bottom line

1. **DBAASP has a real, documented, machine-readable REST API** — not just per-peptide web lookup. It's queryable/paginable at `https://dbaasp.org/peptides` and returns full structured records (including a `smiles` field) at `https://dbaasp.org/peptides/{id}`. As of today it holds **25,069 peptide/multimer records**. This de-risks P0 substantially: "pull DBAASP" is a scriptable bulk pull via pagination, not scraping HTML.
2. **Correction to the source doc: "CycloPs_v2" does not exist under that name.** The tool actually linked from/built for PeptideCLM is **`p2smi`** (Aaron Feller, arXiv:2505.00719, `github.com/AaronFeller/p2smi`, `pip install p2smi`), which explicitly supersedes the CycloPs concept for this use case. Use `p2smi`, not a search for "CycloPs_v2."
3. A **live pilot pull of 48 DBAASP records**, spread across the ID space, gives a first-order chemical-fidelity estimate: **~67% of records carry a curated SMILES string, ~46% have an annotated cyclization bond, ~40% have non-standard-amino-acid annotations, and ~35%/31% have N-/C-terminal modification annotations.** This is a small sample (see caveats in §6) but it is a real number, not a guess, and directly seeds open question #2.
4. Supplementary sources all check out as real and bulk-downloadable at meaningfully different scales — DRAMP (30,260 entries, dedicated `_smiles` bulk files), dbAMP (35,518 entries), APD3/APD6 (6,309 entries, but **bulk file is sequence-only** — activity values are not in the bulk FASTA), HemoPI2 (1,926 peptides with HC50 regression labels), and one source **not in the original doc that you should add**: **Hemolytik2** (13,215 entries / ~8,700 peptides, dedicated hemolysis DB with SMILES, D-amino acid and cyclic/linear flags, REST API — published bioRxiv May 2025, more directly on-task for HC50 than HemoPI2 alone).
5. QMAP (cited in the source doc) is a real, installable benchmark (`pip install qmap-benchmark`, `github.com/anthol42/QMAP`) shipping ML-ready MIC/HC50 data with SMILES and homology-aware splits — worth pulling in directly rather than re-deriving.

---

## 1. DBAASP — verified access mechanism

### 1.1 It has a real REST API, confirmed live

The homepage (`dbaasp.org`) links to `api?page=rest`, which is a client-rendered Angular page (static HTML is ~4KB, mostly JS shell — don't scrape this page itself). The actual OpenAPI 3 spec is served at:

```
GET https://dbaasp.org/v3/api-docs
```

(confirmed: HTTP 200, 14,912 bytes, valid OpenAPI JSON, fetched today). Decoding it gives exactly two paths:

| Path | Method | Purpose |
|---|---|---|
| `/peptides` | GET | Search/list — paginated (`limit`, `offset`), filterable by ~35 fields |
| `/peptides/{peptideId}` | GET | Full single-record detail (`PeptideView`) |

Filters on `/peptides` include (field names exactly as in the spec): `sequence.value`, `sequenceLength.value`, `nTerminus.value`, `cTerminus.value`, `unusualAminoAcid.value`, `intraChainBond.cycleType`, `intraChainBond.chainParticipating`, `interChainBond.*`, `targetSpecies.value`, `targetGroup.value`, `hemolyticAndCytotoxicActivitie.value`, `synthesisType.value`, `threeDStructure.value` (enum: `without_structure`/`pdb`/`md_model`/`pdb_md_model`), `kingdom.value`, `source.value`, `pubchem.value`, article metadata fields, and `limit`/`offset`. This is enough to filter server-side for, e.g., only cyclic peptides (`intraChainBond.cycleType` non-null) or only records with hemolytic data, before paying the cost of full-record pulls.

**Live confirmation of scale**, queried today:
```
GET https://dbaasp.org/peptides?limit=1&offset=0
→ {"totalCount": 25069, "data": [...]}
```
This is up from the >15,700 reported in the DBAASP v3 paper (Pirtskhalava et al., *NAR* 2021, PMC7778994) — the database has grown ~60% since publication, consistent with ongoing curation.

### 1.2 Full-record schema (confirmed from the OpenAPI spec + a live pull)

`GET /peptides/{id}` returns a `PeptideView` with these top-level arrays/fields (verified against a real record, `id=16`, "Distinctin chain 1"):

- `sequence`, `sequenceLength`, `name`, `dbaaspId`
- `getnTerminus` / `getcTerminus` (`MixedView` — terminal modification name, e.g. amidation; `null` if unmodified)
- `synthesisType` (e.g. "Ribosomal", "Synthetic")
- `complexity` (`Monomer` / `Multimer` / `Multi peptide`) — **matters**: multimers (e.g. disulfide-linked heterodimers like Distinctin) have `sequence=""` at the top level and the real chains live under `monomers[]`, each a nested `PeptideView`. A naive "grab `sequence` field" pull will silently drop ~10% of records (see §6 pilot stats) into empty strings.
- `unusualAminoAcids[]` — `{position, modificationType, beforeModification, note}` — this is the D-amino-acid / non-canonical-residue annotation.
- `intrachainBonds[]` / `interchainBonds[]` — `{position1, position2, type, cycleType, chainParticipating, note}` — this is the **cyclization** annotation (head-to-tail, side-chain-to-side-chain, disulfide, etc. — `cycleType` is exactly the taxonomy `p2smi` expects as input, see §2).
- `targetActivities[]` (MIC-equivalent) — `{targetSpecies, activityMeasureGroup (MIC/MBC/MFC/IC50/LD50...), activityMeasureValue, concentration, unit, ph, ionicStrength, saltType, medium, cfu, note, reference, activity}`. **Units are not uniform** — confirmed mix of `µM` and `µg/ml` in the pilot pull (see §6), exactly the normalization the source doc flags.
- `hemoliticCytotoxicActivities[]` (HC50-equivalent) — `{targetCell, activityMeasureForLysisGroup, activityMeasureForLysisValue, concentration, unit, ph, ionicStrength, saltType, note, reference, activity}`. **`targetCell` is not always human RBC** — the pilot pull saw "Human erythrocytes," "Human fibroblasts WI-38," "Vero cells," "Mouse fibroblasts NIH 3T3," "Human leukocytes" all under this same array. This is the field to filter on for open question #1 (lock HC50 to human RBC).
- `smiles[]` — `{smiles, smilesImageUrl, description, manuallyEdited (bool), pubChemCid, lastUpdatedAt}`. This is DBAASP's own curated structure — when present and `manuallyEdited=true`, this is the highest-trust SMILES you'll get from any source and skips conversion entirely, exactly as the source doc hoped.
- `pdbs[]`, `structureModel` (MD trajectory files — DCD/PSF, CHARMM22-parameterized) — not needed for the SMILES-LM pipeline but notable that >3,200 peptides (per the NAR paper) have full MD models, a resource worth remembering for §6.3-style physics-feature ideas in the parent doc.
- `physicoChemicalProperties[]` — precomputed charge, hydrophobic moment, isoelectric point, penetration depth, tilt angle, amphiphilicity index, etc. **Free baseline features** — reuse these directly for the P1 physicochemical-descriptor baseline in the parent doc's §5.5 instead of recomputing.

### 1.3 Concrete pull strategy

```
1. GET /peptides?limit=1&offset=0            → read totalCount (25,069 today)
2. Page through /peptides?limit=200&offset=N  (N = 0, 200, 400, ...)  to enumerate all ids
   (the list endpoint returns SearchResultItemView — thin, just id/name/sequence/top activity —
    do NOT try to harvest MIC/HC50/modifications from this view, it's a search-results summary)
3. For each id, GET /peptides/{id}            → full PeptideView, dump raw JSON to disk keyed by id
4. Parse the cached JSON offline into the unified schema (§4)
```

Step 3 is ~25,000 individual HTTP calls. At a polite ~3–4 req/s (what I used for the pilot; no documented rate limit found — see §8) that's roughly **2–2.5 hours of wall time**, embarrassingly parallel and trivially resumable if you cache each response to `raw/dbaasp/{id}.json` and skip ids already on disk. This is a one-time cost; re-runs should diff against a `lastUpdatedAt`-style check if DBAASP exposes one (the `smiles` sub-object does carry `lastUpdatedAt`; unclear if the peptide record itself does — check on first pull).

**Do not skip step 2/3 and try to read everything off `/peptides` with a huge `limit`.** The list view lacks `smiles`, `unusualAminoAcids`, `intrachainBonds`, full `targetActivities`/`hemoliticCytotoxicActivities` arrays — those only come back from the per-id detail endpoint. This is the single most important implementation detail this verification pass adds to the source doc's "pull DBAASP" line item.

**Licensing flag:** the site gates its documentation page behind agreeing to a "data usage policy and restrictions" PDF (referenced in the nav but I did not locate and read the PDF itself — see §8). Read and record that document before a bulk programmatic pull; DBAASP is described everywhere as free/open-access but the exact terms (attribution requirements, redistribution limits) weren't independently confirmed here.

---

## 2. SMILES conversion tooling

### 2.1 Correction: "CycloPs_v2" → `p2smi`

I could not find any tool named "CycloPs_v2" anywhere — not on the PeptideCLM repo, not on PyPI, not in search. What **is** linked to PeptideCLM and does exactly the job the source doc describes is:

**`p2smi`** — Aaron Feller, `github.com/AaronFeller/p2smi`, pip-installable (`pip install p2smi`), published alongside an arXiv paper (arXiv:2505.00719, *"p2smi: A Python Toolkit for Peptide FASTA-to-SMILES Conversion and Molecular Property Analysis"*). The README states directly that it was **"developed in support of PeptideCLM."** Latest release confirmed: **v1.1.1, November 10 2025** ("user specified cyclization"), so it's maintained on a timescale relevant to this project, not abandoned.

- **Input:** FASTA files, with cyclization encoded via a constraint mask in the header (positions marked `X`=unconstrained, `C`=disulfide-participating, `N`/`Z`=N-/C-terminus-participating). Also accepts plain SMILES and its own `.p2smi` intermediate format.
- **Output:** `.p2smi` files (sequence↔SMILES pairs), JSON molecular-property files, annotated FASTA.
- **Cyclization types supported (5):** disulfide (SS), head-to-tail (HT), side-chain-to-side-chain (SCSC), side-chain-to-N-terminus (SCNT), side-chain-to-C-terminus (SCCT). This taxonomy lines up well with DBAASP's `intrachainBonds[].cycleType` field (§1.2) — the mapping from DBAASP's bond annotation to a `p2smi` FASTA header tag is a small, well-scoped adapter script, not a research problem.

The **original CycloPs** (Duffy & Verniere, 2011, *JCIM* 10.1021/ci100431r) is a separate, older tool at `github.com/fergaljd/cyclops` — a GUI Python app depending on a separate `PepLibGen` package, RDKit, and PIL. I could not confirm a recent commit date from the repo page (GitHub's rendered page didn't surface timestamps to the fetch), but there is no evidence of active maintenance and no mention anywhere of a "v2." Given `p2smi` exists, is explicitly built for this exact model, is actively released, and is pip-installable, **there is no reason to chase down the 2011 CycloPs codebase** — treat the source doc's "CycloPs_v2" reference as meaning `p2smi`, and use that name going forward in any code/tickets so this doesn't get re-searched for later.

### 2.2 Recommended conversion stack, by case

| Case | Tool | Notes |
|---|---|---|
| DBAASP record has a curated `smiles[]` entry, `manuallyEdited=true` | **Use it directly** | Skip conversion. Highest trust. |
| Linear, unmodified, canonical-residue peptide | `RDKit Chem.MolFromSequence` / `MolFromFASTA` | Mechanical, well-tested, fast. |
| Cyclized and/or contains annotated non-canonical residues, terminal mods | **`p2smi`** | Primary tool for the hard cases; matches PeptideCLM's training-distribution conventions since Feller built both. |
| Terminal/backbone modification expressed only as HELM notation (some DRAMP/other-source records) | HELM→SMILES toolchain — **not independently verified in this pass**; candidates worth evaluating: the Pistoia Alliance's open-source HELM toolkit, or RDKit's own (limited) HELM parsing support in recent versions | Flagged as unverified — budget a short spike to confirm which HELM parser handles the specific modification types you actually encounter before committing. |
| Modification metadata present but doesn't map cleanly to any of the above (inconsistent free-text notes, ambiguous position, non-standard nomenclature) | **Fall back to plain-backbone `Chem.MolFromSequence`, and flag the record** | This is the fallback the source doc warns about — see §5/§6 for how to track it, not just accept it silently. |

### 2.3 Validation step (missing from the source doc — add this)

For every generated SMILES, before it enters the training set:
1. **Round-trip check**: `Chem.MolFromSmiles(smi)` must parse without error, and re-canonicalizing should be stable (`Chem.MolToSmiles(Chem.MolFromSmiles(smi))` idempotent).
2. **Formula/atom-count sanity check** against the expected peptide (residue count × average residue mass ± modification mass deltas) to catch silent cyclization/ring-closure bugs of exactly the kind the source doc flags for PeptideCLM v1's pretraining data (ring-numbering bug fixed in v1.1, per doc `02`, §5.2). This is cheap insurance against re-introducing that class of bug in your own conversion step.
3. Log pass/fail per record — this log **is** the raw material for the chemical-fidelity report in §6.

---

## 3. Supplementary sources — verified scale and bulk-download status

| Source | Confirmed scale | Bulk download? | Format | HC50/MIC coverage | Notes |
|---|---|---|---|---|---|
| **DBAASP** | 25,069 records (live, today) | Yes — REST API, paginated (§1) | JSON | Both, per-record, with units/species/assay metadata | Primary source; also has curated `smiles[]` |
| **DRAMP 4.0** | 30,260 total; **11,612 "general" (non-patent) entries**; **2,891 with experimental hemolytic activity**; 2,674 with cytotoxicity data | Yes — categorized bulk files (`general_amps.xlsx/.txt/.fasta`, plus **dedicated `*_smiles.xlsx/.txt/.fasta`** for general/patent/specific/stability subsets) | XLSX, TXT, FASTA | MIC via "Antibacterial/Anti-Gram+/Anti-Gram-" activity-labeled subsets; hemolytic activity flagged in general entries | Good coverage supplement; SMILES bulk files are a direct win — less conversion work than DBAASP for the subset that has them. Source: DRAMP 4.0, *NAR* 2025 (PMC11701585); `dramp.cpu-bioinfor.org` |
| **APD3 / APD6** | 6,309 total peptides (current, as of Jan 2026); "2024 natural AMPs with known activity" subset = 3,306 | Yes, but **bulk FASTA is sequence + name only** — no MIC/HC50 values in the bulk file | FASTA | Values must be scraped per-peptide from the web card, or obtained via a filtered search-and-download-FASTA-of-results workflow | Good for sequence-space coverage/dedup cross-checking; **not** an efficient bulk source of activity labels. `aps.unmc.edu` |
| **dbAMP 3.0** | 35,518 entries (33,065 AMPs + 2,453 antimicrobial proteins), June 2024 release | Download page confirmed to exist (`awi.cuhk.edu.cn/~dbAMP/download2024.php`); exact field-level bulk-file contents **not independently confirmed** (page content did not render in fetch) | Unconfirmed (historically XLS/FASTA) | Includes a hemolytic-toxicity/half-life prediction tool per the 3.0 paper; unclear how much is *curated experimental* data vs. *predicted* — needs a closer look before trusting bulk hemolytic values from this source | Largest raw count of the AMP-focused DBs, but treat MIC/hemolytic completeness claims as unverified until the actual download is inspected. Paper: dbAMP 3.0, *NAR* 2025 (academic.oup.com/nar/article/53/D1/D364) |
| **HemoPI / HemoPI2** | HemoPI-1: 552 hemolytic + 552 non-hemolytic (from Hemolytik + SwissProt negatives); **HemoPI2: 1,926 experimentally validated hemolytic peptides**, with regression-ready HC50 values | Yes — `github.com/raghavagps/HemoPI2`, `pip install hemopi2`, zip dataset on `webs.iiitd.edu.in/raghava/hemopi2/download.html` | CSV/FASTA in zip | HC50 only (that's the point of this source) | Smaller than Hemolytik2 (below) — consider HemoPI2 mainly for its clean regression framing/benchmark comparability, not raw volume. |
| **Hemolytik2** — *not in the source doc; found during this pass, worth adding* | **13,215 entries, ~8,700 unique peptides** | Yes — stated to include a REST API, `webs.iiitd.edu.in/raghava/hemolytik2/` | Unconfirmed exact bulk format (web page states download + API available) | HC50-focused, dedicated hemolysis DB | Curated from APD, UniProt, CAMP-R4, DAMPD. **Explicitly includes SMILES, terminal modifications, stereochemistry (D-/L-), and a linear-vs-cyclic flag** — i.e. it ships close to the same modification metadata DBAASP does, but for hemolysis specifically, and it's newer (bioRxiv May 2025, PMC/biorxiv 2025.05.12.653624) than HemoPI2. **Recommend treating this as a co-primary hemolysis source alongside DBAASP**, not just a "supplement," given the metadata richness. |
| **QMAP** | Benchmark dataset, not a fresh source of new peptides — pools/cleans existing DBs | Yes — `pip install qmap-benchmark`, `github.com/anthol42/QMAP`, "consensus bacterial MIC and mammal HC50" | ML-ready, includes SMILES, supports modified peptides (N-acetylation, C-amidation) and intrachain bond notation, predefined homology-aware train/test splits | Both, pre-cleaned | Confirmed real and current — published bioRxiv Feb 2026, now also *Scientific Reports* (June 2026, DOI in `10.1038/s41598-026-56004-8` family). Use directly for (a) an extra, already-homology-clean data slice, and (b) as the external benchmark to report against, per the parent doc's §5.4/§5.5. |
| **CycPeptMPDB** | Per source doc / PMC10091415 — permeability data, not lysis | Yes (per original PeptideCLM usage) | — | None (wrong assay — permeability, not lysis) | Confirmed relevant only for the "cyclic long game" chemical-space argument in doc `02` §6.2, not as a label source for this task. Not re-verified in depth this pass since it contributes no HC50/MIC labels. |

**Read on overlap:** DRAMP, APD3, and dbAMP are all literature-curation databases drawing on largely the same primary-publication universe as DBAASP (and cite each other as sources — Hemolytik2 explicitly lists APD as an input, for instance). **Expect heavy sequence-level overlap across DBAASP/DRAMP/APD3/dbAMP/Hemolytik2** — this is not a hypothesis, it's how these databases are built (manual curation of overlapping AMP literature). Dedup strategy in §4.3 is not optional.

---

## 4. Unified dataset design

### 4.1 Schema

One row per **(peptide-construct, assay-record)** — i.e. a peptide with two MIC values against two species gets two rows sharing a peptide key, not one row with two columns, so masking works cleanly in the multitask setup from doc `02` §5.1/§5.6.

| Column | Type | Notes |
|---|---|---|
| `peptide_uid` | string | Stable hash of canonical sequence + modification signature (§4.3) — the dedup key |
| `source` | enum | `dbaasp` / `dramp` / `apd3` / `dbamp` / `hemopi2` / `hemolytik2` / `qmap` |
| `source_id` | string | Original record ID in that source, for traceability/audit |
| `sequence_raw` | string | As given by source (1-letter, may include non-standard codes) |
| `sequence_canonical` | string | Normalized 1-letter sequence used for dedup/clustering |
| `is_multimer` | bool | From DBAASP `complexity`; other sources may need inference |
| `is_cyclic` | bool | From explicit cyclization annotation where present |
| `cyclization_type` | enum/null | HT / SCSC / SCNT / SCCT / SS / none / unknown |
| `nterm_mod` | string/null | e.g. acetylation, free |
| `cterm_mod` | string/null | e.g. amidation, free |
| `unusual_residues` | list[struct] | `[{position, from, modification_type}]` |
| `smiles` | string/null | Final SMILES used for the model |
| `smiles_source` | enum | `db_curated` (skip-conversion case) / `rdkit_sequence` / `p2smi` / `helm_toolchain` / `plain_backbone_fallback` |
| `smiles_validated` | bool | Passed the §2.3 round-trip + mass-sanity check |
| `chemical_fidelity_tier` | enum | `full` / `partial` / `backbone_only` — defined precisely in §5.1 |
| `hc50_value` | float/null | In original units, pre-normalization |
| `hc50_unit` | string/null | e.g. µM, µg/mL |
| `hc50_value_uM` | float/null | Post-normalization (needs MW; see §4.2) |
| `hc50_log_uM` | float/null | log-transformed per doc `02` §5.5 |
| `hc50_assay_target_cell` | string/null | e.g. "Human erythrocytes" — the field to filter on for open question #1 |
| `mic_value` | float/null | Original units |
| `mic_unit` | string/null | |
| `mic_value_uM` | float/null | Normalized |
| `mic_log_uM` | float/null | |
| `mic_target_species` | string/null | e.g. "Escherichia coli" |
| `mic_assay_medium` | string/null | e.g. "MHB" — DBAASP exposes this; keep it, QMAP-style benchmarks care about assay consistency |
| `label_quality_flag` | enum/null | e.g. `range_only` (DBAASP sometimes reports a range like "1.58-25.33" rather than a point value — decide a convention: midpoint? geometric mean? flag and pick one, don't silently average) |
| `reference` | string/null | Citation/PMID where available, for the datasheet |
| `duplicate_of` | string/null | `peptide_uid` of the canonical record this was merged into, if applicable |

### 4.2 Unit normalization

Per doc `02` §5.5: normalize everything to **µM**, then log-transform for modeling. Concretely:
- Values already in µM/µg·mL⁻¹ etc.: convert µg/mL → µM using **molecular weight computed from the final SMILES** (`RDKit Descriptors.MolWt`), not a naive residue-count estimate — this matters specifically because modified/cyclic peptides have MW deltas from their unmodified backbone, and using the wrong MW is a quiet source of systematic error concentrated exactly in the chemically-interesting (modified) subset you most want to get right.
- Confirmed from the live DBAASP pull (§6): both µM and µg/mL appear roughly evenly in `targetActivities` and `hemoliticCytotoxicActivities` — this conversion step is not optional, it's needed for a large fraction of records.
- Range-reported values (e.g. DBAASP's `"1.58-25.33"` concentration strings): pick and document one convention (recommend geometric mean, consistent with log-transform downstream) and keep the raw range in a side column so the choice is auditable/reversible.

### 4.3 Dedup strategy

1. **Canonicalize sequence**: strip whitespace, uppercase, resolve any source-specific non-standard single-letter codes to a common vocabulary (this vocabulary itself needs to span DBAASP's `unusualAminoAcid` naming, DRAMP's, etc. — expect ad hoc mapping work here, not a fully mechanical step).
2. **Canonicalize modification signature**: `(nterm_mod, cterm_mod, sorted list of (position, modification_type), cyclization_type)`.
3. **`peptide_uid` = hash(sequence_canonical + modification signature)**. Two records from different sources with the same sequence and same modification signature collapse to the same `peptide_uid`.
4. **Cross-source conflict resolution** when the same `peptide_uid` has multiple label rows for the *same* assay type/species/target-cell from different sources: keep both rows (they're independent replicate measurements, useful signal about label noise) but tag them with a shared `peptide_uid` so cluster-splitting (doc `02` §5.4) treats them as one entity — this is the leakage risk the source doc's homology-split requirement is designed to prevent, and it applies at the *dedup* stage too, not just the final split. If you dedupe imperfectly and DBAASP's copy of a peptide lands in train while DRAMP's copy of the *same* peptide lands in test, you've silently reintroduced the leakage the homology split was supposed to kill.
5. **Fuzzy fallback**: sequence-identical-but-modification-metadata-missing-in-one-source records (e.g. DBAASP has full mod annotation, APD3's bulk FASTA for the "same" peptide has none) should still collapse on sequence alone with a flag `metadata_source_conflict=true`, and the richer record should win for `smiles`/`chemical_fidelity_tier` purposes — don't let a metadata-poor duplicate silently downgrade a metadata-rich one.

### 4.4 Datasheet deliverable

Per doc `02`'s P0 line ("a versioned dataset + datasheet noting assay/species per label"), concretely produce, alongside the dataset file:
- A per-source **row-count and coverage table** (rows contributed, % with HC50, % with MIC, % with both, % after dedup).
- A **species/target-cell frequency table** for MIC and HC50 respectively (this is exactly what open question #1 needs to make the "lock to human RBC + fixed bacterial panel" decision with real numbers instead of a guess).
- The **chemical-fidelity breakdown** from §5, versioned alongside the data (fidelity is a property of *this specific pull*, and will change as sources update — don't let it go stale silently).
- A **dedup summary**: how many raw rows in, how many unique `peptide_uid`s out, overlap matrix between sources (e.g. "N peptides appear in both DBAASP and DRAMP").
- License/attribution notes per source, given the DBAASP terms-of-use gate noted in §1.3 and similar likely restrictions on the others (not independently checked for every source this pass — treat as a checklist item, not done).

---

## 5. Chemical fidelity measurement plan (open question #2)

### 5.1 Define fidelity tiers precisely (this doesn't exist yet — needed before you can measure anything)

- **`full`**: record has explicit modification metadata (terminal mods and/or non-canonical residues and/or cyclization) **and** a SMILES was generated by a modification-aware path (`db_curated`, `p2smi`, or `helm_toolchain`) **and** it passed the §2.3 validation.
- **`partial`**: record has *some* modification metadata, but generation could only account for part of it (e.g. cyclization handled by `p2smi` but a non-canonical residue inside the ring wasn't in its residue table, so it was substituted or dropped) — this tier needs its own sub-flag for *which* modification got dropped, otherwise "partial" hides exactly the information you need to decide whether to fix the gap or accept it.
- **`backbone_only`**: record has no usable modification metadata (missing, inconsistent, or unparseable), and the SMILES was generated by plain `Chem.MolFromSequence` on the unmodified canonical backbone. This is the fallback case the source doc warns about.
- **`failed`**: no valid SMILES could be produced at all (should be near-zero if the sequence field itself is clean; track separately since it's a different failure mode than fidelity loss).

### 5.2 Measurement is a straightforward pass over the conversion log from §2.3

```
for each record in pulled_dataset:
    run conversion pipeline (§2.2), get (smiles, smiles_source, validation_result)
    classify into {full, partial, backbone_only, failed} per §5.1
    log: peptide_uid, source, tier, which modifications (if any) were dropped

report:
    tier counts and % overall
    tier counts and % broken out by source (DBAASP vs DRAMP vs Hemolytik2, etc. — expect DBAASP/Hemolytik2 to skew toward `full` given their richer metadata, and APD3's bulk pull to skew toward `backbone_only` given §3's finding that its bulk file is sequence-only)
    tier counts and % broken out by whether the record has HC50, MIC, or both — this is the number that actually answers "does the SMILES route beat plain ESM-2," since a fidelity win that's concentrated in records without usable labels doesn't help the model
```

### 5.3 A first-order estimate, from a live pilot (do the real thing at full scale, but this is not a guess)

To seed this rather than leave it purely hypothetical, I pulled **48 full DBAASP records**, sampled at even offsets across the entire 25,069-record ID space (not just the first page — this matters, since early IDs skew toward older/simpler literature entries):

| Property (of 48 sampled DBAASP records) | Count | % |
|---|---|---|
| Has curated `smiles[]` entry | 32 | 67% |
| Has ≥1 target (MIC-type) activity | 48 | 100% |
| Has a strictly-labeled "MIC" measure (vs. only MBC/LD50/IC50) | 45 | 94% |
| Has ≥1 hemolytic/cytotoxic activity | 33 | 69% |
| Has both MIC and hemolytic data | 33 | 69% |
| Has ≥1 non-canonical-residue annotation | 19 | 40% |
| Has ≥1 intrachain bond (cyclization signal) | 22 | 46% |
| Has an N-terminal modification | 17 | 35% |
| Has a C-terminal modification | 15 | 31% |
| Is a multimer (needs the `monomers[]` un-nesting handled) | 5 | 10% |
| Target-activity units seen | — | µM (198 activity rows) and µg/mL (191 rows), roughly split |
| Hemolytic `targetCell` values seen | — | "Human erythrocytes" (35 rows) dominant, but also fibroblasts (WI-38, NIH 3T3), Vero cells, human leukocytes (19 rows combined) |

**Caveats on this number, stated plainly:** n=48 gives roughly ±14 percentage points of sampling noise at 95% confidence on a proportion near 50% — this is a *pilot*, useful for planning (e.g. "expect roughly half your DBAASP pull to carry cyclization annotations, budget `p2smi` integration time accordingly") but **not** a substitute for running the full §5.2 measurement over all 25,069 records once the pipeline exists. It does, however, already tell you two decision-relevant things: (a) DBAASP's own curated SMILES field alone (~67%) will *not* get you full coverage — conversion tooling is load-bearing for the other third, and (b) roughly 30% of hemolytic/cytotoxic rows are **not** human-RBC — open question #1's "lock to human RBC" filter will materially shrink the usable hemolysis set, and that shrinkage should be sized (from the full pull, not this sample) before committing to the restriction.

---

## 6. Concrete P0 checklist (translating doc `02` §5.5's P0 line into steps)

1. Read DBAASP's data-usage-policy PDF (gated behind the API docs page) before scripting a bulk pull; note attribution requirements in the datasheet.
2. Build the DBAASP puller: paginate `/peptides` for the full id list, cache each `/peptides/{id}` response to disk, resumable (§1.3). Budget ~2–3 hours wall time at a polite request rate; no documented rate limit found, so start conservative and watch for 429s.
3. Pull DRAMP's bulk `general_amps` + `*_smiles` + activity-labeled files (§3) — this is a single set of file downloads, not a scraping job.
4. Pull Hemolytik2 and HemoPI2 bulk data (§3) as the hemolysis-focused supplements.
5. Pull APD3's FASTA (sequence coverage/dedup cross-check only — don't expect activity values from the bulk file, per §3).
6. Investigate the actual dbAMP 3.0 download page contents directly (this pass could not confirm field-level bulk contents — treat as an open item, not "done").
7. Pull QMAP's packaged dataset via `pip install qmap-benchmark` as an additional pre-cleaned, homology-split-ready slice.
8. Run the SMILES conversion stack (§2.2) with validation (§2.3) over everything pulled.
9. Run the fidelity measurement (§5.2) at full scale — this is the actual answer to open question #2, supersede the §5.3 pilot numbers with it.
10. Run dedup (§4.3), producing `peptide_uid`s and the overlap matrix.
11. Normalize units (§4.2), log-transform.
12. Produce the datasheet (§4.4).
13. Cluster (on PeptideCLM-2 embeddings, per doc `02` §5.4) and split — **out of scope for this document**, handoff to whichever track owns §5.4, but note the dedup step above must complete first or the split will leak across near-duplicate source-overlap pairs.

---

## 7. Open questions raised by this pass (beyond the parent doc's §6.4)

1. **DBAASP rate limits / ToS**: no documented rate limit found; the data-usage-policy PDF gating the API docs page was not read in this pass. Read it before running the full ~25k-record pull.
2. **dbAMP 3.0's actual bulk file contents**: page fetch didn't render usable content; confirmed the download page *exists* but not what's *in* it (field-level MIC/hemolytic completeness, format). Needs a direct follow-up, ideally by actually downloading and opening a file rather than fetching the page.
3. **HELM→SMILES toolchain choice**: flagged in §2.2 as unverified — which specific library handles which specific modification types needs a short hands-on spike once you see what modification vocabulary the pulled data actually contains (don't pre-commit to a HELM library before knowing what you need it to parse).
4. **Whether DBAASP peptide records carry a top-level `lastUpdatedAt`** (for efficient re-pulls) — only confirmed this exists on the nested `smiles` sub-object, not the peptide record itself.
5. **Exact overlap fraction between DBAASP/DRAMP/APD3/dbAMP/Hemolytik2** — asserted as "expect heavy overlap" on curation-process grounds (§3), but the actual overlap percentage is an empirical question only answerable after step 10 of the P0 checklist; don't assume a specific number going in.
6. **Whether DBAASP's `smiles[]` records with `manuallyEdited=false` are trustworthy enough to skip conversion**, or whether only `manuallyEdited=true` should count as `smiles_source=db_curated` — the pilot pull didn't isolate this distinction; worth checking at full-scale measurement time (§5.2).
7. **CycloPs (2011, `fergaljd/cyclops`) exact maintenance status** — could not confirm last-commit date through the tools available this pass; concluded "use `p2smi` instead" on the strength of `p2smi` being newer/maintained/purpose-built rather than on direct proof CycloPs is abandoned.

---

## 8. Source pointers

- DBAASP: `dbaasp.org`; OpenAPI spec live at `dbaasp.org/v3/api-docs`; DBAASP v3 paper, Pirtskhalava et al., *NAR* 2021, PMC7778994.
- `p2smi`: `github.com/AaronFeller/p2smi`; PyPI `p2smi`; arXiv:2505.00719.
- CycloPs (2011, legacy, not recommended): `github.com/fergaljd/cyclops`; Duffy & Verniere, *JCIM* 2011, 10.1021/ci100431r.
- DRAMP 4.0: `dramp.cpu-bioinfor.org`; paper *NAR* 2025, PMC11701585.
- APD3/APD6: `aps.unmc.edu`; downloads at `aps.unmc.edu/downloads` (formerly `aps.unmc.edu/AP/downloads.php`).
- dbAMP 3.0: `awi.cuhk.edu.cn/~dbAMP/download2024.php`; paper *NAR* 2025, `academic.oup.com/nar/article/53/D1/D364`.
- HemoPI2: `github.com/raghavagps/HemoPI2`; PyPI `hemopi2`; `webs.iiitd.edu.in/raghava/hemopi2`.
- Hemolytik2 (new find, not in source doc): `webs.iiitd.edu.in/raghava/hemolytik2`; bioRxiv 2025.05.12.653624.
- QMAP: `github.com/anthol42/QMAP`; PyPI `qmap-benchmark`; bioRxiv 2026.02.03.703041, now *Scientific Reports* 2026.
- CycPeptMPDB: PMC10091415 (referenced for chemical-space relevance only, per doc `02` §6.2; not a lysis-label source).
