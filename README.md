# CLAMP

CLAMP adapts PeptideCLM to predict antimicrobial peptide activity, specifically
membrane permeabilization, measured via HC50 (hemolytic activity) and MIC
(minimum inhibitory concentration). The idea is to take a pretrained peptide
language model and fine-tune it on a unified, deduplicated dataset pulled
together from several public AMP/hemolysis databases.

This project follows the five-phase plan in
`MD_design_docs/08_implementation_roadmap.md`. **Phase 1 (data acquisition &
preprocessing) is complete** — the pipeline below pulls, converts,
deduplicates, and normalizes all five sources into a unified dataset. Phases
2-5 (embedding/splitting, model selection, multitask training, ablations)
are not yet implemented.


## Why five sources

No single database has both good chemical structure coverage (cyclic
peptides, D-amino acids, terminal modifications) and both HC50 and MIC
labels. DBAASP is the main source and the only one pulled live via API; the
other four (DRAMP, Hemolytik2, HemoPI2, QMAP) are bulk downloads that fill in
gaps or add hemolysis-specific data. `data/README.md` has the full
per-source writeup — how each schema was confirmed, what's curated vs.
inferred, and the specific reconciliation problems (multimers, unit mixing,
range-valued labels, etc). Worth reading before touching the source pullers.

## Setup

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # adjust CLAMP_* settings if needed, defaults are fine
```

## Running the pipeline

The `clamp-data` CLI drives everything:

```bash
clamp-data pull dbaasp       # ~2-3 hours, resumable, see the internal guide below first
clamp-data pull dramp        # bulk downloads, a few minutes each
clamp-data pull hemolytik2
clamp-data pull hemopi2
clamp-data pull qmap

clamp-data convert           # sequence -> SMILES for everything pulled
clamp-data dedup             # collapse duplicate peptides across sources
clamp-data normalize         # units -> uM, log-transform
clamp-data datasheet         # coverage/fidelity report, data/datasheet/datasheet.md
```

Or just run `clamp-data run-all` once you've done a pull successfully — it
chains convert through datasheet and skips stages whose inputs haven't
changed (tracked in `data/manifest.json`). Doing the steps individually the
first time is worth it though, so a bug in `convert` doesn't hide whether the
DBAASP pull actually finished.

**Before running the DBAASP pull at full scale**, read
[`docs/dbaasp_pull_guide.md`](docs/dbaasp_pull_guide.md) — there's a blocking
step (reading DBAASP's data-usage-policy PDF) that needs a human, not code.

Raw pulls, interim tables, and the processed dataset all live under `data/`
and are gitignored — only this repo's code and the `data/README.md` writeup
are tracked. See `data/README.md` for the directory layout and what's in each
stage's output.

## Testing

```bash
uv run pytest
```

173 tests, organized by pipeline stage under `tests/data/`, with real
fixture data (not synthetic) pulled from each source under
`tests/data/fixtures/`.

## Project layout

```
src/clamp/
├── cli.py                  # clamp-data entry point
├── config.py                # settings, env-overridable via CLAMP_*
└── data/
    ├── schema.py             # PeptideRecord, the unified data model
    ├── sources/              # one puller per source (dbaasp, dramp, hemolytik2, hemopi2, qmap)
    ├── convert/               # sequence -> SMILES (rdkit, p2smi, HELM stub)
    ├── dedup.py               # peptide_uid, conflict flagging
    ├── normalize.py           # unit conversion, log transform
    ├── datasheet.py            # coverage/fidelity report generation
    └── pipeline.py            # stage orchestration, manifest-based caching
```

## Further reading

- `MD_design_docs/06_task5_data_pipeline_plan.md` — the research trace behind
  the pipeline design, including the live pilot pull that shaped a lot of
  these decisions.
- `MD_design_docs/09_phase1_implementation_design.md` — the implementation
  design doc this code follows, including the Phase 1 runbook and
  definition of done.
- `data/README.md` — source-by-source schema provenance and the
  deduplication strategy, including all the edge cases (same peptide,
  conflicting labels across sources; multimers; range-valued concentrations).




Data were obtained from the DBAASP (https://dbaasp.org) which is an open-access AMP data
resource supported by I.Beritashvili Center of Experimental Biomedicine (IBCEB) Tbilisi,
Georgia and the National Institute of Allergy and Infectious Diseases (NIAID) Office of Cyber
Infrastructure and Computational Biology (OCICB) in Bethesda, MD. These data were collected
and submitted by members of the DBAASP team.

Pirtskhalava M, Armstrong AA, Grigolava M, Chubinidze M, Alimbarashvili E, Vishnepolsky B,
Gabrielian A, Rosenthal A, Hurt DE, Tartakovsky M. DBAASP v3: database of antimicrobial/cytotoxic
activity and structure of peptides as a resource for development of new therapeutics, Nucleic
Acids Research, Volume 49, Issue D1, 8 January 2021, Pages D288–D297,
https://doi.org/10.1093/nar/gkaa991
