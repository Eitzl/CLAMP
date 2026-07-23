# Internal guide: running the DBAASP pull and finishing Phase 1

This is the practical "what do I actually do" guide for the one manual step
in the pipeline plus the full end-to-end run. The design reasoning behind
all of this lives in `MD_design_docs/06_task5_data_pipeline_plan.md` and
`MD_design_docs/09_phase1_implementation_design.md` §14 — this doc is just
the condensed checklist for actually doing it.

## 0. The one thing you can't skip: read DBAASP's usage policy

DBAASP's docs page is gated behind a "data usage policy and restrictions"
PDF. Nobody on this project has actually opened and read it yet — it's
referenced in the site nav but was never tracked down during the research
pass (doc 06 §1.3/§7.1, doc 09 §16 item 4). Before you run a pull at full
~25k-record scale:

1. Go to https://dbaasp.org and find the data usage policy link (it's near
   the API docs page).
2. Read it. Specifically check for: attribution requirements, whether
   redistribution of the pulled data (even derived/processed forms) is
   restricted, and whether there's an explicit rate limit stated anywhere.
3. Write down what it says — attribution requirements especially need to
   end up in the datasheet output eventually. There's no code gate for
   this on purpose (doc 09 explicitly calls it a manual step, not something
   to automate around), so just make a note somewhere durable (a line in
   `data/README.md`'s open items, or wherever the team tracks this) once
   you've read it.

Do this before step 2 below. It's a blocking prerequisite, not a
nice-to-have — if the terms turn out to restrict something we're already
doing (e.g. bulk redistribution), better to know now than after a 2-3 hour
pull.

## 1. Environment setup

```bash
uv sync
cp .env.example .env
```

Defaults in `.env` are sane for a first run:

```
CLAMP_DATA_ROOT=data
CLAMP_DBAASP_BASE_URL=https://dbaasp.org
CLAMP_DBAASP_REQUESTS_PER_SECOND=3.0
CLAMP_DBAASP_PAGE_SIZE=200
CLAMP_HTTP_TIMEOUT_S=30.0
```

`CLAMP_DBAASP_REQUESTS_PER_SECOND=3.0` matches the "polite" rate used
during the original pilot pull. There is no documented rate limit for
DBAASP — this number is a guess, not a confirmed-safe value. Don't crank it
up for a faster pull without watching for 429s first (see §3 below).

## 2. Running the DBAASP pull itself

```bash
clamp-data pull dbaasp
```

What this actually does (`src/clamp/data/sources/dbaasp.py`,
`src/clamp/data/sources/base.py::ApiPuller.pull`):

1. Pages through `GET {base_url}/peptides?limit=200&offset=N` to collect
   every id (this is a thin summary endpoint — no SMILES/activity data, just
   ids and a `totalCount`).
2. For each id not already cached on disk, calls
   `GET {base_url}/peptides/{id}` and writes the raw JSON to
   `data/raw/dbaasp/{id}.json`.
3. Sleeps `1 / CLAMP_DBAASP_REQUESTS_PER_SECOND` between requests, retries
   transient failures (via `tenacity`) with exponential backoff up to 5
   attempts, capped at 60s between retries.

Expect roughly **2-3 hours** wall time for the full ~25k records at the
default rate. This is safe to interrupt (Ctrl-C) and rerun — anything
already written to `data/raw/dbaasp/{id}.json` gets skipped on the next
run, so you're only ever paying for what's left.

Check progress any time by counting files:

```bash
ls data/raw/dbaasp/*.json | wc -l
```

The command prints a `PullReport` JSON when it finishes (or when you let it
run to completion) — `attempted`, `succeeded`, `skipped_cached`, and a
`failed` list of ids that exhausted retries. If `failed` is non-empty, just
rerun `clamp-data pull dbaasp` again — resume logic means it'll only retry
the ones that failed, not redo the whole pull.

## 3. If you see 429s or a lot of failures

The rate limit isn't documented anywhere, so the current 3 req/s is a
conservative guess carried over from the pilot, not a confirmed-safe
number. If the `failed` list in the report is large, or you're watching
logs and see repeated retries:

- Lower `CLAMP_DBAASP_REQUESTS_PER_SECOND` in `.env` (try 1.5-2) and rerun
  — it'll pick up where it left off.
- Don't just increase retry counts to push through — a 429 means "back off
  harder," not "try again immediately," which is why the retry already
  uses exponential backoff rather than a fixed interval.

## 4. The bulk-download sources (much faster, no policy gate found)

```bash
clamp-data pull dramp
clamp-data pull hemolytik2
clamp-data pull hemopi2
clamp-data pull qmap
```

These are one-shot file downloads (or, for Hemolytik2, a handful of
per-facet API calls) — minutes each, not hours. No documented licensing
gate was found for these during the research pass, but that means
"not found," not "confirmed clear" — worth a quick look at each site's terms
if this dataset is ever going to leave the project, same as DBAASP.

## 5. Convert, dedup, normalize, datasheet

```bash
clamp-data convert
clamp-data dedup
clamp-data normalize
clamp-data datasheet
```

Or just `clamp-data run-all` for these four once the pulls are done —
`run-all` also includes the five pulls, but running them separately first
is worth it so a `convert` bug doesn't obscure whether the DBAASP pull
actually finished cleanly.

Each stage's output:

| Stage | Output |
|---|---|
| `convert` | `data/interim/converted.parquet` — every pulled record with a `smiles`/`smiles_source`/`chemical_fidelity_tier` attached |
| `dedup` | `data/interim/deduped.parquet` + `data/interim/dedup_result.json` (overlap matrix, flagged-conflict count) |
| `normalize` | `data/processed/dataset.parquet` — the actual Phase 1 deliverable |
| `datasheet` | `data/datasheet/datasheet.json` + `datasheet.md` |

`run-all` skips a stage entirely if its declared inputs haven't changed
since last time (tracked in `data/manifest.json`) — pass `--force` to
rerun anyway.

## 6. Read the datasheet by hand before calling Phase 1 done

Don't just check that `datasheet.md` got generated — actually read it. It's
the thing that answers the two open questions from
`MD_design_docs/02_peptideclm_transfer_learning_plan.md`:

1. Should HC50 be locked to human RBC only? The pilot pull estimated ~30%
   of hemolytic/cytotoxic rows are non-human-RBC; the datasheet has the
   real number from the full pull.
2. What's the chemical-fidelity floor — i.e. how much of the dataset is
   `full`/`partial` fidelity vs. `backbone_only`/`failed`? The pilot's ~67%
   curated-SMILES estimate had a wide margin of error (n=48); this is where
   you get the real number.

## 7. Definition of done checklist

Straight from `MD_design_docs/09_phase1_implementation_design.md` §15:

- [ ] `data/raw/dbaasp/` has all ~25k (or current `totalCount`) records
- [ ] `data/raw/{dramp,hemolytik2,hemopi2,qmap}/` have the bulk downloads
- [ ] every record has a `smiles_source` + `chemical_fidelity_tier` set
      (`failed` counts, `null`/unset doesn't)
- [ ] `data/processed/dataset.parquet` has one row per (peptide-construct,
      assay-record), `peptide_uid` populated
- [ ] `mic_value_uM`/`hc50_value_uM`/`*_log_uM` populated everywhere a raw
      label existed
- [ ] `data/datasheet/` generated from `dataset.parquet`, not hand-written
- [ ] `uv run pytest` passes for the full suite
- [ ] you've actually read the datasheet and used it to answer the two
      open questions above — a datasheet nobody reads doesn't close them

## 8. What's still genuinely open after all of the above

These are known gaps, not things the pull will surface as errors — see
`data/README.md` §4 for the full list with reasoning. The short version,
roughly in order of how much they might matter for label coverage:

- DBAASP's Multimer/Multi-Peptide complex-level labels are dropped (each
  chain is pulled as its own record instead) — revisit if the datasheet
  shows this loses a meaningful number of labels.
- Hemolytik2 coverage is enumeration-based (no "list all" endpoint) — cross
  check its row count against the ~13,215 figure from the source paper.
- Hemolytik2 sequences sometimes embed literal `Δ`/`*` modification markers
  that aren't stripped or structured yet — likely contributing to that
  source's SMILES-generation failure rate.
- DRAMP's activity columns are free text and deliberately not parsed into
  structured labels — it's treated as a SMILES/sequence source only.
- The HELM toolchain (`src/clamp/data/convert/helm.py`) is still a stub.

None of these block calling Phase 1 done — they're sized/prioritized after
the datasheet gives real numbers, not before.
