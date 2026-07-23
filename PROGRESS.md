# Progress: Phase 4 & 5 Implementation Design Docs

Task: write `MD_design_docs/12_phase4_implementation_design.md` and
`MD_design_docs/13_phase5_implementation_design.md`, in the same spirit/concreteness
as `09_phase1_implementation_design.md`. Docs-only; do not touch `src/`/`tests/`,
do not touch docs 10/11 (owned by a concurrent agent).

- [x] Read roadmap (doc 08), Phase 1 template (doc 09), Phase 4 research plan (doc 04),
      Phase 5 research plans (doc 03 warm-start; roadmap's TI ablation bullet),
      splitting plan (doc 07, for the split-artifact contract), model-selection plan
      (doc 05, for concrete backbone facts), `src/clamp/` layout, `data/README.md`,
      `schema.py`, `config.py`, `pyproject.toml`. — 2026-07-23T16:52:00Z: read all of
      the above; grounding module layout in the actual `src/clamp/data/*` code rather
      than inventing a fictional structure.
- [x] Draft doc 12 (Phase 4) outline: package layout (`models/`, `training/`),
      architecture module (pooling/heads/multitask model), masked multitask loss,
      stratified batch sampler, P2 two-stage curriculum, P3 head-addition +
      copy-vs-Xavier init ablation, checkpoint format, CLI, test plan, runbook, DoD,
      open risks. — 2026-07-23T17:20:00Z: outline settled, matches doc 09's section
      shape (0 scope, N module designs, testing, storage, runbook, DoD, open risks).
- [x] Write doc 12 in full. — 2026-07-23T17:20:00Z: written to
      `MD_design_docs/12_phase4_implementation_design.md`. Includes concrete
      `clamp.models.*`/`clamp.training.*` class/function signatures, an explicit
      §2 "data contract consumed from Phase 2/3" section (since docs 10/11 are
      concurrent and unread), checkpoint directory format, `clamp-train` CLI,
      test plan, runbook, DoD, and 8 flagged open risks.
- [x] Draft doc 13 (Phase 5) outline: two workstreams (PAMPA warm-start
      reproduction + ablation; derived-vs-learned TI ablation), shared statistical
      testing utility, module layout, CLI, test plan, runbook, DoD, open risks.
      — 2026-07-23T17:35:00Z: outline settled; reuses Phase 4's checkpoint/metrics
      modules rather than duplicating them.
- [x] Write doc 13 in full. — 2026-07-23T17:35:00Z: written to
      `MD_design_docs/13_phase5_implementation_design.md`. Covers both roadmap
      Phase 5 workstreams (warm-start-encoder ablation per doc 03; derived-vs-
      learned TI ablation per doc 08's Phase 5 bullet), with a concrete
      freeze-vs-LoRA protocol decision, checkpoint-parity verification, paired
      Wilcoxon significance testing shared with doc 12, CLI, test plan, runbook,
      DoD, and 8 flagged open risks.
- [x] Self-review both docs against doc 09's bar. — 2026-07-23T17:38:00Z: confirmed
      doc 12 (738 lines) and doc 13 (535 lines) are comparable in depth/concreteness
      to doc 09 (604 lines); grepped both for accidental references that would edit
      or depend on unread content in docs 10/11 (only forward-pointer mentions
      found, no collisions); confirmed `git status` shows no changes under
      `src/`/`tests/` and docs 10/11 untouched.
- [x] Commit. — 2026-07-23T17:40:00Z
- [x] Push branch `phase4-5-design-docs`. — 2026-07-23T17:41:00Z
- [x] Open PR against `Eitzl/CLAMP` `master`. — 2026-07-23T17:43:00Z
</content>
