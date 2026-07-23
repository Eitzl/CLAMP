# Progress: Phase 2 & 3 Implementation Design Docs

Tracks work on `MD_design_docs/10_phase2_implementation_design.md` and
`MD_design_docs/11_phase3_implementation_design.md`.

- [x] Read roadmap doc 08 (Phases 2 & 3 scope) and template doc 09 (structure/bar to match) — 2026-07-23 16:47 UTC
- [x] Read research-plan docs 07 (splitting strategy) and 05 (model selection) in full — 2026-07-23 16:47 UTC
- [x] Read existing `src/clamp/` layout, `data/README.md`, `schema.py`, `pipeline.py`, `cli.py`, `config.py`, `pyproject.toml` to ground designs in real code, not fiction — 2026-07-23 16:50 UTC
- [x] Create branch `phase2-3-design-docs` from master — 2026-07-23 16:50 UTC
- [x] Draft doc 10 outline (Phase 2: embedding, clustering, splitting) — 2026-07-23 16:55 UTC
- [x] Write doc 10 in full (`MD_design_docs/10_phase2_implementation_design.md`) — 2026-07-23 17:10 UTC — resolved two ambiguities explicitly: doc07 §3.3 vs §3.4 k-range tension (one joint k, HC50-labeled-size floor), and doc08 vs doc07 §3.2 framing of homology partition as cross-check vs. final split (embedding-cluster LOCO is default, homology partition is escalation fallback)
- [x] Draft doc 11 outline (Phase 3: model selection pilot) — 2026-07-23 17:20 UTC
- [x] Write doc 11 in full (`MD_design_docs/11_phase3_implementation_design.md`) — 2026-07-23 17:40 UTC — named package `clamp.model_selection` (not `modeling`/`train`) deliberately to avoid pre-empting Phase 4's namespace; reuses `clamp.splitting.embed`'s loader/mean_pool rather than duplicating
- [x] Self-review both docs against doc 09's bar (concreteness, options+recommendation, DoD, runbook, open questions) — 2026-07-23 17:50 UTC — verified cross-doc consistency (doc11 reuses doc10's `load_encoder`/`mean_pool` signatures exactly), verified all doc05/doc07 section citations against the source text, confirmed no files under src/ or tests/ were touched (docs-only, `git status` shows only the two new MD files + this one)
- [x] Commit — 2026-07-23 17:52 UTC
- [x] Push branch — 2026-07-23 17:53 UTC
- [x] Open PR to Eitzl/CLAMP master — 2026-07-23 17:56 UTC — https://github.com/Eitzl/CLAMP/pull/5
