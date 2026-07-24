# Phase 1 Code Review Progress

- [x] read design doc (09_phase1_implementation_design.md)
- [x] read data/README.md
- [x] review schema.py
- [x] review config.py + cli.py
- [x] review all 5 source pullers (base, dbaasp, dramp, hemolytik2, hemopi2, qmap)
- [x] review convert/ (validate, p2smi_adapter, rdkit_backbone, helm, __init__ dispatch)
- [x] review dedup.py
- [x] review normalize.py
- [x] review datasheet.py
- [x] review pipeline.py
- [x] review all tests + fixtures
- [x] fix low-risk bug: QMAP MIC consensus NaN guard + regression test
- [x] open PR #4 (https://github.com/Eitzl/CLAMP/pull/4)
- [x] final report
- [x] follow-up (2026-07-23, priority, orchestrator-applied): fixed F2 (case-folding destroyed D/L stereochemistry — canonicalize_sequence and p2smi_adapter no longer uppercase; convert dispatch now routes any lowercase-containing sequence to p2smi) and F3 (mod-code case/whitespace normalization in modification_signature, scoped down from a full name->abbreviation table after confirming real DBAASP data already uses short codes, not spelled-out names) directly onto this branch; 9 new regression tests added, all 172 tests pass
