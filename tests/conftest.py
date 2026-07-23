"""Shared pytest fixtures.

Fixture files under tests/data/fixtures/{source}/ are real records
captured live from each source on 2026-07-22 (doc 09 §12) — a handful of
representative rows per source (DBAASP: a plain monomer, a multimer, an
unusual-residue record, a head-to-tail-bonded record; DRAMP: general_amps
+ general_smiles samples with an overlapping id to exercise the join;
Hemolytik2: a captured API response spanning clean/percentage/cyclic/
non-hemolytic activity-text forms; HemoPI2: a slice of the real GitHub
Dataset/ CSVs; QMAP: a slice of real DBAASPDataset samples spanning DSB/
head-to-tail-AMD/plain bond shapes) — not hand-invented shapes.
"""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "data" / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR
