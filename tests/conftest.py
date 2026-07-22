"""Shared pytest fixtures.

Fixture files under tests/data/fixtures/{source}/ are meant to be real,
captured records per source (doc 09 §12) — currently empty placeholders
(tests/data/fixtures/{dbaasp,dramp,hemolytik2,hemopi2,qmap}/) pending a
first real pull. Populate these before writing the tests that depend on
them.
"""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "data" / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR
