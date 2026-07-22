"""Intended coverage per MD_design_docs/09_phase1_implementation_design.md
§12. Skipped until src/clamp/data/normalize.py is implemented.
"""

import pytest

pytestmark = pytest.mark.skip(reason="normalize.py is a NotImplementedError stub — see doc 09 §12")


class TestNormalizeConcentration:
    def test_uM_passthrough(self):
        pass

    def test_ug_per_ml_converts_using_smiles_derived_mw(self):
        pass

    def test_range_string_collapses_via_geometric_mean(self):
        pass

    def test_none_value_returns_none_not_zero_or_nan(self):
        """log_transform must be able to distinguish "no label" from
        "label is exactly zero"."""

    def test_unrecognized_unit_raises(self):
        pass


class TestLogTransform:
    def test_none_passes_through_as_none(self):
        pass

    def test_positive_value_transforms(self):
        pass
