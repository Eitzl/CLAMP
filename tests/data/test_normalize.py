"""Coverage per MD_design_docs/09_phase1_implementation_design.md §12."""

import math

import pytest

from clamp.data.normalize import geometric_mean, log_transform, normalize_concentration, parse_range, ug_per_ml_to_uM


class TestNormalizeConcentration:
    def test_uM_passthrough(self):
        assert normalize_concentration(10.0, "uM", molecular_weight=1000.0) == 10.0

    @pytest.mark.parametrize("unit", ["uM", "µM", "μM", "UM"])
    def test_uM_unit_aliases_all_accepted(self, unit):
        assert normalize_concentration(10.0, unit, molecular_weight=1000.0) == 10.0

    def test_ug_per_ml_converts_using_smiles_derived_mw(self):
        # 10 ug/ml at MW 1000 -> (10/1000)*1000 = 10 uM
        assert normalize_concentration(10.0, "ug/ml", molecular_weight=1000.0) == pytest.approx(10.0)
        assert normalize_concentration(10.0, "ug/ml", molecular_weight=1000.0) == ug_per_ml_to_uM(10.0, 1000.0)

    @pytest.mark.parametrize("unit", ["ug/ml", "µg/ml", "μg/ml", "UG/ML"])
    def test_ugml_unit_aliases_all_accepted(self, unit):
        assert normalize_concentration(10.0, unit, molecular_weight=1000.0) == pytest.approx(10.0)

    def test_range_string_collapses_via_geometric_mean(self):
        result = normalize_concentration(None, "uM", molecular_weight=1000.0, is_range=True, range_raw="1.58-25.33")
        assert result == pytest.approx(geometric_mean(1.58, 25.33))

    def test_none_value_returns_none_not_zero_or_nan(self):
        """log_transform must be able to distinguish "no label" from
        "label is exactly zero"."""
        result = normalize_concentration(None, "uM", molecular_weight=1000.0)
        assert result is None

    def test_unrecognized_unit_raises(self):
        with pytest.raises(ValueError):
            normalize_concentration(10.0, "mg/L", molecular_weight=1000.0)


class TestBoundaryValues:
    """MIC/HC50 are physical concentrations that shouldn't legitimately be
    <= 0, and molecular weight should never legitimately be 0 for a valid
    SMILES — so these aren't "normal" inputs. But the current behavior on
    them was previously untested and undocumented, meaning a change here
    would be a silent behavior change rather than a caught regression.
    Pinning down what happens TODAY (raises, not a graceful None/clamp) so
    that's a documented contract, not an accident."""

    def test_zero_molecular_weight_raises_zero_division(self):
        with pytest.raises(ZeroDivisionError):
            normalize_concentration(10.0, "ug/ml", molecular_weight=0.0)

    def test_zero_value_log_transform_raises_math_domain_error(self):
        """log10(0) is undefined — this is exactly why normalize_concentration
        returns None (not 0.0) for missing input, per its own docstring;
        but nothing currently stops a genuine 0.0 concentration value from
        reaching log_transform and raising here."""
        with pytest.raises(ValueError):
            log_transform(0.0)

    def test_negative_value_log_transform_raises_math_domain_error(self):
        with pytest.raises(ValueError):
            log_transform(-5.0)

    def test_zero_concentration_passes_through_normalize_concentration_unchanged(self):
        """normalize_concentration itself does NOT reject 0.0 — only the
        downstream log_transform call blows up on it. Documented here so
        the boundary between the two functions' responsibilities is clear."""
        assert normalize_concentration(0.0, "uM", molecular_weight=1000.0) == 0.0


class TestParseRange:
    def test_parses_lo_hi(self):
        assert parse_range("1.58-25.33") == (1.58, 25.33)

    def test_handles_spaces(self):
        assert parse_range("1.58 - 25.33") == (1.58, 25.33)


class TestGeometricMean:
    def test_known_value(self):
        assert geometric_mean(4.0, 9.0) == pytest.approx(6.0)


class TestLogTransform:
    def test_none_passes_through_as_none(self):
        assert log_transform(None) is None

    def test_positive_value_transforms(self):
        assert log_transform(10.0) == pytest.approx(1.0)

    def test_distinguishes_missing_from_zero(self):
        assert log_transform(None) is None
        assert log_transform(1.0) == 0.0
        assert not math.isnan(log_transform(1.0))
