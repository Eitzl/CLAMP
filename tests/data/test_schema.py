"""schema.py is fully implemented (no stubs) — these are real tests, not
placeholders, and double as an import smoke test for the whole package."""

import pytest
from pydantic import ValidationError

from clamp.data.schema import (
    CyclizationType,
    FidelityTier,
    PeptideRecord,
    Source,
    UnusualResidue,
)


def test_minimal_record_constructs():
    record = PeptideRecord(source=Source.DBAASP, source_id="16", sequence_raw="KWKLFKKIEK")
    assert record.is_cyclic is False
    assert record.cyclization_type == CyclizationType.NONE
    assert record.unusual_residues == []


def test_missing_required_field_raises():
    with pytest.raises(ValidationError):
        PeptideRecord(source=Source.DBAASP, sequence_raw="KWKLFKKIEK")  # missing source_id


def test_invalid_enum_value_raises():
    with pytest.raises(ValidationError):
        PeptideRecord(source="not_a_real_source", source_id="16", sequence_raw="KWKLFKKIEK")


def test_unusual_residue_round_trips():
    record = PeptideRecord(
        source=Source.DBAASP,
        source_id="16",
        sequence_raw="kWKLFKKIEK",
        unusual_residues=[UnusualResidue(position=1, from_residue="K", modification_type="D-amino-acid")],
    )
    assert record.unusual_residues[0].modification_type == "D-amino-acid"


def test_fidelity_tier_defaults_unset():
    record = PeptideRecord(source=Source.QMAP, source_id="1", sequence_raw="KWKLFKKIEK")
    assert record.chemical_fidelity_tier is None
    assert record.smiles_source is None


@pytest.mark.parametrize("tier", list(FidelityTier))
def test_all_fidelity_tiers_assignable(tier):
    record = PeptideRecord(
        source=Source.DBAASP, source_id="16", sequence_raw="KWKLFKKIEK", chemical_fidelity_tier=tier
    )
    assert record.chemical_fidelity_tier == tier
