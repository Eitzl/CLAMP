"""SMILES conversion dispatch.

Implements the decision table in MD_design_docs/06_task5_data_pipeline_plan.md
§2.2, as an explicit dispatch (not a buried if-chain) so each path is
independently testable and the fidelity tier falls out of which path ran.
See MD_design_docs/09_phase1_implementation_design.md §6.
"""

from pydantic import BaseModel

from clamp.data.convert.p2smi_adapter import convert_via_p2smi
from clamp.data.convert.rdkit_backbone import sequence_to_smiles
from clamp.data.convert.validate import mass_sanity_check, round_trips
from clamp.data.schema import FidelityTier, PeptideRecord, SmilesSource

# Standard average residue masses (Da) — i.e. amino acid mass minus the
# water lost on peptide-bond formation. Used ONLY as the pre-conversion
# expected-mass estimate for the mass-sanity gate (doc 06 §4.2), never for
# label normalization (which always uses the RDKit-computed MW of the
# actual generated SMILES). A flat per-residue average (e.g. 110 Da) was
# tried first and rejected: it's off by >20% for real Lys/Trp/Phe-heavy
# sequences, which would fail nearly every legitimate conversion at the
# tolerance doc 06 specifies — a real per-residue table is what makes the
# mass-sanity check catch the class of bug it's meant to (doc 02 §5.2's
# ring-numbering bug), not composition variance.
_RESIDUE_MASS_DA = {
    "G": 57.0519, "A": 71.0788, "S": 87.0782, "P": 97.1167, "V": 99.1326,
    "T": 101.1051, "C": 103.1388, "L": 113.1594, "I": 113.1594, "N": 114.1038,
    "D": 115.0886, "Q": 128.1307, "K": 128.1741, "E": 129.1155, "M": 131.1926,
    "H": 137.1411, "F": 147.1766, "R": 156.1875, "Y": 163.1760, "W": 186.2132,
}
_WATER_MASS_DA = 18.0153
_FALLBACK_RESIDUE_MASS_DA = 110.0  # for any character outside the 20 standard residues


def _estimate_peptide_mass(sequence: str) -> float:
    residue_sum = sum(_RESIDUE_MASS_DA.get(ch, _FALLBACK_RESIDUE_MASS_DA) for ch in sequence.upper())
    return residue_sum + _WATER_MASS_DA if sequence else 0.0


class ConversionResult(BaseModel):
    smiles: str | None
    smiles_source: SmilesSource
    fidelity_tier: FidelityTier
    dropped_modifications: list[str]
    validated: bool


def convert_record(record: PeptideRecord, curated_smiles: str | None = None) -> ConversionResult:
    """Dispatch a record to the appropriate conversion path.

    curated_smiles -> validate directly (db_curated)
    cyclic / has unusual residues / has terminal mods -> p2smi, falling
      back to plain-backbone if p2smi can't place the sequence at all
    otherwise -> plain RDKit Chem.MolFromSequence
    """
    if curated_smiles is not None:
        return _validate(curated_smiles, SmilesSource.DB_CURATED, record, dropped_modifications=[])

    needs_p2smi = bool(record.is_cyclic or record.unusual_residues or record.nterm_mod or record.cterm_mod)
    if needs_p2smi:
        smiles, dropped = convert_via_p2smi(record)
        if smiles is not None:
            return _validate(smiles, SmilesSource.P2SMI, record, dropped_modifications=dropped)

        fallback_smiles = sequence_to_smiles(record)
        if fallback_smiles is None:
            return ConversionResult(
                smiles=None,
                smiles_source=SmilesSource.P2SMI,
                fidelity_tier=FidelityTier.FAILED,
                dropped_modifications=dropped,
                validated=False,
            )
        return _validate(
            fallback_smiles,
            SmilesSource.PLAIN_BACKBONE_FALLBACK,
            record,
            dropped_modifications=[*dropped, "p2smi_conversion_failed_used_plain_backbone"],
        )

    smiles = sequence_to_smiles(record)
    if smiles is None:
        return ConversionResult(
            smiles=None,
            smiles_source=SmilesSource.RDKIT_SEQUENCE,
            fidelity_tier=FidelityTier.FAILED,
            dropped_modifications=[],
            validated=False,
        )
    return _validate(smiles, SmilesSource.RDKIT_SEQUENCE, record, dropped_modifications=[])


def _validate(
    smiles: str,
    source: SmilesSource,
    record: PeptideRecord,
    dropped_modifications: list[str],
) -> ConversionResult:
    """Round-trip parse + mass-sanity check, doc 06 §2.3.

    The mass-sanity check is skipped (round_trips alone still applies) for
    a DB_CURATED record that carries a modification _estimate_peptide_mass
    has no way to account for (terminal acylation, cyclization, an
    unusual residue) — found via a real ~90k-row pilot pull: DBAASP id 51
    ("KLlK" with an N-terminal palmitoyl/C16 modification) has a correct,
    round-trip-clean curated SMILES whose real mass is ~738 Da against a
    naive bare-backbone estimate of ~501 Da (47% off, far outside the 5%
    default tolerance) — the curated SMILES is right, the estimator just
    can't see the modification. Quantified impact: 1,307 of 4,011 FAILED
    rows in that pilot (32.6%), 697 of which had exactly this shape. Only
    applies to DB_CURATED, not p2smi/RDKit-generated paths — those are
    generated FROM the same modification fields, so a mismatch there is
    still a meaningful signal, not a blind spot in the estimator.
    """
    expected_mass = _estimate_peptide_mass(record.sequence_canonical or record.sequence_raw)
    has_unestimated_modification = bool(
        record.nterm_mod or record.cterm_mod or record.is_cyclic or record.unusual_residues
    )
    skip_mass_check = source == SmilesSource.DB_CURATED and has_unestimated_modification
    valid = round_trips(smiles) and (skip_mass_check or mass_sanity_check(smiles, expected_mass))

    if not valid:
        return ConversionResult(
            smiles=smiles,
            smiles_source=source,
            fidelity_tier=FidelityTier.FAILED,
            dropped_modifications=dropped_modifications,
            validated=False,
        )

    # PLAIN_BACKBONE_FALLBACK is checked first, not after dropped_modifications:
    # convert_record always appends a marker string when it falls back
    # (see "p2smi_conversion_failed_used_plain_backbone" above), so
    # dropped_modifications is *never* empty on this path — checking it
    # first would make BACKBONE_ONLY unreachable dead code, silently
    # reporting every fallback as merely PARTIAL instead of flagging the
    # more severe "lost the p2smi structure entirely" case.
    if source == SmilesSource.PLAIN_BACKBONE_FALLBACK:
        tier = FidelityTier.BACKBONE_ONLY
    elif dropped_modifications:
        tier = FidelityTier.PARTIAL
    else:
        tier = FidelityTier.FULL

    return ConversionResult(
        smiles=smiles,
        smiles_source=source,
        fidelity_tier=tier,
        dropped_modifications=dropped_modifications,
        validated=True,
    )
