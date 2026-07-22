"""SMILES conversion dispatch.

Implements the decision table in MD_design_docs/06_task5_data_pipeline_plan.md
§2.2, as an explicit dispatch (not a buried if-chain) so each path is
independently testable and the fidelity tier falls out of which path ran.
See MD_design_docs/09_phase1_implementation_design.md §6.
"""

from pydantic import BaseModel

from clamp.data.schema import FidelityTier, PeptideRecord, SmilesSource


class ConversionResult(BaseModel):
    smiles: str | None
    smiles_source: SmilesSource
    fidelity_tier: FidelityTier
    dropped_modifications: list[str]
    validated: bool


def convert_record(record: PeptideRecord, curated_smiles: str | None = None) -> ConversionResult:
    """Dispatch a record to the appropriate conversion path.

    curated_smiles -> validate directly (db_curated)
    cyclic / has unusual residues / has terminal mods -> p2smi
    otherwise -> plain RDKit Chem.MolFromSequence
    """
    raise NotImplementedError
