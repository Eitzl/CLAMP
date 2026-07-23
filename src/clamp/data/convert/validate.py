"""SMILES validation: round-trip parse + mass-sanity check.

MD_design_docs/06_task5_data_pipeline_plan.md §2.3. This is the guard
against silently reintroducing a class of bug PeptideCLM v1's pretraining
data actually had (ring-numbering bug fixed in v1.1, per doc 02 §5.2) —
flagged in doc 09 §12 as the primary hypothesis-testing target.
"""

from rdkit import Chem
from rdkit.Chem import Descriptors


def round_trips(smiles: str) -> bool:
    """Chem.MolFromSmiles parses without error, and re-canonicalizing is
    idempotent."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    once = Chem.MolToSmiles(mol)
    twice_mol = Chem.MolFromSmiles(once)
    if twice_mol is None:
        return False
    return Chem.MolToSmiles(twice_mol) == once


def mass_sanity_check(smiles: str, expected_mass: float, tolerance_frac: float = 0.05) -> bool:
    """RDKit-computed MolWt of `smiles` is within tolerance_frac of
    expected_mass (residue-count-derived estimate, pre-conversion)."""
    if expected_mass <= 0:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return abs(Descriptors.MolWt(mol) - expected_mass) / expected_mass <= tolerance_frac
