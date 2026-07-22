"""Plain-backbone SMILES generation for linear, unmodified, canonical-residue
peptides via RDKit.

MD_design_docs/06_task5_data_pipeline_plan.md §2.2 ("Linear, unmodified,
canonical-residue peptide" row). This is also the fallback path when
modification metadata doesn't map cleanly to p2smi/HELM (doc 06 §2.2's
"Fall back to plain-backbone" row) — callers set
smiles_source=PLAIN_BACKBONE_FALLBACK instead of RDKIT_SEQUENCE in that
case; this module only produces the SMILES, it doesn't decide which
smiles_source label applies.
"""

from clamp.data.schema import PeptideRecord


def sequence_to_smiles(record: PeptideRecord) -> str:
    """RDKit Chem.MolFromSequence(record.sequence_canonical) -> canonical
    SMILES string."""
    raise NotImplementedError
