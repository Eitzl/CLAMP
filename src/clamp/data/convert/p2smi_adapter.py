"""p2smi adapter for cyclized / non-canonical-residue / terminally-modified
peptides.

MD_design_docs/06_task5_data_pipeline_plan.md §2.1-§2.2. p2smi
(github.com/AaronFeller/p2smi, built in support of PeptideCLM) takes FASTA
input with cyclization encoded via a constraint mask in the header
(X=unconstrained, C=disulfide-participating, N/Z=N-/C-terminus-
participating) and supports 5 cyclization types: SS, HT, SCSC, SCNT, SCCT —
these map directly to schema.CyclizationType.

The DBAASP `intrachainBonds[].cycleType` -> p2smi header-tag mapping is
"a small, well-scoped adapter script, not a research problem" per doc 06
§2.1 — implement it here once p2smi's exact header format is confirmed
against the installed version.
"""

from clamp.data.schema import PeptideRecord


def build_p2smi_fasta_header(record: PeptideRecord) -> str:
    """Construct the p2smi constraint-mask FASTA header for this record from
    its cyclization_type and unusual_residues."""
    raise NotImplementedError


def convert_via_p2smi(record: PeptideRecord) -> tuple[str, list[str]]:
    """Run p2smi on record, returning (smiles, dropped_modifications).

    dropped_modifications is non-empty when p2smi's residue table can't
    represent something in record.unusual_residues — this is exactly the
    signal that should downgrade the result to FidelityTier.PARTIAL rather
    than FULL (doc 06 §5.1).
    """
    raise NotImplementedError
