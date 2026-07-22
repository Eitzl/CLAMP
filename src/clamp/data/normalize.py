"""Unit normalization + log transform.

MD_design_docs/06_task5_data_pipeline_plan.md §4.2, scaffolded in
MD_design_docs/09_phase1_implementation_design.md §8. Molecular weight
input must be the RDKit-computed MW of the *generated SMILES*
(convert.convert_record's output), not a residue-count estimate — this
matters most for exactly the modified/cyclic peptides the SMILES-LM
approach exists to handle correctly (data/README.md §2).
"""


def ug_per_ml_to_uM(value_ug_ml: float, molecular_weight: float) -> float:
    raise NotImplementedError


def parse_range(raw: str) -> tuple[float, float]:
    """Parse a DBAASP-style range string, e.g. "1.58-25.33"."""
    raise NotImplementedError


def geometric_mean(lo: float, hi: float) -> float:
    raise NotImplementedError


def normalize_concentration(
    value: float | None,
    unit: str,
    molecular_weight: float,
    is_range: bool = False,
    range_raw: str | None = None,
) -> float | None:
    """Convert to uM. Ranges collapse via geometric_mean per the documented
    convention (doc 06 §4.2). Returns None (not 0/NaN) for missing input —
    log_transform must be able to distinguish "no label" from "label is
    exactly zero"."""
    raise NotImplementedError


def log_transform(value_uM: float | None) -> float | None:
    raise NotImplementedError
