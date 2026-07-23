"""Unit normalization + log transform.

MD_design_docs/06_task5_data_pipeline_plan.md §4.2, scaffolded in
MD_design_docs/09_phase1_implementation_design.md §8. Molecular weight
input must be the RDKit-computed MW of the *generated SMILES*
(convert.convert_record's output), not a residue-count estimate — this
matters most for exactly the modified/cyclic peptides the SMILES-LM
approach exists to handle correctly (data/README.md §2).
"""

import math

# Unit aliases observed live across sources: DBAASP's own JSON uses the
# MICRO SIGN (U+00B5, "µM"); free text elsewhere sometimes uses GREEK
# SMALL LETTER MU (U+03BC, "μM") instead — both spellings are handled so a
# real-vs-typed-Unicode mismatch doesn't silently raise as "unrecognized".
_UM_UNIT_ALIASES = {"um", "µm", "μm"}
_UGML_UNIT_ALIASES = {"ug/ml", "µg/ml", "μg/ml"}


def ug_per_ml_to_uM(value_ug_ml: float, molecular_weight: float) -> float:
    return (value_ug_ml / molecular_weight) * 1000


def parse_range(raw: str) -> tuple[float, float]:
    """Parse a DBAASP-style range string, e.g. "1.58-25.33"."""
    lo, hi = (float(x) for x in raw.replace(" ", "").split("-"))
    return lo, hi


def geometric_mean(lo: float, hi: float) -> float:
    return math.sqrt(lo * hi)


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
    if is_range and range_raw:
        lo, hi = parse_range(range_raw)
        value = geometric_mean(lo, hi)
    if value is None:
        return None
    unit_key = unit.strip().lower()
    if unit_key in _UGML_UNIT_ALIASES:
        return ug_per_ml_to_uM(value, molecular_weight)
    if unit_key in _UM_UNIT_ALIASES:
        return value
    raise ValueError(f"unrecognized unit: {unit!r}")


def log_transform(value_uM: float | None) -> float | None:
    return math.log10(value_uM) if value_uM is not None else None
