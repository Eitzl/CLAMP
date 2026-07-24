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

# 8 rows (0.003%) in the full DBAASP pull are literal source-database
# curation errors, not conversion bugs -- e.g. id 24641's raw concentration
# field is the string "2E46" (-> 2e46 uM), id 10002's is "1814000" ug/mL
# (-> ~1e6 uM, 1.8 kg/mL, physically impossible to dissolve). The largest
# value below this cutoff in the real full-scale data is ~2.19e5 uM, a
# ~5x gap, so 1e6 cleanly separates genuine (if extreme) assay values from
# these typos without touching anything real. No real HC50/MIC assay
# reports a molar-scale concentration.
_MAX_PLAUSIBLE_UM = 1e6


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
    exactly zero".

    A real full-scale DBAASP pull surfaced ~90 rows (0.036% of the full
    dataset) whose range string itself carries a stray inequality
    qualifier on one or both bounds (e.g. "16->32", "<=0.03-0.12",
    "8 - =>16") — not the plain hyphen-range parse_range expects. Given
    the negligible volume, these are treated as unparseable and dropped
    (label -> None for this row) rather than special-cased, the same
    "anything else unparseable" simplification DBAASP's own
    _parse_concentration already applies."""
    if is_range and range_raw:
        try:
            lo, hi = parse_range(range_raw)
        except ValueError:
            return None
        value = geometric_mean(lo, hi)
    if value is None:
        return None
    if unit is None:
        # 27 rows (0.011%) in the full DBAASP pull have a numeric
        # concentration but no unit field at all — can't convert without
        # knowing the unit, so treat the same as a missing label.
        return None
    unit_key = unit.strip().lower()
    if unit_key in _UGML_UNIT_ALIASES:
        converted = ug_per_ml_to_uM(value, molecular_weight)
    elif unit_key in _UM_UNIT_ALIASES:
        converted = value
    else:
        raise ValueError(f"unrecognized unit: {unit!r}")
    return None if converted >= _MAX_PLAUSIBLE_UM else converted


def log_transform(value_uM: float | None) -> float | None:
    """log10 is undefined at/below zero. A real full-scale pull surfaced 2
    range-valued rows with a literal "0" lower bound (e.g. "0-6.06"),
    whose geometric_mean is exactly 0 — not a plausible concentration
    measurement, so treated as missing rather than raising."""
    if value_uM is None or value_uM <= 0:
        return None
    return math.log10(value_uM)
