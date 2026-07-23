"""p2smi adapter for cyclized / non-canonical-residue / terminally-modified
peptides.

Implemented directly against the installed `p2smi==1.1.1` package source
(github.com/aaronfeller/p2smi), not just its docs, since the docs alone
under-specify the exact control flow:

- Our `CyclizationType` enum values (HT, SS, SCSC, SCNT, SCCT) already
  match p2smi's own tag vocabulary by construction (doc 06 §2.1) — no
  translation table needed for the tag itself.
- `p2smi.fasta2smi.constraint_resolver(sequence, tag)` is what actually
  places the bond: it inspects the sequence's own chemistry (e.g. which
  residues are disulfide-capable) and returns a resolved position mask.
  We use this instead of hand-building a mask, because our schema has no
  field for "which exact residue positions form this bond" — only the
  cyclization *type*.
- Confirmed by reading `constraint_resolver` directly: when a requested
  cyclization type can't actually be placed on the given sequence (e.g.
  SS requested but fewer than two disulfide-capable residues), it does
  **not** raise — it silently falls back to `pattern=""` (linear). We
  detect that silent fallback (empty pattern despite a non-empty tag)
  and record it as a dropped modification, since relying on the
  exception path alone would miss this.
- Unusual residues (from PeptideRecord.unusual_residues) are recorded as
  dropped modifications rather than attempted-and-substituted: our schema
  doesn't carry a vocabulary mapping from free-text modification_type
  strings to one of p2smi's ~thousands of named non-canonical residues,
  and data/README.md §2 is explicit that this vocabulary is source-specific
  and should grow from real data, not be guessed. The base sequence (using
  the standard-letter approximation at that position) still converts, so
  the result is PARTIAL fidelity rather than FAILED — accurate backbone,
  lost side-chain/stereochemistry detail.
"""

from p2smi import fasta2smi
from p2smi.utilities import smilesgen

from clamp.data.schema import CyclizationType, PeptideRecord


def build_p2smi_fasta_header(record: PeptideRecord) -> str:
    """The `|CONSTRAINT` half of a p2smi FASTA header (`>name|CONSTRAINT`).
    Empty string for records with no cyclization (or an unresolved
    UNKNOWN cyclization_type, which p2smi has no tag for)."""
    if record.cyclization_type in (CyclizationType.NONE, CyclizationType.UNKNOWN):
        return ""
    return record.cyclization_type.value


def convert_via_p2smi(record: PeptideRecord) -> tuple[str | None, list[str]]:
    """Run p2smi on record, returning (smiles, dropped_modifications).

    dropped_modifications is non-empty when p2smi's residue table can't
    represent something in record.unusual_residues, or its
    constraint_resolver can't place the requested cyclization on this
    sequence — this is exactly the signal that should downgrade the
    result to FidelityTier.PARTIAL rather than FULL (doc 06 §5.1).
    Returns (None, dropped) only if the base sequence itself contains a
    character p2smi doesn't recognize at all.
    """
    # Case is NOT normalized before this point: p2smi's own residue table
    # (p2smi.utilities.aminoacids.all_aminos) uses case to distinguish D-
    # from L-form standard residues (e.g. 'a' = D-Alanine, 'A' = L-Alanine)
    # — uppercasing here would silently generate the wrong stereochemistry
    # for any D-containing sequence.
    sequence = list(record.sequence_canonical or record.sequence_raw)
    dropped: list[str] = []

    tag = build_p2smi_fasta_header(record)
    pattern = ""
    if tag:
        try:
            _, pattern = fasta2smi.constraint_resolver(sequence, tag)
        except (fasta2smi.InvalidConstraintError, smilesgen.UndefinedAminoError):
            # constraint_resolver's happy path only raises
            # InvalidConstraintError, but it calls smilesgen.what_constraints
            # internally, which raises UndefinedAminoError (a *different*
            # exception hierarchy — defined in smilesgen, not fasta2smi) the
            # moment the sequence has a letter it doesn't recognize. Caught
            # here so an unmappable sequence still degrades to the documented
            # (None, dropped) contract via the final constrained_peptide_smiles
            # call below, instead of crashing this function outright.
            pattern = ""
        if not pattern:
            dropped.append(f"cyclization:{tag}:unresolvable_on_sequence")
    elif record.cyclization_type == CyclizationType.UNKNOWN:
        dropped.append("cyclization:unknown_type:not_representable")

    if record.nterm_mod and not tag:
        dropped.append(f"nterm_mod:{record.nterm_mod}:not_representable")
    if record.cterm_mod and not tag:
        dropped.append(f"cterm_mod:{record.cterm_mod}:not_representable")

    for residue in record.unusual_residues:
        dropped.append(f"unusual_residue:{residue.position}:{residue.modification_type}:not_representable")

    try:
        _, _, smiles = smilesgen.constrained_peptide_smiles(sequence, pattern)
    except (KeyError, smilesgen.UndefinedAminoError):
        dropped.append("sequence:contains_residue_unknown_to_p2smi")
        return None, dropped

    return smiles, dropped
