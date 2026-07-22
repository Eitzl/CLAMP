"""Unified peptide-record data model.

Executable form of the schema in MD_design_docs/06_task5_data_pipeline_plan.md
§4.1, as scaffolded in MD_design_docs/09_phase1_implementation_design.md §3.

One PeptideRecord = one (peptide-construct, assay-record) row, per doc 06
§4.1 — a peptide with MIC measured against two species is two records
sharing a peptide_uid, not one record with two MIC columns. This is what
lets the Phase 4 masked-multitask loss (doc 04) treat HC50/MIC presence as
an independent per-row mask.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class Source(StrEnum):
    DBAASP = "dbaasp"
    DRAMP = "dramp"
    APD3 = "apd3"
    HEMOPI2 = "hemopi2"
    HEMOLYTIK2 = "hemolytik2"
    QMAP = "qmap"


class CyclizationType(StrEnum):
    HEAD_TO_TAIL = "HT"
    SIDE_CHAIN_TO_SIDE_CHAIN = "SCSC"
    SIDE_CHAIN_TO_N_TERM = "SCNT"
    SIDE_CHAIN_TO_C_TERM = "SCCT"
    DISULFIDE = "SS"
    NONE = "none"
    UNKNOWN = "unknown"


class SmilesSource(StrEnum):
    DB_CURATED = "db_curated"
    RDKIT_SEQUENCE = "rdkit_sequence"
    P2SMI = "p2smi"
    HELM_TOOLCHAIN = "helm_toolchain"
    PLAIN_BACKBONE_FALLBACK = "plain_backbone_fallback"


class FidelityTier(StrEnum):
    FULL = "full"
    PARTIAL = "partial"
    BACKBONE_ONLY = "backbone_only"
    FAILED = "failed"


class LabelQualityFlag(StrEnum):
    """Set by dedup.py when a label needs a second look before modeling.
    See data/README.md §3.5 for the case this exists to catch: two sources
    reporting wildly different values for the same peptide_uid under the
    same assay conditions, which is more likely a unit/transcription error
    than genuine replicate noise."""

    RANGE_ONLY = "range_only"
    HIGH_REPLICATE_DISPERSION = "high_replicate_dispersion"
    METADATA_SOURCE_CONFLICT = "metadata_source_conflict"


class UnusualResidue(BaseModel):
    position: int
    from_residue: str | None = None
    modification_type: str


class PeptideRecord(BaseModel):
    # identity / provenance
    source: Source
    source_id: str
    sequence_raw: str
    sequence_canonical: str | None = None  # filled by dedup.canonicalize_sequence
    peptide_uid: str | None = None  # filled by dedup.compute_peptide_uid

    # structure
    is_multimer: bool = False
    is_cyclic: bool = False
    cyclization_type: CyclizationType = CyclizationType.NONE
    nterm_mod: str | None = None
    cterm_mod: str | None = None
    unusual_residues: list[UnusualResidue] = Field(default_factory=list)

    # chemistry (filled by convert.convert_record)
    smiles: str | None = None
    smiles_source: SmilesSource | None = None
    smiles_validated: bool = False
    chemical_fidelity_tier: FidelityTier | None = None
    dropped_modifications: list[str] = Field(default_factory=list)

    # labels (raw, pre-normalization)
    hc50_value: float | None = None
    hc50_unit: str | None = None
    hc50_assay_target_cell: str | None = None
    mic_value: float | None = None
    mic_unit: str | None = None
    mic_target_species: str | None = None
    mic_assay_medium: str | None = None
    label_is_range: bool = False
    label_range_raw: str | None = None
    label_quality_flag: LabelQualityFlag | None = None

    # labels (filled by normalize.normalize_concentration / log_transform)
    hc50_value_uM: float | None = None
    hc50_log_uM: float | None = None
    mic_value_uM: float | None = None
    mic_log_uM: float | None = None

    reference: str | None = None
    duplicate_of: str | None = None
