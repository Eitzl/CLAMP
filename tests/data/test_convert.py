"""Coverage per MD_design_docs/09_phase1_implementation_design.md §12."""

from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st
from rdkit import Chem
from rdkit.Chem import Descriptors

from clamp.data.convert import convert_record
from clamp.data.convert.p2smi_adapter import build_p2smi_fasta_header, convert_via_p2smi
from clamp.data.convert.rdkit_backbone import sequence_to_smiles
from clamp.data.convert.validate import mass_sanity_check, round_trips
from clamp.data.schema import CyclizationType, FidelityTier, PeptideRecord, Source, SmilesSource, UnusualResidue

_LINEAR_SEQUENCE = "KWKLFKKIEK"
_CANONICAL_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def _linear_record(sequence: str = _LINEAR_SEQUENCE, **kwargs) -> PeptideRecord:
    return PeptideRecord(source=Source.DBAASP, source_id="1", sequence_raw=sequence, **kwargs)


class TestRoundTrips:
    def test_valid_smiles_round_trips(self):
        smiles = sequence_to_smiles(_linear_record())
        assert round_trips(smiles) is True

    def test_malformed_smiles_fails(self):
        assert round_trips("not a smiles!!") is False

    @given(st.text(alphabet=_CANONICAL_ALPHABET, min_size=1, max_size=25))
    @settings(max_examples=50)
    def test_round_trips_property_holds_for_any_canonical_linear_peptide(self, sequence):
        """Actually property-based (hypothesis generates novel sequences
        here, not a fixed sample) — but scoped honestly: this only exercises
        the plain linear RDKit backbone path (rdkit_backbone.py), which
        never touches ring-closure numbering at all. It's a real regression
        guard for "RDKit's own canonical SMILES round-trips," just not for
        the ring-numbering bug class specifically — see
        test_p2smi_cyclic_round_trips_property below for that."""
        smiles = sequence_to_smiles(_linear_record(sequence))
        assert smiles is not None
        assert round_trips(smiles) is True

    @given(
        st.text(alphabet=_CANONICAL_ALPHABET, min_size=3, max_size=20),
        st.sampled_from([CyclizationType.HEAD_TO_TAIL, CyclizationType.DISULFIDE]),
    )
    @settings(max_examples=30)
    def test_p2smi_cyclic_round_trips_property(self, sequence, cyclization_type):
        """This is the property test that actually matches the bug class
        the docstring above used to (mis)claim for the plain linear path:
        PeptideCLM v1's ring-numbering bug (doc 02 §5.2) lives in exactly
        this ring-closure code, not in linear backbone generation. Skips
        sequences p2smi can't place the constraint on at all (empty
        smiles) rather than asserting on them — that's convert_via_p2smi's
        own documented "unresolvable" signal, tested separately in
        TestP2smiAdapter, not a round-trip failure."""
        record = _linear_record(sequence, is_cyclic=True, cyclization_type=cyclization_type)
        smiles, _dropped = convert_via_p2smi(record)
        if smiles is None:
            return
        assert round_trips(smiles) is True

    def test_mutated_ring_closure_digit_fails_round_trip(self):
        """A concrete regression case for the ring-numbering bug class:
        corrupting a ring-closure digit should make the SMILES either fail
        to parse or fail to round-trip, not silently produce a different
        valid molecule that still "passes"."""
        smiles = sequence_to_smiles(_linear_record("ACDEFGHIKC"))
        mutated = smiles.replace("1", "9", 1) if "1" in smiles else smiles + ")"
        # Either it fails to parse (round_trips -> False) or, if RDKit is
        # lenient enough to reinterpret it, it must no longer describe the
        # same molecule as the original.
        if round_trips(mutated):
            assert Chem.CanonSmiles(mutated) != Chem.CanonSmiles(smiles)


class TestMassSanityCheck:
    def test_matching_mass_passes(self):
        smiles = sequence_to_smiles(_linear_record())
        from clamp.data.convert import _estimate_peptide_mass

        expected = _estimate_peptide_mass(_LINEAR_SEQUENCE)
        assert mass_sanity_check(smiles, expected) is True

    def test_mismatched_mass_fails_within_tolerance_bound(self):
        smiles = sequence_to_smiles(_linear_record())
        assert mass_sanity_check(smiles, 1.0) is False

    def test_invalid_smiles_fails(self):
        assert mass_sanity_check("garbage", 1000.0) is False

    def test_tolerance_boundary_is_actually_enforced_at_5_percent(self):
        """The previous version of this test only proved "wildly wrong mass
        fails" (expected_mass=1.0 against a ~1300 Da real peptide) — that
        would pass identically whether tolerance_frac were 5%, 50%, or
        500%. This pins down the documented default (0.05) at both edges."""
        smiles = sequence_to_smiles(_linear_record())
        true_mass = Descriptors.MolWt(Chem.MolFromSmiles(smiles))
        # mass_sanity_check's fractional error is |true - expected| / expected
        # (expected is the denominator, not true_mass) — dividing true_mass
        # by (1 + target_frac) makes that fraction land on target_frac
        # exactly: |true - true/(1+t)| / (true/(1+t)) simplifies to t.
        just_inside = true_mass / 1.049  # exactly 4.9% off -> within the 5% default
        just_outside = true_mass / 1.051  # exactly 5.1% off -> outside the 5% default
        assert mass_sanity_check(smiles, just_inside) is True
        assert mass_sanity_check(smiles, just_outside) is False
        # and the tolerance_frac parameter itself must actually be load-bearing,
        # not ignored
        assert mass_sanity_check(smiles, just_outside, tolerance_frac=0.10) is True


class TestConvertDispatch:
    def test_curated_smiles_short_circuits_to_db_curated(self):
        smiles = sequence_to_smiles(_linear_record())
        result = convert_record(_linear_record(), curated_smiles=smiles)
        assert result.smiles_source == SmilesSource.DB_CURATED
        assert result.fidelity_tier == FidelityTier.FULL
        assert result.validated is True

    def test_malformed_curated_smiles_yields_failed_tier(self):
        result = convert_record(_linear_record(), curated_smiles="not a smiles!!")
        assert result.fidelity_tier == FidelityTier.FAILED
        assert result.validated is False

    def test_curated_smiles_with_terminal_modification_is_not_falsely_failed(self):
        """Regression guard for a real bug found via a ~90k-row pilot pull
        (DBAASP id 51, "KLlK" with an N-terminal palmitoyl/C16 modification):
        a correct, round-trip-clean curated SMILES for a terminally-modified
        peptide was getting marked FAILED because _estimate_peptide_mass
        has no way to account for the modification, so the mass-sanity
        check compared the real (heavier, modified) SMILES against a bare
        -backbone estimate and failed — 1,307 of 4,011 FAILED rows in that
        pilot (32.6%) had exactly this shape. This palmitoylated-KLLK
        SMILES (~739 Da) vs. the bare-backbone estimate (~501 Da) is a real
        47% mismatch — the same magnitude as the pilot's actual finding."""
        palmitoylated_smiles = (
            "CCCCCCCCCCCCCCCC(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCCCN)C(=O)O"
        )
        record = _linear_record("KLLK", nterm_mod="C16")
        result = convert_record(record, curated_smiles=palmitoylated_smiles)
        assert result.fidelity_tier == FidelityTier.FULL
        assert result.validated is True
        assert result.smiles == palmitoylated_smiles

    def test_curated_smiles_without_declared_modification_still_gets_mass_checked(self):
        """The skip only applies when the record actually declares a
        modification the estimator can't see — a record with NO nterm_mod/
        cterm_mod/is_cyclic/unusual_residues but a curated SMILES that
        doesn't match its bare sequence at all must still fail. Otherwise
        every DB_CURATED record would silently bypass the mass check."""
        record = _linear_record("KLLK")  # no modifications declared
        palmitoylated_smiles = (
            "CCCCCCCCCCCCCCCC(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCCCN)C(=O)O"
        )
        result = convert_record(record, curated_smiles=palmitoylated_smiles)
        assert result.fidelity_tier == FidelityTier.FAILED
        assert result.validated is False

    def test_cyclic_record_routes_to_p2smi(self):
        record = _linear_record(is_cyclic=True, cyclization_type=CyclizationType.HEAD_TO_TAIL)
        result = convert_record(record)
        assert result.smiles_source == SmilesSource.P2SMI
        assert result.fidelity_tier == FidelityTier.FULL
        assert result.dropped_modifications == []

    def test_linear_unmodified_record_routes_to_rdkit(self):
        result = convert_record(_linear_record())
        assert result.smiles_source == SmilesSource.RDKIT_SEQUENCE
        assert result.fidelity_tier == FidelityTier.FULL

    def test_unusual_residue_downgrades_to_partial_tier(self):
        """doc 06 §5.1: partial fidelity tier when p2smi's residue table
        can't represent something in unusual_residues."""
        record = _linear_record(
            unusual_residues=[UnusualResidue(position=1, from_residue="K", modification_type="D-amino-acid")]
        )
        result = convert_record(record)
        assert result.fidelity_tier == FidelityTier.PARTIAL
        assert result.validated is True
        assert any("unusual_residue" in d for d in result.dropped_modifications)

    def test_unknown_cyclization_type_downgrades_to_partial_not_full(self):
        record = _linear_record(is_cyclic=True, cyclization_type=CyclizationType.UNKNOWN)
        result = convert_record(record)
        assert result.fidelity_tier == FidelityTier.PARTIAL

    def test_nterm_mod_only_routes_to_p2smi_and_is_dropped(self):
        """No cyclization, no unusual residues — nterm_mod alone must still
        trigger needs_p2smi and get recorded as a dropped modification;
        previously untested (0% coverage on this exact branch)."""
        record = _linear_record(nterm_mod="Acetylation")
        result = convert_record(record)
        assert result.smiles_source == SmilesSource.P2SMI
        assert result.fidelity_tier == FidelityTier.PARTIAL
        assert any(d.startswith("nterm_mod:Acetylation") for d in result.dropped_modifications)

    def test_cterm_mod_only_routes_to_p2smi_and_is_dropped(self):
        record = _linear_record(cterm_mod="Amidation")
        result = convert_record(record)
        assert result.fidelity_tier == FidelityTier.PARTIAL
        assert any(d.startswith("cterm_mod:Amidation") for d in result.dropped_modifications)

    def test_lowercase_d_residue_alone_routes_to_p2smi(self):
        """No cyclization/unusual_residues/mods declared — a bare lowercase
        letter in the sequence (QMAP's D-amino-acid convention) must still
        trigger p2smi routing. Previously this fell straight through to the
        plain RDKit path, which ignores letter case entirely and always
        emits the L-form regardless (confirmed directly: Chem.MolFromSequence
        produces identical output for 'A' and 'a')."""
        result = convert_record(_linear_record("KaK"))
        assert result.smiles_source == SmilesSource.P2SMI

    def test_both_p2smi_and_backbone_fallback_fail_yields_failed_tier(self):
        """Forces convert_via_p2smi's terminal (None, dropped) return path —
        the ONLY way convert_record even considers backbone fallback — by
        using a sequence with a character p2smi's residue table (and, as it
        happens, RDKit's own Chem.MolFromSequence) doesn't recognize at
        all. This previously crashed outright with an uncaught
        UndefinedAminoError (constraint_resolver calls smilesgen.what_constraints
        internally, which raises that from a different exception hierarchy
        than the InvalidConstraintError the adapter was catching) — fixed
        in p2smi_adapter.py; this test pins the fix down."""
        record = _linear_record("ACDEFGHIK1", is_cyclic=True, cyclization_type=CyclizationType.HEAD_TO_TAIL)
        result = convert_record(record)
        assert result.smiles is None
        assert result.fidelity_tier == FidelityTier.FAILED
        assert result.validated is False
        assert any("contains_residue_unknown_to_p2smi" in d for d in result.dropped_modifications)

    def test_p2smi_failure_with_working_backbone_fallback_yields_backbone_only(self):
        """Empirically, p2smi and RDKit's Chem.MolFromSequence fail on the
        exact same set of unrecognized letters (checked directly against
        both real libraries: U, O, B, Z, J, X are rejected by both) — so a
        real sequence that makes p2smi fail while RDKit still succeeds may
        not be constructible at all with the current implementation. This
        test isolates convert_record's own dispatch/tier-assignment logic
        from that chemistry question by mocking convert_via_p2smi and
        sequence_to_smiles directly, so the BACKBONE_ONLY branch
        (convert/__init__.py) still gets real coverage rather than staying
        untestable."""
        record = _linear_record(is_cyclic=True, cyclization_type=CyclizationType.HEAD_TO_TAIL)
        # Use the record's own real, correct plain-backbone SMILES as the
        # mocked fallback so mass_sanity_check/round_trips pass on genuine
        # chemistry, not a fabricated value that happens to be tiny.
        real_backbone_smiles = sequence_to_smiles(record)
        with (
            patch("clamp.data.convert.convert_via_p2smi", return_value=(None, ["some_reason"])),
            patch("clamp.data.convert.sequence_to_smiles", return_value=real_backbone_smiles),
        ):
            result = convert_record(record)
        assert result.smiles == real_backbone_smiles
        assert result.smiles_source == SmilesSource.PLAIN_BACKBONE_FALLBACK
        assert result.fidelity_tier == FidelityTier.BACKBONE_ONLY
        assert "p2smi_conversion_failed_used_plain_backbone" in result.dropped_modifications


class TestP2smiAdapter:
    """Against the 5 p2smi cyclization types (doc 06 §2.1)."""

    def test_head_to_tail_tag(self):
        record = _linear_record(is_cyclic=True, cyclization_type=CyclizationType.HEAD_TO_TAIL)
        assert build_p2smi_fasta_header(record) == "HT"
        smiles, dropped = convert_via_p2smi(record)
        assert smiles is not None
        assert dropped == []

    def test_disulfide_tag(self):
        record = _linear_record(sequence="ACDEFGHIKC", is_cyclic=True, cyclization_type=CyclizationType.DISULFIDE)
        assert build_p2smi_fasta_header(record) == "SS"
        smiles, dropped = convert_via_p2smi(record)
        assert smiles is not None
        assert dropped == []

    def test_side_chain_to_side_chain_tag(self):
        record = _linear_record(
            sequence="KACDEFGHIKD", is_cyclic=True, cyclization_type=CyclizationType.SIDE_CHAIN_TO_SIDE_CHAIN
        )
        smiles, dropped = convert_via_p2smi(record)
        assert smiles is not None
        assert dropped == []

    def test_none_cyclization_yields_empty_tag(self):
        assert build_p2smi_fasta_header(_linear_record()) == ""

    def test_lowercase_residue_preserves_d_stereochemistry(self):
        """Regression guard for the case-folding bug: p2smi's own residue
        table uses case to distinguish D/L (e.g. 'a' = D-Alanine vs 'A' =
        L-Alanine, per p2smi.utilities.aminoacids.all_aminos) —
        uppercasing before conversion silently produced an all-L SMILES
        for any D-containing peptide regardless of what the source data
        actually said."""
        upper_smiles, _ = convert_via_p2smi(_linear_record("KAK"))
        lower_smiles, _ = convert_via_p2smi(_linear_record("KaK"))
        assert upper_smiles is not None
        assert lower_smiles is not None
        assert upper_smiles != lower_smiles

    def test_unplaceable_disulfide_on_cys_free_sequence_is_dropped_not_silent(self):
        """A sequence with no disulfide-capable residues can't actually
        form the requested SS bond — p2smi silently falls back to linear
        rather than raising, so convert_via_p2smi must detect that and
        record it as a dropped modification rather than reporting a false
        FULL-fidelity success."""
        record = _linear_record(sequence="AAAAAAA", is_cyclic=True, cyclization_type=CyclizationType.DISULFIDE)
        smiles, dropped = convert_via_p2smi(record)
        assert smiles is not None
        assert any("unresolvable_on_sequence" in d for d in dropped)
