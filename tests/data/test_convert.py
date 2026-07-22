"""Intended coverage per MD_design_docs/09_phase1_implementation_design.md
§12. Skipped until src/clamp/data/convert/* are implemented — see NEXT
STEPS in the scaffold review. Un-skip and flesh out each test as its target
function gets a real body.
"""

import pytest

pytestmark = pytest.mark.skip(reason="convert/* is a NotImplementedError stub — see doc 09 §12 for intended coverage")


class TestRoundTrips:
    def test_valid_smiles_round_trips(self):
        """A known-good SMILES parses and re-canonicalizes idempotently."""

    def test_malformed_smiles_fails(self):
        """Garbage input returns False, not an exception."""

    @pytest.mark.parametrize("seed", range(20))
    def test_hypothesis_style_mutation_property(self, seed):
        """Property-based (hypothesis) test generating/mutating known-valid
        SMILES — this is the class of bug PeptideCLM v1's ring-numbering
        issue (doc 02 §5.2) exemplifies; see doc 09 §12."""


class TestMassSanityCheck:
    def test_matching_mass_passes(self):
        pass

    def test_mismatched_mass_fails_within_tolerance_bound(self):
        pass


class TestConvertDispatch:
    def test_curated_smiles_short_circuits_to_db_curated(self):
        pass

    def test_cyclic_record_routes_to_p2smi(self):
        pass

    def test_linear_unmodified_record_routes_to_rdkit(self):
        pass

    def test_p2smi_dropped_modification_downgrades_to_partial_tier(self):
        """doc 06 §5.1: partial fidelity tier when p2smi's residue table
        can't represent something in unusual_residues."""


class TestP2smiAdapter:
    """Against real fixture records spanning the 5 p2smi cyclization types
    (HT, SCSC, SCNT, SCCT, SS) per doc 06 §2.1 — requires
    tests/data/fixtures/dbaasp/ to be populated first."""

    def test_head_to_tail_cyclization_header(self):
        pass

    def test_disulfide_cyclization_header(self):
        pass
