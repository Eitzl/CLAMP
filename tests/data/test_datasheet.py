"""Direct unit coverage for datasheet.py's build_datasheet, in particular
peptide_level_coverage — added this session after a real ~90k-row pilot
pull found that per_source_coverage's row-level pct_with_both is
structurally 0.0% for every source by construction of the
one-row-per-assay-record schema, which materially undersells "how many
peptides have both labels" (see build_datasheet's docstring).
"""

import pandas as pd

from clamp.data.datasheet import build_datasheet
from clamp.data.dedup import DedupResult


def _dataset(rows: list[dict]) -> pd.DataFrame:
    columns = [
        "source",
        "peptide_uid",
        "mic_value_uM",
        "hc50_value_uM",
        "mic_target_species",
        "hc50_assay_target_cell",
        "chemical_fidelity_tier",
    ]
    return pd.DataFrame(rows, columns=columns)


def _empty_dedup_result() -> DedupResult:
    return DedupResult(records=[], overlap_matrix={}, raw_row_count=0, unique_peptide_uid_count=0, flagged_conflict_count=0)


class TestPeptideLevelCoverage:
    def test_complementary_rows_across_sources_count_as_both_at_peptide_level(self):
        """Same peptide_uid, MIC from one source, HC50 from another — 0%
        row-level pct_with_both, but 100% peptide-level."""
        dataset = _dataset(
            [
                {
                    "source": "dbaasp",
                    "peptide_uid": "uid1",
                    "mic_value_uM": 10.0,
                    "hc50_value_uM": None,
                    "mic_target_species": "E. coli",
                    "hc50_assay_target_cell": None,
                    "chemical_fidelity_tier": "full",
                },
                {
                    "source": "hemolytik2",
                    "peptide_uid": "uid1",
                    "mic_value_uM": None,
                    "hc50_value_uM": 50.0,
                    "mic_target_species": None,
                    "hc50_assay_target_cell": "Human RBC",
                    "chemical_fidelity_tier": "full",
                },
            ]
        )
        tables = build_datasheet(dataset, _empty_dedup_result())

        row_level = {r["source"]: r for r in tables.per_source_coverage}
        assert row_level["dbaasp"]["pct_with_both"] == 0.0
        assert row_level["hemolytik2"]["pct_with_both"] == 0.0

        peptide_level = {r["source"]: r for r in tables.peptide_level_coverage}
        assert peptide_level["dbaasp"]["pct_with_both"] == 100.0
        assert peptide_level["hemolytik2"]["pct_with_both"] == 100.0
        all_row = next(r for r in tables.peptide_level_coverage if r["source"] == "ALL")
        assert all_row["unique_peptide_uids"] == 1
        assert all_row["pct_with_both"] == 100.0

    def test_unrelated_peptides_do_not_count_as_both(self):
        dataset = _dataset(
            [
                {
                    "source": "dbaasp",
                    "peptide_uid": "uid1",
                    "mic_value_uM": 10.0,
                    "hc50_value_uM": None,
                    "mic_target_species": "E. coli",
                    "hc50_assay_target_cell": None,
                    "chemical_fidelity_tier": "full",
                },
                {
                    "source": "dbaasp",
                    "peptide_uid": "uid2",
                    "mic_value_uM": None,
                    "hc50_value_uM": 50.0,
                    "mic_target_species": None,
                    "hc50_assay_target_cell": "Human RBC",
                    "chemical_fidelity_tier": "full",
                },
            ]
        )
        tables = build_datasheet(dataset, _empty_dedup_result())
        dbaasp_row = next(r for r in tables.peptide_level_coverage if r["source"] == "dbaasp")
        assert dbaasp_row["unique_peptide_uids"] == 2
        assert dbaasp_row["pct_with_mic"] == 50.0
        assert dbaasp_row["pct_with_hc50"] == 50.0
        assert dbaasp_row["pct_with_both"] == 0.0

    def test_does_not_mutate_caller_dataframe(self):
        """build_datasheet copies before adding its temporary helper
        columns — the caller's DataFrame must come back unchanged."""
        dataset = _dataset(
            [
                {
                    "source": "dbaasp",
                    "peptide_uid": "uid1",
                    "mic_value_uM": 10.0,
                    "hc50_value_uM": None,
                    "mic_target_species": "E. coli",
                    "hc50_assay_target_cell": None,
                    "chemical_fidelity_tier": "full",
                }
            ]
        )
        columns_before = list(dataset.columns)
        build_datasheet(dataset, _empty_dedup_result())
        assert list(dataset.columns) == columns_before
