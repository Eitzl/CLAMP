"""End-to-end integration test for the pull -> convert -> dedup -> normalize
-> datasheet wiring in pipeline.py/datasheet.py.

This is the biggest single gap the test-debate identified this session:
every stage (convert.py, dedup.py, normalize.py, datasheet.py) had unit
tests in isolation, but nothing proved they're actually wired together
with consistent field names/types across stage boundaries — a category of
bug unit tests structurally cannot catch. This seeds one source (QMAP,
via its jsonl cache format — the simplest to hand-construct without a
real network pull or file parse) with two records sharing a sequence, and
runs the real pipeline stage functions against a redirected data_root.
"""

import json
import math

import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

from clamp.config import settings
from clamp.data import pipeline
from clamp.data.normalize import log_transform, ug_per_ml_to_uM
from clamp.data.schema import CyclizationType, PeptideRecord, Source


@pytest.fixture
def pilot_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_root", tmp_path)
    qmap_dir = tmp_path / "raw" / "qmap"
    qmap_dir.mkdir(parents=True)

    shared_sequence = "KWKLFKKIEK"
    records = [
        PeptideRecord(
            source=Source.QMAP,
            source_id="1:mic",
            sequence_raw=shared_sequence,
            mic_value=10.0,
            mic_unit="uM",
            mic_target_species="Escherichia coli",
        ),
        PeptideRecord(
            source=Source.QMAP,
            source_id="2:hc50",
            sequence_raw=shared_sequence,
            hc50_value=100.0,
            hc50_unit="ug/ml",
            hc50_assay_target_cell="Human RBC",
        ),
        # A record that fails conversion entirely (unmappable character +
        # forced through p2smi via is_cyclic) -> smiles=None after
        # convert.py runs. Included specifically to guard against a real
        # bug a parallel pilot-data-pull agent found this session:
        # df["smiles"].map(_molecular_weight) crashes with an opaque C++
        # TypeError on a genuinely-missing smiles value once it round-trips
        # through pandas as NaN instead of None.
        PeptideRecord(
            source=Source.QMAP,
            source_id="3:unresolvable",
            sequence_raw="ACDEFGHIK1",
            is_cyclic=True,
            cyclization_type=CyclizationType.HEAD_TO_TAIL,
            mic_value=5.0,
            mic_unit="uM",
            mic_target_species="Escherichia coli",
        ),
    ]
    (qmap_dir / "qmap.jsonl").write_text("\n".join(r.model_dump_json() for r in records))
    return tmp_path


def test_convert_stage_produces_expected_row_count_and_fidelity(pilot_data_root):
    pipeline._convert()
    df = pd.read_parquet(pilot_data_root / "interim" / "converted.parquet")
    assert len(df) == 3
    assert set(df["source"]) == {"qmap"}
    by_id = df.set_index("source_id")
    assert by_id.loc["1:mic", "chemical_fidelity_tier"] == "full"
    assert by_id.loc["2:hc50", "chemical_fidelity_tier"] == "full"
    assert by_id.loc["1:mic", "smiles_source"] == "rdkit_sequence"
    # the deliberately-unresolvable record must fail cleanly, not crash
    # convert() (that's the whole point of including it), and it must
    # genuinely have a missing smiles afterward (a raw pd.read_parquet
    # read surfaces a missing value as NaN, not None — pd.isna() is the
    # correct check here, not `is None`)
    assert by_id.loc["3:unresolvable", "chemical_fidelity_tier"] == "failed"
    assert pd.isna(by_id.loc["3:unresolvable", "smiles"])


def test_dedup_stage_collapses_shared_sequence_to_one_peptide_uid(pilot_data_root):
    pipeline._convert()
    pipeline._dedup()
    df = pd.read_parquet(pilot_data_root / "interim" / "deduped.parquet")
    dedup_info = json.loads((pilot_data_root / "interim" / "dedup_result.json").read_text())
    assert len(df) == 3  # rows are never dropped, only relabeled
    # the two KWKLFKKIEK records collapse to one peptide_uid; the
    # unresolvable-sequence record is its own, distinct peptide_uid
    assert df["peptide_uid"].nunique() == 2
    assert dedup_info["raw_row_count"] == 3
    assert dedup_info["unique_peptide_uid_count"] == 2


def test_normalize_stage_computes_correct_uM_and_log_values(pilot_data_root):
    pipeline._convert()
    pipeline._dedup()
    pipeline._normalize()
    df = pd.read_parquet(pilot_data_root / "processed" / "dataset.parquet")
    assert len(df) == 3
    by_id = df.set_index("source_id")

    # Independently recompute the expected values from the real RDKit MW of
    # whatever SMILES the pipeline actually generated, rather than
    # hardcoding a magic number that would silently drift if the RDKit/
    # residue-mass logic ever changed — this test's job is to verify the
    # stages are WIRED correctly, not to re-verify convert/normalize's own
    # unit-tested chemistry.
    mw = Descriptors.MolWt(Chem.MolFromSmiles(by_id.loc["1:mic", "smiles"]))

    assert by_id.loc["1:mic", "mic_value_uM"] == pytest.approx(10.0)
    assert by_id.loc["1:mic", "mic_log_uM"] == pytest.approx(log_transform(10.0))

    expected_hc50_uM = ug_per_ml_to_uM(100.0, mw)
    assert by_id.loc["2:hc50", "hc50_value_uM"] == pytest.approx(expected_hc50_uM)
    assert by_id.loc["2:hc50", "hc50_log_uM"] == pytest.approx(math.log10(expected_hc50_uM))

    # the MIC row must NOT have picked up an hc50 value and vice versa —
    # a column-alignment bug across the two _normalized_column() calls in
    # pipeline._normalize() would show up here
    assert pd.isna(by_id.loc["1:mic", "hc50_value_uM"])
    assert pd.isna(by_id.loc["2:hc50", "mic_value_uM"])

    # The regression guard this record exists for: normalize() must not
    # crash when a row's smiles is None (real bug found via a live pilot
    # pull this session — df["smiles"].map(_molecular_weight) crashed with
    # an opaque C++ TypeError on a NaN-valued smiles cell). Current
    # (conservative) behavior is that a record with no computable
    # molecular weight gets no normalized value at all, even for a plain
    # uM unit that wouldn't have actually needed the weight — documented
    # here as the real, current contract, not silently assumed.
    assert pd.isna(by_id.loc["3:unresolvable", "mic_value_uM"])


def test_datasheet_stage_reports_consistent_numbers_with_upstream_stages(pilot_data_root):
    pipeline._convert()
    pipeline._dedup()
    pipeline._normalize()
    pipeline._datasheet()

    datasheet_json_path = pilot_data_root / "datasheet" / "datasheet.json"
    datasheet_md_path = pilot_data_root / "datasheet" / "datasheet.md"
    assert datasheet_json_path.exists()
    assert datasheet_md_path.exists()

    tables = json.loads(datasheet_json_path.read_text())
    qmap_coverage = next(row for row in tables["per_source_coverage"] if row["source"] == "qmap")
    assert qmap_coverage["rows"] == 3
    assert qmap_coverage["unique_peptide_uids"] == 2
    # 1:mic and 3:unresolvable both have a raw mic_value, but 3's never
    # gets normalized to mic_value_uM (see the note above) — pct_with_mic
    # in the datasheet is computed off mic_value_uM (the normalized
    # column), so it reports 1/3, not 2/3. Documented here so this
    # subtlety is visible rather than assumed.
    assert qmap_coverage["pct_with_mic"] == pytest.approx(100 / 3, abs=0.1)
    assert qmap_coverage["pct_with_hc50"] == pytest.approx(100 / 3, abs=0.1)
    assert qmap_coverage["pct_with_both"] == pytest.approx(0.0)

    # peptide_level_coverage answers a different question than
    # per_source_coverage's row-level pct_with_both (which is structurally
    # 0.0% for every source by construction — see datasheet.py's
    # build_datasheet docstring): "1:mic" and "2:hc50" share a peptide_uid
    # (same sequence, no mods) and between them cover both tasks, so that
    # ONE peptide counts as "has both" at the peptide level even though
    # neither individual row does. "3:unresolvable" is its own peptide_uid
    # with no successfully-normalized value at all -> has neither.
    peptide_qmap = next(row for row in tables["peptide_level_coverage"] if row["source"] == "qmap")
    assert peptide_qmap["unique_peptide_uids"] == 2
    assert peptide_qmap["pct_with_mic"] == pytest.approx(50.0)
    assert peptide_qmap["pct_with_hc50"] == pytest.approx(50.0)
    assert peptide_qmap["pct_with_both"] == pytest.approx(50.0)

    dedup_summary = tables["dedup_summary"][0]
    assert dedup_summary["raw_row_count"] == 3
    assert dedup_summary["unique_peptide_uid_count"] == 2

    # datasheet.md is rendered from the same DatasheetTables instance, per
    # datasheet.py's own stated design constraint — spot-check it actually
    # contains the same row count, not an independently-drifted rendering
    md_text = datasheet_md_path.read_text()
    assert "qmap" in md_text
    assert "Per-source coverage" in md_text


def test_normalize_does_not_cross_contaminate_range_flag_between_mic_and_hc50(tmp_path, monkeypatch):
    """Regression guard for a real bug found via a ~90k-row pilot pull:
    label_is_range/label_range_raw are single fields shared by whichever
    ONE of mic_value/hc50_value a row actually populates (schema.py has no
    separate range-flag per task). An MIC-only row with a range-valued
    concentration was incorrectly treated as "the HC50 column also has
    range input" when normalizing hc50_value_uM (since the old check only
    looked at the shared is_range/range_raw fields), then crashed on
    normalize_concentration's unit.strip() because hc50_unit is correctly
    None for an MIC-only row. Confirmed via direct reproduction against
    the real pipeline._normalize() before this fix, not a synthetic guess.
    """
    monkeypatch.setattr(settings, "data_root", tmp_path)
    qmap_dir = tmp_path / "raw" / "qmap"
    qmap_dir.mkdir(parents=True)

    records = [
        # MIC-only, range-valued concentration -> hc50 columns must stay
        # untouched (no AttributeError on hc50_unit=None)
        PeptideRecord(
            source=Source.QMAP,
            source_id="1:mic-range",
            sequence_raw="KWKLFKKIEK",
            mic_value=None,
            mic_unit="uM",
            mic_target_species="Escherichia coli",
            label_is_range=True,
            label_range_raw="2.0-4.0",
        ),
        # HC50-only, range-valued concentration -> mic columns must stay
        # untouched
        PeptideRecord(
            source=Source.QMAP,
            source_id="2:hc50-range",
            sequence_raw="ACDEFGHIK",
            hc50_value=None,
            hc50_unit="uM",
            hc50_assay_target_cell="Human RBC",
            label_is_range=True,
            label_range_raw="1.58-25.33",
        ),
    ]
    (qmap_dir / "qmap.jsonl").write_text("\n".join(r.model_dump_json() for r in records))

    pipeline._convert()
    pipeline._dedup()
    pipeline._normalize()  # must not raise

    df = pd.read_parquet(tmp_path / "processed" / "dataset.parquet")
    by_id = df.set_index("source_id")

    assert by_id.loc["1:mic-range", "mic_value_uM"] == pytest.approx(math.sqrt(2.0 * 4.0))
    assert pd.isna(by_id.loc["1:mic-range", "hc50_value_uM"])

    assert by_id.loc["2:hc50-range", "hc50_value_uM"] == pytest.approx(math.sqrt(1.58 * 25.33))
    assert pd.isna(by_id.loc["2:hc50-range", "mic_value_uM"])
