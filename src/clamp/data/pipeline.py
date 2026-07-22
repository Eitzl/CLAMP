"""Phase 1 pipeline orchestration: manifest-file-based staging, not a
workflow engine — see MD_design_docs/09_phase1_implementation_design.md
§10 for why (linear DAG, one machine, no scheduling/distribution need).

Revisit `dvc` specifically (not Prefect/Luigi) if/when the "versioned
dataset" requirement from MD_design_docs/02_peptideclm_transfer_learning_plan.md
P0 deliverable becomes load-bearing rather than a nice property.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Stage:
    name: str
    run: Callable[[], None]
    inputs: list[str]
    output: str


def _pull_dbaasp() -> None:
    raise NotImplementedError


def _pull_dramp() -> None:
    raise NotImplementedError


def _pull_hemolytik2() -> None:
    raise NotImplementedError


def _pull_hemopi2() -> None:
    raise NotImplementedError


def _pull_qmap() -> None:
    raise NotImplementedError


def _convert() -> None:
    raise NotImplementedError


def _dedup() -> None:
    raise NotImplementedError


def _normalize() -> None:
    raise NotImplementedError


def _datasheet() -> None:
    raise NotImplementedError


STAGES: list[Stage] = [
    Stage("pull_dbaasp", _pull_dbaasp, inputs=[], output="raw/dbaasp/"),
    Stage("pull_dramp", _pull_dramp, inputs=[], output="raw/dramp/"),
    Stage("pull_hemolytik2", _pull_hemolytik2, inputs=[], output="raw/hemolytik2/"),
    Stage("pull_hemopi2", _pull_hemopi2, inputs=[], output="raw/hemopi2/"),
    Stage("pull_qmap", _pull_qmap, inputs=[], output="raw/qmap/"),
    Stage("convert", _convert, inputs=["raw/*"], output="interim/converted.parquet"),
    Stage("dedup", _dedup, inputs=["interim/converted.parquet"], output="interim/deduped.parquet"),
    Stage("normalize", _normalize, inputs=["interim/deduped.parquet"], output="processed/dataset.parquet"),
    Stage("datasheet", _datasheet, inputs=["processed/dataset.parquet"], output="datasheet/"),
]


def run_pipeline(stages: list[Stage] = STAGES, force: bool = False) -> None:
    """Loads/updates settings.manifest_path, skips a stage if its inputs are
    unchanged and force=False, otherwise runs it and updates the manifest.
    See MD_design_docs/09_phase1_implementation_design.md §10 for the
    manifest schema ({stage: {inputs_hash, output_path, completed_at}})."""
    raise NotImplementedError
