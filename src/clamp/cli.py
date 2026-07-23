"""`clamp-data` CLI entry point.

MD_design_docs/09_phase1_implementation_design.md §11 and §14 (runbook).
Run subcommands individually the first time through a fresh pull so a
failure in `convert` doesn't obscure whether `pull dbaasp` actually
finished cleanly; `run-all` is the convenience path for reruns.
"""

import typer

from clamp.config import settings
from clamp.data.pipeline import STAGES, run_pipeline
from clamp.data.sources import dbaasp, dramp, hemolytik2, hemopi2, qmap

app = typer.Typer()

_PULLERS = {
    "dbaasp": lambda: dbaasp.DbaaspPuller().pull(settings.raw_dir / "dbaasp"),
    "dramp": lambda: dramp.DrampDownloader().pull(settings.raw_dir / "dramp"),
    "hemolytik2": lambda: hemolytik2.Hemolytik2Downloader().pull(settings.raw_dir / "hemolytik2"),
    "hemopi2": lambda: hemopi2.HemoPI2Downloader().pull(settings.raw_dir / "hemopi2"),
    "qmap": lambda: qmap.QmapDownloader().pull(settings.raw_dir / "qmap"),
}

_STAGES_BY_NAME = {stage.name: stage for stage in STAGES}


@app.command()
def pull(source: str, force: bool = False) -> None:
    """Run a single source's pull stage, e.g. `clamp-data pull dbaasp`."""
    if source not in _PULLERS:
        raise typer.BadParameter(f"unknown source {source!r}, expected one of {sorted(_PULLERS)}")
    report = _PULLERS[source]()
    typer.echo(report.model_dump_json(indent=2))


@app.command()
def convert(force: bool = False) -> None:
    _STAGES_BY_NAME["convert"].run()


@app.command()
def dedup(force: bool = False) -> None:
    _STAGES_BY_NAME["dedup"].run()


@app.command()
def normalize(force: bool = False) -> None:
    _STAGES_BY_NAME["normalize"].run()


@app.command()
def datasheet() -> None:
    _STAGES_BY_NAME["datasheet"].run()


@app.command(name="run-all")
def run_all(force: bool = False) -> None:
    """The full Phase 1 pipeline, doc 08's Phase 1 bullets end to end."""
    run_pipeline(STAGES, force=force)


if __name__ == "__main__":
    app()
