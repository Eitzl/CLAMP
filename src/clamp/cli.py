"""`clamp-data` CLI entry point.

MD_design_docs/09_phase1_implementation_design.md §11 and §14 (runbook).
Run subcommands individually the first time through a fresh pull so a
failure in `convert` doesn't obscure whether `pull dbaasp` actually
finished cleanly; `run-all` is the convenience path for reruns.
"""

import typer

from clamp.data.pipeline import STAGES, run_pipeline

app = typer.Typer()


@app.command()
def pull(source: str, force: bool = False) -> None:
    """Run a single source's pull stage, e.g. `clamp-data pull dbaasp`."""
    raise NotImplementedError


@app.command()
def convert(force: bool = False) -> None:
    raise NotImplementedError


@app.command()
def dedup(force: bool = False) -> None:
    raise NotImplementedError


@app.command()
def normalize(force: bool = False) -> None:
    raise NotImplementedError


@app.command()
def datasheet() -> None:
    raise NotImplementedError


@app.command(name="run-all")
def run_all(force: bool = False) -> None:
    """The full Phase 1 pipeline end to end."""
    run_pipeline(STAGES, force=force)


if __name__ == "__main__":
    app()
