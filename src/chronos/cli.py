"""Command-line entry point (`chronos`)."""

from __future__ import annotations

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command("init-db")
def init_db(path: str = typer.Option(None, help="SQLite path (defaults to config)")) -> None:
    """Create or migrate the ChronosGraph database."""
    from chronos.config import Paths
    from chronos.storage.db import Database

    db = Database(path or Paths().db)
    typer.echo(f"schema version {db.schema_version()} at {db.path}")


@app.command("ingest-demo")
def ingest_demo() -> None:
    """Ingest data/demo_sessions (implemented in Part 2)."""
    typer.echo("ingest-demo: not implemented yet (Part 2)", err=True)
    raise typer.Exit(1)


@app.command("eval")
def eval_cmd(manifest: str = typer.Option(..., help="Path to a manifest JSON")) -> None:
    """Run an evaluation manifest (implemented in Part 3)."""
    typer.echo(f"eval {manifest}: not implemented yet (Part 3)", err=True)
    raise typer.Exit(1)


@app.command("report")
def report() -> None:
    """Build the results report (implemented in Part 4)."""
    typer.echo("report: not implemented yet (Part 4)", err=True)
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
