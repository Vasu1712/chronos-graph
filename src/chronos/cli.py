"""Command-line entry point (`chronos` / `python -m chronos.cli`)."""

from __future__ import annotations

import logging
from pathlib import Path

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


def _memory_table(db, conversation_id: str) -> list[str]:
    rows = db.conn.execute(
        "SELECT m.memory_id, e.canonical_name, m.attribute_key, m.status, m.polarity, m.fact_text, "
        "m.valid_from, m.valid_from_precision, m.valid_to, m.valid_to_precision, m.event_date, "
        "m.event_date_precision, m.recorded_session_ordinal, m.recorded_turn_index "
        "FROM memories m JOIN entities e ON e.entity_id = m.subject_entity_id "
        "WHERE m.conversation_id = ? ORDER BY m.recorded_session_ordinal, m.recorded_turn_index, m.rowid",
        (conversation_id,),
    ).fetchall()
    out = []
    for r in rows:
        if r["event_date_precision"]:
            when = f"event {r['event_date']} ({r['event_date_precision']})"
        else:
            vt = f"{r['valid_to']} ({r['valid_to_precision']})" if r["valid_to"] else "now"
            when = f"{r['valid_from'] or '?'} ({r['valid_from_precision']}) -> {vt}"
        key = r["attribute_key"] or "-"
        out.append(
            f"s{r['recorded_session_ordinal']}t{r['recorded_turn_index']:<3} {r['status']:<10} {r['polarity']} "
            f"{key:<16} {r['fact_text']}  [{when}]"
        )
    return out


@app.command("ingest")
def ingest(
    file: Path = typer.Argument(
        ..., exists=True, help="Conversation JSON (see chronos.ingestion.files)"
    ),
    db: Path = typer.Option(None, help="SQLite path (default: artifacts/chronos.sqlite)"),
    scripted: bool = typer.Option(
        False, help="Replay the file's scripted extractor/classifier outputs"
    ),
    embedder: str = typer.Option("bge", help="bge (real) or hash (offline bag-of-words)"),
    verbose: bool = typer.Option(False, "-v", help="Log per-session details"),
) -> None:
    """Ingest every session of a conversation file and print the resulting memories."""
    from chronos.config import Paths
    from chronos.ingestion.files import load_conversation_file
    from chronos.ingestion.pipeline import IngestionPipeline
    from chronos.storage.db import Database

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )
    conv, data = load_conversation_file(file)
    if scripted:
        from chronos.ingestion.scripted import ScriptedLLM

        if "scripted" not in data:
            raise typer.BadParameter("file has no 'scripted' block")
        llm = ScriptedLLM(conv, data["scripted"])
    else:
        from chronos.llm.cache import ResponseCache
        from chronos.llm.local_client import LocalLLM

        llm = LocalLLM(cache=ResponseCache(Paths().llm_cache))
    if embedder == "hash":
        from chronos.llm.embed import HashingEmbedder

        emb = HashingEmbedder()
    else:
        from chronos.llm.embed import Embedder

        emb = Embedder()

    database = Database(db or Paths().db)
    pipe = IngestionPipeline(database, llm, emb)
    results = pipe.ingest_conversation(conv)
    for r in results:
        s = r.summary()
        if r.skipped:
            typer.echo(f"{r.session_id}: already ingested (no-op)")
            continue
        ex = s["extraction"]
        typer.echo(
            f"{r.session_id}: {ex['kept']}/{ex['raw_facts']} facts kept, dropped {ex['dropped'] or 0}, "
            f"relations {s['relations']}"
        )
    typer.echo("")
    for line in _memory_table(database, conv.conversation_id):
        typer.echo(line)
    database.close()


@app.command("ingest-demo")
def ingest_demo(
    scripted: bool = typer.Option(False, help="Use the demo's scripted outputs (offline)"),
) -> None:
    """Ingest data/demo_sessions/demo.json into a fresh artifacts/demo.sqlite."""
    from chronos.config import REPO_ROOT

    path = REPO_ROOT / "artifacts" / "demo.sqlite"
    for suffix in ("", "-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)
    ingest(REPO_ROOT / "data" / "demo_sessions" / "demo.json", db=path, scripted=scripted,
           embedder="hash" if scripted else "bge", verbose=False)  # fmt: skip
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
