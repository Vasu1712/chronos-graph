"""Command-line entry point (`chronos` / `python -m chronos.cli`), rendered with rich.

chronos ingest <path>                       ingest a conversation file
chronos ask "<question>" [--conversation] [--as-of]
chronos inspect <memory_id | query_id>      fact, status, interval, chain, quotes, turns, NLI verdicts
chronos history <entity> [--attribute]      every version of an entity's facts
chronos stats                               database counts and answer latency
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console(emoji=False)  # memory ids contain ":m:", which rich would turn into an emoji

STATUS_STYLE = {
    "ACTIVE": "green",
    "SUPERSEDED": "yellow",
    "RETRACTED": "red",
    "CONFLICTED": "magenta",
}


# ------------------------------------------------------------------------------ helpers
def _open_db(db: Path | None):
    from chronos.config import Paths
    from chronos.storage.db import Database

    return Database(db or Paths().db)


def _conversation(database, conversation: str | None) -> str:
    if conversation:
        return conversation
    rows = [
        r[0]
        for r in database.conn.execute("SELECT conversation_id FROM conversations ORDER BY rowid")
    ]
    if len(rows) == 1:
        return rows[0]
    raise typer.BadParameter(
        f"--conversation is required; the database holds {len(rows)}: {', '.join(rows[:10])}"
    )


def _status(s: str) -> str:
    return f"[{STATUS_STYLE.get(s, 'white')}]{s}[/]"


def _interval(r: Any) -> str:
    from chronos.ingestion.resolver import MemRow
    from chronos.ingestion.temporal import describe

    cols = MemRow.__dataclass_fields__
    return describe(MemRow(*[r[c] for c in cols]).rt)


def _memory_table(db, conversation_id: str) -> list[str]:
    """Plain-text memory listing (also used by eval/probes.py reports)."""
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


def _memories_rich(db, conversation_id: str, where: str = "", params: tuple = ()) -> Table:
    t = Table(show_lines=False, header_style="bold")
    for col in ("rec", "status", "subject", "attribute", "fact", "interval", "memory_id"):
        t.add_column(col, overflow="fold")
    rows = db.conn.execute(
        "SELECT m.*, e.canonical_name FROM memories m JOIN entities e ON e.entity_id = m.subject_entity_id "
        f"WHERE m.conversation_id = ? {where} ORDER BY m.recorded_session_ordinal, m.recorded_turn_index, m.rowid",
        (conversation_id, *params),
    ).fetchall()
    for r in rows:
        neg = " [dim](NEG)[/]" if r["polarity"] == "NEG" else ""
        t.add_row(f"s{r['recorded_session_ordinal']}t{r['recorded_turn_index']}", _status(r["status"]),
                  escape(r["canonical_name"]), r["attribute_key"] or "-", escape(r["fact_text"]) + neg, _interval(r), r["memory_id"])  # fmt: skip
    return t


def _turn_line(db, turn_uid: str) -> str:
    r = db.conn.execute(
        "SELECT session_ts, speaker, text FROM episodes WHERE turn_uid = ?", (turn_uid,)
    ).fetchone()
    if r is None:
        return f"[red]{escape(turn_uid)} (not found)[/]"
    date = (r[0] or "")[:10]
    prefix = escape(f"[{turn_uid} | {date} | {r[1]}]")
    return f"[cyan]{prefix}[/] {escape(r[2])}"


def _build_agent(
    database, *, embedder: str = "bge", gate: str = "full", nli_threshold: float | None = None
):
    from chronos.agent.memory_agent import MemoryAgent
    from chronos.config import AgentConfig, Paths
    from chronos.llm.cache import ResponseCache
    from chronos.llm.local_client import LocalLLM
    from chronos.llm.nli import NLI

    if embedder == "hash":
        from chronos.llm.embed import HashingEmbedder

        emb = HashingEmbedder()
    else:
        from chronos.llm.embed import Embedder

        emb = Embedder()
    cfg = AgentConfig(
        answerability=gate,
        **({"nli_threshold": nli_threshold} if nli_threshold is not None else {}),
    )
    return MemoryAgent(
        database, LocalLLM(cache=ResponseCache(Paths().llm_cache)), emb, NLI(), agent=cfg
    )


def _check_embedder(database, conversation_id: str, embedder_name: str) -> None:
    models = {
        r[0]
        for r in database.conn.execute(
            "SELECT DISTINCT v.model FROM episode_embeddings v JOIN episodes e USING(turn_uid) WHERE e.conversation_id = ?",
            (conversation_id,),
        )
    }
    if models and embedder_name not in models:
        raise typer.BadParameter(
            f"conversation was embedded with {sorted(models)}, not {embedder_name}; pass --embedder"
        )


# ------------------------------------------------------------------------------ commands
@app.command("init-db")
def init_db(path: str = typer.Option(None, help="SQLite path (defaults to config)")) -> None:
    """Create or migrate the ChronosGraph database."""
    database = _open_db(Path(path) if path else None)
    console.print(f"schema version {database.schema_version()} at {database.path}")


@app.command("ingest")
def ingest(
    path: Path = typer.Argument(
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
    conv, data = load_conversation_file(path)
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

    database = _open_db(db)
    results = IngestionPipeline(database, llm, emb).ingest_conversation(conv)
    for r in results:
        if r.skipped:
            console.print(f"{r.session_id}: already ingested (no-op)")
            continue
        s = r.summary()
        ex = s["extraction"]
        console.print(f"{r.session_id}: {ex['kept']}/{ex['raw_facts']} facts kept, dropped {ex['dropped'] or 0}, "
                      f"relations {s['relations']}")  # fmt: skip
    console.print(_memories_rich(database, conv.conversation_id))
    database.close()


@app.command("ask")
def ask(
    question: str = typer.Argument(...),
    conversation: str = typer.Option(None, help="conversation_id (optional if the DB holds one)"),
    as_of: str = typer.Option(None, "--as-of", help="answer as of this date (YYYY-MM-DD)"),
    db: Path = typer.Option(None),
    asker: str = typer.Option(None, help="speaker asking the question (default: first speaker)"),
    gate: str = typer.Option("full", help="answerability gate: off | speaker | full"),
    embedder: str = typer.Option("bge", help="must match the embedder used at ingest"),
    show_context: bool = typer.Option(False, "--context", help="print the packed context"),
    as_json: bool = typer.Option(False, "--json", help="print the answer_log summary as JSON"),
) -> None:
    """Answer a question from memory, with cited turns."""
    database = _open_db(db)
    conv_id = _conversation(database, conversation)
    agent = _build_agent(database, embedder=embedder, gate=gate)
    _check_embedder(database, conv_id, getattr(agent.retriever.embedder, "name", ""))
    ans = agent.answer(conv_id, question, as_of=as_of, asker=asker)
    if as_json:
        print(json.dumps(ans.summary(), indent=1, default=str))
        return
    _render_answer(database, ans, show_context)
    database.close()


def _render_answer(database, ans, show_context: bool = False) -> None:
    a = ans.analysis
    meta = f"intent {a.intent.value}" + (
        f", about {a.subject_speaker_name}" if a.subject_speaker_name else ""
    )
    if a.time:
        meta += f", time {a.time.lo}..{a.time.hi} ({a.time.precision})"
    style = "red" if ans.abstained else "green"
    body = escape(ans.final_answer) + (
        f"\n[dim]abstained at {ans.abstained_at}: {escape(ans.abstention_reason or '')}[/]"
        if ans.abstained
        else ""
    )
    console.print(
        Panel(
            body,
            title=f"[bold]{escape(ans.question)}[/]",
            subtitle=f"{ans.query_id} | {escape(meta)}",
            border_style=style,
        )
    )
    for i, c in enumerate(ans.claims, start=1):
        console.print(f"[bold]{i}. {escape(c.text)}[/]")
        for uid in c.cited_turn_uids:
            console.print(f"   {_turn_line(database, uid)}")
        if c.cited_memory_ids:
            console.print(f"   [dim]memories: {', '.join(c.cited_memory_ids)}[/]")
    if show_context:
        console.print(
            Panel(
                escape(ans.packed.text or "(empty)"), title=f"context ({ans.packed.tokens} tokens)"
            )
        )
    t = Table(title="latency (s)", header_style="bold")
    for step in (
        "analyze",
        "retrieve",
        "pack",
        "gate",
        "generate",
        "verify",
        "regenerate",
        "total",
    ):
        t.add_column(step, justify="right")
    t.add_row(*[f"{ans.timings.get(s, 0):.2f}" for s in ("analyze", "retrieve", "pack", "gate", "generate",
                                                          "verify", "regenerate", "total")])  # fmt: skip
    console.print(t)


@app.command("inspect")
def inspect(
    item_id: str = typer.Argument(..., help="memory_id or query_id"), db: Path = typer.Option(None)
) -> None:
    """Show a memory (fact, status, interval, version chain, quotes, source turns, NLI verdicts) or a logged query."""
    database = _open_db(db)
    row = database.conn.execute(
        "SELECT m.*, e.canonical_name FROM memories m JOIN entities e ON e.entity_id = m.subject_entity_id "
        "WHERE m.memory_id = ?",
        (item_id,),
    ).fetchone()
    if row is not None:
        _inspect_memory(database, row)
    else:
        from chronos.agent.verifier import AnswerLog

        q = AnswerLog(database.conn).get(item_id)
        if q is None:
            raise typer.BadParameter(f"{item_id} is neither a memory_id nor a query_id")
        _inspect_query(database, q)
    database.close()


def _inspect_memory(database, r) -> None:
    from chronos.ingestion.resolver import MemRow
    from chronos.retrieval.packer import Packer

    conn = database.conn
    lines = [
        f"[bold]{escape(r['fact_text'])}[/]",
        f"subject: {escape(r['canonical_name'])}   attribute: {r['attribute_key'] or '-'}   polarity: {r['polarity']}   "
        f"mode: {r['mode']}   corroboration: {r['corroboration']}",
        f"status: {_status(r['status'])}   interval: {_interval(r)}",
        f"recorded: session {r['recorded_session_ordinal']} turn {r['recorded_turn_index']}"
        + (f"   retired: session {r['retired_session_ordinal']} turn {r['retired_turn_index']}"
           if r["retired_session_ordinal"] is not None else ""),
    ]  # fmt: skip
    if r["conflict_set_id"]:
        lines.append(f"conflict set: {r['conflict_set_id']}")
    console.print(
        Panel("\n".join(lines), title=Text(r["memory_id"]))
    )  # Text: no emoji/markup parsing

    chain = Packer(database).chain(MemRow(*[r[c] for c in MemRow.__dataclass_fields__]))
    t = Table(title="version chain (oldest first)", header_style="bold")
    for col in ("", "status", "fact", "interval", "memory_id"):
        t.add_column(col, overflow="fold")
    for c in chain:
        mark = "->" if c.memory_id == r["memory_id"] else ""
        rr = conn.execute("SELECT * FROM memories WHERE memory_id = ?", (c.memory_id,)).fetchone()
        t.add_row(mark, _status(c.status), escape(c.fact_text), _interval(rr), c.memory_id)
    console.print(t)

    src = conn.execute(
        "SELECT s.role, s.turn_uid, s.quote, s.quote_start, s.quote_end FROM fact_sources s JOIN episodes e USING(turn_uid) "
        "WHERE s.memory_id = ? ORDER BY e.session_ordinal, e.turn_index",
        (r["memory_id"],),
    ).fetchall()
    t = Table(title="quotes", header_style="bold")
    for col in ("role", "turn", "quote", "span"):
        t.add_column(col, overflow="fold")
    for s in src:
        t.add_row(
            s["role"],
            s["turn_uid"],
            escape(f'"{s["quote"]}"'),
            f"{s['quote_start']}..{s['quote_end']}",
        )
    console.print(t)
    console.print("[bold]source turns (+-1)[/]")
    shown: set[str] = set()
    for s in src:
        e = conn.execute(
            "SELECT session_id, turn_index FROM episodes WHERE turn_uid = ?", (s["turn_uid"],)
        ).fetchone()
        for (uid,) in conn.execute(
            "SELECT turn_uid FROM episodes WHERE session_id = ? AND turn_index BETWEEN ? AND ? ORDER BY turn_index",
            (e[0], e[1] - 1, e[1] + 1),
        ):
            if uid not in shown:
                shown.add(uid)
                mark = "[bold]>[/]" if uid == s["turn_uid"] else " "
                console.print(f"{mark} {_turn_line(database, uid)}")

    verdicts = _nli_verdicts_for(database, r["memory_id"], {s["turn_uid"] for s in src})
    t = Table(title="NLI verdicts in answer_log", header_style="bold")
    for col in ("query_id", "claim", "entailment", "supported"):
        t.add_column(col, overflow="fold")
    for v in verdicts:
        t.add_row(v["query_id"], escape(v["text"]), f"{v['nli_entailment']:.3f}" if v["nli_entailment"] is not None else "-",
                  "yes" if v["supported"] else "[red]no[/]")  # fmt: skip
    console.print(
        t if verdicts else "[dim]no answer_log claims cite this memory or its source turns[/]"
    )


def _nli_verdicts_for(database, memory_id: str, source_turns: set[str]) -> list[dict]:
    out = []
    for qid, vj in database.conn.execute(
        "SELECT query_id, verifier_json FROM answer_log WHERE verifier_json IS NOT NULL"
    ):
        for rnd in json.loads(vj).get("rounds", []):
            for c in rnd.get("claims", []):
                if memory_id in c.get("cited_memory_ids", []) or source_turns & set(
                    c.get("cited_turn_uids", [])
                ):
                    out.append({"query_id": qid, **c})
    return out


def _inspect_query(database, q: dict) -> None:
    resp = q["response_json"] or {}
    console.print(Panel(
        f"[bold]{escape(q['query'])}[/]\n\n{escape(resp.get('final_answer', ''))}"
        + (f"\n[dim]abstained at {resp.get('abstained_at')}: {resp.get('abstention_reason')}[/]" if resp.get("abstained") else ""),
        title=q["query_id"], subtitle=f"{q['conversation_id']} | {q['created_at']}",
    ))  # fmt: skip
    a = resp.get("analysis", {})
    console.print(escape(f"intent {a.get('intent')}  speaker {a.get('subject_speaker')}  attribute {a.get('attribute_key')}  "
                  f"time {a.get('time')}  packed {resp.get('packed')}"))  # fmt: skip
    console.print(escape(f"gate: {resp.get('gate')}"))
    for rnd in (q["verifier_json"] or {}).get("rounds", []):
        t = Table(title=f"verifier round {rnd['round']}", header_style="bold")
        for col in ("claim", "kept citations", "dropped", "entailment", "supported"):
            t.add_column(col, overflow="fold")
        for c in rnd.get("claims", []):
            dropped = ", ".join(f"{d['turn_uid']} ({d['reason']})" for d in c["dropped"])
            ent = f"{c['nli_entailment']:.3f}" if c["nli_entailment"] is not None else "-"
            t.add_row(
                escape(c["text"]),
                "\n".join(c["kept_turn_uids"]),
                dropped,
                ent,
                "yes" if c["supported"] else "[red]no[/]",
            )
        console.print(t)
    for c in resp.get("claims", []):
        console.print(f"[bold]{escape(c['text'])}[/]")
        for uid in c["cited_turn_uids"]:
            console.print(f"   {_turn_line(database, uid)}")
    timings = resp.get("timings", {})
    if timings:
        console.print(
            "latency: " + "  ".join(f"{k} {v:.2f}s" for k, v in timings.items() if "." not in k)
        )


@app.command("history")
def history(
    entity: str = typer.Argument(..., help="entity name or alias"),
    attribute: str = typer.Option(None, help="restrict to one attribute key"),
    conversation: str = typer.Option(None),
    db: Path = typer.Option(None),
) -> None:
    """Every version of an entity's facts, with status and interval."""
    database = _open_db(db)
    conv_id = _conversation(database, conversation)
    ids = [
        r[0]
        for r in database.conn.execute(
            "SELECT DISTINCT e.entity_id FROM entities e LEFT JOIN entity_aliases a USING(entity_id) "
            "WHERE e.conversation_id = ? AND (lower(e.canonical_name) = lower(?) OR lower(a.alias) = lower(?))",
            (conv_id, entity, entity),
        )
    ]
    if not ids:
        raise typer.BadParameter(f"no entity named {entity!r} in {conv_id}")
    where = f"AND m.subject_entity_id IN ({','.join('?' * len(ids))})"
    params: tuple = tuple(ids)
    if attribute:
        where += " AND m.attribute_key = ?"
        params += (attribute,)
    console.print(_memories_rich(database, conv_id, where, params))
    database.close()


@app.command("stats")
def stats(db: Path = typer.Option(None)) -> None:
    """Database counts and answer latency."""
    database = _open_db(db)
    c = database.conn
    t = Table(
        title=f"ChronosGraph {database.path} (schema v{database.schema_version()})",
        header_style="bold",
    )
    t.add_column("item")
    t.add_column("count", justify="right")
    for label, sql in [
        ("conversations", "SELECT COUNT(*) FROM conversations"),
        (
            "sessions (ingested)",
            "SELECT COUNT(*) || ' (' || SUM(ingested_at IS NOT NULL) || ')' FROM sessions",
        ),
        ("episodes", "SELECT COUNT(*) FROM episodes"),
        ("episode vectors", "SELECT COUNT(*) FROM episode_embeddings"),
        ("entities", "SELECT COUNT(*) FROM entities"),
        ("fact sources", "SELECT COUNT(*) FROM fact_sources"),
        ("answers logged", "SELECT COUNT(*) FROM answer_log"),
    ]:
        t.add_row(label, str(c.execute(sql).fetchone()[0]))
    for status, n in c.execute(
        "SELECT status, COUNT(*) FROM memories GROUP BY status ORDER BY status"
    ):
        t.add_row(f"memories {_status(status)}", str(n))
    console.print(t)
    totals = [json.loads(r[0]).get("timings", {}).get("total") for r in c.execute(
        "SELECT response_json FROM answer_log WHERE response_json IS NOT NULL")]  # fmt: skip
    totals = [x for x in totals if x is not None]
    if totals:
        console.print(
            f"answers: {len(totals)}, mean latency {sum(totals) / len(totals):.2f}s, max {max(totals):.2f}s"
        )
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
    scripted: bool = typer.Option(
        False, help="Use the Alice/Bob demo's scripted outputs (offline)"
    ),
) -> None:
    """Ingest the demo into a fresh artifacts/demo.sqlite (live: user_story.json; --scripted: demo.json)."""
    scripted: bool = typer.Option(False, help="Use the demo's scripted outputs (offline)"),
) -> None:
    """Ingest data/demo_sessions/demo.json into a fresh artifacts/demo.sqlite."""
    from chronos.config import REPO_ROOT

    path = REPO_ROOT / "artifacts" / "demo.sqlite"
    for suffix in ("", "-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)
    name = "demo.json" if scripted else "user_story.json"
    ingest(REPO_ROOT / "data" / "demo_sessions" / name, db=path, scripted=scripted,
           embedder="hash" if scripted else "bge", verbose=False)  # fmt: skip
    ingest(REPO_ROOT / "data" / "demo_sessions" / "demo.json", db=path, scripted=scripted,
           embedder="hash" if scripted else "bge", verbose=False)  # fmt: skip
@app.command("ingest-demo")
def ingest_demo() -> None:
    """Ingest data/demo_sessions (implemented in Part 2)."""
    typer.echo("ingest-demo: not implemented yet (Part 2)", err=True)
    raise typer.Exit(1)


@app.command("eval")
def eval_cmd(manifest: str = typer.Option(..., help="Path to a manifest JSON")) -> None:
    """Run an evaluation manifest (implemented in Part 4)."""
    console.print(f"eval {manifest}: not implemented yet")
    """Run an evaluation manifest (implemented in Part 3)."""
    typer.echo(f"eval {manifest}: not implemented yet (Part 3)", err=True)
    raise typer.Exit(1)


@app.command("report")
def report() -> None:
    """Build the results report (implemented in Part 4)."""
    console.print("report: not implemented yet")
    typer.echo("report: not implemented yet (Part 4)", err=True)
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
