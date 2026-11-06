"""Ingest every conversation / LongMemEval-S-24k history of a manifest once.

  uv run python -m eval.ingest_all --manifest eval/manifests/quick_v1.json

Output: artifacts/memory_<manifest>.sqlite plus artifacts/ingest_logs/<manifest>.jsonl (one record
per session with wall time and LLM usage, one per finished conversation with a state summary).
Resumable and safe to stop at any time: a session is one transaction, so a killed run leaves no
partial session, and the next run skips every session already ingested. Wrap long runs in
`caffeinate -i` (the Makefile does).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from chronos.ingestion.pipeline import IngestionPipeline
from chronos.ingestion.types import Conversation
from chronos.storage.db import Database
from eval.common import (
    ManifestData,
    append_jsonl,
    ingest_log_path,
    ingestion_config_for,
    llm_cache_path,
    load_manifest_data,
    memory_db_path,
    now_iso,
    read_jsonl,
    stop_on_sigterm,
    timed_step,
)


def summarize_calls(records: list[Any]) -> dict[str, Any]:
    by: dict[str, dict[str, float]] = {}
    for r in records:
        d = by.setdefault(r.purpose, {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                      "latency_s": 0.0, "wall_s": 0.0})  # fmt: skip
        d["calls"] += 1
        d["cached"] += int(r.cached)
        d["prompt_tokens"] += r.prompt_tokens
        d["completion_tokens"] += r.completion_tokens
        d["latency_s"] += r.latency_s or 0.0
        d["wall_s"] += r.wall_s
    return {
        "calls": sum(int(d["calls"]) for d in by.values()),
        "cached_calls": sum(int(d["cached"]) for d in by.values()),
        "prompt_tokens": sum(int(d["prompt_tokens"]) for d in by.values()),
        "completion_tokens": sum(int(d["completion_tokens"]) for d in by.values()),
        "llm_seconds": sum(d["latency_s"] for d in by.values()),
        "by_purpose": by,
    }


def conversation_summary(db: Database, conversation_id: str) -> dict[str, Any]:
    """State of one ingested conversation, kept in the log so reports need no database."""
    c = db.conn
    status = dict(c.execute("SELECT status, COUNT(*) FROM memories WHERE conversation_id = ? GROUP BY status",
                            (conversation_id,)).fetchall())  # fmt: skip
    keys = Counter(r[0] or "(none)" for r in c.execute(
        "SELECT attribute_key FROM memories WHERE conversation_id = ?", (conversation_id,)))  # fmt: skip
    ents = c.execute(
        "SELECT canonical_name, entity_type, descriptor, entity_id FROM entities "
        "WHERE conversation_id = ? AND is_speaker = 0 ORDER BY rowid", (conversation_id,)).fetchall()  # fmt: skip
    by_name: dict[str, list[str]] = {}
    for e in ents:
        by_name.setdefault((e[0].casefold(), e[1]), []).append(e[2] or "")
    relations = dict(c.execute(
        "SELECT applied_relation, COUNT(*) FROM resolver_log WHERE conversation_id = ? GROUP BY 1",
        (conversation_id,)).fetchall())  # fmt: skip
    return {
        "memories": sum(status.values()),
        "memories_by_status": status,
        "attribute_keys": dict(keys.most_common(20)),
        "entities": len(ents),
        "entity_name_duplicates": [
            {"name": k[0], "type": k[1], "count": len(v), "descriptors": v}
            for k, v in by_name.items()
            if len(v) > 1
        ],  # fmt: skip
        "relations": relations,
    }


def ingest_conversation(
    pipe_factory, db: Database, conv: Conversation, *, manifest: str, log_path: Path, llm: Any,
    variant: str | None = None, progress=print,
) -> dict[str, Any]:  # fmt: skip
    """Ingest the not-yet-ingested sessions of one conversation; returns what was done."""
    done = {
        r[0] for r in db.conn.execute(
            "SELECT session_id FROM sessions WHERE conversation_id = ? AND ingested_at IS NOT NULL",
            (conv.conversation_id,))
    }  # fmt: skip
    todo = [s for s in conv.sessions if s.session_id not in done]
    out = {"conversation_id": conv.conversation_id, "sessions_total": len(conv.sessions),
           "sessions_skipped": len(conv.sessions) - len(todo), "sessions_ingested": 0, "seconds": 0.0}  # fmt: skip
    if not todo:
        return out
    pipe: IngestionPipeline = pipe_factory(conv)
    pipe.register_conversation(conv.conversation_id, conv.dataset, conv.speakers)
    for session in todo:
        mark = len(llm.records)
        started = time.perf_counter()
        result = pipe.ingest_session(conv.conversation_id, session)
        seconds = time.perf_counter() - started
        usage = summarize_calls(llm.records[mark:])
        ex = result.extraction.as_dict() if result.extraction else None
        append_jsonl(log_path, {
            "type": "session", "manifest": manifest, "variant": variant, "at": now_iso(),
            "dataset": conv.dataset, "conversation_id": conv.conversation_id,
            "session_id": session.session_id, "session_ordinal": session.ordinal,
            "turns": len(session.turns), "seconds": seconds, "llm": usage,
            "extraction": ex, "relations": dict(result.relation_counts),
            "memories_written": len(result.decisions),
        })  # fmt: skip
        out["sessions_ingested"] += 1
        out["seconds"] += seconds
        progress(f"  {session.session_id}: {len(session.turns)} turns, {seconds:6.1f}s, "
                 f"{usage['calls']} LLM calls ({usage['cached_calls']} cached), "
                 f"{len(result.decisions)} facts")  # fmt: skip
    seconds_total = sum(r["seconds"] for r in read_jsonl(log_path)
                        if r.get("type") == "session" and r["conversation_id"] == conv.conversation_id)  # fmt: skip
    append_jsonl(log_path, {
        "type": "conversation", "manifest": manifest, "variant": variant, "at": now_iso(),
        "dataset": conv.dataset, "conversation_id": conv.conversation_id,
        "sessions": len(conv.sessions), "turns": sum(len(s.turns) for s in conv.sessions),
        "ingest_seconds": seconds_total, **conversation_summary(db, conv.conversation_id),
    })  # fmt: skip
    return out


def ingest_manifest(
    data: ManifestData, db_path: Path, llm: Any, embedder: Any, *, log_path: Path | None = None,
    policy: str = "bitemporal", recorded: dict | None = None, variant: str | None = None,
    limit: int | None = None, only: str | None = None, progress=print,
) -> dict[str, Any]:  # fmt: skip
    log_path = log_path or ingest_log_path(data.name, variant)
    db = Database(db_path)
    convs = [c for c in data.conversations.values() if only is None or only in c.conversation_id]
    if limit is not None:
        convs = convs[:limit]
    started = time.perf_counter()
    totals = {"conversations": len(convs), "sessions_ingested": 0, "sessions_skipped": 0}

    def pipe_factory(conv: Conversation) -> IngestionPipeline:
        rec = None
        if recorded is not None:
            rec = {k: v for k, v in recorded.items() if k.startswith(f"{conv.conversation_id}:")}
        return IngestionPipeline(
            db, llm, embedder, config=ingestion_config_for(conv), policy=policy, recorded=rec
        )

    try:
        for i, conv in enumerate(convs, start=1):
            progress(
                f"[{i}/{len(convs)}] {conv.conversation_id} ({conv.dataset}, {len(conv.sessions)} sessions)"
            )
            res = ingest_conversation(pipe_factory, db, conv, manifest=data.name, log_path=log_path,
                                      llm=llm, variant=variant, progress=progress)  # fmt: skip
            totals["sessions_ingested"] += res["sessions_ingested"]
            totals["sessions_skipped"] += res["sessions_skipped"]
            if res["sessions_ingested"] == 0:
                progress("  already ingested")
    finally:
        db.close()
    totals["seconds"] = time.perf_counter() - started
    return totals


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--db", type=Path, help="default: artifacts/memory_<manifest>.sqlite")
    ap.add_argument("--embedder", choices=["bge", "hash"], default="bge")
    ap.add_argument("--limit", type=int, help="only the first N conversations (smoke test)")
    ap.add_argument("--only", help="only conversations whose id contains this text")
    args = ap.parse_args(argv)
    stop_on_sigterm()

    from chronos.llm.cache import ResponseCache
    from chronos.llm.local_client import LocalLLM

    data = load_manifest_data(args.manifest)
    if args.embedder == "hash":
        from chronos.llm.embed import HashingEmbedder

        embedder = HashingEmbedder()
    else:
        from chronos.llm.embed import Embedder

        embedder = Embedder()
    llm = LocalLLM(cache=ResponseCache(llm_cache_path()))
    db_path = args.db or memory_db_path(data.name)
    print(
        f"ingesting {len(data.conversations)} conversations of {data.name} into {db_path}",
        flush=True,
    )
    with timed_step(data.name, "ingest"):
        totals = ingest_manifest(data, db_path, llm, embedder, limit=args.limit, only=args.only,
                                 progress=lambda m: print(m, flush=True))  # fmt: skip
    print(f"done: {totals}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
