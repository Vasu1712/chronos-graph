"""Sample memories from the dev conversation into a CSV for hand labelling.

  uv run python scripts/audit_facts.py --ingest     # ingest the dev LoCoMo conversation first (live model)
  uv run python scripts/audit_facts.py              # sample from an already-ingested database

Columns: memory_id, quote, fact_text, attribute_key, status, interval, correct?, note
`correct?` and `note` are left blank for the human labeller. Sampling is seeded (default 42).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from chronos.config import REPO_ROOT, SEED, Paths
from chronos.ingestion.resolver import MemRow
from chronos.ingestion.temporal import describe
from chronos.storage.db import Database

COLUMNS = [
    "memory_id",
    "quote",
    "fact_text",
    "attribute_key",
    "status",
    "interval",
    "correct?",
    "note",
]


def dev_conversation_id() -> str:
    manifest = json.loads((Paths().manifests / "dev_v1.json").read_text())
    return manifest["locomo"]["conversations"][0]


def ingest_dev(db: Database, conversation_id: str) -> None:
    from chronos.ingestion.pipeline import IngestionPipeline
    from chronos.llm.cache import ResponseCache
    from chronos.llm.embed import Embedder
    from chronos.llm.local_client import LocalLLM
    from eval.loaders.locomo import load_locomo

    convs, _, _ = load_locomo(Paths().data_raw / "locomo10.json")
    conv = next(c for c in convs if c.conversation_id == conversation_id)
    pipe = IngestionPipeline(db, LocalLLM(cache=ResponseCache(Paths().llm_cache)), Embedder())
    pipe.register_conversation(conv.conversation_id, conv.dataset, conv.speakers)
    for s in conv.sessions:
        r = pipe.ingest_session(conv.conversation_id, s)
        print(json.dumps(r.summary()), flush=True)


def sample_rows(db: Database, conversation_id: str, n: int, seed: int) -> list[dict[str, str]]:
    cols = ", ".join(MemRow.__dataclass_fields__)
    mems = [MemRow(*r) for r in db.conn.execute(
        f"SELECT {cols} FROM memories WHERE conversation_id = ? ORDER BY memory_id", (conversation_id,)
    )]  # fmt: skip
    picked = random.Random(f"{seed}:audit:{conversation_id}").sample(mems, min(n, len(mems)))
    picked.sort(key=lambda m: (m.key, m.memory_id))
    out = []
    for m in picked:
        sources = db.conn.execute(
            "SELECT turn_uid, role, quote FROM fact_sources WHERE memory_id = ? "
            "ORDER BY CASE role WHEN 'ASSERTS' THEN 0 WHEN 'CORRECTS' THEN 1 ELSE 2 END, turn_uid",
            (m.memory_id,),
        ).fetchall()
        quote = " | ".join(f"[{uid} {role}] {q}" for uid, role, q in sources)
        out.append({
            "memory_id": m.memory_id, "quote": quote, "fact_text": m.fact_text,
            "attribute_key": m.attribute_key or "", "status": m.status, "interval": describe(m.rt),
            "correct?": "", "note": "",
        })  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--db", type=Path, default=REPO_ROOT / "artifacts" / "dev.sqlite")
    ap.add_argument("--conversation", default=None, help="default: the dev_v1 LoCoMo conversation")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--ingest", action="store_true", help="ingest the conversation first (live model)"
    )
    ap.add_argument(
        "--out", type=Path, default=REPO_ROOT / "artifacts" / "audit" / "dev_facts_audit.csv"
    )
    args = ap.parse_args()

    conv_id = args.conversation or dev_conversation_id()
    with Database(args.db) as db:
        if args.ingest:
            ingest_dev(db, conv_id)
        rows = sample_rows(db, conv_id, args.n, args.seed)
    if not rows:
        raise SystemExit(f"no memories for {conv_id} in {args.db}; run with --ingest first")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows for {conv_id} to {args.out} (unlabelled)")


if __name__ == "__main__":
    main()
