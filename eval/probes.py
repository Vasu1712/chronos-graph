"""Run synthetic probes (data/synthetic_probes) and check their expected final memories.

  uv run python -m eval.probes --mode scripted     # offline: scripted extractor/classifier
  uv run python -m eval.probes --mode live         # real Qwen2.5-7B via Ollama + bge embeddings

A probe passes when every expected memory exists with the expected status / interval, the set of
current (ACTIVE, open, POS) values of each listed attribute matches, and no forbidden value is
current. The live report is written to artifacts/probe_report_<mode>.json.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from chronos.config import REPO_ROOT
from chronos.ingestion.files import conversation_from_dict
from chronos.ingestion.pipeline import IngestionPipeline
from chronos.storage.db import Database

PROBE_DIR = REPO_ROOT / "data" / "synthetic_probes"
FIELDS = (
    "status",
    "polarity",
    "valid_from",
    "valid_from_precision",
    "valid_to",
    "valid_to_precision",
)


def load_probes(only: str | None = None) -> list[dict[str, Any]]:
    probes = [json.loads(p.read_text()) for p in sorted(PROBE_DIR.glob("*.json"))]
    return [p for p in probes if not only or only in p["probe_id"] or only == p["case"]]


# ------------------------------------------------------------------------------ checking
def _rows(conn: sqlite3.Connection, conv: str, subject: str) -> list[sqlite3.Row]:
    ids = {
        r[0]
        for r in conn.execute(
            "SELECT e.entity_id FROM entities e LEFT JOIN entity_aliases a USING(entity_id) "
            "WHERE e.conversation_id = ? AND (lower(e.canonical_name) = lower(?) OR lower(a.alias) = lower(?))",
            (conv, subject, subject),
        )
    }
    rows = conn.execute(
        "SELECT * FROM memories WHERE conversation_id = ? ORDER BY rowid", (conv,)
    ).fetchall()
    return [r for r in rows if r["subject_entity_id"] in ids]


def _contains(row: sqlite3.Row, spec: dict) -> bool:
    want = spec["contains"]
    want = [want] if isinstance(want, str) else want
    text = row["fact_text"].casefold()
    if not any(w.casefold() in text for w in want):
        return False
    if "attribute_key" in spec and row["attribute_key"] != spec["attribute_key"]:
        return False
    return not ("polarity" in spec and row["polarity"] != spec["polarity"])


def _field_errors(row: sqlite3.Row, spec: dict) -> list[str]:
    errs = []
    for f in FIELDS:
        if f not in spec or f == "polarity":
            continue
        want, got = spec[f], row[f]
        ok = (
            (got is not None)
            if want == "*"
            else (got in want if isinstance(want, list) else got == want)
        )
        if not ok:
            errs.append(f"{f}={got!r} (want {want!r})")
    return errs


def _is_current(row: sqlite3.Row) -> bool:
    return row["status"] == "ACTIVE" and row["valid_to"] is None


def check(conn: sqlite3.Connection, probe: dict[str, Any]) -> list[str]:
    conv, exp = probe["conversation_id"], probe["expected"]
    failures: list[str] = []
    for spec in exp.get("memories", []):
        cands = [r for r in _rows(conn, conv, spec["subject"]) if _contains(r, spec)]
        label = f"memory {spec['subject']}~{spec['contains']!r}"
        if not cands:
            failures.append(f"{label}: not found")
            continue
        errs = [_field_errors(r, spec) for r in cands]
        if all(errs):
            best = min(errs, key=len)
            failures.append(f"{label}: " + "; ".join(best))
    for spec in exp.get("current", []):
        cur = [
            r for r in _rows(conn, conv, spec["subject"])
            if _is_current(r) and r["attribute_key"] == spec["attribute_key"] and r["polarity"] == "POS"
        ]  # fmt: skip
        texts = [r["fact_text"] for r in cur]
        ok = bool(cur) and all(
            any(w.casefold() in t.casefold() for w in spec["contains"]) for t in texts
        )
        if not ok:
            failures.append(
                f"current {spec['subject']}.{spec['attribute_key']} = {texts} (want {spec['contains']})"
            )
    for spec in exp.get("absent", []):
        bad = [
            r["fact_text"]
            for r in _rows(conn, conv, spec["subject"])
            if _is_current(r) and _contains(r, spec)
        ]
        if bad:
            failures.append(f"must not be current: {bad}")
    return failures


# ------------------------------------------------------------------------------ running
def run_probe(
    probe: dict[str, Any], *, llm: Any, embedder: Any, db_path: str | Path
) -> dict[str, Any]:
    conv = conversation_from_dict(probe)
    with Database(db_path) as db:
        pipe = IngestionPipeline(db, llm, embedder)
        results = pipe.ingest_conversation(conv)
        failures = check(db.conn, probe)
        extraction: Counter = Counter()
        dropped: Counter = Counter()
        for r in results:
            if r.extraction:
                extraction.update(
                    {k: v for k, v in r.extraction.as_dict().items() if isinstance(v, int)}
                )
                dropped.update(r.extraction.dropped)
        from chronos.cli import _memory_table

        return {
            "probe_id": probe["probe_id"],
            "case": probe["case"],
            "passed": not failures,
            "failures": failures,
            "extraction": dict(extraction),
            "dropped": dict(dropped),
            "relations": dict(sum((r.relation_counts for r in results), Counter())),
            "decisions": [
                {
                    "fact": d.fact_text,
                    "llm": d.llm_relation,
                    "applied": d.relation,
                    "notes": d.notes,
                }
                for r in results
                for d in r.decisions
            ],
            "memories": _memory_table(db, conv.conversation_id),
        }


def run_all(mode: str, only: str | None = None) -> list[dict[str, Any]]:
    probes = load_probes(only)
    if mode == "live":
        from chronos.config import Paths
        from chronos.llm.cache import ResponseCache
        from chronos.llm.embed import Embedder
        from chronos.llm.local_client import LocalLLM

        llm = LocalLLM(cache=ResponseCache(Paths().llm_cache))
        embedder = Embedder()
    else:
        from chronos.llm.embed import HashingEmbedder

        embedder = HashingEmbedder()
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for p in probes:
            if mode == "scripted":
                from chronos.ingestion.scripted import ScriptedLLM

                llm = ScriptedLLM(conversation_from_dict(p), p["scripted"])
            out.append(
                run_probe(
                    p, llm=llm, embedder=embedder, db_path=Path(tmp) / f"{p['probe_id']}.sqlite"
                )
            )
    return out


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    per_case: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    dropped: Counter = Counter()
    extraction: Counter = Counter()
    for r in results:
        per_case[r["case"]][0] += r["passed"]
        per_case[r["case"]][1] += 1
        dropped.update(r["dropped"])
        extraction.update(r["extraction"])
    return {
        "per_case": {c: {"passed": p, "total": t} for c, (p, t) in sorted(per_case.items())},
        "passed": sum(r["passed"] for r in results),
        "total": len(results),
        "extraction": dict(extraction),
        "dropped": dict(dropped),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", choices=["scripted", "live"], default="scripted")
    ap.add_argument("--only", help="probe id substring or case name")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    results = run_all(args.mode, args.only)
    summary = summarize(results)
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"{mark} {r['probe_id']:<20} {'; '.join(r['failures'])}")
    print()
    for case, v in summary["per_case"].items():
        print(f"{case:<24} {v['passed']}/{v['total']}")
    print(f"{'TOTAL':<24} {summary['passed']}/{summary['total']}")
    print(f"extraction: {summary['extraction']}")
    print(f"dropped: {summary['dropped']}")
    out = args.out or REPO_ROOT / "artifacts" / f"probe_report_{args.mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"mode": args.mode, "summary": summary, "results": results}, indent=1) + "\n"
    )
    print(f"report: {out}")


if __name__ == "__main__":
    main()
