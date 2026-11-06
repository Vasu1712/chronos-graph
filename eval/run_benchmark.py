"""Run systems (and ablation variants) over a manifest's questions.

  uv run python -m eval.run_benchmark --manifest eval/manifests/quick_v1.json --systems chronos,B3_chunk_rag
  uv run python -m eval.run_benchmark --manifest eval/manifests/test_v1.json --systems all --ablations all

One JSONL line per (system, question) goes to artifacts/raw_outputs/<manifest>/<system>.jsonl.
Resumable: questions already present are skipped, and every LLM call is cached, so an interrupted run
costs nothing to continue. Requires the manifest to be ingested first (eval.ingest_all). A system
that cannot answer a question (B2 when the history exceeds the 32k window) writes a `skipped` line
instead of a wrong answer.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chronos.ingestion.pipeline import recorded_decisions
from chronos.storage.db import Database
from eval.ablations import POLICY_DBS, Ablation, load_ablations, resolve_selection
from eval.common import (
    ManifestData,
    append_jsonl,
    llm_cache_path,
    load_manifest_data,
    memory_db_path,
    raw_output_path,
    read_jsonl,
    stop_on_sigterm,
    timed_step,
)
from eval.ingest_all import ingest_manifest
from eval.loaders.types import QAItem
from eval.systems import MAIN_SYSTEMS, OPTIONAL_SYSTEMS, Runtime, Skipped, build_system


@dataclass
class RunSpec:
    system_id: str
    system: str
    spec: dict[str, Any] = field(default_factory=dict)
    select: Callable[[QAItem], bool] = lambda qa: True


def plan(systems: list[str], ablations: list[Ablation]) -> list[RunSpec]:
    """Main systems first (plus any system an ablation reuses), then each ablation variant."""
    names = list(dict.fromkeys(systems))
    for ab in ablations:
        for alias in sorted(ab.aliases):
            if alias not in names:
                names.append(alias)
    specs = [RunSpec(n, n) for n in names]
    for ab in ablations:
        for v in ab.variants:
            if v.alias_of:
                continue
            specs.append(RunSpec(v.system_id, v.system, dict(v.spec), ab.selects))
    return specs


def ensure_policy_db(data: ManifestData, variant: str, embedder: Any, progress=print) -> Path:
    """Build artifacts/memory_<manifest>__<variant>.sqlite by replaying the full run's recorded
    decisions under another state policy. Extraction comes from the LLM cache: a cache miss is an
    error, never a new model call."""
    from chronos.llm.cache import ResponseCache
    from chronos.llm.local_client import LocalLLM

    if variant not in POLICY_DBS:
        raise ValueError(f"unknown policy database {variant!r}")
    path = memory_db_path(data.name, variant)
    src = Database(memory_db_path(data.name))
    try:
        recorded = recorded_decisions(src.conn)
    finally:
        src.close()
    llm = LocalLLM(cache=ResponseCache(llm_cache_path()), cache_only=True)
    progress(f"building the {variant} state database from {len(recorded)} recorded decisions")
    with timed_step(data.name, f"replay:{variant}"):
        ingest_manifest(
            data,
            path,
            llm,
            embedder,
            policy=variant,
            recorded=recorded,
            variant=variant,
            progress=lambda m: None,
        )
    return path


def run_system(
    spec: RunSpec, rt: Runtime, data: ManifestData, *, limit: int | None = None, progress=print
) -> dict[str, int]:
    out = raw_output_path(data.name, spec.system_id)
    done = {r["question_id"] for r in read_jsonl(out)}
    questions = [q for q in data.qas if spec.select(q)]
    if limit is not None:
        questions = questions[:limit]
    todo = [q for q in questions if q.question_id not in done]
    counts = {
        "questions": len(questions),
        "already_done": len(questions) - len(todo),
        "answered": 0,
        "skipped": 0,
        "errors": 0,
    }
    if not todo:
        return counts
    system = build_system(spec.system, rt, **(spec.spec | {"system_id": spec.system_id}))
    started = time.perf_counter()
    for i, qa in enumerate(todo, start=1):
        t0 = time.perf_counter()
        try:
            rec = system.answer(qa)
        except Skipped as why:
            from eval.systems.common import base_record

            append_jsonl(out, base_record(spec.system_id, data, qa) | {"skipped": str(why)})
            counts["skipped"] += 1
            progress(
                f"  [{spec.system_id}] {i}/{len(todo)} {qa.question_id} skipped: {str(why)[:90]}"
            )
            continue
        except Exception:  # keep going: this question is retried by the next run
            counts["errors"] += 1
            traceback.print_exc()
            progress(
                f"  [{spec.system_id}] {i}/{len(todo)} {qa.question_id} ERROR (will retry on the next run)"
            )
            continue
        append_jsonl(out, rec)
        counts["answered"] += 1
        elapsed = time.perf_counter() - started
        eta = elapsed / i * (len(todo) - i)
        ans = ("[abstained] " if rec["abstained"] else "") + rec["final_answer"].replace("\n", " ")[
            :70
        ]
        progress(f"  [{spec.system_id}] {i}/{len(todo)} {qa.category[:11]:<11} {time.perf_counter() - t0:5.1f}s "
                 f"eta {eta / 60:5.1f}m  {ans}")  # fmt: skip
    return counts


def make_runtime(data: ManifestData, embedder_name: str = "bge") -> Runtime:
    from chronos.llm.cache import ResponseCache
    from chronos.llm.local_client import LocalLLM
    from chronos.llm.nli import NLI

    if embedder_name == "hash":
        from chronos.llm.embed import HashingEmbedder

        embedder: Any = HashingEmbedder()
    else:
        from chronos.llm.embed import Embedder

        embedder = Embedder()
    llm = LocalLLM(cache=ResponseCache(llm_cache_path()))

    def open_db(variant: str | None) -> Database:
        path = memory_db_path(data.name, variant)
        if not path.exists():
            raise SystemExit(f"{path} does not exist: run eval.ingest_all first")
        return Database(path)

    return Runtime(data=data, llm=llm, embedder=embedder, nli=NLI(), open_db=open_db)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument(
        "--systems",
        default="chronos",
        help=f"comma list or 'all'; known: {', '.join(MAIN_SYSTEMS + OPTIONAL_SYSTEMS)}",
    )
    ap.add_argument("--ablations", default="", help="A,B,C,D / names / all")
    ap.add_argument("--state-probes", choices=["none", "scripted", "live"], default="none",
                    help="also run the 30 synthetic probes under each state policy (ablation B)")  # fmt: skip
    ap.add_argument("--limit", type=int, help="first N questions per run (smoke test)")
    ap.add_argument("--embedder", choices=["bge", "hash"], default="bge")
    args = ap.parse_args(argv)
    stop_on_sigterm()

    data = load_manifest_data(args.manifest)
    systems = (
        list(MAIN_SYSTEMS)
        if args.systems == "all"
        else [s.strip() for s in args.systems.split(",") if s.strip()]
    )
    ablations = resolve_selection(args.ablations, load_ablations())
    specs = plan(systems, ablations)
    print(
        f"{data.name}: {len(data.qas)} questions; runs: {', '.join(s.system_id for s in specs)}",
        flush=True,
    )
    rt = make_runtime(data, args.embedder)
    try:
        for variant in sorted(
            {v for ab in ablations for v in ab.db_variants}
            | ({"append_only"} if "B5_append_only" in {s.system for s in specs} else set())
        ):
            if not _db_complete(data, variant):
                ensure_policy_db(data, variant, rt.embedder)
        errors = 0
        for spec in specs:
            with timed_step(data.name, f"answer:{spec.system_id}"):
                counts = run_system(
                    spec, rt, data, limit=args.limit, progress=lambda m: print(m, flush=True)
                )
            errors += counts["errors"]
            print(f"{spec.system_id}: {counts}", flush=True)
    finally:
        rt.close()
    if args.state_probes != "none":
        from eval.state_probes import run_state_probes

        with timed_step(data.name, f"state_probes:{args.state_probes}"):
            print(run_state_probes(args.state_probes)["summary"])
    return 1 if errors else 0


def _db_complete(data: ManifestData, variant: str) -> bool:
    path = memory_db_path(data.name, variant)
    if not path.exists():
        return False
    db = Database(path)
    try:
        ingested = {
            r[0]
            for r in db.conn.execute(
                "SELECT session_id FROM sessions WHERE ingested_at IS NOT NULL"
            )
        }
    finally:
        db.close()
    return all(s.session_id in ingested for c in data.conversations.values() for s in c.sessions)


if __name__ == "__main__":
    sys.exit(main())
