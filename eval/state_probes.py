"""Ablation B on the 30 synthetic probes: final memory state under each state policy.

SYNTHETIC: hand-written conversations (data/synthetic_probes). Reported separately from the benchmark
results and labelled as such. For each probe the full (bi-temporal) policy is ingested first; the
arrival-order and append-only policies then replay its recorded classifier decisions.

  mode scripted   the probes' scripted outputs (an ideal extractor/classifier): isolates the state logic
  mode live       Qwen2.5-7B through the LLM cache / Ollama: the realistic extractor and classifier
"""

from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from chronos.ingestion.files import conversation_from_dict
from chronos.ingestion.pipeline import IngestionPipeline, recorded_decisions
from chronos.ingestion.scripted import ScriptedLLM
from chronos.llm.embed import HashingEmbedder
from chronos.storage.db import Database
from eval.common import artifacts
from eval.probes import check, load_probes

POLICIES = ("append_only", "arrival_order", "bitemporal")


class _ExtractionOnly:
    """Wraps an LLM so a replay can extract but never classify."""

    def __init__(self, inner: Any) -> None:
        self.inner, self.records = inner, getattr(inner, "records", [])

    def structured(self, messages, model):
        if model.__name__ == "RelationOut":
            raise AssertionError("a replay must not call the classifier")
        return self.inner.structured(messages, model)


def run_state_probes(
    mode: str, out: Path | None = None, *, only: str | None = None
) -> dict[str, Any]:
    if mode == "live":
        from chronos.llm.cache import ResponseCache
        from chronos.llm.embed import Embedder
        from chronos.llm.local_client import LocalLLM
        from eval.common import llm_cache_path

        live = LocalLLM(cache=ResponseCache(llm_cache_path()))
        embedder: Any = Embedder()
    else:
        live, embedder = None, HashingEmbedder()
    details = []
    with tempfile.TemporaryDirectory() as tmp:
        for probe in load_probes(only):
            conv = conversation_from_dict(probe)
            llm = live or ScriptedLLM(conv, probe["scripted"])
            dbs: dict[str, Database] = {}
            dbs["bitemporal"] = Database(Path(tmp) / f"{probe['probe_id']}__bitemporal.sqlite")
            IngestionPipeline(dbs["bitemporal"], llm, embedder).ingest_conversation(conv)
            recorded = recorded_decisions(dbs["bitemporal"].conn)
            replay_llm = _ExtractionOnly(llm)
            for policy in ("arrival_order", "append_only"):
                dbs[policy] = Database(Path(tmp) / f"{probe['probe_id']}__{policy}.sqlite")
                IngestionPipeline(
                    dbs[policy], replay_llm, embedder, policy=policy, recorded=recorded
                ).ingest_conversation(conv)
            for policy in POLICIES:
                failures = check(dbs[policy].conn, probe)
                # strict = every expected memory incl. its exact interval; current = only the checks
                # on which value is current / must not be (the part arrival order can still get right)
                current_ok = not any(
                    f.startswith(("current ", "must not be current")) for f in failures
                )
                details.append({"probe_id": probe["probe_id"], "case": probe["case"], "policy": policy,
                                "passed": not failures, "current_ok": current_ok, "failures": failures})  # fmt: skip
            for d in dbs.values():
                d.close()
    table: dict[str, dict[str, list[int]]] = {p: defaultdict(lambda: [0, 0, 0]) for p in POLICIES}
    for d in details:
        cell = table[d["policy"]][d["case"]]
        cell[0] += d["passed"]
        cell[1] += 1
        cell[2] += d["current_ok"]
    result = {
        "label": "synthetic", "mode": mode, "details": details,
        "per_policy_case": {p: {c: {"passed": v[0], "total": v[1], "current_ok": v[2]} for c, v in sorted(cases.items())} for p, cases in table.items()},
        "summary": {p: f"{sum(v[0] for v in cases.values())}/{sum(v[1] for v in cases.values())}" for p, cases in table.items()},
    }  # fmt: skip
    out = out or artifacts() / f"state_probes_{mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    return result
