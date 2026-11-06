"""System registry: build a system (main or ablation variant) by its id."""

from __future__ import annotations

from typing import Any

from eval.systems.baselines import ChunkRAG, FullContext, SlidingWindow, TurnsPlusNotes
from eval.systems.chronos_system import ChronosSystem
from eval.systems.common import BUDGET_TOKENS, Runtime, Skipped

MAIN_SYSTEMS = (
    "chronos",
    "B1_sliding_window",
    "B2_full_context",
    "B3_chunk_rag",
    "B4_turns_plus_notes",
    "B5_append_only",
)
OPTIONAL_SYSTEMS = ("B6_mem0",)
B6_NOT_RUN = (
    "B6_mem0 is not run: mem0ai installs cleanly (Python 3.11, Ollama + local embeddings), but its "
    "memories carry no turn provenance, so under the shared turn-citation verifier every claim would be "
    "dropped. A fair comparison would need a verifier exemption, which breaks the rule that every system "
    "shares the generator, verifier and answer schema."
)


def build_system(name: str, rt: Runtime, **spec: Any):
    """`spec` may carry an ablation variant: db, retrieval, agent, budget, system_id."""
    budget = spec.get("budget", BUDGET_TOKENS)
    sid = spec.get("system_id")
    if name == "chronos":
        return ChronosSystem(rt, variant=spec.get("db"), retrieval=spec.get("retrieval"), agent=spec.get("agent"),
                             budget=budget, system_id=sid)  # fmt: skip
    if name == "B5_append_only":
        return ChronosSystem(rt, variant="append_only", budget=budget, system_id=sid or name)
    if name == "B1_sliding_window":
        return SlidingWindow(rt, budget=budget, system_id=sid)
    if name == "B2_full_context":
        return FullContext(rt, budget=budget, system_id=sid)
    if name == "B3_chunk_rag":
        return ChunkRAG(rt, budget=budget, system_id=sid)
    if name == "B4_turns_plus_notes":
        return TurnsPlusNotes(rt, budget=budget, system_id=sid)
    if name == "B6_mem0":
        raise SystemExit(B6_NOT_RUN)
    raise KeyError(f"unknown system {name!r}")


__all__ = ["MAIN_SYSTEMS", "OPTIONAL_SYSTEMS", "Runtime", "Skipped", "build_system"]
