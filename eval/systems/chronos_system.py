"""ChronosGraph (and its ablation variants) behind the same interface as the baselines."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from chronos.agent.memory_agent import MemoryAgent
from chronos.config import AgentConfig, RetrievalConfig
from eval.loaders.types import QAItem
from eval.systems.common import (
    BUDGET_TOKENS,
    Packed,
    Runtime,
    asker_of,
    base_record,
    finish_record,
    question_date_of,
)


class ChronosSystem:
    id = "chronos"

    def __init__(
        self,
        rt: Runtime,
        *,
        variant: str | None = None,
        retrieval: dict[str, Any] | None = None,
        agent: dict[str, Any] | None = None,
        budget: int = BUDGET_TOKENS,
        system_id: str | None = None,
    ) -> None:
        self.rt = rt
        self.id = system_id or self.id
        self.db = rt.db(variant)
        rcfg = replace(RetrievalConfig(), token_budget=budget, **(retrieval or {}))
        acfg = replace(AgentConfig(), **(agent or {}))
        self.agent = MemoryAgent(self.db, rt.llm, rt.embedder, rt.nli, retrieval=rcfg, agent=acfg)

    def answer(self, qa: QAItem) -> dict[str, Any]:
        rec = base_record(self.id, self.rt.data, qa)
        started = time.perf_counter()
        qdate = question_date_of(qa, self.db)
        ans = self.agent.answer(
            qa.conversation_id, qa.question, question_date=qdate, asker=asker_of(qa)
        )
        total = time.perf_counter() - started
        packed = Packed(
            ans.packed.text, ans.packed.turn_uids, ans.packed.memory_ids, ans.packed.tokens,
            ans.packed.budget, ans.packed.truncated,
        )  # fmt: skip
        final = type("F", (), {"final_answer": ans.final_answer, "abstained": ans.abstained,
                               "abstention_reason": ans.abstention_reason, "claims": ans.claims})()  # fmt: skip
        s = ans.summary()
        return finish_record(
            rec, final=final, record=ans.verifier_record, stage=ans.abstained_at, packed=packed,
            gate=s["gate"], analysis=s["analysis"], timings=ans.timings, calls=ans.calls,
            total_wall=total, query_id=ans.query_id,
        )  # fmt: skip
