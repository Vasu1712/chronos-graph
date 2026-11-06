"""What every system shares: the runtime, the raw-output record, and context-building helpers."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from chronos.agent.answering import generate_and_verify
from chronos.agent.verifier import AnswerLog, Verifier
from chronos.config import AgentConfig, RetrievalConfig
from chronos.llm.tokens import count_tokens
from chronos.retrieval.packer import clip
from chronos.storage.db import Database
from eval.common import LME_ASKER, ManifestData, now_iso
from eval.loaders.types import QAItem

BUDGET_TOKENS = 1500
TURN_CLIP_CHARS = (
    RetrievalConfig().max_source_turn_chars
)  # baselines clip turns like ChronosGraph's sources


class Skipped(Exception):
    """A system cannot answer this question (recorded as a skip, never as a wrong answer)."""


@dataclass
class Runtime:
    """Models and databases shared by every system in one process."""

    data: ManifestData
    llm: Any
    embedder: Any
    nli: Any
    open_db: Any  # callable(variant: str | None) -> Database
    _dbs: dict[str | None, Database] = field(default_factory=dict)

    def db(self, variant: str | None = None) -> Database:
        if variant not in self._dbs:
            self._dbs[variant] = self.open_db(variant)
        return self._dbs[variant]

    def close(self) -> None:
        for d in self._dbs.values():
            d.close()
        self._dbs.clear()


@dataclass
class Packed:
    text: str
    turn_uids: list[str]
    memory_ids: list[str] = field(default_factory=list)
    tokens: int = 0
    budget: int | None = BUDGET_TOKENS
    truncated: bool = False


def turn_line(
    uid: str, date: str, speaker: str, text: str, limit: int | None = TURN_CLIP_CHARS
) -> str:
    return f"[{uid} | {date[:10]} | {speaker}] {clip(text, limit) if limit else text}"


def fit_lines(lines: list[str], budget: int) -> tuple[list[str], bool]:
    """Longest prefix of `lines` whose joined text fits `budget` tokens, and whether it was cut."""
    out: list[str] = []
    for ln in lines:
        if count_tokens("\n".join([*out, ln])) > budget:
            return out, True
        out.append(ln)
    return out, False


def question_date_of(qa: QAItem, db: Database) -> str | None:
    if qa.question_date:
        return qa.question_date
    row = db.conn.execute(
        "SELECT MAX(session_ts) FROM sessions WHERE conversation_id = ?", (qa.conversation_id,)
    ).fetchone()
    return row[0] if row else None


def asker_of(qa: QAItem) -> str | None:
    return LME_ASKER if qa.dataset == "longmemeval" else None


def reconstruct_latency(total_wall: float, calls: list[dict[str, Any]]) -> float:
    """Wall time with each LLM call's cost replaced by its original latency (a cached call replays
    what it cost when it was made; calls from before usage tracking keep their wall time)."""
    return (
        total_wall
        - sum(c["wall_s"] for c in calls)
        + sum(c["latency_s"] if c.get("latency_s") is not None else c["wall_s"] for c in calls)
    )


def base_record(system: str, data: ManifestData, qa: QAItem) -> dict[str, Any]:
    return {
        "system": system, "manifest": data.name, "dataset": qa.dataset, "question_id": qa.question_id,
        "conversation_id": qa.conversation_id, "category": qa.category, "question": qa.question,
        "gold_answer": qa.answer, "gold_unanswerable": qa.is_abstention,
        "gold_evidence_turn_uids": list(qa.evidence_turn_uids),
        "gold_evidence_sessions": list(qa.evidence_session_ordinals), "created_at": now_iso(),
    }  # fmt: skip


def finish_record(rec: dict[str, Any], *, final, record, stage, packed: Packed, gate, analysis, timings,
                  calls: list[dict[str, Any]], total_wall: float, query_id: str) -> dict[str, Any]:  # fmt: skip
    rec |= {
        "query_id": query_id,
        "final_answer": final.final_answer, "abstained": final.abstained,
        "abstained_at": stage, "abstention_reason": final.abstention_reason,
        "claims": [c.model_dump() for c in final.claims],
        "verifier": record, "gate": gate, "analysis": analysis,
        "context": {"tokens": packed.tokens, "budget": packed.budget, "truncated": packed.truncated,
                    "turn_uids": packed.turn_uids, "memory_ids": packed.memory_ids, "text": packed.text},
        "llm_calls": calls, "timings": timings,
        "latency_s": reconstruct_latency(total_wall, calls),
    }  # fmt: skip
    return rec


class BaselineSystem:
    """B1-B4: build a context, then the shared generate -> verify -> repair path."""

    id = "baseline"

    def __init__(self, rt: Runtime, *, budget: int = BUDGET_TOKENS, agent: AgentConfig | None = None,
                 variant: str | None = None, system_id: str | None = None) -> None:  # fmt: skip
        self.rt = rt
        self.budget = budget
        self.agent = agent or AgentConfig()
        self.db = rt.db(variant)
        self.id = system_id or self.id
        self.verifier = Verifier(self.db.conn, rt.nli, self.agent.nli_threshold)
        self.log = AnswerLog(self.db.conn)

    def build_context(self, qa: QAItem, qdate: str | None) -> Packed:  # pragma: no cover - abstract
        raise NotImplementedError

    def answer(self, qa: QAItem) -> dict[str, Any]:
        rt = self.rt
        rec = base_record(self.id, rt.data, qa)
        mark = len(rt.llm.records)
        started = time.perf_counter()
        qdate = question_date_of(qa, self.db)
        t = time.perf_counter()
        packed = self.build_context(qa, qdate)
        timings = {"retrieve": time.perf_counter() - t}
        query_id = f"q:{uuid.uuid4().hex[:12]}"
        self.log.open(
            query_id, qa.conversation_id, qa.question, packed.memory_ids, packed.turn_uids
        )
        ans = generate_and_verify(
            llm=rt.llm, verifier=self.verifier, query_id=query_id, question=qa.question,
            context=packed.text, allowed_turn_uids=packed.turn_uids, question_date=qdate,
            asker=asker_of(qa), question_type=None, regenerate_once=self.agent.regenerate_once,
        )  # fmt: skip
        timings |= ans.timings
        total = time.perf_counter() - started
        timings["total"] = total
        calls = [r.as_dict() for r in rt.llm.records[mark:]]
        return finish_record(rec, final=ans.final, record=ans.record, stage=ans.stage, packed=packed,
                             gate=None, analysis=None, timings=timings, calls=calls, total_wall=total,
                             query_id=query_id)  # fmt: skip
