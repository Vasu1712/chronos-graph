"""MemoryAgent.answer(): query analysis -> retrieval -> packing -> gate -> generation -> verification.

Every answer is written to answer_log: the packed memory_ids / turn_uids (the allowed citation
set) before generation, and the final response, gate result, per-step timings and per-claim
verifier record (NLI scores, dropped citations, both rounds) after it.

The question date defaults to the latest session timestamp of the conversation, which is the
"now" of a recorded conversation (LoCoMo has no question dates; LongMemEval passes its own).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from chronos.agent.answerability import AnswerabilityGate, GateResult
from chronos.agent.generator import Claim, GenerationOut, Generator, question_type
from chronos.agent.verifier import AnswerLog, Verifier, verify_and_repair
from chronos.config import AgentConfig, RetrievalConfig
from chronos.retrieval.packer import PackedContext, Packer
from chronos.retrieval.query import QueryAnalysis, analyze
from chronos.retrieval.retriever import Retriever
from chronos.storage.db import Database

STEPS = ("analyze", "retrieve", "pack", "gate", "generate", "verify", "regenerate", "total")


@dataclass
class AgentAnswer:
    query_id: str
    question: str
    final_answer: str
    claims: list[Claim]
    abstained: bool
    abstention_reason: str | None
    abstained_at: str | None  # gate | model | verifier | None
    analysis: QueryAnalysis
    packed: PackedContext
    gate: GateResult
    timings: dict[str, float] = field(default_factory=dict)
    llm_calls: int = 0

    def summary(self) -> dict[str, Any]:
        a = self.analysis
        return {
            "query_id": self.query_id,
            "final_answer": self.final_answer,
            "claims": [c.model_dump() for c in self.claims],
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "abstained_at": self.abstained_at,
            "analysis": {
                "intent": a.intent.value, "cues": a.cues, "entities": [e.name for e in a.entities],
                "subject_speaker": a.subject_speaker_name, "attribute_key": a.attribute_key,
                "time": asdict(a.time) if a.time else None, "question_date": a.question_date,
            },
            "packed": {"tokens": self.packed.tokens, "budget": self.packed.budget, "truncated": self.packed.truncated,
                       "memories": len(self.packed.memory_ids), "turns": len(self.packed.turn_uids)},
            "gate": self.gate.as_dict(),
            "timings": self.timings,
            "llm_calls": self.llm_calls,
        }  # fmt: skip


class _TimedGenerator(Generator):
    """Generator that records the wall time of every call (first call = generate, rest = regenerate)."""

    def __init__(self, llm: Any) -> None:
        super().__init__(llm)
        self.durations: list[float] = []

    def generate(self, *args: Any, **kwargs: Any) -> GenerationOut:
        t = time.perf_counter()
        try:
            return super().generate(*args, **kwargs)
        finally:
            self.durations.append(time.perf_counter() - t)


class MemoryAgent:
    def __init__(
        self,
        db: Database,
        llm: Any,
        embedder: Any,
        nli: Any,
        *,
        retrieval: RetrievalConfig | None = None,
        agent: AgentConfig | None = None,
        count_tokens: Any = None,
    ) -> None:
        self.db = db
        self.llm = llm
        self.retrieval_config = retrieval or RetrievalConfig()
        self.agent_config = agent or AgentConfig()
        self.retriever = Retriever(db, embedder, self.retrieval_config)
        self.packer = Packer(db, self.retrieval_config, count_tokens)
        self.gate = AnswerabilityGate(db, llm, self.agent_config.answerability)
        self.verifier = Verifier(db.conn, nli, self.agent_config.nli_threshold)
        self.log = AnswerLog(db.conn)

    def default_question_date(self, conversation_id: str) -> str | None:
        row = self.db.conn.execute(
            "SELECT MAX(session_ts) FROM sessions WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()
        return row[0] if row else None

    def answer(
        self,
        conversation_id: str,
        question: str,
        as_of: str | None = None,
        *,
        question_date: str | None = None,
        asker: str | None = None,
    ) -> AgentAnswer:
        timings: dict[str, float] = dict.fromkeys(STEPS, 0.0)
        t_all = time.perf_counter()
        calls_before = getattr(self.llm, "calls", None)
        qdate = question_date or as_of or self.default_question_date(conversation_id)

        t = time.perf_counter()
        analysis = analyze(question, conn=self.db.conn, conversation_id=conversation_id,
                           question_date=qdate, as_of=as_of, asker=asker)  # fmt: skip
        timings["analyze"] = time.perf_counter() - t

        t = time.perf_counter()
        result = self.retriever.retrieve(analysis, conversation_id)
        timings["retrieve"] = time.perf_counter() - t
        timings |= {f"retrieve.{k}": v for k, v in result.timings.items()}

        t = time.perf_counter()
        packed = self.packer.pack(result)
        timings["pack"] = time.perf_counter() - t

        query_id = f"q:{uuid.uuid4().hex[:12]}"
        self.log.open(query_id, conversation_id, question, packed.memory_ids, packed.turn_uids)

        t = time.perf_counter()
        gate = self.gate.check(analysis, packed)
        timings["gate"] = time.perf_counter() - t

        record: dict[str, Any] = {"rounds": []}
        if gate.abstain:
            final = GenerationOut(
                final_answer="I don't know.",
                claims=[],
                abstained=True,
                abstention_reason=gate.reason,
            )
            stage = "gate"
        else:
            gen = _TimedGenerator(self.llm)
            qtype = question_type(analysis)
            first = gen.generate(
                question,
                packed.text,
                packed.turn_uids,
                question_date=qdate,
                asker=analysis.asker_name,
                question_type=qtype,
            )
            t = time.perf_counter()
            final, record = verify_and_repair(
                query_id=query_id, question=question, context=packed.text, allowed_turn_uids=packed.turn_uids,
                generator=gen, verifier=self.verifier, first=first, question_date=qdate,
                regenerate_once=self.agent_config.regenerate_once,
                asker=analysis.asker_name,
                question_type=qtype,
            )  # fmt: skip
            timings["generate"] = gen.durations[0]
            timings["regenerate"] = sum(gen.durations[1:])
            timings["verify"] = time.perf_counter() - t - timings["regenerate"]
            stage = None
            if final.abstained:
                stage = (
                    "model"
                    if first.abstained
                    or (record["rounds"] and record["rounds"][-1].get("abstained_by_model"))
                    else "verifier"
                )

        timings["total"] = time.perf_counter() - t_all
        calls = (getattr(self.llm, "calls", 0) - calls_before) if calls_before is not None else 0
        ans = AgentAnswer(query_id, question, final.final_answer, final.claims, final.abstained,
                          final.abstention_reason, stage, analysis, packed, gate, timings, calls)  # fmt: skip
        self.log.finish(query_id, ans.summary(), record)
        return ans
