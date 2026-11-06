"""Answerability gate (SPEC §6 step 7).

(a) Deterministic speaker mismatch. The question is about speaker A (QueryAnalysis). Among the
    packed memories, the "relevant" ones match the question's attribute key or one of its topic
    terms. If there are relevant memories and every one is about a *different speaker* B, flag
    speaker_mismatch ("What is Melanie's job?" answered only by Caroline's job).
    Contrary evidence: a packed turn spoken by A that mentions a topic term, or (full mode) an
    ANSWERABLE verdict whose supporting turns include one spoken by A.
(b) One LLM call (full mode): ANSWERABLE | PARTIAL | NOT_ANSWERABLE, missing_detail,
    supporting_turn_uids (filtered to turns that are actually in the context).

Abstain when:  off      never
               speaker  speaker_mismatch and no contrary evidence
               full     NOT_ANSWERABLE, or speaker_mismatch and no contrary evidence,
                        or an empty context (no LLM call needed)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from chronos.retrieval.packer import PackedContext
from chronos.retrieval.query import QueryAnalysis
from chronos.storage.db import Database

MODES = ("off", "speaker", "full")


class GateOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str
    verdict: Literal["ANSWERABLE", "PARTIAL", "NOT_ANSWERABLE"]
    missing_detail: str | None
    supporting_turn_uids: list[str]


@dataclass
class GateResult:
    mode: str
    abstain: bool = False
    reason: str | None = None
    speaker_mismatch: bool = False
    mismatch_detail: str | None = None
    contrary_evidence: bool = False
    verdict: str | None = None
    missing_detail: str | None = None
    supporting_turn_uids: list[str] = field(default_factory=list)
    llm_called: bool = False

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


GATE_SYSTEM = """You check whether retrieved conversation evidence answers a question.
Judge only from the EVIDENCE, never from general knowledge.
ANSWERABLE      the evidence states the answer for the exact person or thing asked about.
PARTIAL         the evidence is related and gives part of the answer, but a key detail is missing.
NOT_ANSWERABLE  nothing in the evidence answers it, or the evidence is about someone else
                (for example the question asks about a sister and no sister is ever mentioned).
A question whose answer needs counting, adding, comparing dates, picking the latest value or
combining several turns is ANSWERABLE when the facts it needs are in the evidence. A turn that
shows "[= ...]" after a date phrase gives the resolved date. When unsure between PARTIAL and
NOT_ANSWERABLE, choose PARTIAL.
Give a one-sentence reason, the verdict, the missing detail (or null), and the turn ids
(the first field inside [ ... | ... | ... ]) that support the answer."""


def _stem(word: str) -> str:
    """Tiny suffix stripper: running/runs/ran-> run, lives/lived/living -> liv."""
    w = word.lower().strip("'")
    for suf in ("ing", "ed", "es", "s", "ly"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    if len(w) >= 4 and w[-1] == w[-2] and w[-1] not in "aeiou":
        w = w[:-1]
    return w


def _term_hit(terms: list[str], text: str) -> bool:
    stems = {_stem(w) for w in re.findall(r"[a-z][a-z'-]+", text.lower())}
    for term in terms:
        t = _stem(term)
        if len(t) >= 3 and any(s == t or (len(t) >= 4 and s.startswith(t)) for s in stems):
            return True
    return False


class AnswerabilityGate:
    def __init__(self, db: Database, llm: Any | None, mode: str = "full") -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.db = db
        self.llm = llm
        self.mode = mode

    # ------------------------------------------------------------------ deterministic part
    def speaker_mismatch(
        self, analysis: QueryAnalysis, packed: PackedContext
    ) -> tuple[bool, str | None, bool]:
        """Returns (mismatch, detail, contrary_evidence_from_turns)."""
        a_id = analysis.subject_speaker_id
        if a_id is None or not packed.memory_ids:
            return False, None, False
        conn = self.db.conn
        marks = ",".join("?" * len(packed.memory_ids))
        rows = conn.execute(
            f"SELECT m.memory_id, m.fact_text, m.attribute_key, m.subject_entity_id, e.is_speaker, e.canonical_name "
            f"FROM memories m JOIN entities e ON e.entity_id = m.subject_entity_id WHERE m.memory_id IN ({marks})",
            packed.memory_ids,
        ).fetchall()
        terms = analysis.topic_terms
        relevant = [
            r for r in rows
            if (analysis.attribute_key and r["attribute_key"] == analysis.attribute_key)
            or (terms and _term_hit(terms, r["fact_text"]))
        ]  # fmt: skip
        if not relevant:
            return False, None, False
        # every speaker the question names is a legitimate subject ("Calvin and Dave want to meet
        # where?" is not about Calvin alone), so only memories about unnamed speakers are a mismatch
        named = {e.entity_id for e in analysis.entities if e.is_speaker} | {a_id}
        if not all(r["is_speaker"] and r["subject_entity_id"] not in named for r in relevant):
            return False, None, False
        others = sorted({r["canonical_name"] for r in relevant})
        detail = f"question is about {analysis.subject_speaker_name}; relevant memories are about {', '.join(others)}"
        a_name = analysis.subject_speaker_name
        contrary = False
        if packed.turn_uids:
            tmarks = ",".join("?" * len(packed.turn_uids))
            for sp, text in conn.execute(
                f"SELECT speaker, text FROM episodes WHERE turn_uid IN ({tmarks})", packed.turn_uids
            ):
                if sp == a_name and terms and _term_hit(terms, text):
                    contrary = True
                    break
        return True, detail, contrary

    # ------------------------------------------------------------------ main
    def check(self, analysis: QueryAnalysis, packed: PackedContext) -> GateResult:
        res = GateResult(self.mode)
        if self.mode == "off":
            return res
        res.speaker_mismatch, res.mismatch_detail, res.contrary_evidence = self.speaker_mismatch(
            analysis, packed
        )

        if self.mode == "full":
            if not packed.text.strip():
                res.verdict, res.abstain, res.reason = (
                    "NOT_ANSWERABLE",
                    True,
                    "no evidence retrieved",
                )
                return res
            out: GateOut = self.llm.structured(self.messages(analysis, packed), GateOut)
            res.llm_called = True
            allowed = set(packed.turn_uids)
            res.verdict, res.missing_detail = out.verdict, out.missing_detail
            res.supporting_turn_uids = [u for u in out.supporting_turn_uids if u in allowed]
            if res.speaker_mismatch and not res.contrary_evidence and out.verdict == "ANSWERABLE":
                res.contrary_evidence = self._spoken_by(
                    res.supporting_turn_uids, analysis.subject_speaker_name
                )
            if out.verdict == "NOT_ANSWERABLE":
                res.abstain, res.reason = (
                    True,
                    f"not answerable: {out.missing_detail or out.reason}",
                )
                return res

        if res.speaker_mismatch and not res.contrary_evidence:
            res.abstain, res.reason = True, f"speaker mismatch: {res.mismatch_detail}"
        return res

    def _spoken_by(self, turn_uids: list[str], speaker: str | None) -> bool:
        if not turn_uids or not speaker:
            return False
        marks = ",".join("?" * len(turn_uids))
        return bool(self.db.conn.execute(
            f"SELECT 1 FROM episodes WHERE turn_uid IN ({marks}) AND speaker = ? LIMIT 1", (*turn_uids, speaker)
        ).fetchone())  # fmt: skip

    @staticmethod
    def messages(analysis: QueryAnalysis, packed: PackedContext) -> list[dict[str, str]]:
        about = f" (about {analysis.subject_speaker_name})" if analysis.subject_speaker_name else ""
        when = f"\nQuestion date: {analysis.question_date[:10]}" if analysis.question_date else ""
        return [
            {"role": "system", "content": GATE_SYSTEM},
            {
                "role": "user",
                "content": f"QUESTION{about}: {analysis.question}{when}\n\nEVIDENCE:\n{packed.text}",
            },
        ]
