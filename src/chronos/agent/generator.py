"""Structured, citation-bound answer generation (SPEC §6 step 8).

Usable by baselines: `generate(question, context, allowed_turn_uids)` takes any packed context
string and the list of turn ids the answer may cite; it knows nothing about ChronosGraph tiers.
Optional hints: `asker` (who "I" is) and `question_type` (e.g. "AS_OF 2019-01-01..2019-12-31"),
which ChronosGraph derives from rule-based query analysis and a baseline may omit.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

ALLOWED_IDS_LISTED_MAX = 150


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    cited_turn_uids: list[str]
    cited_memory_ids: list[str]


class GenerationOut(BaseModel):
    # Field order is the order a constrained decoder writes them: claims (with their cited turns)
    # come before final_answer, so the 7B model looks for evidence before committing to an
    # answer. With final_answer first it wrote "I don't know." before reading the context.
    model_config = ConfigDict(extra="forbid")

    claims: list[Claim]
    final_answer: str
    abstained: bool
    abstention_reason: str | None


GENERATOR_SYSTEM = """You answer questions about people from their past conversations, using only the CONTEXT.

Rules:
- Every factual claim must cite at least one turn id from ALLOWED TURN IDS: the first field of a
  line "[turn_id | date | speaker] text", copied exactly. Memory lines look like
  "(memory <memory_id>) fact"; put the memory ids you relied on in cited_memory_ids, never in
  cited_turn_uids.
- Write each claim as one short standalone sentence in the third person with names
  ("Meera lives in Bengaluru."), stating only what its cited turns say.
- If a QUESTION TYPE line is given, follow it:
  CURRENT         the present: use ACTIVE facts with an open interval ("to now"); mention earlier
                  or SUPERSEDED values only if the question asks about a change.
  AS_OF <period>  that period: use the facts whose interval covers it, even if they are no longer
                  current; a later value does not cancel an earlier one.
  HISTORY, TEMPORAL_ORDER  changes over time: use the version history and the dates.
  OPEN            use whatever in the context answers the question.
  Without a QUESTION TYPE line, treat present-tense questions as CURRENT.
- RETRACTED facts were never true; never state them as facts.
- Resolve relative dates ("last week") against the date of the turn that says them. When a turn
  shows "[= ...]" after a date phrase, that is the resolved date: use it in your answer.
- Answer whenever the CONTEXT contains the information, even if you must count, add up, compare
  dates, pick the most recent value, or combine two turns. A short answer from the evidence is
  better than abstaining. If the context gives only part of the answer, answer with that part.
- Abstain only if no line of the CONTEXT is about what the question asks, or the context is about
  a different person than the one asked about. Then set claims to [], final_answer to
  "I don't know.", abstained to true, and give abstention_reason.
- final_answer is a short direct answer that only combines your claims."""


FORCE_ANSWER = """INSTRUCTION: Your previous attempt declined to answer, but the question is being asked \
about this conversation and the CONTEXT was retrieved for it. Answer now: give the best answer the \
CONTEXT supports, even if it is only partly stated or needs counting, adding, comparing dates or \
combining turns. Write the claims that support it, citing ALLOWED TURN IDS. Set abstained to false."""


class Generator:
    """Builds the prompt and calls the model.

    The model's own "abstained" flag is not trusted on its own. eval-50 showed a 7B model declining
    86% of answerable questions in every system, even with the gold turn in the context (B4: 85% of
    gold turns in context, 77% false abstention), so a decline with no claims gets exactly one
    forced-answer retry. Whether to abstain is decided by the answerability gate, and by the
    verifier when nothing citeable survives, not by the generator's reluctance.
    """

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.forced_retries = 0  # forced-answer retries made (diagnostics)

    @staticmethod
    def messages(
        question: str,
        context: str,
        allowed_turn_uids: list[str],
        *,
        question_date: str | None = None,
        feedback: str | None = None,
        asker: str | None = None,
        question_type: str | None = None,
        force_answer: bool = False,
    ) -> list[dict[str, str]]:
        parts = [f"QUESTION: {question}"]
        if question_type:
            parts.append(f"QUESTION TYPE: {question_type}")
        if asker:
            parts.append(f'ASKED BY: {asker} ("I", "me", "my" in the question mean {asker})')
        if question_date:
            parts.append(f"QUESTION DATE: {question_date[:10]}")
        if len(allowed_turn_uids) > ALLOWED_IDS_LISTED_MAX:
            # a whole-history context (baseline B2) would spend thousands of tokens repeating ids
            allowed = (
                f"every turn id that starts a context line, inside [...] "
                f"({len(allowed_turn_uids)} turns)"
            )
        else:
            allowed = ", ".join(allowed_turn_uids) or "(none)"
        parts += ["", "CONTEXT:", context or "(empty)", "", "ALLOWED TURN IDS: " + allowed]
        if feedback:
            parts += ["", "FEEDBACK ON YOUR PREVIOUS ANSWER:", feedback]
        if force_answer:
            parts += ["", FORCE_ANSWER]
        return [
            {"role": "system", "content": GENERATOR_SYSTEM},
            {"role": "user", "content": "\n".join(parts)},
        ]

    def generate(
        self,
        question: str,
        context: str,
        allowed_turn_uids: list[str],
        *,
        question_date: str | None = None,
        feedback: str | None = None,
        asker: str | None = None,
        question_type: str | None = None,
    ) -> GenerationOut:
        kw = dict(
            question_date=question_date, feedback=feedback, asker=asker, question_type=question_type
        )
        out = self.llm.structured(
            self.messages(question, context, allowed_turn_uids, **kw), GenerationOut
        )
        if out.abstained and out.claims:  # contradictory: it wrote claims, so it did answer
            return out.model_copy(update={"abstained": False, "abstention_reason": None})
        if out.abstained and context.strip():
            self.forced_retries += 1
            retry = self.llm.structured(
                self.messages(question, context, allowed_turn_uids, force_answer=True, **kw),
                GenerationOut,
            )
            if retry.claims:
                return retry.model_copy(update={"abstained": False, "abstention_reason": None})
            out = out.model_copy(
                update={"abstention_reason": out.abstention_reason or retry.abstention_reason}
            )
        return out


def question_type(analysis) -> str:
    """Render a QueryAnalysis intent (and time window) as the generator's QUESTION TYPE hint."""
    t = analysis.time
    if analysis.intent.value == "AS_OF" and t is not None:
        return f"AS_OF {t.lo}..{t.hi}"
    return analysis.intent.value
