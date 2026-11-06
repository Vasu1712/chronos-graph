"""Generation + verification, shared by ChronosGraph and every baseline.

Everything after "which context do we show the generator" is identical across systems: the same
generator prompt and schema, the same verifier (cited turns exist, were in the context, NLI
entailment >= threshold), the same single repair round, the same abstention rules.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from chronos.agent.generator import GenerationOut, Generator
from chronos.agent.verifier import Verifier, verify_and_repair


class TimedGenerator(Generator):
    """Generator that records the wall time of every call (first = generate, rest = regenerate)."""

    def __init__(self, llm: Any) -> None:
        super().__init__(llm)
        self.durations: list[float] = []

    def generate(self, *args: Any, **kwargs: Any) -> GenerationOut:
        t = time.perf_counter()
        try:
            return super().generate(*args, **kwargs)
        finally:
            self.durations.append(time.perf_counter() - t)


@dataclass
class Answered:
    final: GenerationOut
    record: dict[str, Any]  # verifier record: rounds, per-claim NLI scores, feedback
    stage: str | None  # None, or where the abstention happened: "model" | "verifier"
    first: GenerationOut
    timings: dict[str, float] = field(default_factory=dict)  # generate / verify / regenerate


def generate_and_verify(
    *,
    llm: Any,
    verifier: Verifier,
    query_id: str,
    question: str,
    context: str,
    allowed_turn_uids: list[str],
    question_date: str | None = None,
    asker: str | None = None,
    question_type: str | None = None,
    regenerate_once: bool = True,
) -> Answered:
    gen = TimedGenerator(llm)
    first = gen.generate(
        question, context, allowed_turn_uids, question_date=question_date, asker=asker,
        question_type=question_type,
    )  # fmt: skip
    t = time.perf_counter()
    final, record = verify_and_repair(
        query_id=query_id, question=question, context=context, allowed_turn_uids=allowed_turn_uids,
        generator=gen, verifier=verifier, first=first, question_date=question_date,
        regenerate_once=regenerate_once, asker=asker, question_type=question_type,
    )  # fmt: skip
    regenerate = sum(gen.durations[1:])
    timings = {"generate": gen.durations[0], "regenerate": regenerate,
               "verify": time.perf_counter() - t - regenerate}  # fmt: skip
    stage = None
    if final.abstained:
        by_model = first.abstained or bool(
            record["rounds"] and record["rounds"][-1].get("abstained_by_model")
        )
        stage = "model" if by_model else "verifier"
    return Answered(final, record, stage, first, timings)
