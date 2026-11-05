"""LongMemEval-S (longmemeval_s.json) loader and haystack reduction.

Each question carries its own history ("haystack"): parallel lists `haystack_session_ids`,
`haystack_dates`, `haystack_sessions` (turns: role, content, optional has_answer).

Ordering: session_ordinal is the 1-based haystack position. Haystacks are already sorted at day
granularity, but within a day the clock times are not monotone (211/500 haystacks), so we do
NOT re-sort by timestamp. The same session id can appear twice in one haystack (15 cases,
identical content, different dates); positions keep them distinct.

Abstention questions are those whose question_id contains "_abs".
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from chronos.storage.db import make_turn_uid
from eval.loaders.types import Conversation, QAItem, Session, Turn

log = logging.getLogger(__name__)

DATASET = "longmemeval"
TOKENIZER_REPO = "Qwen/Qwen2.5-7B-Instruct"  # count tokens the way the system LLM sees them


@dataclass(frozen=True)
class LMEQuestion:
    qa: QAItem
    question_type: str
    history: Conversation
    answer_session_ids: tuple[str, ...]  # dataset session ids

    @property
    def question_id(self) -> str:
        return self.qa.question_id

    def answer_ordinals(self) -> set[int]:
        ans = set(self.answer_session_ids)
        return {s.ordinal for s in self.history.sessions if s.source_ref in ans}


def parse_lme_datetime(s: str) -> str:
    """'2023/05/30 (Tue) 23:40' -> '2023-05-30T23:40:00'."""
    return datetime.strptime(s.strip(), "%Y/%m/%d (%a) %H:%M").isoformat()


def is_abstention(question_id: str) -> bool:
    return "_abs" in question_id


def _build(raw: dict) -> LMEQuestion:
    qid = raw["question_id"]
    sessions: list[Session] = []
    has_answer_uids: list[str] = []
    for pos, (sid, date, turns_raw) in enumerate(
        zip(
            raw["haystack_session_ids"],
            raw["haystack_dates"],
            raw["haystack_sessions"],
            strict=True,
        ),
        start=1,
    ):
        turns: list[Turn] = []
        for ti, t in enumerate(turns_raw, start=1):
            uid = make_turn_uid(DATASET, qid, pos, ti)
            flag = bool(t.get("has_answer", False))
            if flag:
                has_answer_uids.append(uid)
            turns.append(
                Turn(
                    turn_uid=uid,
                    session_ordinal=pos,
                    turn_index=ti,
                    speaker=t["role"],
                    text=t["content"],
                    source_ref=f"{sid}:{ti - 1}",  # dataset session id + 0-based turn index
                    has_answer=flag,
                )
            )
        sessions.append(
            Session(
                session_id=f"{qid}:s{pos:02d}",
                ordinal=pos,
                timestamp=parse_lme_datetime(date),
                turns=tuple(turns),
                source_ref=sid,
            )
        )
    answer_ids = tuple(raw["answer_session_ids"])
    ans = set(answer_ids)
    qa = QAItem(
        dataset=DATASET,
        question_id=qid,
        conversation_id=qid,
        question=raw["question"],
        answer=str(raw["answer"]),
        category=raw["question_type"],
        evidence_turn_uids=tuple(has_answer_uids),
        evidence_session_ordinals=tuple(s.ordinal for s in sessions if s.source_ref in ans),
        is_abstention=is_abstention(qid),
        question_date=parse_lme_datetime(raw["question_date"]),
    )
    history = Conversation(DATASET, qid, ("user", "assistant"), tuple(sessions))
    return LMEQuestion(
        qa=qa, question_type=raw["question_type"], history=history, answer_session_ids=answer_ids
    )


def load_longmemeval(path: str | Path) -> list[LMEQuestion]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [_build(raw) for raw in data]


# ------------------------------------------------------------------------------ tokens
@lru_cache(maxsize=1)
def _tokenizer():
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    return Tokenizer.from_file(hf_hub_download(TOKENIZER_REPO, "tokenizer.json"))


def render_session(session: Session) -> str:
    return "\n".join(f"{t.speaker}: {t.text}" for t in session.turns)


def qwen_token_counter(texts: Sequence[str]) -> list[int]:
    return [len(e.ids) for e in _tokenizer().encode_batch(list(texts), add_special_tokens=False)]


TokenCounter = Callable[[Sequence[str]], list[int]]


def session_token_counts(q: LMEQuestion, counter: TokenCounter | None = None) -> dict[int, int]:
    counter = counter or qwen_token_counter
    sessions = q.history.sessions
    return dict(
        zip(
            (s.ordinal for s in sessions),
            counter([render_session(s) for s in sessions]),
            strict=True,
        )
    )


def reduce_haystack(
    question: LMEQuestion,
    max_tokens: int = 24000,
    seed: int = 42,
    *,
    counter: TokenCounter | None = None,
) -> LMEQuestion:
    """Keep every answer session, then add distractors in seeded random order while they fit.

    - Answer sessions are always kept, even if they alone exceed the budget (logged).
    - Distractors are shuffled with random.Random(f"{seed}:{question_id}") (stable across runs
      and processes) and added greedily; one that does not fit is skipped, not a stop signal.
    - Kept sessions retain their original ordinals and turn_uids and stay in haystack order.
    """
    counts = session_token_counts(question, counter)
    answer = question.answer_ordinals()
    keep = set(answer)
    used = sum(counts[o] for o in answer)
    if used > max_tokens:
        log.warning(
            "%s: answer sessions alone use %d > %d tokens", question.question_id, used, max_tokens
        )

    distractors = [s.ordinal for s in question.history.sessions if s.ordinal not in answer]
    random.Random(f"{seed}:{question.question_id}").shuffle(distractors)
    for o in distractors:
        if used + counts[o] <= max_tokens:
            keep.add(o)
            used += counts[o]

    sessions = tuple(s for s in question.history.sessions if s.ordinal in keep)
    return replace(question, history=replace(question.history, sessions=sessions))
