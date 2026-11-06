"""Dataset-neutral records produced by the loaders and consumed by ingestion / evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chronos.ingestion.types import Conversation, Session, Turn

__all__ = ["Conversation", "QAItem", "Session", "Turn"]


@dataclass(frozen=True)
class QAItem:
    dataset: str
    question_id: str
    conversation_id: str
    question: str
    answer: str  # reference answer used for judging
    category: str
    evidence_turn_uids: tuple[str, ...] = ()
    evidence_session_ordinals: tuple[int, ...] = ()
    is_abstention: bool = False
    question_date: str | None = None  # ISO-8601 (LongMemEval)
    meta: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)
