"""Conversation records shared by ingestion and the dataset loaders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    turn_uid: str  # "{dataset}:{conversation}:s{session:02d}:t{turn:03d}"
    session_ordinal: int
    turn_index: int
    speaker: str
    text: str  # verbatim utterance
    source_ref: str  # dataset's own id (LoCoMo dia_id; LongMemEval "<session_id>:<turn>")
    has_answer: bool = False  # LongMemEval evidence flag
    image_caption: str | None = None  # LoCoMo shared-photo BLIP caption

    @property
    def display_text(self) -> str:
        """Utterance plus a textual stand-in for a shared image, if any."""
        if self.image_caption:
            return f"{self.text} [shares a photo: {self.image_caption}]"
        return self.text


@dataclass(frozen=True)
class Session:
    session_id: str  # unique within the corpus: "{conversation}:s{ordinal:02d}"
    ordinal: int  # 1-based transaction-time order
    timestamp: str | None  # ISO-8601
    turns: tuple[Turn, ...]
    source_ref: str  # dataset's own session id


@dataclass(frozen=True)
class Conversation:
    dataset: str
    conversation_id: str
    speakers: tuple[str, ...]
    sessions: tuple[Session, ...]

    def turns(self) -> list[Turn]:
        return [t for s in self.sessions for t in s.turns]
