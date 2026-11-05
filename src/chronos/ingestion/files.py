"""Conversation JSON files (synthetic probes, demo sessions) -> Conversation records.

Format:
  {"conversation_id": "probe-x", "speakers": ["A", "B"],
   "sessions": [{"ts": "2023-05-17T10:00:00", "turns": [{"speaker": "A", "text": "..."}]}],
   ...optional "scripted" / "expected" blocks used by tests}
Sessions are numbered 1.. in file order; turns 1.. within a session.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chronos.ingestion.types import Conversation, Session, Turn
from chronos.storage.db import make_turn_uid

DATASET = "synthetic"


def conversation_from_dict(data: dict[str, Any], dataset: str = DATASET) -> Conversation:
    conv_id = data["conversation_id"]
    sessions = []
    for s_i, s in enumerate(data["sessions"], start=1):
        turns = tuple(
            Turn(
                turn_uid=make_turn_uid(dataset, conv_id, s_i, t_i),
                session_ordinal=s_i,
                turn_index=t_i,
                speaker=t["speaker"],
                text=t["text"],
                source_ref=f"s{s_i}:t{t_i}",
            )
            for t_i, t in enumerate(s["turns"], start=1)
        )
        sessions.append(Session(f"{conv_id}:s{s_i:02d}", s_i, s["ts"], turns, f"session_{s_i}"))
    return Conversation(dataset, conv_id, tuple(data["speakers"]), tuple(sessions))


def load_conversation_file(path: str | Path) -> tuple[Conversation, dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return conversation_from_dict(data), data
