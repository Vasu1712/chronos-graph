"""LoCoMo (locomo10.json) loader.

- Conversations: `sample_id` (e.g. "conv-26"), speakers `speaker_a` / `speaker_b`.
- Sessions: keys `session_N` with timestamps from `session_N_date_time`
  ("1:56 pm on 8 May, 2023"). Some conversations carry `session_N_date_time` entries with no
  `session_N` content; those are skipped and reported.
- Turns: `dia_id` "D3:7" maps to turn_uid "locomo:conv-26:s03:t007" (both 1-based).
- QA categories: 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial.
  For category 5 the reference answer is `adversarial_answer`.
- Evidence ids are normalized: strings holding several ids ("D8:6; D9:17") are split, typos
  like "D:11:26" / "D30:05" are repaired. Ids that cannot be parsed or point at a missing turn
  are logged and counted in the LoadReport, never raised.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from chronos.storage.db import make_turn_uid
from eval.loaders.types import Conversation, QAItem, Session, Turn

log = logging.getLogger(__name__)

DATASET = "locomo"

CATEGORY_MAP = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}

_SESSION_KEY = re.compile(r"session_(\d+)")
_DIA_ID = re.compile(r"D\s*:?\s*(\d+)\s*:\s*(\d+)")


@dataclass
class LoadReport:
    conversations: int = 0
    sessions: int = 0
    turns: int = 0
    questions: int = 0
    category_counts: Counter = field(default_factory=Counter)
    evidence_ids: int = 0  # raw evidence strings seen
    evidence_resolved: int = 0  # normalized ids that resolved to one turn
    evidence_repaired: list[tuple[str, str, str]] = field(default_factory=list)  # (qid, raw, fixed)
    evidence_malformed: list[tuple[str, str, str]] = field(default_factory=list)  # (qid, raw, why)
    dateless_sessions_skipped: list[tuple[str, int]] = field(default_factory=list)
    questions_without_evidence: int = 0


def parse_locomo_datetime(s: str) -> str:
    """'1:56 pm on 8 May, 2023' -> '2023-05-08T13:56:00'."""
    return datetime.strptime(s.strip(), "%I:%M %p on %d %B, %Y").isoformat()


def normalize_evidence(raw: str) -> list[tuple[int, int]]:
    """Extract every (session, turn) pair from one raw evidence string."""
    return [(int(a), int(b)) for a, b in _DIA_ID.findall(raw)]


def load_locomo(path: str | Path) -> tuple[list[Conversation], list[QAItem], LoadReport]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    report = LoadReport()
    conversations: list[Conversation] = []
    questions: list[QAItem] = []

    for sample in data:
        conv_id = sample["sample_id"]
        conv = sample["conversation"]
        speakers = (conv["speaker_a"], conv["speaker_b"])
        session_nums = sorted(int(m.group(1)) for k in conv if (m := _SESSION_KEY.fullmatch(k)))
        dated = sorted(
            int(m.group(1)) for k in conv if (m := re.fullmatch(r"session_(\d+)_date_time", k))
        )
        for n in sorted(set(dated) - set(session_nums)):
            report.dateless_sessions_skipped.append((conv_id, n))

        sessions: list[Session] = []
        by_dia: dict[str, str] = {}  # canonical "D{s}:{t}" -> turn_uid
        for n in session_nums:
            ts_raw = conv.get(f"session_{n}_date_time")
            ts = parse_locomo_datetime(ts_raw) if ts_raw else None
            turns: list[Turn] = []
            for raw_turn in conv[f"session_{n}"]:
                pairs = normalize_evidence(raw_turn["dia_id"])
                if len(pairs) != 1 or pairs[0][0] != n:
                    raise ValueError(f"{conv_id}: unexpected dia_id {raw_turn['dia_id']!r}")
                _, t = pairs[0]
                uid = make_turn_uid(DATASET, conv_id, n, t)
                canonical = f"D{n}:{t}"
                if canonical in by_dia:
                    raise ValueError(f"{conv_id}: duplicate dia_id {canonical}")
                by_dia[canonical] = uid
                turns.append(
                    Turn(
                        turn_uid=uid,
                        session_ordinal=n,
                        turn_index=t,
                        speaker=raw_turn["speaker"],
                        text=raw_turn["text"],
                        source_ref=raw_turn["dia_id"],
                        image_caption=raw_turn.get("blip_caption"),
                    )
                )
            sessions.append(
                Session(
                    session_id=f"{conv_id}:s{n:02d}",
                    ordinal=n,
                    timestamp=ts,
                    turns=tuple(turns),
                    source_ref=f"session_{n}",
                )
            )
            report.sessions += 1
            report.turns += len(turns)

        conversations.append(Conversation(DATASET, conv_id, speakers, tuple(sessions)))
        report.conversations += 1

        for qi, qa in enumerate(sample["qa"]):
            qid = f"{DATASET}:{conv_id}:q{qi:04d}"
            cat_num = int(qa["category"])
            category = CATEGORY_MAP[cat_num]
            reference = qa.get("adversarial_answer") if cat_num == 5 else qa.get("answer")
            reference = "" if reference is None else str(reference)

            uids: list[str] = []
            if not qa["evidence"]:
                report.questions_without_evidence += 1
            for raw in qa["evidence"]:
                report.evidence_ids += 1
                pairs = normalize_evidence(raw)
                if not pairs:
                    report.evidence_malformed.append((qid, raw, "unparseable"))
                    continue
                canonicals = [f"D{s}:{t}" for s, t in pairs]
                if ", ".join(canonicals) != raw:
                    report.evidence_repaired.append((qid, raw, ", ".join(canonicals)))
                for canonical in canonicals:
                    uid = by_dia.get(canonical)
                    if uid is None:
                        report.evidence_malformed.append((qid, raw, f"no turn {canonical}"))
                        continue
                    if uid not in uids:
                        uids.append(uid)
                    report.evidence_resolved += 1

            questions.append(
                QAItem(
                    dataset=DATASET,
                    question_id=qid,
                    conversation_id=conv_id,
                    question=qa["question"],
                    answer=reference,
                    category=category,
                    evidence_turn_uids=tuple(uids),
                    evidence_session_ordinals=tuple(sorted({_ordinal(u) for u in uids})),
                    is_abstention=cat_num == 5,
                    meta={
                        "category_id": cat_num,
                        "raw_evidence": list(qa["evidence"]),
                        "answer_field": qa.get("answer"),
                    },
                )
            )
            report.questions += 1
            report.category_counts[category] += 1

    for qid, raw, why in report.evidence_malformed:
        log.warning("locomo malformed evidence %s: %r (%s)", qid, raw, why)
    if report.dateless_sessions_skipped:
        log.info(
            "locomo: skipped %d dated-but-empty sessions", len(report.dateless_sessions_skipped)
        )
    return conversations, questions, report


def _ordinal(turn_uid: str) -> int:
    return int(turn_uid.rsplit(":", 2)[-2][1:])
