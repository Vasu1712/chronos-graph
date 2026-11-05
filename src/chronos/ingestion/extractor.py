"""Fact extractor (SPEC §3 steps 1-2).

A finished session is processed in chunks of up to `chunk_turns` turns, each with the previous
`context_turns` turns shown read-only. Prompt layout, static part first so Ollama can reuse its
prompt cache across calls:

  system: instructions, coreference rules, attribute ontology, 4 few-shot examples (static)
  user:   speakers, session date, known entities, context turns, target turns (dynamic)

Code-side validation of every fact:
  - turn_index must name one of the chunk's target turns, else dropped
  - quote must be an exact substring of that turn after whitespace normalization, else a fuzzy
    match (rapidfuzz partial ratio >= threshold) whose span is stored, else dropped
  - attribute_key must exist in the ontology, else set to null
  - FUTURE_PLAN facts never carry an attribute_key (a plan does not change current state)
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict
from rapidfuzz import fuzz

from chronos.config import IngestionConfig
from chronos.ingestion.temporal import TimeExpr, session_date
from chronos.ingestion.types import Session, Turn
from chronos.ontology import Ontology, load_ontology

log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------ LLM schema
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TimeOut(_Strict):
    kind: Literal["ABSOLUTE", "RELATIVE", "NONE"]
    value: str | None
    offset: int | None
    unit: Literal["DAY", "WEEK", "MONTH", "YEAR"] | None
    tense: Literal["CURRENT", "PAST_ENDED", "FUTURE_PLAN"]

    def to_expr(self) -> TimeExpr:
        return TimeExpr(**self.model_dump())


class EntityMention(_Strict):
    name: str
    entity_type: Literal["PERSON", "ORG", "PLACE", "OTHER"]
    relation: str | None  # PERSON only: relation to the speaker ("brother", "coworker")


class ExtractedFact(_Strict):
    turn_index: int
    subject: str
    fact_text: str
    attribute_key: str | None
    polarity: Literal["POS", "NEG"]
    mode: Literal["EXPLICIT", "IMPLIED"]
    is_event: bool
    entities_mentioned: list[EntityMention]
    time: TimeOut
    is_correction: bool
    quote: str


class ExtractionOutput(_Strict):
    facts: list[ExtractedFact]


class StructuredLLM(Protocol):
    def structured(self, messages: list[dict[str, str]], response_model: type[Any]) -> Any: ...


# ------------------------------------------------------------------------------ results
@dataclass(frozen=True)
class ValidFact:
    turn: Turn
    ordinal_in_turn: int  # position among this turn's kept facts (stable ids)
    subject: str
    fact_text: str
    attribute_key: str | None
    polarity: str
    mode: str
    is_event: bool
    entities: tuple[EntityMention, ...]
    time: TimeExpr
    is_correction: bool
    quote: str  # verbatim span of turn.text
    quote_start: int
    quote_end: int
    quote_fuzzy: bool


@dataclass
class ExtractionStats:
    chunks: int = 0
    chunks_failed: int = 0
    raw_facts: int = 0
    kept: int = 0
    dropped: Counter = field(default_factory=Counter)  # reason -> count
    fuzzy_quotes: int = 0
    attribute_key_nulled: int = 0
    pronoun_in_fact_text: int = 0

    @property
    def dropped_total(self) -> int:
        return sum(self.dropped.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunks": self.chunks,
            "chunks_failed": self.chunks_failed,
            "raw_facts": self.raw_facts,
            "kept": self.kept,
            "dropped": dict(self.dropped),
            "fuzzy_quotes": self.fuzzy_quotes,
            "attribute_key_nulled": self.attribute_key_nulled,
            "pronoun_in_fact_text": self.pronoun_in_fact_text,
        }


# ------------------------------------------------------------------------------ quote matching
def _normalize_ws(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to one space and strip; return text and index map to original."""
    out: list[str] = []
    idx: list[int] = []
    prev_space = True  # strips leading whitespace
    for i, ch in enumerate(text):
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                idx.append(i)
            prev_space = True
        else:
            out.append(ch)
            idx.append(i)
            prev_space = False
    if out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


def match_quote(quote: str, text: str, threshold: float = 90.0) -> tuple[int, int, bool] | None:
    """Locate `quote` in `text`. Returns (start, end, fuzzy) as offsets into `text`, or None."""
    nq, _ = _normalize_ws(quote.strip().strip('"').strip())
    nt, idx = _normalize_ws(text)
    if not nq or not nt:
        return None
    pos = nt.find(nq)
    if pos >= 0:
        return idx[pos], idx[pos + len(nq) - 1] + 1, False
    if len(nq) > len(nt):
        if fuzz.ratio(nq, nt) >= threshold:
            return idx[0], idx[-1] + 1, True
        return None
    al = fuzz.partial_ratio_alignment(nq, nt)
    if al is None or al.score < threshold or al.dest_end <= al.dest_start:
        return None
    return idx[al.dest_start], idx[al.dest_end - 1] + 1, True


_PRONOUN = re.compile(r"\b(i|me|my|mine|myself|you|your|yours|we|us|our)\b", re.IGNORECASE)


# ------------------------------------------------------------------------------ prompt
_INSTRUCTIONS = """You extract long-term memory facts from a two-person chat.

For each TARGET turn, list the self-contained atomic facts it states about people, their lives,
plans, possessions, relationships, preferences and experiences. Skip greetings, questions,
small talk and anything only mentioned in CONTEXT turns. A turn may yield zero facts.

Rules:
- Coreference: "I/me/my" is the speaker of that turn; "you/your" is the other participant.
  `subject` is always a name (a speaker name or another entity's name), never a pronoun.
- `fact_text`: one third-person sentence that makes sense on its own, using names, with no
  pronouns (write "Lena moved to Porto", not "I moved there").
- `quote`: copy the exact words from that turn that support the fact (a contiguous span).
- `attribute_key`: one key from the list below if the fact is the subject's current value of
  that attribute, else null. Fill template slots with a short snake_case noun
  (favorite:book). Temporary stays, trips and visits are NOT residence.
- `polarity`: NEG when the fact says something is not or no longer the case, else POS.
- `mode`: EXPLICIT when stated outright, IMPLIED when it clearly follows from what was said.
- `is_event`: true for a one-off happening (went hiking, had surgery, attended a concert),
  false for a state that holds over time (lives in, works at, likes, owns, is married).
- `is_correction`: true when the speaker corrects something they just said
  ("sorry, I meant", "actually, no").
- `entities_mentioned`: people, organizations and places in the fact other than the subject's
  speaker self. For a PERSON give `relation` to the speaker (brother, coworker, friend...) when
  stated, else null.
- `time`: normalize the time expression; never compute dates yourself.
  kind ABSOLUTE: value "YYYY", "YYYY-MM" or "YYYY-MM-DD" (use the session year if omitted).
  kind RELATIVE: signed offset and unit from the session date: yesterday -1 DAY,
    last week -1 WEEK, two months ago -2 MONTH, last year -1 YEAR, next month 1 MONTH.
    Vague ("a while ago", "years ago"): kind RELATIVE, offset null, unit null, value = the words.
  kind NONE: no time words.
  tense: CURRENT if true now, PAST_ENDED if it was true but has ended ("used to", "back then",
    "previously"), FUTURE_PLAN if planned or intended.

Attribute keys:
{ontology}

Return JSON: {{"facts": [...]}}."""


def _example(
    session_date_s: str, weekday: str, turns: list[tuple[int, str, str]], facts: list[dict]
) -> str:
    lines = [
        "Speakers: Lena, Omar",
        f"Session date: {session_date_s} ({weekday})",
        "Known entities: Lena (PERSON, speaker), Omar (PERSON, speaker)",
        "CONTEXT turns: (none)",
        "TARGET turns:",
        *[f"[t{i}] {s}: {t}" for i, s, t in turns],
    ]
    out = ExtractionOutput.model_validate({"facts": facts})  # examples must satisfy the schema
    return "\n".join(lines) + "\nOutput:\n" + json.dumps(out.model_dump(), ensure_ascii=False)


def _t(kind="NONE", value=None, offset=None, unit=None, tense="CURRENT") -> dict:
    return {"kind": kind, "value": value, "offset": offset, "unit": unit, "tense": tense}


def _f(
    turn,
    subject,
    text,
    quote,
    key=None,
    time=None,
    ents=(),
    event=False,
    corr=False,
    pol="POS",
    mode="EXPLICIT",
):
    return {
        "turn_index": turn,
        "subject": subject,
        "fact_text": text,
        "attribute_key": key,
        "polarity": pol,
        "mode": mode,
        "is_event": event,
        "entities_mentioned": list(ents),
        "time": time or _t(),
        "is_correction": corr,
        "quote": quote,
    }


def _place(name):
    return {"name": name, "entity_type": "PLACE", "relation": None}


def _org(name):
    return {"name": name, "entity_type": "ORG", "relation": None}


FEW_SHOT = [
    # 1. a move (plus the implied previous residence)
    _example(
        "2022-09-14",
        "Wednesday",
        [
            (3, "Lena", "We finally moved to Porto last month, the old flat in Lisbon was too small."),
            (4, "Omar", "Congrats! How's the new place?"),
        ],
        [
            _f(3, "Lena", "Lena moved to Porto.", "We finally moved to Porto last month",
               key="residence", time=_t("RELATIVE", None, -1, "MONTH"), ents=[_place("Porto")]),
            _f(3, "Lena", "Lena previously lived in a flat in Lisbon.", "the old flat in Lisbon was too small",
               key="residence", time=_t(tense="PAST_ENDED"), ents=[_place("Lisbon")], mode="IMPLIED"),
        ],
    ),
    # 2. an in-session correction
    _example(
        "2022-10-02",
        "Sunday",
        [
            (2, "Omar", "I'm teaching at Riverside High now."),
            (3, "Lena", "Riverside? That's great."),
            (4, "Omar", "Sorry, I meant Hillcrest High. Riverside was just an interview."),
        ],
        [
            _f(2, "Omar", "Omar works as a teacher.", "I'm teaching", key="job_title"),
            _f(2, "Omar", "Omar teaches at Riverside High.", "I'm teaching at Riverside High now",
               key="employer", ents=[_org("Riverside High")]),
            _f(4, "Omar", "Omar teaches at Hillcrest High.", "Sorry, I meant Hillcrest High",
               key="employer", ents=[_org("Hillcrest High")], corr=True),
        ],
    ),
    # 3. a past fact next to the current one
    _example(
        "2023-01-20",
        "Friday",
        [
            (6, "Lena", "Back in 2015 I was vegetarian for a whole year, but these days I eat everything."),
        ],
        [
            _f(6, "Lena", "Lena was vegetarian in 2015.", "Back in 2015 I was vegetarian for a whole year",
               key="diet", time=_t("ABSOLUTE", "2015", tense="PAST_ENDED")),
            _f(6, "Lena", "Lena eats everything and has no dietary restriction.", "these days I eat everything",
               key="diet"),
        ],
    ),
    # 4. a point event involving another person
    _example(
        "2023-03-08",
        "Wednesday",
        [
            (9, "Omar", "My sister Ana and I ran the city half marathon on Sunday!"),
            (10, "Lena", "You two are amazing."),
        ],
        [
            _f(9, "Omar", "Omar ran the city half marathon with his sister Ana.",
               "My sister Ana and I ran the city half marathon on Sunday", event=True,
               time=_t("RELATIVE", None, -3, "DAY", "PAST_ENDED"),
               ents=[{"name": "Ana", "entity_type": "PERSON", "relation": "sister"}]),
            _f(9, "Ana", "Ana is Omar's sister.", "My sister Ana",
               ents=[{"name": "Ana", "entity_type": "PERSON", "relation": "sister"}]),
        ],
    ),
]  # fmt: skip


def system_prompt(ontology: Ontology) -> str:
    parts = [_INSTRUCTIONS.format(ontology=ontology.render_for_prompt()), "", "Examples:"]
    for i, ex in enumerate(FEW_SHOT, start=1):
        parts += ["", f"### Example {i}", ex]
    return "\n".join(parts)


def _fmt_turn(t: Turn) -> str:
    return f"[t{t.turn_index}] {t.speaker}: {t.text}"


def user_prompt(
    speakers: Sequence[str],
    session_ts: str,
    known_entities: Sequence[str],
    context: Sequence[Turn],
    targets: Sequence[Turn],
) -> str:
    d: date = session_date(session_ts)
    lines = [
        f"Speakers: {', '.join(speakers)}",
        f"Session date: {d.isoformat()} ({d.strftime('%A')})",
        f"Known entities: {'; '.join(known_entities) if known_entities else '(none)'}",
        "CONTEXT turns:" + ("" if context else " (none)"),
        *[_fmt_turn(t) for t in context],
        "TARGET turns:",
        *[_fmt_turn(t) for t in targets],
        "Output:",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------------------ extractor
class Extractor:
    def __init__(
        self,
        llm: StructuredLLM,
        ontology: Ontology | None = None,
        config: IngestionConfig | None = None,
    ) -> None:
        self.llm = llm
        self.ontology = ontology or load_ontology()
        self.config = config or IngestionConfig()
        self._system = system_prompt(self.ontology)

    def chunks(self, turns: Sequence[Turn]) -> list[tuple[list[Turn], list[Turn]]]:
        n, c = self.config.chunk_turns, self.config.context_turns
        return [
            (list(turns[max(0, i - c) : i]), list(turns[i : i + n]))
            for i in range(0, len(turns), n)
        ]

    def extract_session(
        self,
        session: Session,
        speakers: Sequence[str],
        known_entities: Sequence[str] = (),
    ) -> tuple[list[ValidFact], ExtractionStats]:
        stats = ExtractionStats()
        facts: list[ValidFact] = []
        per_turn: Counter = Counter()
        seen: set[tuple[str, str]] = set()
        for context, targets in self.chunks(session.turns):
            stats.chunks += 1
            messages = [
                {"role": "system", "content": self._system},
                {
                    "role": "user",
                    "content": user_prompt(
                        speakers, session.timestamp or "", known_entities, context, targets
                    ),
                },
            ]
            try:
                out: ExtractionOutput = self.llm.structured(messages, ExtractionOutput)
            except Exception as exc:  # one bad chunk must not lose the session
                stats.chunks_failed += 1
                log.warning("%s: extraction chunk failed: %s", session.session_id, exc)
                continue
            by_index = {t.turn_index: t for t in targets}
            for raw in out.facts:
                stats.raw_facts += 1
                vf = self._validate(raw, by_index, stats)
                if vf is None:
                    continue
                key = (vf.turn.turn_uid, vf.fact_text.casefold())
                if key in seen:
                    stats.dropped["duplicate"] += 1
                    continue
                seen.add(key)
                vf = _with_ordinal(vf, per_turn[vf.turn.turn_uid])
                per_turn[vf.turn.turn_uid] += 1
                facts.append(vf)
                stats.kept += 1
        log.info(
            "%s: extracted %d raw facts, kept %d, dropped %s", session.session_id,
            stats.raw_facts, stats.kept, dict(stats.dropped) or "{}",
        )  # fmt: skip
        return facts, stats

    def _validate(
        self, f: ExtractedFact, by_index: dict[int, Turn], stats: ExtractionStats
    ) -> ValidFact | None:
        turn = by_index.get(f.turn_index)
        if turn is None:
            stats.dropped["bad_turn_index"] += 1
            return None
        subject, fact_text = f.subject.strip(), f.fact_text.strip()
        if not subject or not fact_text:
            stats.dropped["empty"] += 1
            return None
        span = match_quote(f.quote, turn.text, self.config.quote_fuzzy_threshold)
        if span is None:
            stats.dropped["quote_not_found"] += 1
            return None
        start, end, fuzzy = span
        stats.fuzzy_quotes += int(fuzzy)
        key = self.ontology.canonicalize(f.attribute_key)
        if f.attribute_key and key is None:
            stats.attribute_key_nulled += 1
        time = f.time.to_expr()
        if time.tense == "FUTURE_PLAN":
            key = None
        if _PRONOUN.search(fact_text):
            stats.pronoun_in_fact_text += 1
        return ValidFact(
            turn=turn,
            ordinal_in_turn=0,
            subject=subject,
            fact_text=fact_text,
            attribute_key=key,
            polarity=f.polarity,
            mode=f.mode,
            is_event=f.is_event,
            entities=tuple(f.entities_mentioned),
            time=time,
            is_correction=f.is_correction,
            quote=turn.text[start:end],
            quote_start=start,
            quote_end=end,
            quote_fuzzy=fuzzy,
        )


def _with_ordinal(vf: ValidFact, k: int) -> ValidFact:
    from dataclasses import replace

    return replace(vf, ordinal_in_turn=k)
