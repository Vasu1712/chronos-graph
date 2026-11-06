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
from chronos.ingestion.temporal import TimeExpr, session_date, span
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


class CorrectionOut(_Strict):
    wrong: str  # the mistaken value, exactly as the speaker first said it ("Priya")
    right: str  # the value that replaces it ("Priyanka")


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
    correction: CorrectionOut | None
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
    correction_wrong: str | None = None
    correction_right: str | None = None
    wrong_start: int | None = None  # span of the wrong value in turn.text, if it is there
    wrong_end: int | None = None
    recent_onset: bool = (
        False  # quote only describes how a state recently began (see is_recent_onset)
    )


@dataclass
class ExtractionStats:
    chunks: int = 0
    chunks_failed: int = 0
    raw_facts: int = 0
    kept: int = 0
    dropped: Counter = field(default_factory=Counter)  # reason -> count
    fuzzy_quotes: int = 0
    turn_reassigned: int = 0  # quote found in a different target turn than turn_index said
    tense_onset_fixed: int = 0  # PAST_ENDED "moved to X last week" read as a current state
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
            "turn_reassigned": self.turn_reassigned,
            "tense_onset_fixed": self.tense_onset_fixed,
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


_ONSET = re.compile(
    r"\b(moved (?:to|into|in)|relocated|settled (?:in|into)|joined|started (?:at|working|my|a|as)|"
    r"became|switched (?:to|from)|got (?:engaged|married|promoted|a job)|enrolled|signed (?:a|the|up))\b",
    re.IGNORECASE,
)
_ENDED = re.compile(
    r"\b(used to|back in|back then|previously|no longer|anymore|until|before that|former|old (?:job|flat|place|house)|"
    r"for (?:a|one|two|three|\d+) (?:year|month|week|while)s?|left|quit|moved (?:away|out))\b",
    re.IGNORECASE,
)
RECENT_ONSET_DAYS = 183


def is_recent_onset(time: TimeExpr, quote: str, session_ts: str) -> bool:
    """A PAST_ENDED state whose quote only describes how it *began*, recently, is still current.

    The 7B extractor reads "We moved to Bengaluru last week" as an ended past state despite the
    prompt; that would store the new residence as history. Deterministic and narrow: needs an
    onset verb, no ending cue, and an onset with no date or ending within RECENT_ONSET_DAYS of the
    session (so "in 2015 I moved to Delhi" stays historical).
    """
    if time.tense != "PAST_ENDED" or not _ONSET.search(quote) or _ENDED.search(quote):
        return False
    if time.kind == "NONE":
        return True
    sp = span(time, session_ts)
    if sp is None:
        return False
    return (session_date(session_ts) - sp[1]).days <= RECENT_ONSET_DAYS


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
- `turn_index`: the number N of the TARGET line "[tN] ..." that contains the quote.
- `quote`: copy the exact words from that turn that support the fact (a contiguous span).
- `attribute_key`: one key from the list below if the fact is the subject's current value of
  that attribute, else null. Fill template slots with a short snake_case noun
  (favorite:book). Temporary stays, trips and visits are NOT residence.
- `polarity`: NEG when the fact says something is not or no longer the case, else POS.
- `mode`: EXPLICIT when stated outright, IMPLIED when it clearly follows from what was said.
- `is_event`: true for a one-off happening (went hiking, had surgery, attended a concert),
  false for a state that holds over time (lives in, works at, likes, owns, is married).
- `is_correction`: true when the speaker corrects something they just said
  ("sorry, I meant", "actually, no"). Then also fill `correction` with the mistaken value as
  first said (`wrong`) and the value replacing it (`right`), and still list the original
  mistaken statement as its own fact if the speaker stated it. Otherwise `correction` is null.
- `entities_mentioned`: people, organizations and places in the fact other than the subject's
  speaker self. For a PERSON give `relation` to the speaker (brother, coworker, friend...) when
  stated, else null.
- `time`: normalize the time expression; never compute dates yourself.
  kind ABSOLUTE: value "YYYY", "YYYY-MM" or "YYYY-MM-DD" (use the session year if omitted).
  kind RELATIVE: signed offset and unit from the session date: yesterday -1 DAY,
    last week -1 WEEK, two months ago -2 MONTH, last year -1 YEAR, next month 1 MONTH.
    Vague ("a while ago", "years ago"): kind RELATIVE, offset null, unit null, value = the words.
  kind NONE: no time words.
  tense: CURRENT if it is true now. A state that began in the past and still holds is CURRENT,
    and its time is when it began ("moved to Porto last month", "joined a new team in May",
    "started a pottery class two weeks ago" are all CURRENT). PAST_ENDED only if the state is
    over: the speaker says or clearly implies it ended ("used to", "back then", "previously",
    "my old flat", "until 2020"). FUTURE_PLAN if planned or intended.

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
    correction=None,
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
        "correction": correction,
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
               key="employer", ents=[_org("Hillcrest High")], corr=True,
               correction={"wrong": "Riverside High", "right": "Hillcrest High"}),
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


def _fmt_turn(t: Turn, max_chars: int | None = None) -> str:
    text = t.text
    if max_chars is not None and len(text) > max_chars:
        text = (
            text[:max_chars] + " [...]"
        )  # display only: quotes are still validated against t.text
    return f"[t{t.turn_index}] {t.speaker}: {text}"


def user_prompt(
    speakers: Sequence[str],
    session_ts: str,
    known_entities: Sequence[str],
    context: Sequence[Turn],
    targets: Sequence[Turn],
    max_chars: int | None = None,
) -> str:
    d: date = session_date(session_ts)
    lines = [
        f"Speakers: {', '.join(speakers)}",
        f"Session date: {d.isoformat()} ({d.strftime('%A')})",
        f"Known entities: {'; '.join(known_entities) if known_entities else '(none)'}",
        "CONTEXT turns:" + ("" if context else " (none)"),
        *[_fmt_turn(t, max_chars) for t in context],
        "TARGET turns:",
        *[_fmt_turn(t, max_chars) for t in targets],
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
        """(context, targets) pairs. Targets are the turns of `target_speakers` (all turns by default),
        up to `chunk_turns` per call; context is the `context_turns` turns just before the first
        target, whoever spoke them."""
        n, c = self.config.chunk_turns, self.config.context_turns
        wanted = self.config.target_speakers
        idx = [i for i, t in enumerate(turns) if wanted is None or t.speaker in wanted]
        out = []
        for k in range(0, len(idx), n):
            group = idx[k : k + n]
            out.append((list(turns[max(0, group[0] - c) : group[0]]), [turns[i] for i in group]))
        return out

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
                        speakers,
                        session.timestamp or "",
                        known_entities,
                        context,
                        targets,
                        self.config.max_prompt_turn_chars,
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
                vf = self._validate(raw, by_index, stats, session.timestamp)
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
        self,
        f: ExtractedFact,
        by_index: dict[int, Turn],
        stats: ExtractionStats,
        session_ts: str | None = None,
    ) -> ValidFact | None:
        subject, fact_text = f.subject.strip(), f.fact_text.strip()
        if not subject or not fact_text:
            stats.dropped["empty"] += 1
            return None
        turn = by_index.get(f.turn_index)
        span = match_quote(f.quote, turn.text, self.config.quote_fuzzy_threshold) if turn else None
        if span is None:
            # The model sometimes names the wrong turn. Accept the fact only if its quote is found
            # in exactly one other target turn of this chunk; otherwise drop it.
            hits = [
                (t, sp)
                for t in by_index.values()
                if t is not turn
                and (sp := match_quote(f.quote, t.text, self.config.quote_fuzzy_threshold))
            ]
            if len(hits) != 1:
                reason = (
                    "ambiguous_turn"
                    if hits
                    else ("bad_turn_index" if turn is None else "quote_not_found")
                )
                stats.dropped[reason] += 1
                return None
            turn, span = hits[0]
            stats.turn_reassigned += 1
        start, end, fuzzy = span
        stats.fuzzy_quotes += int(fuzzy)
        key = self.ontology.canonicalize(f.attribute_key)
        if f.attribute_key and key is None:
            stats.attribute_key_nulled += 1
        time = f.time.to_expr()
        as_past = time.model_copy(update={"tense": "PAST_ENDED"})
        onset = bool(
            key and session_ts and is_recent_onset(as_past, turn.text[start:end], session_ts)
        )
        if onset and time.tense == "PAST_ENDED":
            time = time.model_copy(update={"tense": "CURRENT"})
            stats.tense_onset_fixed += 1
        if time.tense == "FUTURE_PLAN":
            key = None
        if _PRONOUN.search(fact_text):
            stats.pronoun_in_fact_text += 1
        wrong = right = None
        w_start = w_end = None
        if (
            f.is_correction
            and f.correction
            and f.correction.wrong.strip()
            and f.correction.right.strip()
        ):
            wrong, right = f.correction.wrong.strip(), f.correction.right.strip()
            if wrong.casefold() == right.casefold():
                wrong = right = None
            elif (m := re.search(rf"\b{re.escape(wrong)}\b", turn.text, re.IGNORECASE)) is not None:
                w_start, w_end = m.start(), m.end()
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
            correction_wrong=wrong,
            correction_right=right,
            wrong_start=w_start,
            wrong_end=w_end,
            recent_onset=onset,
        )


def _with_ordinal(vf: ValidFact, k: int) -> ValidFact:
    from dataclasses import replace

    return replace(vf, ordinal_in_turn=k)
