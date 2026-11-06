"""Query analysis by rules and entity string matching, no LLM (SPEC §6 step 1).

Intent, in order of precedence:
  TEMPORAL_ORDER  "when", "how long", "which came first", "how many days", "what year", ...
  HISTORY         "before", "used to", "previously", "first", "originally", "ever", ...
  AS_OF           an explicit time ("in May 2023", "in 2019", "last year") or "back when",
                  "at that time", or an explicit as_of argument
  CURRENT         "now", "currently", "these days", or a present-tense question (is/are/do/does...)
  OPEN            everything else (past tense, modal, why/would questions)

Entities are linked by exact or alias match (normalized, whole words) against the
conversation's entities. The subject speaker is the first speaker named in the question, else
the asker when the question is in the first person ("my", "I"), else the other participant for
"you/your". Time expressions are resolved with chronos.ingestion.temporal against the question
date.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

from chronos.ingestion.entities import normalize_name
from chronos.ingestion.temporal import Precision, TimeExpr, session_date, span


class Intent(StrEnum):
    CURRENT = "CURRENT"
    AS_OF = "AS_OF"
    HISTORY = "HISTORY"
    TEMPORAL_ORDER = "TEMPORAL_ORDER"
    OPEN = "OPEN"


@dataclass(frozen=True)
class TimeConstraint:
    lo: str  # YYYY-MM-DD
    hi: str
    precision: Precision
    text: str


@dataclass(frozen=True)
class EntityLink:
    entity_id: str
    name: str
    is_speaker: bool
    matched: str  # the question text that matched


@dataclass
class QueryAnalysis:
    question: str
    intent: Intent
    cues: list[str] = field(default_factory=list)
    entities: list[EntityLink] = field(default_factory=list)
    subject_speaker_id: str | None = None
    subject_speaker_name: str | None = None
    asker_id: str | None = None
    asker_name: str | None = None
    time: TimeConstraint | None = None
    attribute_key: str | None = None
    topic_terms: list[str] = field(default_factory=list)
    question_date: str | None = None

    @property
    def entity_ids(self) -> list[str]:
        return [e.entity_id for e in self.entities]


# ------------------------------------------------------------------------------ intent cues
_TEMPORAL_ORDER = [
    r"^when\b", r"\bwhen (did|was|were|is|does|do|will|has|had)\b", r"\bhow long\b",
    r"\bwhich (came|happened|was|did \w+) first\b", r"\bwhat came first\b", r"\bhow many (days|weeks|months|years)\b",
    r"\bwhat (date|day|month|year|time)\b", r"\bin what order\b", r"\bbefore or after\b", r"\bhow soon\b",
    r"\bhow recently\b", r"\bsince when\b",
]  # fmt: skip
_HISTORY = [
    r"\bbefore\b", r"\bused to\b", r"\bpreviously\b", r"\bprevious\b", r"\bfirst\b", r"\boriginally\b",
    r"\bformerly\b", r"\bin the past\b", r"\bever\b", r"\bearlier\b", r"\ball the (places|jobs|companies)\b",
    r"\bhistory\b", r"\bprior\b", r"\bold (job|place|address|company)\b",
]  # fmt: skip
_AS_OF_WORDS = [
    r"\bback when\b",
    r"\bat that time\b",
    r"\bat the time\b",
    r"\bas of\b",
    r"\bback then\b",
    r"\bduring\b",
]
_CURRENT_WORDS = [r"\bnow\b", r"\bcurrently\b", r"\bcurrent\b", r"\bthese days\b", r"\bat the moment\b",
                  r"\bstill\b", r"\btoday\b", r"\bnowadays\b", r"\bright now\b"]  # fmt: skip
_PRESENT_AUX = re.compile(
    r"^(?:(?:what|where|who|whom|which|how|why)\s+(?:\w+\s+){0,2}?)?(is|are|am|do|does|has|have)\b",
    re.IGNORECASE,
)
_PAST_OR_MODAL = re.compile(r"\b(did|was|were|had|would|could|might|should|will)\b", re.IGNORECASE)

FIRST_PERSON = re.compile(r"\b(i|me|my|mine|myself|i'm|i've|i'd)\b", re.IGNORECASE)
SECOND_PERSON = re.compile(r"\b(you|your|yours|yourself)\b", re.IGNORECASE)

# ------------------------------------------------------------------------------ attribute guess
_ATTRIBUTE_CUES: list[tuple[str, str]] = [
    (
        r"\b(live|lives|living|lived|based|reside|resides|moved|move|city|address|home)\b",
        "residence",
    ),
    (r"\b(grew up|hometown|originally from|born)\b", "hometown"),
    (
        r"\b(work for|works for|work at|works at|worked at|employer|company|job at|joined)\b",
        "employer",
    ),
    (r"\b(what does (?:\w+\s){1,3}?do|what do i do)\b", "job_title"),
    (r"\b(work|works|working)\b", "employer"),
    (r"\b(job title|role|occupation|profession|do for a living)\b", "job_title"),
    (r"\b(school|university|college|study|studies|studying|enrolled)\b", "school"),
    (r"\b(single|married|engaged|divorced|relationship status|dating)\b", "relationship_status"),
    (r"\b(partner|girlfriend|boyfriend|wife|husband|spouse|fiance|fiancee)\b", "partner"),
    (r"\b(diet|vegan|vegetarian|keto|pescatarian|gluten)\b", "diet"),
]
_FAVORITE = re.compile(r"\bfavou?rite\s+([a-z]+(?:\s[a-z]+)?)", re.IGNORECASE)

_STOP = set(
    ["a", "an", "the", "of", "to", "in", "on", "at", "for", "from", "by", "with", "about", "and", "or", "but", "if", "is", "are", "am", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had", "what", "which", "who", "whom", "whose", "where", "when", "why", "how", "that", "this", "these", "those", "it", "its", "i", "me", "my", "mine", "you", "your", "we", "our", "they", "their", "he", "his", "she", "her", "them", "there", "here", "not", "no", "yes", "any", "some", "much", "many", "more", "most", "so", "than", "then", "also", "can", "could", "would", "should", "will", "might", "may", "must", "just", "very", "really", "ever", "still", "now", "currently", "first", "before", "after", "used", "previously", "during", "back", "time", "times", "long", "ago", "since", "year", "years", "month", "months", "week", "weeks", "day", "days"]
)  # fmt: skip

# ------------------------------------------------------------------------------ time parsing
_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"], start=1)}  # fmt: skip
_MONTHS |= {k[:3]: v for k, v in list(_MONTHS.items())} | {"sept": 9}
_MON = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
_NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
              "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}  # fmt: skip


def _expr_from_question(q: str, qdate: date) -> tuple[TimeExpr, str] | None:
    """Find the first time expression in the question and normalize it like the extractor does."""
    text = q
    if m := re.search(r"\b((?:19|20)\d{2})-(\d{2})-(\d{2})\b", text):
        return TimeExpr(kind="ABSOLUTE", value=m.group(0)), m.group(0)
    # 3 May 2023 / May 3, 2023 / May 3rd
    pats = [
        rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<mon>{_MON})\b,?\s*(?P<year>(?:19|20)\d{{2}})?",
        rf"\b(?P<mon>{_MON})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\b,?\s*(?P<year>(?:19|20)\d{{2}})?",
    ]
    for pat in pats:
        if m := re.search(pat, text, re.IGNORECASE):
            if m.group("mon").lower() == "may" and not m.group("mon")[0].isupper():
                continue  # the modal verb "may"
            mon = _MONTHS[m.group("mon").lower()]
            year = int(m.group("year")) if m.group("year") else _recent_year(mon, qdate)
            return TimeExpr(
                kind="ABSOLUTE", value=f"{year:04d}-{mon:02d}-{int(m.group('day')):02d}"
            ), m.group(0)
    if m := re.search(
        rf"\b(?P<mon>{_MON})\s+(?:of\s+)?(?P<year>(?:19|20)\d{{2}})\b", text, re.IGNORECASE
    ):
        mon = _MONTHS[m.group("mon").lower()]
        return TimeExpr(kind="ABSOLUTE", value=f"{int(m.group('year')):04d}-{mon:02d}"), m.group(0)
    m = re.search(rf"\b(?:in|during|since|until|by|of)\s+(?P<mon>{_MON})\b", text, re.IGNORECASE)
    if m and not (m.group("mon").lower() == "may" and not m.group("mon")[0].isupper()):
        mon = _MONTHS[m.group("mon").lower()]
        return TimeExpr(
            kind="ABSOLUTE", value=f"{_recent_year(mon, qdate):04d}-{mon:02d}"
        ), m.group(0)
    if m := re.search(r"\b(?:19|20)\d{2}\b", text):
        return TimeExpr(kind="ABSOLUTE", value=m.group(0)), m.group(0)
    low = text.lower()
    if m := re.search(r"\byesterday\b", low):
        return TimeExpr(kind="RELATIVE", offset=-1, unit="DAY"), m.group(0)
    if m := re.search(r"\b(last|this|next)\s+(week|month|year)\b", low):
        off = {"last": -1, "this": 0, "next": 1}[m.group(1)]
        return TimeExpr(kind="RELATIVE", offset=off, unit=m.group(2).upper()), m.group(0)
    if m := re.search(
        r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(day|week|month|year)s?\s+ago\b",
        low,
    ):
        n = int(m.group(1)) if m.group(1).isdigit() else _NUM_WORDS[m.group(1)]
        return TimeExpr(kind="RELATIVE", offset=-n, unit=m.group(2).upper()), m.group(0)
    return None


def _recent_year(month: int, qdate: date) -> int:
    """A month without a year means its most recent occurrence up to the question date."""
    return qdate.year if month <= qdate.month else qdate.year - 1


def parse_time(question: str, question_date: str | None) -> TimeConstraint | None:
    qdate = session_date(question_date) if question_date else date.today()
    found = _expr_from_question(question, qdate)
    if found is None:
        return None
    expr, text = found
    sp = span(expr, qdate)
    if sp is None:
        return None
    lo, hi, prec = sp
    return TimeConstraint(lo.isoformat(), hi.isoformat(), prec, text)


def as_of_constraint(as_of: str) -> TimeConstraint:
    d = session_date(as_of).isoformat()
    return TimeConstraint(d, d, Precision.DAY, f"as of {d}")


# ------------------------------------------------------------------------------ entities
def _entity_names(
    conn: sqlite3.Connection, conversation_id: str
) -> list[tuple[str, str, bool, str]]:
    rows = conn.execute(
        "SELECT e.entity_id, e.canonical_name, e.is_speaker, a.alias FROM entities e "
        "LEFT JOIN entity_aliases a USING(entity_id) WHERE e.conversation_id = ? ORDER BY e.rowid",
        (conversation_id,),
    ).fetchall()
    out = []
    for eid, name, is_speaker, alias in rows:
        out.append((eid, name, bool(is_speaker), name))
        if alias:
            out.append((eid, name, bool(is_speaker), alias))
    return out


def link_entities(
    conn: sqlite3.Connection, conversation_id: str, question: str
) -> list[tuple[int, EntityLink]]:
    """(position, link) for every entity whose name or alias appears as whole words."""
    norm_q = f" {normalize_name(question)} "
    found: dict[str, tuple[int, EntityLink]] = {}
    for eid, name, is_speaker, surface in _entity_names(conn, conversation_id):
        n = normalize_name(surface)
        if len(n) < 2 or n in _STOP:
            continue
        pos = norm_q.find(f" {n} ")
        if pos < 0:
            continue
        if eid not in found or pos < found[eid][0] or len(n) > len(found[eid][1].matched):
            found[eid] = (pos, EntityLink(eid, name, is_speaker, surface))
    return sorted(found.values(), key=lambda x: x[0])


def speakers(conn: sqlite3.Connection, conversation_id: str) -> list[tuple[str, str]]:
    return [
        (r[0], r[1])
        for r in conn.execute(
            "SELECT entity_id, canonical_name FROM entities WHERE conversation_id = ? AND is_speaker = 1 ORDER BY rowid",
            (conversation_id,),
        )
    ]


# ------------------------------------------------------------------------------ main
def _matches(patterns: list[str], text: str) -> list[str]:
    return [m.group(0) for p in patterns if (m := re.search(p, text, re.IGNORECASE))]


def detect_intent(
    question: str, time: TimeConstraint | None, as_of: str | None = None
) -> tuple[Intent, list[str]]:
    q = question.strip()
    if as_of:
        return Intent.AS_OF, [f"as_of={as_of}"]
    if cues := _matches(_TEMPORAL_ORDER, q):
        return Intent.TEMPORAL_ORDER, cues
    if cues := _matches(_HISTORY, q):
        return Intent.HISTORY, cues
    if time is not None:
        return Intent.AS_OF, [time.text]
    if cues := _matches(_AS_OF_WORDS, q):
        return Intent.AS_OF, cues
    if cues := _matches(_CURRENT_WORDS, q):
        return Intent.CURRENT, cues
    if (m := _PRESENT_AUX.search(q)) and not _PAST_OR_MODAL.search(q):
        return Intent.CURRENT, [f"present tense: {m.group(1)}"]
    return Intent.OPEN, []


def guess_attribute(question: str) -> str | None:
    if m := _FAVORITE.search(question):
        return f"favorite:{re.sub(r'[^a-z0-9]+', '_', m.group(1).lower()).strip('_')}"
    for pat, key in _ATTRIBUTE_CUES:
        if re.search(pat, question, re.IGNORECASE):
            return key
    return None


def topic_terms(question: str, entity_names: list[str]) -> list[str]:
    names = {t for n in entity_names for t in normalize_name(n).split()}
    words = re.findall(r"[a-z][a-z'-]+", question.lower())
    return [w for w in dict.fromkeys(words) if w not in _STOP and w not in names and len(w) > 2]


def analyze(
    question: str,
    *,
    conn: sqlite3.Connection,
    conversation_id: str,
    question_date: str | None = None,
    as_of: str | None = None,
    asker: str | None = None,
) -> QueryAnalysis:
    """Analyze a question. `asker` is a speaker name; defaults to the conversation's first speaker."""
    time = as_of_constraint(as_of) if as_of else parse_time(question, question_date)
    intent, cues = detect_intent(question, time, as_of)
    links = link_entities(conn, conversation_id, question)
    sp = speakers(conn, conversation_id)
    asker_id = next((eid for eid, name in sp if asker and name == asker), sp[0][0] if sp else None)

    subject_id = subject_name = None
    named_speakers = [lk for _, lk in links if lk.is_speaker]
    if named_speakers:
        subject_id, subject_name = named_speakers[0].entity_id, named_speakers[0].name
    elif FIRST_PERSON.search(question) and asker_id:
        subject_id = asker_id
    elif SECOND_PERSON.search(question):
        others = [eid for eid, _ in sp if eid != asker_id]
        subject_id = others[0] if len(others) == 1 else None
    if subject_id and not subject_name:
        subject_name = next((name for eid, name in sp if eid == subject_id), None)

    entities = [lk for _, lk in links]
    return QueryAnalysis(
        question=question,
        intent=intent,
        cues=cues,
        entities=entities,
        subject_speaker_id=subject_id,
        subject_speaker_name=subject_name,
        asker_id=asker_id,
        asker_name=next((name for eid, name in sp if eid == asker_id), None),
        time=time,
        attribute_key=guess_attribute(question),
        topic_terms=topic_terms(
            question, [e.name for e in entities] + [e.matched for e in entities]
        ),
        question_date=question_date,
    )
