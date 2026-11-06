"""State resolver: candidate lookup, relation classification, bi-temporal writes (SPEC §3-5).

The LLM only picks a relation (and which candidate it is about). Everything that changes state
is decided here, deterministically:

  normalization of the LLM's choice
    - no usable target                       -> NEW (unless there is exactly one candidate)
    - is_correction and UPDATES              -> CORRECTS
    - CORRECTS without is_correction         -> UPDATES (same attribute key) or NEW
    - CORRECTS of a value settled by a correction in the same turn -> NEW
    - CORRECTS of a value established by a correction, by a fact that does not name it as the
      wrong value                            -> NEW
    - CONFLICT with a target from another turn -> UPDATES (the later statement wins)
    - DUPLICATE with opposite polarity       -> UPDATES
    - NEG fact against a POS target via UPDATES/DUPLICATE -> NEGATES (close the target)
    - UPDATES/CORRECTS/NEGATES need an open ACTIVE target; otherwise the open same-key
      candidate is used, else NEW
    - UPDATES by a fact whose own interval is closed -> HISTORICAL (an ended state never
      replaces the current one)
    - HISTORICAL for a recently begun state ("moved to X last week") -> UPDATES
    - HISTORICAL with an open interval       -> time re-resolved as PAST_ENDED

  write rules, order key = (session_ordinal, turn_index)
    NEW / EXTENDS / HISTORICAL  insert ACTIVE (HISTORICAL keeps its closed interval)
    DUPLICATE   REAFFIRMS source on the target; corroboration += 1 for a new distinct turn
    UPDATES     temporal.compare(new.valid_from, old.valid_from):
                  LATER -> old SUPERSEDED, valid_to = new.valid_from, retired at new key
                  EARLIER -> new inserted SUPERSEDED behind old
                  INCOMPARABLE -> the later order key wins
    CORRECTS    old RETRACTED (never true), new ACTIVE with a CORRECTS source
    CONFLICT    both CONFLICTED with a shared conflict_set_id
    NEGATES     old SUPERSEDED with valid_to from the NEG fact; NEG fact inserted ACTIVE
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from chronos.config import IngestionConfig
from chronos.ingestion.temporal import (
    Cmp,
    Precision,
    ResolvedTime,
    TimeExpr,
    compare,
    describe,
    resolve,
    session_date,
)
from chronos.llm.embed import from_blob, to_blob

log = logging.getLogger(__name__)

APPEND_ONLY_AS_NEW = {"UPDATES", "NEGATES", "CONFLICT", "HISTORICAL", "EXTENDS"}
RELATIONS = ("NEW", "DUPLICATE", "EXTENDS", "UPDATES", "CORRECTS", "HISTORICAL", "CONFLICT")


# ------------------------------------------------------------------------------ data
@dataclass(frozen=True)
class NewFact:
    memory_id: str
    conversation_id: str
    subject_entity_id: str
    subject_name: str
    attribute_key: str | None
    fact_text: str
    polarity: str
    mode: str
    is_event: bool
    time_expr: TimeExpr
    session_ts: str
    rt: ResolvedTime
    session_ordinal: int
    turn_index: int
    turn_uid: str
    quote: str
    quote_start: int | None
    quote_end: int | None
    is_correction: bool
    mention_entity_ids: tuple[str, ...] = ()
    embedding: np.ndarray | None = field(default=None, compare=False)
    # structured correction from the extractor ("Priya, sorry, I meant Priyanka")
    correction_wrong: str | None = None
    correction_right: str | None = None
    wrong_quote: tuple[str, int, int] | None = None  # verbatim span of the wrong value in the turn
    recent_onset: bool = False  # extractor: the quote only says how this state recently began

    @property
    def key(self) -> tuple[int, int]:
        return (self.session_ordinal, self.turn_index)


@dataclass(frozen=True)
class MemRow:
    memory_id: str
    subject_entity_id: str
    attribute_key: str | None
    fact_text: str
    polarity: str
    status: str
    valid_from: str | None
    valid_from_precision: str | None
    valid_to: str | None
    valid_to_precision: str | None
    event_date: str | None
    event_date_precision: str | None
    recorded_session_ordinal: int
    recorded_turn_index: int
    supersedes_id: str | None
    superseded_by_id: str | None
    conflict_set_id: str | None
    embedding: bytes | None

    @property
    def key(self) -> tuple[int, int]:
        return (self.recorded_session_ordinal, self.recorded_turn_index)

    @property
    def is_open(self) -> bool:
        return self.status == "ACTIVE" and self.valid_to is None

    @property
    def rt(self) -> ResolvedTime:
        p = lambda v: Precision(v) if v else None  # noqa: E731
        return ResolvedTime(
            self.valid_from, p(self.valid_from_precision), self.valid_to, p(self.valid_to_precision),
            self.event_date, p(self.event_date_precision),
        )  # fmt: skip


_MEM_COLS = ", ".join(MemRow.__dataclass_fields__)


@dataclass
class Decision:
    memory_id: str
    fact_text: str
    llm_relation: str | None  # None when no LLM call was made
    relation: str  # what was applied
    target_id: str | None
    candidates: int
    notes: list[str] = field(default_factory=list)
    llm_target_id: str | None = None  # candidate memory the classifier named (before normalization)


class RelationOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str
    relation: Literal[
        "NEW", "DUPLICATE", "EXTENDS", "UPDATES", "CORRECTS", "HISTORICAL", "CONFLICT"
    ]
    target: str | None  # candidate label "C1".."C5", or null for NEW


# ------------------------------------------------------------------------------ prompt
CLASSIFY_SYSTEM = """You maintain a memory of facts about people in a conversation.
Given a NEW fact and existing CANDIDATE memories about the same subject, choose how the new fact
relates to the single most relevant candidate:

NEW         it is about something none of the candidates cover.
DUPLICATE   it restates a candidate with no new information.
EXTENDS     same topic as a candidate, compatible with it, adds detail; both remain true.
UPDATES     the situation changed over time, so it replaces the candidate's value
            (moved somewhere else, changed job, stopped doing something).
CORRECTS    the speaker says the candidate was a mistake ("sorry, I meant ..."): it was never true.
HISTORICAL  it describes an earlier, finished period and does not change the candidate's
            current value ("back in 2019 I lived in ...").
CONFLICT    it contradicts a candidate stated in the same turn, with no cue for which is right.

Temporary situations (trips, visits, a week somewhere) do not update where someone lives.
Give a one-sentence reason, then the relation, then the candidate label (null for NEW)."""


def _fmt_mem(label: str, m: MemRow) -> str:
    when = describe(m.rt)
    key = f", attribute {m.attribute_key}" if m.attribute_key else ""
    return (
        f"{label}: {m.fact_text} [{m.status}{key}, {'NEG' if m.polarity == 'NEG' else 'POS'}, {when}, "
        f"recorded session {m.recorded_session_ordinal} turn {m.recorded_turn_index}]"
    )


def classify_messages(f: NewFact, cands: Sequence[MemRow]) -> list[dict[str, str]]:
    key = f", attribute {f.attribute_key}" if f.attribute_key else ""
    user = "\n".join(
        [
            f"NEW fact: {f.fact_text} [{'NEG' if f.polarity == 'NEG' else 'POS'}{key}, {describe(f.rt)}, "
            f"session {f.session_ordinal} turn {f.turn_index}"
            f"{', speaker marked this as a correction' if f.is_correction else ''}]",
            f'Quote: "{f.quote}"',
            "CANDIDATES:",
            *[_fmt_mem(f"C{i}", m) for i, m in enumerate(cands, start=1)],
        ]
    )
    return [{"role": "system", "content": CLASSIFY_SYSTEM}, {"role": "user", "content": user}]


# ------------------------------------------------------------------------------ resolver
POLICIES = ("bitemporal", "arrival_order", "append_only")


class StateResolver:
    """Applies the write rules. `policy` selects the state model (ablation B):

    bitemporal     the full system (default)
    arrival_order  supersession by arrival only: the fact's time expression is discarded (only its
                   tense is kept), so every UPDATES supersedes forward and "an ended state never
                   replaces the current one" (normalization rule 7) never fires
    append_only    nothing is ever superseded, closed or conflicted: UPDATES / NEGATES / CONFLICT /
                   HISTORICAL / EXTENDS all insert a new ACTIVE memory; DUPLICATE still only
                   reaffirms and CORRECTS still retracts

    `recorded` maps memory_id -> (classifier relation, classifier target memory_id) from a previous
    run's resolver_log; with it no LLM is called, which is how a policy is replayed.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        llm: Any,
        config: IngestionConfig | None = None,
        *,
        policy: str = "bitemporal",
        recorded: dict[str, tuple[str | None, str | None]] | None = None,
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        self.conn = conn
        self.llm = llm
        self.config = config or IngestionConfig()
        self.policy = policy
        self.recorded = recorded
        self.llm_calls = 0

    # ------------------------------------------------------------------ candidates
    def _rows(self, where: str, params: Sequence[Any]) -> list[MemRow]:
        return [
            MemRow(*r)
            for r in self.conn.execute(
                f"SELECT {_MEM_COLS} FROM memories WHERE {where}", params
            ).fetchall()
        ]

    def get(self, memory_id: str) -> MemRow | None:
        rows = self._rows("memory_id = ?", (memory_id,))
        return rows[0] if rows else None

    def candidates(self, f: NewFact) -> list[MemRow]:
        base = (
            "conversation_id = ? AND subject_entity_id = ? AND status = 'ACTIVE' AND memory_id != ?"
        )
        if f.attribute_key:
            rows = self._rows(
                base + " AND attribute_key = ? ORDER BY (valid_to IS NOT NULL), "
                "recorded_session_ordinal DESC, recorded_turn_index DESC, rowid DESC",
                (f.conversation_id, f.subject_entity_id, f.memory_id, f.attribute_key),
            )
            return rows[: self.config.candidate_top_k]
        if f.embedding is None:
            return []
        rows = [
            r
            for r in self._rows(
                base + " ORDER BY rowid", (f.conversation_id, f.subject_entity_id, f.memory_id)
            )
            if r.embedding
        ]
        if not rows:
            return []
        sims = np.stack([from_blob(r.embedding) for r in rows]) @ f.embedding
        order = sorted(range(len(rows)), key=lambda i: (-float(sims[i]), i))
        keep = [i for i in order if sims[i] >= self.config.candidate_min_similarity]
        return [rows[i] for i in keep[: self.config.candidate_top_k]]

    # ------------------------------------------------------------------ main entry
    def apply(self, f: NewFact) -> Decision:
        dec = self._apply(f)
        if dec.relation != "SKIPPED_EXISTS":
            self._log(f, dec)
        return dec

    def _log(self, f: NewFact, dec: Decision) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO resolver_log(memory_id, conversation_id, session_ordinal, turn_index, "
            "llm_relation, llm_target_id, applied_relation, applied_target_id, n_candidates, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f.memory_id, f.conversation_id, f.session_ordinal, f.turn_index, dec.llm_relation,
                dec.llm_target_id, dec.relation, dec.target_id, dec.candidates, json.dumps(dec.notes),
            ),
        )  # fmt: skip

    def _apply(self, f: NewFact) -> Decision:
        if self.get(f.memory_id) is not None:
            return Decision(f.memory_id, f.fact_text, None, "SKIPPED_EXISTS", None, 0)
        if self.policy == "arrival_order":
            f = self._ignore_event_time(f)
        if f.is_correction and f.correction_wrong:
            dec = self._structured_correction(f)
            if dec is not None:
                return dec
        cands = self.candidates(f)
        if not cands:
            self._insert(f, "ACTIVE")
            return Decision(f.memory_id, f.fact_text, None, "NEW", None, 0)

        labels = {f"C{i}": m for i, m in enumerate(cands, start=1)}
        notes: list[str] = []
        if self.recorded is not None:
            out = self._recorded_relation(f, labels, notes)
        else:
            self.llm_calls += 1
            out = self.llm.structured(classify_messages(f, cands), RelationOut)
        target = labels.get((out.target or "").strip().upper())
        dec = Decision(
            f.memory_id, f.fact_text, out.relation, out.relation, None, len(cands), notes
        )
        dec.llm_target_id = target.memory_id if target else None
        if self.recorded is not None and f.memory_id not in self.recorded:
            dec.llm_relation = None  # nothing was recorded for this fact in the source run
        rel, target, f = self._normalize(out.relation, target, f, cands, dec)
        if self.policy == "append_only" and rel in APPEND_ONLY_AS_NEW:
            dec.notes.append(f"append-only: {rel} -> NEW")
            rel, target = "NEW", None
        dec.relation, dec.target_id = rel, target.memory_id if target else None
        self._write(rel, f, target)
        return dec

    def _recorded_relation(
        self, f: NewFact, labels: dict[str, MemRow], notes: list[str]
    ) -> RelationOut:
        """The classifier's answer from the source run, mapped onto this run's candidate labels."""
        rec = self.recorded.get(f.memory_id)
        if rec is None or rec[0] is None or rec[0] == "NEW":
            return RelationOut(reason="replay", relation="NEW", target=None)
        relation, target_id = rec
        if target_id is None:
            return RelationOut(reason="replay", relation=relation, target=None)
        label = next((lab for lab, m in labels.items() if m.memory_id == target_id), None)
        if label is None:
            notes.append("replay: recorded target is not a candidate under this policy -> NEW")
            return RelationOut(reason="replay", relation="NEW", target=None)
        return RelationOut(reason="replay", relation=relation, target=label)

    def _ignore_event_time(self, f: NewFact) -> NewFact:
        expr = TimeExpr(kind="NONE", tense=f.time_expr.tense)
        return replace(f, time_expr=expr, rt=resolve(expr, f.session_ts, is_event=f.is_event))

    # ------------------------------------------------------------------ structured corrections
    def _structured_correction(self, f: NewFact) -> Decision | None:
        """CORRECTS without a classifier call when the extractor named the wrong value.

        Target: the latest open ACTIVE memory of the same subject (and attribute key, if any),
        recorded no later than this fact, that mentions the wrong value but not the right one.
        If the model never extracted that original statement (common when both values are in
        one turn), it is created from this turn (fact_text with right -> wrong, quote = the wrong
        value's span) and then retracted, so the record of what was said survives.
        """
        target = self._correction_target(f)
        note = "structured correction: retracted existing memory"
        if target is None and self._wrong_value_was_just_settled(f):
            return None  # it would undo this turn's own correction: let the guarded classifier path decide
        if target is None:
            target = self._synthesize_original(f)
            note = "structured correction: original statement created from the turn, then retracted"
        if target is None:
            return None
        self._write("CORRECTS", f, target)
        return Decision(f.memory_id, f.fact_text, None, "CORRECTS", target.memory_id, 0, [note])

    def _correction_target(self, f: NewFact) -> MemRow | None:
        wrong = re.compile(rf"\b{re.escape(f.correction_wrong)}\b", re.IGNORECASE)
        right = (
            re.compile(rf"\b{re.escape(f.correction_right)}\b", re.IGNORECASE)
            if f.correction_right
            else None
        )
        where = ("conversation_id = ? AND subject_entity_id = ? AND status = 'ACTIVE' AND valid_to IS NULL "
                 "AND memory_id != ?")  # fmt: skip
        params: list[Any] = [f.conversation_id, f.subject_entity_id, f.memory_id]
        if f.attribute_key:
            where += " AND attribute_key = ?"
            params.append(f.attribute_key)
        rows = self._rows(
            where + " ORDER BY recorded_session_ordinal DESC, recorded_turn_index DESC, rowid DESC",
            params,
        )
        for r in rows:
            if (
                r.key <= f.key
                and not self._settled_by_correction_in_turn(r, f)
                and wrong.search(r.fact_text)
                and not (right and right.search(r.fact_text))
            ):
                return r
        return None

    def _wrong_value_was_just_settled(self, f: NewFact) -> bool:
        wrong = re.compile(rf"\b{re.escape(f.correction_wrong)}\b", re.IGNORECASE)
        rows = self._rows(
            "memory_id IN (SELECT memory_id FROM fact_sources WHERE turn_uid = ? AND role = 'CORRECTS') "
            "AND subject_entity_id = ?",
            (f.turn_uid, f.subject_entity_id),
        )
        return any(wrong.search(r.fact_text) for r in rows)

    def _established_by_correction(self, m: MemRow) -> bool:
        return bool(self.conn.execute(
            "SELECT 1 FROM fact_sources WHERE memory_id = ? AND role = 'CORRECTS'", (m.memory_id,)
        ).fetchone())  # fmt: skip

    @staticmethod
    def _names_wrong(f: NewFact, m: MemRow) -> bool:
        return (
            bool(f.correction_wrong)
            and re.search(rf"\b{re.escape(f.correction_wrong)}\b", m.fact_text, re.IGNORECASE)
            is not None
        )

    def _settled_by_correction_in_turn(self, m: MemRow, f: NewFact) -> bool:
        """True if `m` is the value a speaker settled on by correcting themselves in f's turn.

        Another fact from the same turn must not retract it ("Pune, sorry, I meant Mumbai" may
        also yield "Neha used to live in Pune"); a correction in a later turn still can.
        """
        return bool(self.conn.execute(
            "SELECT 1 FROM fact_sources WHERE memory_id = ? AND turn_uid = ? AND role = 'CORRECTS'",
            (m.memory_id, f.turn_uid),
        ).fetchone())  # fmt: skip

    def _synthesize_original(self, f: NewFact) -> MemRow | None:
        if not f.correction_right:
            return None
        m = re.search(rf"\b{re.escape(f.correction_right)}\b", f.fact_text, re.IGNORECASE)
        if m is None:
            return None
        text = f.fact_text[: m.start()] + f.correction_wrong + f.fact_text[m.end() :]
        quote, qs, qe = f.wrong_quote if f.wrong_quote else (f.quote, f.quote_start, f.quote_end)
        orig = replace(f, memory_id=f"{f.memory_id}-orig", fact_text=text, is_correction=False, quote=quote,
                       quote_start=qs, quote_end=qe, mention_entity_ids=(), correction_wrong=None,
                       correction_right=None, wrong_quote=None)  # fmt: skip
        if self.get(orig.memory_id) is None:
            self._insert(orig, "ACTIVE")
        return self.get(orig.memory_id)

    # ------------------------------------------------------------------ deterministic rules
    def _normalize(
        self, rel: str, target: MemRow | None, f: NewFact, cands: list[MemRow], dec: Decision
    ) -> tuple[str, MemRow | None, NewFact]:
        note = dec.notes.append
        if rel != "NEW" and target is None:
            if len(cands) == 1:
                target = cands[0]
                note("target defaulted to the only candidate")
            else:
                note(f"{rel} without a valid target -> NEW")
                return "NEW", None, f
        if rel == "NEW":
            return "NEW", None, f

        if f.is_correction and rel == "UPDATES":
            rel = "CORRECTS"
            note("correction cue: UPDATES -> CORRECTS")
        if rel == "CORRECTS" and not f.is_correction:
            # CORRECTS means the speaker took a statement back; without the extractor's correction
            # cue it is at most a change of the same single-valued attribute
            same_slot = f.attribute_key is not None and f.attribute_key == target.attribute_key
            note(f"CORRECTS without a correction cue -> {'UPDATES' if same_slot else 'NEW'}")
            if not same_slot:
                return "NEW", None, f
            rel = "UPDATES"
        if rel == "CORRECTS" and self._settled_by_correction_in_turn(target, f):
            note("target was itself established by a correction in this turn -> NEW")
            return "NEW", None, f
        if (
            rel == "CORRECTS"
            and self._established_by_correction(target)
            and not self._names_wrong(f, target)
        ):
            note(
                "target was established by a correction and this fact does not name it as wrong -> NEW"
            )
            return "NEW", None, f
        if rel == "CONFLICT" and target.key != f.key:
            rel = "UPDATES"
            note("CONFLICT across turns -> UPDATES")
        if rel == "DUPLICATE" and f.polarity != target.polarity:
            rel = "UPDATES"
            note("DUPLICATE with opposite polarity -> UPDATES")
        if f.polarity == "NEG" and target.polarity == "POS" and rel == "UPDATES":
            rel = "NEGATES"

        if rel in ("UPDATES", "CORRECTS", "NEGATES", "CONFLICT") and not target.is_open:
            alt = next(
                (
                    c
                    for c in cands
                    if c.is_open and f.attribute_key and c.attribute_key == f.attribute_key
                ),
                None,
            )
            if alt is not None:
                note(f"{rel} target not open -> retargeted to open same-key candidate")
                target = alt
            else:
                note(f"{rel} target not open -> NEW")
                return "NEW", None, f

        if rel == "HISTORICAL" and f.recent_onset and not f.rt.is_closed:
            note("HISTORICAL for a recently begun state -> UPDATES")
            rel = "UPDATES"
        if rel == "UPDATES" and f.rt.is_closed and self.policy != "arrival_order":
            note("UPDATES by a closed-interval fact -> HISTORICAL")
            rel = "HISTORICAL"
        if rel == "HISTORICAL" and not f.rt.is_closed and not f.is_event:
            past = f.time_expr.model_copy(update={"tense": "PAST_ENDED"})
            f = replace(f, rt=resolve(past, f.session_ts, is_event=False))
            note("HISTORICAL with open interval -> time re-resolved as PAST_ENDED")
        return rel, target, f

    def _write(self, rel: str, f: NewFact, t: MemRow | None) -> None:
        if rel in ("NEW", "EXTENDS", "HISTORICAL"):
            self._insert(f, "ACTIVE")
        elif rel == "DUPLICATE":
            self._reaffirm(t, f)
        elif rel == "UPDATES":
            cmp = compare(f.rt.start, t.rt.start)
            if cmp == Cmp.LATER or (cmp == Cmp.INCOMPARABLE and f.key >= t.key):
                self._supersede_forward(t, f)
            else:
                self._insert_behind(t, f)
        elif rel == "CORRECTS":
            # insert first: superseded_by_id is a foreign key to the new row
            self._insert(f, "ACTIVE", supersedes_id=t.memory_id, role="CORRECTS")
            self.conn.execute(
                "UPDATE memories SET status = 'RETRACTED', retired_session_ordinal = ?, retired_turn_index = ?, "
                "superseded_by_id = ? WHERE memory_id = ?",
                (*f.key, f.memory_id, t.memory_id),
            )
            self._source(t.memory_id, f, "RETRACTS")
        elif rel == "CONFLICT":
            cs = t.conflict_set_id or f"cs:{t.memory_id}"
            self.conn.execute(
                "UPDATE memories SET status = 'CONFLICTED', conflict_set_id = ? WHERE memory_id = ?",
                (cs, t.memory_id),
            )
            self._insert(f, "CONFLICTED", conflict_set_id=cs)
        elif rel == "NEGATES":
            self._supersede_forward(t, f)
        else:  # pragma: no cover
            raise ValueError(rel)

    # ------------------------------------------------------------------ write primitives
    def _end_point(self, f: NewFact) -> tuple[str, str]:
        """Where an old fact ends when `f` replaces it: f.valid_from, else 'by this session'."""
        if f.rt.valid_from is not None:
            return f.rt.valid_from, str(f.rt.valid_from_precision)
        if f.is_event and f.rt.event_date is not None:
            return f.rt.event_date, str(f.rt.event_date_precision)
        return session_date(f.session_ts).isoformat(), Precision.BEFORE.value

    def _supersede_forward(self, t: MemRow, f: NewFact) -> None:
        end, prec = self._end_point(f)
        self._insert(f, "ACTIVE", supersedes_id=t.memory_id)  # first: FK target of superseded_by_id
        self.conn.execute(
            "UPDATE memories SET status = 'SUPERSEDED', valid_to = ?, valid_to_precision = ?, "
            "retired_session_ordinal = ?, retired_turn_index = ?, superseded_by_id = ? WHERE memory_id = ?",
            (end, prec, *f.key, f.memory_id, t.memory_id),
        )

    def _insert_behind(self, t: MemRow, f: NewFact) -> None:
        """`f` happened before the current fact `t`: insert it SUPERSEDED, linked behind `t`."""
        if t.valid_from is not None:
            f = replace(
                f,
                rt=replace(
                    f.rt,
                    valid_to=t.valid_from,
                    valid_to_precision=Precision(t.valid_from_precision),
                ),
            )
        else:
            f = replace(f, rt=replace(f.rt, valid_to=None, valid_to_precision=None))
        prev = t.supersedes_id
        self._insert(
            f, "SUPERSEDED", supersedes_id=prev, superseded_by_id=t.memory_id, retired=f.key
        )
        if prev is not None:
            self.conn.execute(
                "UPDATE memories SET superseded_by_id = ? WHERE memory_id = ?", (f.memory_id, prev)
            )
        self.conn.execute(
            "UPDATE memories SET supersedes_id = ? WHERE memory_id = ?", (f.memory_id, t.memory_id)
        )

    def _reaffirm(self, t: MemRow, f: NewFact) -> None:
        new_turn = not self.conn.execute(
            "SELECT 1 FROM fact_sources WHERE memory_id = ? AND turn_uid = ?",
            (t.memory_id, f.turn_uid),
        ).fetchone()
        if new_turn:
            self._source(t.memory_id, f, "REAFFIRMS")
            self.conn.execute(
                "UPDATE memories SET corroboration = corroboration + 1 WHERE memory_id = ?",
                (t.memory_id,),
            )

    def _source(self, memory_id: str, f: NewFact, role: str) -> None:
        self.conn.execute(
            "INSERT INTO fact_sources(memory_id, turn_uid, quote, quote_start, quote_end, role) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
            (memory_id, f.turn_uid, f.quote, f.quote_start, f.quote_end, role),
        )

    def _insert(
        self,
        f: NewFact,
        status: str,
        *,
        supersedes_id: str | None = None,
        superseded_by_id: str | None = None,
        conflict_set_id: str | None = None,
        retired: tuple[int, int] | None = None,
        role: str = "ASSERTS",
    ) -> None:
        rt = f.rt
        s = lambda p: str(p) if p is not None else None  # noqa: E731
        self.conn.execute(
            "INSERT INTO memories(memory_id, conversation_id, subject_entity_id, attribute_key, fact_text, "
            "polarity, mode, corroboration, status, valid_from, valid_from_precision, valid_to, valid_to_precision, "
            "event_date, event_date_precision, recorded_session_ordinal, recorded_turn_index, "
            "retired_session_ordinal, retired_turn_index, supersedes_id, superseded_by_id, conflict_set_id, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f.memory_id, f.conversation_id, f.subject_entity_id, f.attribute_key, f.fact_text,
                f.polarity, f.mode, status, rt.valid_from, s(rt.valid_from_precision), rt.valid_to,
                s(rt.valid_to_precision), rt.event_date, s(rt.event_date_precision), *f.key,
                *(retired or (None, None)), supersedes_id, superseded_by_id, conflict_set_id,
                to_blob(f.embedding) if f.embedding is not None else None,
            ),
        )  # fmt: skip
        self._source(f.memory_id, f, role)
        self.conn.execute(
            "INSERT OR IGNORE INTO fact_entities(memory_id, entity_id, role) VALUES (?, ?, 'SUBJECT')",
            (f.memory_id, f.subject_entity_id),
        )
        for eid in f.mention_entity_ids:
            if eid != f.subject_entity_id:
                self.conn.execute(
                    "INSERT OR IGNORE INTO fact_entities(memory_id, entity_id, role) VALUES (?, ?, 'MENTION')",
                    (f.memory_id, eid),
                )
