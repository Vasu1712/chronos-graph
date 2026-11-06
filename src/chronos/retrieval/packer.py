"""Context packer (SPEC §6 steps 5-6).

Takes the fused, filtered candidates in rank order and adds blocks until the next one would push
the rendered text over the token budget (Qwen2.5 tokenizer). Rendering groups memories by subject:

  ## Alice
  - (memory demo:m:abc) Alice moved to Milwaukee. | ACTIVE, residence | from 2023-10-01 (MONTH) to now | sources: t1
    history: Alice lives in Chicago. (memory ..., SUPERSEDED, from ... to ...) -> Alice moved to Milwaukee. (...)
    [turn_uid | 2023-11-05 | Alice] We moved to Milwaukee last month.
  ## Other relevant turns
    [turn_uid | date | speaker] text

Per memory: status, attribute, interval with precision, conflict partners, the version chain when
the intent needs history (HISTORY, TEMPORAL_ORDER, AS_OF), and its source turns plus a
+-`episode_window` turn window. Turns are printed once; later blocks reference them by id.
The result lists exactly the memory_ids and turn_uids that appear in the text.

Long turns are clipped for display (LongMemEval assistant answers run to 1000+ tokens): a memory's
source turn keeps `max_source_turn_chars`, centred on the supporting quote; ±window neighbours keep
`max_neighbor_turn_chars`. With `expand_sources=False` (retrieval ablation) memory lines list their
source turn ids but no turn text is attached; those ids are still allowed citations.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field

from chronos import textdates
from chronos.config import RetrievalConfig
from chronos.ingestion.resolver import MemRow
from chronos.ingestion.temporal import describe
from chronos.llm.tokens import count_tokens
from chronos.retrieval.query import Intent
from chronos.retrieval.retriever import RetrievalResult
from chronos.storage.db import Database

qwen_token_count = count_tokens  # name kept for callers and tests

_HISTORY_INTENTS = {Intent.HISTORY, Intent.TEMPORAL_ORDER, Intent.AS_OF}
_MEM_COLS = ", ".join(MemRow.__dataclass_fields__)


@dataclass
class PackedContext:
    text: str
    memory_ids: list[str]
    turn_uids: list[str]
    tokens: int
    budget: int
    truncated: bool  # True if some candidates did not fit

    @property
    def allowed_turn_uids(self) -> list[str]:
        return self.turn_uids


def clip(text: str, limit: int, focus: tuple[int, int] | None = None) -> str:
    """Shorten `text` to about `limit` chars, centred on `focus` (a char span) when given."""
    if len(text) <= limit:
        return text
    if focus is None:
        return text[:limit].rstrip() + " […]"
    mid = (focus[0] + focus[1]) // 2
    start = max(0, min(mid - limit // 2, len(text) - limit))
    end = start + limit
    return (
        ("[…] " if start > 0 else "")
        + text[start:end].strip()
        + (" […]" if end < len(text) else "")
    )


@dataclass
class _Turn:
    turn_uid: str
    date: str
    speaker: str
    text: str  # already clipped for display
    session_id: str
    turn_index: int

    def line(self) -> str:
        # relative date words resolved in code, identically for every system (chronos.textdates)
        return f"[{self.turn_uid} | {self.date} | {self.speaker}] {textdates.annotate(self.text, self.date)}"


@dataclass
class _Block:
    kind: str  # memory | episode
    subject: str | None
    lines_head: list[str]  # memory lines (no turns)
    memory_ids: list[str]
    turns: list[_Turn]  # in display order
    listed_turn_uids: list[str] = field(default_factory=list)  # source ids named on memory lines


class Packer:
    def __init__(
        self,
        db: Database,
        config: RetrievalConfig | None = None,
        count_tokens: Callable[[str], int] | None = None,
    ) -> None:
        self.db = db
        self.config = config or RetrievalConfig()
        self.count_tokens = count_tokens or qwen_token_count

    # ------------------------------------------------------------------ db helpers
    @property
    def conn(self) -> sqlite3.Connection:
        return self.db.conn

    def _mem(self, memory_id: str) -> MemRow | None:
        r = self.conn.execute(
            f"SELECT {_MEM_COLS} FROM memories WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        return MemRow(*r) if r else None

    def _subject_name(self, entity_id: str) -> str:
        r = self.conn.execute(
            "SELECT canonical_name FROM entities WHERE entity_id = ?", (entity_id,)
        ).fetchone()
        return r[0] if r else entity_id

    def _source_rows(self, memory_id: str) -> list[tuple[str, int | None, int | None]]:
        return [
            (r[0], r[1], r[2])
            for r in self.conn.execute(
                "SELECT s.turn_uid, s.quote_start, s.quote_end FROM fact_sources s "
                "JOIN episodes e USING(turn_uid) WHERE s.memory_id = ? "
                "ORDER BY e.session_ordinal, e.turn_index, s.rowid",
                (memory_id,),
            )
        ]

    def _sources(self, memory_id: str) -> list[str]:
        return list(dict.fromkeys(uid for uid, _, _ in self._source_rows(memory_id)))

    def _make_turn(
        self, r: sqlite3.Row | tuple, *, source: bool, focus: tuple[int, int] | None
    ) -> _Turn:
        cfg = self.config
        limit = cfg.max_source_turn_chars if source else cfg.max_neighbor_turn_chars
        return _Turn(
            r[0], (r[1] or "")[:10], r[2], clip(r[3], limit, focus if source else None), r[4], r[5]
        )

    def _turn(self, turn_uid: str, focus: tuple[int, int] | None = None) -> _Turn | None:
        """A turn shown as a source (clipped around `focus`, the supporting quote, if known)."""
        r = self.conn.execute(
            "SELECT turn_uid, session_ts, speaker, text, session_id, turn_index FROM episodes WHERE turn_uid = ?",
            (turn_uid,),
        ).fetchone()
        return self._make_turn(r, source=True, focus=focus) if r else None

    def _window(self, turn_uid: str, focus: tuple[int, int] | None = None) -> list[_Turn]:
        """The source turn plus +-`episode_window` neighbours (neighbours are clipped harder)."""
        center = self._turn(turn_uid, focus)
        if center is None:
            return []
        w = self.config.episode_window
        rows = self.conn.execute(
            "SELECT turn_uid, session_ts, speaker, text, session_id, turn_index FROM episodes "
            "WHERE session_id = ? AND turn_index BETWEEN ? AND ? ORDER BY turn_index",
            (center.session_id, center.turn_index - w, center.turn_index + w),
        ).fetchall()
        return [
            center if r[0] == turn_uid else self._make_turn(r, source=False, focus=None)
            for r in rows
        ]

    def _source_turns(self, memory_id: str, *, detail: bool = True) -> list[_Turn]:
        """Source turns for a memory; honours expand_sources.

        detail=True: every source turn with its +-window neighbours. detail=False: only the first
        source turn, centred on its quote, with no neighbours (the cheap first packing pass).
        """
        if not self.config.expand_sources:
            return []
        out: list[_Turn] = []
        seen: set[str] = set()
        for uid, qs, qe in self._source_rows(memory_id):
            if uid in seen:
                continue
            seen.add(uid)
            focus = (qs, qe) if qs is not None and qe is not None else None
            if detail:
                out += self._window(uid, focus)
            elif (t := self._turn(uid, focus)) is not None:
                return [t]
        return out

    def chain(self, m: MemRow) -> list[MemRow]:
        """Full version chain through supersedes_id / superseded_by_id, oldest first."""
        back: list[MemRow] = []
        cur, seen = m, {m.memory_id}
        while cur.supersedes_id and cur.supersedes_id not in seen:
            nxt = self._mem(cur.supersedes_id)
            if nxt is None:
                break
            back.append(nxt)
            seen.add(nxt.memory_id)
            cur = nxt
        fwd: list[MemRow] = []
        cur = m
        while cur.superseded_by_id and cur.superseded_by_id not in seen:
            nxt = self._mem(cur.superseded_by_id)
            if nxt is None:
                break
            fwd.append(nxt)
            seen.add(nxt.memory_id)
            cur = nxt
        return [*reversed(back), m, *fwd]

    # ------------------------------------------------------------------ blocks
    @staticmethod
    def _mem_line(m: MemRow, sources: list[str]) -> str:
        bits = [m.status]
        if m.attribute_key:
            bits.append(m.attribute_key)
        if m.polarity == "NEG":
            bits.append("NEGATIVE")
        return f"- (memory {m.memory_id}) {m.fact_text} | {', '.join(bits)} | {describe(m.rt)} | sources: {', '.join(sources) or 'none'}"

    def _memory_block(self, memory_id: str, intent: Intent, detail: bool = True) -> _Block | None:
        m = self._mem(memory_id)
        if m is None:
            return None
        sources = self._sources(m.memory_id)
        head = [self._mem_line(m, sources)]
        ids = [m.memory_id]
        listed = list(sources)
        turns = self._source_turns(m.memory_id, detail=detail)
        if m.conflict_set_id:
            for (other_id,) in self.conn.execute(
                "SELECT memory_id FROM memories WHERE conflict_set_id = ? AND memory_id != ? ORDER BY rowid",
                (m.conflict_set_id, m.memory_id),
            ):
                o = self._mem(other_id)
                o_src = self._sources(other_id)
                head.append("  conflicts with: " + self._mem_line(o, o_src)[2:])
                ids.append(other_id)
                listed += o_src
                turns += self._source_turns(other_id, detail=detail)
        if intent in _HISTORY_INTENTS:
            chain = self.chain(m)
            if len(chain) > 1:
                parts = [
                    f"{c.fact_text} (memory {c.memory_id}, {c.status}, {describe(c.rt)})"
                    for c in chain
                ]
                head.append("  history: " + " -> ".join(parts))
                for c in chain:
                    if c.memory_id not in ids:
                        ids.append(c.memory_id)
                        listed += self._sources(c.memory_id)
                        turns += self._source_turns(c.memory_id, detail=detail)
        return _Block("memory", self._subject_name(m.subject_entity_id), head, ids, turns, listed)

    def _episode_block(self, turn_uid: str, detail: bool = True) -> _Block | None:
        turns = self._window(turn_uid) if self.config.expand_sources and detail else []
        if not turns and (t := self._turn(turn_uid)):
            turns = [t]
        return _Block("episode", None, [], [], turns) if turns else None

    # ------------------------------------------------------------------ render
    @staticmethod
    def render(blocks: list[_Block]) -> tuple[str, list[str], list[str]]:
        printed: set[str] = set()
        mem_ids: list[str] = []
        turn_ids: list[str] = []
        groups: dict[str | None, list[_Block]] = {}
        for b in blocks:
            groups.setdefault(b.subject if b.kind == "memory" else None, []).append(b)
        lines: list[str] = []

        def emit_turns(b: _Block) -> None:
            for t in sorted(b.turns, key=lambda t: (t.session_id, t.turn_index)):
                if t.turn_uid in printed:
                    continue
                printed.add(t.turn_uid)
                turn_ids.append(t.turn_uid)
                lines.append(f"  {t.line()}")

        for subject, bs in groups.items():
            if subject is None:
                continue
            lines.append(f"## {subject}")
            for b in bs:
                lines += b.lines_head
                for mid in b.memory_ids:
                    if mid not in mem_ids:
                        mem_ids.append(mid)
                emit_turns(b)
        if None in groups:
            eps = [b for b in groups[None] if any(t.turn_uid not in printed for t in b.turns)]
            if eps:
                lines.append("## Other relevant turns")
                for b in eps:
                    emit_turns(b)
        for b in blocks:  # ids named on memory lines are allowed citations even without turn text
            for uid in b.listed_turn_uids:
                if uid not in turn_ids:
                    turn_ids.append(uid)
        return "\n".join(lines), mem_ids, turn_ids

    def pack(self, result: RetrievalResult, budget: int | None = None) -> PackedContext:
        """Two passes over the fused ranking, so the budget buys breadth before detail.

        Pass 1 adds, in rank order, each memory's line with its first source turn (or an episode's
        turn alone). Pass 2 then upgrades the chosen blocks, in rank order, with every source turn
        and its +-window neighbours while the budget lasts. One pass with full detail per block
        (the earlier design) spent the budget on neighbours of the first few hits: in eval-50 the
        full system kept the gold turn in context for 29% of multi-hop and single-hop questions
        against 69% without expansion.
        """
        budget = budget or self.config.token_budget
        intent = result.analysis.intent
        chosen: list[_Block] = []
        keys: list[tuple[str, str]] = []  # (kind, id) of each chosen block
        best = ("", [], [], 0)
        truncated = False
        seen_mem: set[str] = set()

        def make(kind: str, cid: str, detail: bool) -> _Block | None:
            return (
                self._memory_block(cid, intent, detail)
                if kind == "memory"
                else self._episode_block(cid, detail)
            )

        for c in result.fused:
            if c.kind == "memory" and c.id in seen_mem:
                continue  # already shown inside another block's chain or conflict set
            b = make(c.kind, c.id, detail=False)
            if b is None:
                continue
            text, mids, tids = self.render([*chosen, b])
            n = self.count_tokens(text)
            if n > budget:
                truncated = True
                continue  # this block does not fit; a smaller, lower-ranked one still may
            chosen.append(b)
            keys.append((c.kind, c.id))
            seen_mem.update(b.memory_ids)
            best = (text, mids, tids, n)

        for i, (kind, cid) in enumerate(keys):
            fuller = make(kind, cid, detail=True)
            if fuller is None or len(fuller.turns) <= len(chosen[i].turns):
                continue
            trial = [*chosen[:i], fuller, *chosen[i + 1 :]]
            text, mids, tids = self.render(trial)
            n = self.count_tokens(text)
            if n > budget:
                truncated = True
                continue
            chosen = trial
            best = (text, mids, tids, n)
        text, mids, tids, n = best
        return PackedContext(text, mids, tids, n, budget, truncated)
