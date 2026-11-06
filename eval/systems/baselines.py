"""Baselines B1-B4. All use the Part 3 generator, verifier and answer schema, and render context
as "[turn_uid | date | speaker] text" under the same token budget (B2 excepted)."""

from __future__ import annotations

import numpy as np

from chronos.config import RetrievalConfig
from chronos.llm.tokens import count_chat_tokens, count_tokens
from chronos.retrieval.query import analyze
from chronos.retrieval.retriever import Retriever
from eval.loaders.types import QAItem
from eval.systems.common import BaselineSystem, Packed, Skipped, fit_lines, turn_line

NUM_CTX = 32768
OUTPUT_RESERVE = 1024  # tokens kept free for the model's answer


def load_turns(db, conversation_id: str) -> list[dict]:
    """Every turn of a conversation in chronological order, straight from the immutable log."""
    rows = db.conn.execute(
        "SELECT turn_uid, session_ordinal, turn_index, speaker, session_ts, text FROM episodes "
        "WHERE conversation_id = ? ORDER BY session_ordinal, turn_index",
        (conversation_id,),
    ).fetchall()
    return [dict(r) for r in rows]


class SlidingWindow(BaselineSystem):
    """B1: the most recent turns that fit the budget."""

    id = "B1_sliding_window"

    def build_context(self, qa: QAItem, qdate: str | None) -> Packed:
        turns = load_turns(self.db, qa.conversation_id)
        picked: list[dict] = []
        used = 0
        for t in reversed(turns):
            ln = turn_line(t["turn_uid"], t["session_ts"] or "", t["speaker"], t["text"])
            n = count_tokens(ln) + 1
            if used + n > self.budget:
                break
            picked.append(t)
            used += n
        picked.reverse()
        lines = [
            turn_line(t["turn_uid"], t["session_ts"] or "", t["speaker"], t["text"]) for t in picked
        ]
        lines, cut = fit_lines(lines, self.budget)  # joined-text count is the authority
        text = "\n".join(lines)
        return Packed(text, [t["turn_uid"] for t in picked][-len(lines) :] if lines else [],
                      tokens=count_tokens(text), budget=self.budget, truncated=len(picked) < len(turns))  # fmt: skip


class FullContext(BaselineSystem):
    """B2: the whole history. Only valid if the complete prompt fits Ollama's 32k window."""

    id = "B2_full_context"

    def build_context(self, qa: QAItem, qdate: str | None) -> Packed:
        if qa.dataset != "locomo":
            raise Skipped("full-context is defined for LoCoMo only")
        turns = load_turns(self.db, qa.conversation_id)
        text = "\n".join(
            turn_line(t["turn_uid"], t["session_ts"] or "", t["speaker"], t["text"], None)
            for t in turns
        )
        uids = [t["turn_uid"] for t in turns]
        from chronos.agent.generator import Generator

        prompt = count_chat_tokens(Generator.messages(qa.question, text, uids, question_date=qdate))
        if prompt > NUM_CTX - OUTPUT_RESERVE:
            raise Skipped(f"context_overflow: prompt is {prompt} tokens, window is {NUM_CTX} "
                          f"(minus {OUTPUT_RESERVE} for the answer)")  # fmt: skip
        return Packed(text, uids, tokens=count_tokens(text), budget=None)


class ChunkRAG(BaselineSystem):
    """B3: 3-turn chunks embedded with bge-small, top-k by cosine until the budget is full."""

    id = "B3_chunk_rag"
    CHUNK = 3

    def __init__(self, *args, **kw) -> None:
        super().__init__(*args, **kw)
        self._index: dict[str, tuple[list[list[dict]], np.ndarray]] = {}

    def index(self, conversation_id: str) -> tuple[list[list[dict]], np.ndarray]:
        if conversation_id not in self._index:
            turns = load_turns(self.db, conversation_id)
            chunks: list[list[dict]] = []
            for sess in sorted({t["session_ordinal"] for t in turns}):
                st = [t for t in turns if t["session_ordinal"] == sess]
                chunks += [st[i : i + self.CHUNK] for i in range(0, len(st), self.CHUNK)]
            vecs = self.rt.embedder.encode_passages(
                ["\n".join(f"{t['speaker']}: {t['text']}" for t in c) for c in chunks]
            )
            self._index[conversation_id] = (chunks, vecs)
        return self._index[conversation_id]

    def build_context(self, qa: QAItem, qdate: str | None) -> Packed:
        chunks, vecs = self.index(qa.conversation_id)
        q = self.rt.embedder.encode_queries([qa.question])[0]
        order = np.argsort(-(vecs @ q), kind="stable")
        chosen: list[int] = []
        truncated = False
        for i in order:
            trial = sorted([*chosen, int(i)])
            lines = [
                turn_line(t["turn_uid"], t["session_ts"] or "", t["speaker"], t["text"])
                for k in trial
                for t in chunks[k]
            ]
            if count_tokens("\n".join(lines)) > self.budget:
                truncated = True
                continue
            chosen = trial
        lines = [
            turn_line(t["turn_uid"], t["session_ts"] or "", t["speaker"], t["text"])
            for k in chosen
            for t in chunks[k]
        ]
        text = "\n".join(lines)
        return Packed(text, [t["turn_uid"] for k in chosen for t in chunks[k]], tokens=count_tokens(text),
                      budget=self.budget, truncated=truncated)  # fmt: skip


class TurnsPlusNotes(BaselineSystem):
    """B4: dense + BM25 over raw turns and over dated fact notes, fused with RRF.

    The notes are ChronosGraph's extracted facts stripped of everything that makes them a state
    store: no status, no interval, no version chain, no temporal filter, no retraction (a RETRACTED
    fact is still a note), no entity expansion. A note is shown as
    "[source_turn_uid | date | speaker] NOTE: fact" with the date it was recorded.
    """

    id = "B4_turns_plus_notes"

    def __init__(self, *args, **kw) -> None:
        super().__init__(*args, **kw)
        cfg = RetrievalConfig(use_entity_expansion=False, use_temporal_filter=False)
        self.retriever = Retriever(self.db, self.rt.embedder, cfg)

    def _first_source(self, memory_id: str) -> dict | None:
        r = self.db.conn.execute(
            "SELECT e.turn_uid, e.session_ordinal, e.turn_index, e.speaker, e.session_ts FROM fact_sources s "
            "JOIN episodes e USING(turn_uid) WHERE s.memory_id = ? ORDER BY e.session_ordinal, e.turn_index LIMIT 1",
            (memory_id,),
        ).fetchone()
        return dict(r) if r else None

    def build_context(self, qa: QAItem, qdate: str | None) -> Packed:
        analysis = analyze(
            qa.question, conn=self.db.conn, conversation_id=qa.conversation_id, question_date=qdate
        )
        result = self.retriever.retrieve(analysis, qa.conversation_id)
        items: list[
            tuple[tuple[int, int], str, str, str | None]
        ] = []  # (order key, line, uid, memory_id)
        for c in result.fused:
            if c.kind == "episode":
                r = self.db.conn.execute(
                    "SELECT turn_uid, session_ordinal, turn_index, speaker, session_ts, text FROM episodes WHERE turn_uid = ?",
                    (c.id,),
                ).fetchone()
                if r:
                    items.append(((r["session_ordinal"], r["turn_index"]),
                                  turn_line(r["turn_uid"], r["session_ts"] or "", r["speaker"], r["text"]), r["turn_uid"], None))  # fmt: skip
            else:
                m = self.db.conn.execute(
                    "SELECT fact_text FROM memories WHERE memory_id = ?", (c.id,)
                ).fetchone()
                src = self._first_source(c.id)
                if m and src:
                    ln = f"[{src['turn_uid']} | {(src['session_ts'] or '')[:10]} | {src['speaker']}] NOTE: {m['fact_text']}"
                    items.append(
                        ((src["session_ordinal"], src["turn_index"]), ln, src["turn_uid"], c.id)
                    )
        chosen: list[tuple] = []
        truncated = False
        for it in items:  # rank order; skip what does not fit
            trial = sorted([*chosen, it], key=lambda x: x[0])
            if count_tokens("\n".join(x[1] for x in trial)) > self.budget:
                truncated = True
                continue
            chosen = trial
        text = "\n".join(x[1] for x in chosen)
        return Packed(text, list(dict.fromkeys(x[2] for x in chosen)), [x[3] for x in chosen if x[3]],
                      tokens=count_tokens(text), budget=self.budget, truncated=truncated)  # fmt: skip
