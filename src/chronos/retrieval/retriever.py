"""Hybrid candidate retrieval (SPEC §6 steps 2-4).

Ranked lists, top `top_k_per_list` each, every one switchable for ablations:
  dense_memories    cosine(query, memories.embedding)        (NumPy over BLOBs loaded from SQLite)
  bm25_memories     FTS5 BM25 over memories.fact_text
  dense_episodes    cosine(query, episode_embeddings)
  bm25_episodes     FTS5 BM25 over episodes.text
  entity_memories   1-hop: memories linked via fact_entities to the query's entities (named ones,
                    plus the speaker that "I"/"my"/"you" resolves to), ordered by dense similarity
                    (or recency if dense memories are off)

All lists are fused with Reciprocal Rank Fusion, score = sum 1 / (k + rank), rank from 1, over
items keyed (kind, id) so memories and episodes compete in one ranking. The temporal filter by
intent is then applied to memory items:
  CURRENT                    ACTIVE or CONFLICTED with open valid_to
  AS_OF [lo, hi]             not RETRACTED and the interval overlaps [lo, hi]
                             (a SESSION/UNKNOWN/BEFORE start counts as "unknown", i.e. unbounded;
                             valid_to is exclusive because a superseding fact starts on it)
  HISTORY / TEMPORAL_ORDER   not RETRACTED (version chains are added by the packer)
  OPEN                       not RETRACTED
Episodes are never filtered: they are the immutable record.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from chronos.config import RetrievalConfig
from chronos.ingestion.pipeline import store_episode_embeddings
from chronos.llm.embed import from_blob
from chronos.retrieval.query import Intent, QueryAnalysis
from chronos.storage.db import Database, fts5_query

LISTS = ("dense_memories", "bm25_memories", "dense_episodes", "bm25_episodes", "entity_memories")
_OPEN_START = {"SESSION", "UNKNOWN", "BEFORE", None}


@dataclass
class Candidate:
    kind: str  # "memory" | "episode"
    id: str
    score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    analysis: QueryAnalysis
    fused: list[Candidate]  # after the temporal filter, best first
    lists: dict[str, list[str]]  # raw ranked ids per list (for debugging / ablations)
    filtered_out: list[str]  # memory ids removed by the temporal filter
    timings: dict[str, float]

    @property
    def memory_ids(self) -> list[str]:
        return [c.id for c in self.fused if c.kind == "memory"]

    @property
    def turn_uids(self) -> list[str]:
        return [c.id for c in self.fused if c.kind == "episode"]


def rrf(lists: dict[str, list[tuple[str, str]]], k: int = 60) -> list[Candidate]:
    """Reciprocal Rank Fusion over lists of (kind, id); ties broken by first appearance."""
    cands: dict[tuple[str, str], Candidate] = {}
    order: dict[tuple[str, str], int] = {}
    for name, items in lists.items():
        for rank, key in enumerate(items, start=1):
            c = cands.setdefault(key, Candidate(key[0], key[1]))
            order.setdefault(key, len(order))
            c.score += 1.0 / (k + rank)
            c.ranks[name] = rank
    return sorted(cands.values(), key=lambda c: (-c.score, order[(c.kind, c.id)]))


def expansion_entities(analysis: QueryAnalysis) -> list[str]:
    """Entities named in the question plus the speaker a first/second-person question is about."""
    ids = list(analysis.entity_ids)
    if analysis.subject_speaker_id and analysis.subject_speaker_id not in ids:
        ids.append(analysis.subject_speaker_id)
    return ids


def passes_temporal_filter(row: sqlite3.Row | dict, analysis: QueryAnalysis) -> bool:
    status = row["status"]
    if analysis.intent == Intent.CURRENT:
        return status in ("ACTIVE", "CONFLICTED") and row["valid_to"] is None
    if status == "RETRACTED":
        return False
    if analysis.intent == Intent.AS_OF and analysis.time is not None:
        lo, hi = analysis.time.lo, analysis.time.hi
        if row["event_date_precision"] is not None and row["valid_from"] is None:
            return row["event_date"] is not None and lo <= row["event_date"] <= hi
        start_known = (
            row["valid_from"] is not None and row["valid_from_precision"] not in _OPEN_START
        )
        if start_known and row["valid_from"] > hi:
            return False
        return row["valid_to"] is None or row["valid_to"] > lo
    return True


class _Index:
    def __init__(self, ids: list[str], matrix: np.ndarray, stamp: tuple) -> None:
        self.ids, self.matrix, self.stamp = ids, matrix, stamp

    def search(self, q: np.ndarray, k: int, allowed: set[str] | None = None) -> list[str]:
        if not self.ids:
            return []
        scores = self.matrix @ q
        order = np.argsort(-scores, kind="stable")
        out = []
        for i in order:
            if allowed is None or self.ids[i] in allowed:
                out.append(self.ids[i])
                if len(out) >= k:
                    break
        return out


class Retriever:
    """In-memory NumPy matrices per conversation, loaded from SQLite BLOBs and refreshed on change."""

    def __init__(self, db: Database, embedder: Any, config: RetrievalConfig | None = None) -> None:
        self.db = db
        self.embedder = embedder
        self.config = config or RetrievalConfig()
        self._mem: dict[str, _Index] = {}
        self._ep: dict[str, _Index] = {}

    # ------------------------------------------------------------------ indexes
    def _memory_index(self, conversation_id: str) -> _Index:
        conn = self.db.conn
        stamp = tuple(conn.execute(
            "SELECT COUNT(*), MAX(rowid) FROM memories WHERE conversation_id = ?", (conversation_id,)
        ).fetchone())  # fmt: skip
        idx = self._mem.get(conversation_id)
        if idx is None or idx.stamp != stamp:
            rows = conn.execute(
                "SELECT memory_id, embedding FROM memories WHERE conversation_id = ? AND embedding IS NOT NULL ORDER BY rowid",
                (conversation_id,),
            ).fetchall()
            idx = _Index([r[0] for r in rows], _stack([r[1] for r in rows]), stamp)
            self._mem[conversation_id] = idx
        return idx

    def _episode_index(self, conversation_id: str) -> _Index:
        conn = self.db.conn
        stamp = tuple(conn.execute(
            "SELECT COUNT(*), MAX(rowid) FROM episodes WHERE conversation_id = ?", (conversation_id,)
        ).fetchone())  # fmt: skip
        idx = self._ep.get(conversation_id)
        if idx is None or idx.stamp != stamp:
            missing = conn.execute(
                "SELECT e.turn_uid, e.text FROM episodes e LEFT JOIN episode_embeddings v USING(turn_uid) "
                "WHERE e.conversation_id = ? AND v.turn_uid IS NULL ORDER BY e.rowid",
                (conversation_id,),
            ).fetchall()
            if missing:  # databases ingested before migration 2: backfill once and persist
                with self.db.transaction() as tx:
                    store_episode_embeddings(tx, self.embedder, [(r[0], r[1]) for r in missing])
            rows = conn.execute(
                "SELECT v.turn_uid, v.embedding FROM episode_embeddings v JOIN episodes e USING(turn_uid) "
                "WHERE e.conversation_id = ? ORDER BY e.rowid",
                (conversation_id,),
            ).fetchall()
            idx = _Index([r[0] for r in rows], _stack([r[1] for r in rows]), stamp)
            self._ep[conversation_id] = idx
        return idx

    # ------------------------------------------------------------------ lists
    def _bm25_memories(self, conversation_id: str, question: str, k: int) -> list[str]:
        return [
            r[0]
            for r in self.db.conn.execute(
                "SELECT m.memory_id FROM memories_fts JOIN memories m ON m.rowid = memories_fts.rowid "
                "WHERE memories_fts MATCH ? AND m.conversation_id = ? ORDER BY bm25(memories_fts) LIMIT ?",
                (fts5_query(question), conversation_id, k),
            )
        ]

    def _entity_memories(
        self, conversation_id: str, entity_ids: list[str], q: np.ndarray | None, k: int
    ) -> list[str]:
        if not entity_ids:
            return []
        marks = ",".join("?" * len(entity_ids))
        linked = [
            r[0]
            for r in self.db.conn.execute(
                f"SELECT DISTINCT fe.memory_id FROM fact_entities fe JOIN memories m USING(memory_id) "
                f"WHERE m.conversation_id = ? AND fe.entity_id IN ({marks}) "
                f"ORDER BY m.recorded_session_ordinal DESC, m.recorded_turn_index DESC, m.rowid DESC",
                (conversation_id, *entity_ids),
            )
        ]
        if q is not None and linked:
            return self._memory_index(conversation_id).search(q, k, allowed=set(linked))
        return linked[:k]

    # ------------------------------------------------------------------ main
    def retrieve(self, analysis: QueryAnalysis, conversation_id: str) -> RetrievalResult:
        cfg, k = self.config, self.config.top_k_per_list
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        need_dense = cfg.use_dense_memories or cfg.use_dense_episodes or cfg.use_entity_expansion
        q = self.embedder.encode_queries([analysis.question])[0] if need_dense else None
        timings["embed_query"] = time.perf_counter() - t0

        lists: dict[str, list[str]] = {}

        def timed(name: str, fn) -> None:
            t = time.perf_counter()
            lists[name] = fn()
            timings[name] = time.perf_counter() - t

        if cfg.use_dense_memories:
            timed("dense_memories", lambda: self._memory_index(conversation_id).search(q, k))
        if cfg.use_bm25_memories:
            timed(
                "bm25_memories", lambda: self._bm25_memories(conversation_id, analysis.question, k)
            )
        if cfg.use_dense_episodes:
            timed("dense_episodes", lambda: self._episode_index(conversation_id).search(q, k))
        if cfg.use_bm25_episodes:
            timed("bm25_episodes", lambda: [r["turn_uid"] for r in self.db.search_episodes_fts(
                analysis.question, conversation_id=conversation_id, limit=k)])  # fmt: skip
        if cfg.use_entity_expansion:
            q_for_entities = q if cfg.use_dense_memories else None
            timed(
                "entity_memories",
                lambda: self._entity_memories(
                    conversation_id, expansion_entities(analysis), q_for_entities, k
                ),
            )

        t = time.perf_counter()
        keyed = {
            name: [("episode" if "episodes" in name else "memory", i) for i in ids]
            for name, ids in lists.items()
        }
        fused = rrf(keyed, cfg.rrf_k)
        filtered_out: list[str] = []
        if cfg.use_temporal_filter:
            mem_ids = [c.id for c in fused if c.kind == "memory"]
            rows = {r["memory_id"]: r for r in _memory_rows(self.db.conn, mem_ids)}
            keep = []
            for c in fused:
                if c.kind == "memory" and not passes_temporal_filter(rows[c.id], analysis):
                    filtered_out.append(c.id)
                else:
                    keep.append(c)
            fused = keep
        timings["fuse_filter"] = time.perf_counter() - t
        return RetrievalResult(analysis, fused, lists, filtered_out, timings)


def _memory_rows(conn: sqlite3.Connection, ids: list[str]) -> list[sqlite3.Row]:
    out: list[sqlite3.Row] = []
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        out += conn.execute(
            f"SELECT * FROM memories WHERE memory_id IN ({','.join('?' * len(chunk))})", chunk
        ).fetchall()
    return out


def _stack(blobs: list[bytes]) -> np.ndarray:
    if not blobs:
        return np.zeros((0, 1), dtype=np.float32)
    return np.stack([from_blob(b) for b in blobs])
