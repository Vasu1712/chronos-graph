"""SQLite access layer: WAL, foreign keys, versioned migrations, per-session transactions."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    applied_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
)
"""


def _load_sql(name: str) -> str:
    return resources.files("chronos.storage").joinpath(name).read_text(encoding="utf-8")


# (version, name, sql). Append new migrations; never edit an applied one.
MIGRATIONS: list[tuple[int, str, str]] = [
    (1, "initial_schema", _load_sql("schema.sql")),
    (2, "episode_embeddings", _load_sql("schema_v2_episode_embeddings.sql")),
]


def make_turn_uid(dataset: str, conversation: str, session: int, turn: int) -> str:
    """SPEC turn_uid: "{dataset}:{conversation}:s{session:02d}:t{turn:03d}"."""
    return f"{dataset}:{conversation}:s{session:02d}:t{turn:03d}"


class ImmutableEpisodeError(sqlite3.IntegrityError):
    """Raised when an existing turn_uid is re-appended with different content."""


@dataclass
class SessionTx:
    """Handle yielded by Database.session_transaction()."""

    conn: sqlite3.Connection
    session_id: str
    already_ingested: bool


class Database:
    def __init__(self, path: str | Path, *, migrate: bool = True) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None: we issue BEGIN/COMMIT ourselves so transaction scope is explicit.
        self.conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.execute("PRAGMA busy_timeout = 5000")
        if migrate:
            self.migrate()

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ migrations
    def schema_version(self) -> int:
        self.conn.execute(SCHEMA_VERSION_TABLE)
        row = self.conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
        return int(row[0] or 0)

    def migrate(self) -> int:
        current = self.schema_version()
        for version, name, sql in MIGRATIONS:
            if version <= current:
                continue
            # executescript() would auto-commit; run statements inside one explicit transaction.
            with self.transaction() as conn:
                for stmt in _split_sql(sql):
                    conn.execute(stmt)
                conn.execute(
                    "INSERT INTO schema_migrations(version, name) VALUES (?, ?)", (version, name)
                )
        return self.schema_version()

    # ------------------------------------------------------------------ transactions
    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """BEGIN IMMEDIATE ... COMMIT, rolling back on any exception."""
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    @contextmanager
    def session_transaction(self, session_id: str) -> Iterator[SessionTx]:
        """One transaction per ingested session (SPEC §3).

        Yields a SessionTx. If the session was already ingested, `already_ingested` is True and
        callers must write nothing (re-ingest is a no-op). On a clean exit the session is
        stamped `ingested_at`; any exception rolls back every write made for the session.
        """
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT ingested_at FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown session_id {session_id!r}")
            tx = SessionTx(conn=conn, session_id=session_id, already_ingested=row[0] is not None)
            yield tx
            if not tx.already_ingested:
                conn.execute(
                    "UPDATE sessions SET ingested_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
                    "WHERE session_id = ?",
                    (session_id,),
                )

    # ------------------------------------------------------------------ Tier 1 helpers
    def add_conversation(
        self,
        conversation_id: str,
        dataset: str,
        *,
        source_ref: str | None = None,
        participants: Sequence[str] = (),
    ) -> None:
        self.conn.execute(
            "INSERT INTO conversations(conversation_id, dataset, source_ref, participants) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(conversation_id) DO NOTHING",
            (conversation_id, dataset, source_ref, json.dumps(list(participants))),
        )

    def add_session(
        self,
        session_id: str,
        conversation_id: str,
        session_ordinal: int,
        session_ts: str | None,
        *,
        source_ref: str | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO sessions(session_id, conversation_id, session_ordinal, session_ts, source_ref) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT(session_id) DO NOTHING",
            (session_id, conversation_id, session_ordinal, session_ts, source_ref),
        )

    def add_entity(
        self,
        entity_id: str,
        conversation_id: str,
        canonical_name: str,
        *,
        entity_type: str | None = None,
        is_speaker: bool = False,
        descriptor: str | None = None,
        embedding: bytes | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO entities(entity_id, conversation_id, canonical_name, entity_type, "
            "is_speaker, descriptor, embedding) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(entity_id) DO NOTHING",
            (
                entity_id,
                conversation_id,
                canonical_name,
                entity_type,
                int(is_speaker),
                descriptor,
                embedding,
            ),
        )

    def append_episode(self, ep: dict[str, Any]) -> bool:
        """Append one episode. Returns True if inserted, False if an identical row exists.

        Re-appending a turn_uid with different content raises ImmutableEpisodeError.
        """
        cols = (
            "turn_uid",
            "conversation_id",
            "session_id",
            "session_ordinal",
            "turn_index",
            "speaker",
            "speaker_entity_id",
            "session_ts",
            "text",
            "source_ref",
        )
        values = tuple(ep.get(c) for c in cols)
        cur = self.conn.execute(
            f"INSERT INTO episodes({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
            "ON CONFLICT(turn_uid) DO NOTHING",
            values,
        )
        if cur.rowcount == 1:
            return True
        existing = self.conn.execute(
            f"SELECT {', '.join(cols)} FROM episodes WHERE turn_uid = ?", (ep["turn_uid"],)
        ).fetchone()
        if tuple(existing) != values:
            raise ImmutableEpisodeError(
                f"episode {ep['turn_uid']} already exists with other content"
            )
        return False

    def search_episodes_fts(
        self, query: str, *, conversation_id: str | None = None, limit: int = 20
    ) -> list[sqlite3.Row]:
        """BM25-ranked FTS5 search over episodes.text (lower bm25 = better)."""
        sql = (
            "SELECT e.*, bm25(episodes_fts) AS score FROM episodes_fts "
            "JOIN episodes e ON e.rowid = episodes_fts.rowid "
            "WHERE episodes_fts MATCH ?"
        )
        params: list[Any] = [fts5_query(query)]
        if conversation_id is not None:
            sql += " AND e.conversation_id = ?"
            params.append(conversation_id)
        sql += " ORDER BY score LIMIT ?"
        params.append(limit)
        return self.conn.execute(sql, params).fetchall()


def fts5_query(text: str) -> str:
    """Turn free text into a safe FTS5 OR-query of quoted terms."""
    terms = ["".join(ch for ch in tok if ch.isalnum()) for tok in text.split()]
    terms = [t for t in terms if t]
    return " OR ".join(f'"{t}"' for t in terms) if terms else '""'


def _split_sql(script: str) -> list[str]:
    """Split a script into complete statements (trigger bodies contain inner semicolons)."""
    statements: list[str] = []
    buf: list[str] = []
    for line in script.splitlines():
        stripped = line.strip()
        if not buf and (not stripped or stripped.startswith("--")):
            continue
        buf.append(line)
        candidate = "\n".join(buf)
        if sqlite3.complete_statement(candidate):
            statements.append(candidate.strip())
            buf = []
    if "".join(buf).strip():
        raise ValueError(f"incomplete SQL statement: {''.join(buf)[:200]}")
    return statements
