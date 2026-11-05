"""SQLite response cache shared by every model client.

Key = sha256 over canonical JSON of (provider, model, quantization, prompt, schema, params).
Only validated responses are stored, so a cache hit is always usable.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

_DDL = """
CREATE TABLE IF NOT EXISTS llm_cache (
    key         TEXT PRIMARY KEY,
    provider    TEXT NOT NULL,
    model       TEXT NOT NULL,
    response    TEXT NOT NULL,
    meta        TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
)
"""


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def cache_key(
    *,
    provider: str,
    model: str,
    quantization: str | None,
    prompt: Any,
    schema: Any,
    params: dict[str, Any],
) -> str:
    payload = {
        "provider": provider,
        "model": model,
        "quantization": quantization,
        "prompt": prompt,
        "schema": schema,
        "params": params,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute(_DDL)
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT response FROM llm_cache WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def put(
        self,
        key: str,
        response: Any,
        *,
        provider: str,
        model: str,
        meta: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO llm_cache(key, provider, model, response, meta) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET response = excluded.response, meta = excluded.meta",
                (key, provider, model, json.dumps(response), canonical_json(meta or {})),
            )

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0])

    def close(self) -> None:
        self._conn.close()
