"""Canonical dump of every ChronosGraph table, for equality checks between databases."""

TABLES = ("conversations", "sessions", "episodes", "entities", "entity_aliases", "memories",
          "fact_sources", "fact_entities", "answer_log")  # fmt: skip
TIMESTAMPS = {"created_at", "ingested_at"}


def dump(conn, *, keep_timestamps: bool = False) -> dict:
    out = {}
    for t in TABLES:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({t})")]
        if not keep_timestamps:
            cols = [c for c in cols if c not in TIMESTAMPS]
        rows = conn.execute(f"SELECT {', '.join(cols)} FROM {t}").fetchall()
        out[t] = sorted((tuple(r) for r in rows), key=repr)
    return out
