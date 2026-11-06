-- Migration 2: dense vectors for episodes (Tier 3 dense retrieval over turns).
-- A separate table keeps `episodes` immutable; vectors are derived data and can be rebuilt.
CREATE TABLE episode_embeddings (
    turn_uid   TEXT PRIMARY KEY REFERENCES episodes(turn_uid),
    model      TEXT NOT NULL,
    embedding  BLOB NOT NULL
);
