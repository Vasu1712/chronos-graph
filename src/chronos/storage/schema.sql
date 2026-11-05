-- ChronosGraph schema, version 1. See SPEC.md §2 and §4.
-- Applied inside a single transaction by chronos.storage.db.Database.migrate().
-- JSON columns are TEXT guarded by json_valid() (a declared type of JSON would get NUMERIC affinity).

------------------------------------------------------------------------------------------
-- Tier 1: Immutable Episodic Log
------------------------------------------------------------------------------------------

CREATE TABLE conversations (
    conversation_id  TEXT PRIMARY KEY,
    dataset          TEXT NOT NULL,
    source_ref       TEXT,
    participants     TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(participants)),
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE sessions (
    session_id       TEXT PRIMARY KEY,
    conversation_id  TEXT NOT NULL REFERENCES conversations(conversation_id),
    session_ordinal  INTEGER NOT NULL,
    session_ts       TEXT,                 -- ISO-8601
    source_ref       TEXT,
    ingested_at      TEXT,                 -- set when the ingestion transaction commits
    UNIQUE (conversation_id, session_ordinal)
);

CREATE TABLE episodes (
    turn_uid           TEXT PRIMARY KEY,   -- "{dataset}:{conversation}:s{session:02d}:t{turn:03d}"
    conversation_id    TEXT NOT NULL REFERENCES conversations(conversation_id),
    session_id         TEXT NOT NULL REFERENCES sessions(session_id),
    session_ordinal    INTEGER NOT NULL,
    turn_index         INTEGER NOT NULL,
    speaker            TEXT NOT NULL,
    speaker_entity_id  TEXT REFERENCES entities(entity_id),
    session_ts         TEXT,               -- ISO-8601
    text               TEXT NOT NULL,
    source_ref         TEXT,               -- dataset's own id, e.g. LoCoMo "D3:7"
    UNIQUE (conversation_id, session_ordinal, turn_index)
);

CREATE INDEX idx_episodes_order ON episodes(conversation_id, session_ordinal, turn_index);

-- Episodes are never updated or deleted.
CREATE TRIGGER episodes_no_update BEFORE UPDATE ON episodes
BEGIN
    SELECT RAISE(ABORT, 'episodes are immutable');
END;

CREATE TRIGGER episodes_no_delete BEFORE DELETE ON episodes
BEGIN
    SELECT RAISE(ABORT, 'episodes are immutable');
END;

CREATE VIRTUAL TABLE episodes_fts USING fts5(
    text,
    content = 'episodes',
    content_rowid = 'rowid',
    tokenize = 'porter unicode61'
);

CREATE TRIGGER episodes_fts_ai AFTER INSERT ON episodes
BEGIN
    INSERT INTO episodes_fts(rowid, text) VALUES (new.rowid, new.text);
END;

------------------------------------------------------------------------------------------
-- Tier 2: Bi-Temporal Fact and Entity Store
------------------------------------------------------------------------------------------

CREATE TABLE entities (
    entity_id        TEXT PRIMARY KEY,
    conversation_id  TEXT NOT NULL REFERENCES conversations(conversation_id),
    canonical_name   TEXT NOT NULL,
    entity_type      TEXT,
    is_speaker       INTEGER NOT NULL DEFAULT 0 CHECK (is_speaker IN (0, 1)),
    descriptor       TEXT,
    embedding        BLOB
);

CREATE INDEX idx_entities_conv ON entities(conversation_id);

CREATE TABLE entity_aliases (
    entity_id  TEXT NOT NULL REFERENCES entities(entity_id),
    alias      TEXT NOT NULL,
    turn_uid   TEXT REFERENCES episodes(turn_uid),   -- NULL for pre-registered speaker names
    UNIQUE (entity_id, alias, turn_uid)
);

CREATE INDEX idx_entity_aliases_alias ON entity_aliases(alias COLLATE NOCASE);

CREATE TABLE memories (
    memory_id                 TEXT PRIMARY KEY,
    conversation_id           TEXT NOT NULL REFERENCES conversations(conversation_id),
    subject_entity_id         TEXT NOT NULL REFERENCES entities(entity_id),
    attribute_key             TEXT,                                  -- NULL = multi-valued
    fact_text                 TEXT NOT NULL,
    polarity                  TEXT NOT NULL DEFAULT 'POS' CHECK (polarity IN ('POS', 'NEG')),
    mode                      TEXT NOT NULL DEFAULT 'EXPLICIT' CHECK (mode IN ('EXPLICIT', 'IMPLIED')),
    corroboration             INTEGER NOT NULL DEFAULT 1,
    status                    TEXT NOT NULL DEFAULT 'ACTIVE'
                              CHECK (status IN ('ACTIVE', 'SUPERSEDED', 'RETRACTED', 'CONFLICTED')),

    -- event time: when it was true in the world
    valid_from                TEXT,
    valid_from_precision      TEXT CHECK (valid_from_precision IN
                                ('DAY', 'WEEK', 'MONTH', 'YEAR', 'SESSION', 'BEFORE', 'UNKNOWN')),
    valid_to                  TEXT,
    valid_to_precision        TEXT CHECK (valid_to_precision IN
                                ('DAY', 'WEEK', 'MONTH', 'YEAR', 'SESSION', 'BEFORE', 'UNKNOWN')),

    -- only for point events ("we went hiking on May 3")
    event_date                TEXT,
    event_date_precision      TEXT CHECK (event_date_precision IN
                                ('DAY', 'WEEK', 'MONTH', 'YEAR', 'SESSION', 'BEFORE', 'UNKNOWN')),

    -- transaction time: when the system learned it / stopped believing it
    recorded_session_ordinal  INTEGER NOT NULL,
    recorded_turn_index       INTEGER NOT NULL,
    retired_session_ordinal   INTEGER,
    retired_turn_index        INTEGER,

    supersedes_id             TEXT REFERENCES memories(memory_id),
    superseded_by_id          TEXT REFERENCES memories(memory_id),
    conflict_set_id           TEXT,
    embedding                 BLOB
);

CREATE INDEX idx_memories_subject_attr ON memories(conversation_id, subject_entity_id, attribute_key, status);
CREATE INDEX idx_memories_status ON memories(conversation_id, status);
CREATE INDEX idx_memories_conflict ON memories(conflict_set_id) WHERE conflict_set_id IS NOT NULL;

CREATE VIRTUAL TABLE memories_fts USING fts5(
    fact_text,
    content = 'memories',
    content_rowid = 'rowid',
    tokenize = 'porter unicode61'
);

CREATE TRIGGER memories_fts_ai AFTER INSERT ON memories
BEGIN
    INSERT INTO memories_fts(rowid, fact_text) VALUES (new.rowid, new.fact_text);
END;

CREATE TRIGGER memories_fts_ad AFTER DELETE ON memories
BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, fact_text) VALUES ('delete', old.rowid, old.fact_text);
END;

CREATE TRIGGER memories_fts_au AFTER UPDATE OF fact_text ON memories
BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, fact_text) VALUES ('delete', old.rowid, old.fact_text);
    INSERT INTO memories_fts(rowid, fact_text) VALUES (new.rowid, new.fact_text);
END;

CREATE TABLE fact_sources (
    memory_id    TEXT NOT NULL REFERENCES memories(memory_id),
    turn_uid     TEXT NOT NULL REFERENCES episodes(turn_uid),   -- real FK, not a JSON list
    quote        TEXT,
    quote_start  INTEGER,
    quote_end    INTEGER,
    role         TEXT NOT NULL CHECK (role IN ('ASSERTS', 'REAFFIRMS', 'CORRECTS', 'RETRACTS')),
    UNIQUE (memory_id, turn_uid, role)
);

CREATE INDEX idx_fact_sources_turn ON fact_sources(turn_uid);

CREATE TABLE fact_entities (
    memory_id  TEXT NOT NULL REFERENCES memories(memory_id),
    entity_id  TEXT NOT NULL REFERENCES entities(entity_id),
    role       TEXT NOT NULL CHECK (role IN ('SUBJECT', 'OBJECT', 'MENTION')),
    PRIMARY KEY (memory_id, entity_id, role)
);

CREATE INDEX idx_fact_entities_entity ON fact_entities(entity_id);

------------------------------------------------------------------------------------------
-- Tier 3: answer provenance
------------------------------------------------------------------------------------------

CREATE TABLE answer_log (
    query_id              TEXT PRIMARY KEY,
    conversation_id       TEXT NOT NULL REFERENCES conversations(conversation_id),
    query                 TEXT NOT NULL,
    retrieved_memory_ids  TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(retrieved_memory_ids)),
    retrieved_turn_uids   TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(retrieved_turn_uids)),
    response_json         TEXT CHECK (response_json IS NULL OR json_valid(response_json)),
    verifier_json         TEXT CHECK (verifier_json IS NULL OR json_valid(verifier_json)),
    created_at            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
