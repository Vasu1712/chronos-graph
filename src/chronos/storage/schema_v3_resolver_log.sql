-- Migration 3: what the state resolver decided for each extracted fact.
-- Used to replay ingestion under other state policies (append-only, arrival-order) without any new
-- LLM call: the classifier's raw answer is recorded next to what was applied. memory_id is the id the
-- fact has or would have had; it is not a foreign key because DUPLICATE facts insert no memory.
CREATE TABLE resolver_log (
    memory_id        TEXT PRIMARY KEY,
    conversation_id  TEXT NOT NULL,
    session_ordinal  INTEGER NOT NULL,
    turn_index       INTEGER NOT NULL,
    llm_relation     TEXT,            -- classifier's relation; NULL if no classifier call was made
    llm_target_id    TEXT,            -- memory_id of the candidate the classifier named, if any
    applied_relation TEXT NOT NULL,   -- relation after the deterministic normalization
    applied_target_id TEXT,
    n_candidates     INTEGER NOT NULL,
    notes            TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(notes))
);

CREATE INDEX idx_resolver_log_conv ON resolver_log(conversation_id, session_ordinal, turn_index);
