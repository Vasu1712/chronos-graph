import sqlite3

import pytest

from chronos.storage.db import Database, ImmutableEpisodeError, make_turn_uid

EXPECTED_TABLES = {
    "conversations",
    "sessions",
    "episodes",
    "episodes_fts",
    "entities",
    "entity_aliases",
    "memories",
    "memories_fts",
    "fact_sources",
    "fact_entities",
    "answer_log",
    "schema_migrations",
    "episode_embeddings",
}


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "t.sqlite")
    yield d
    d.close()


def _seed(db: Database) -> str:
    db.add_conversation("conv-1", "locomo", participants=["Caroline", "Melanie"])
    db.add_session("conv-1:s01", "conv-1", 1, "2023-05-08T13:56:00")
    db.add_entity("conv-1:e:caroline", "conv-1", "Caroline", entity_type="PERSON", is_speaker=True)
    uid = make_turn_uid("locomo", "conv-1", 1, 0)
    db.append_episode(
        {
            "turn_uid": uid,
            "conversation_id": "conv-1",
            "session_id": "conv-1:s01",
            "session_ordinal": 1,
            "turn_index": 0,
            "speaker": "Caroline",
            "speaker_entity_id": "conv-1:e:caroline",
            "session_ts": "2023-05-08T13:56:00",
            "text": "I went to an LGBTQ support group yesterday and it was powerful.",
            "source_ref": "D1:1",
        }
    )
    return uid


def test_turn_uid_format():
    assert make_turn_uid("locomo", "conv-26", 3, 7) == "locomo:conv-26:s03:t007"


def test_schema_creates(db):
    names = {
        r[0] for r in db.conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table')")
    }
    assert names >= EXPECTED_TABLES
    assert db.schema_version() == 2
    assert db.schema_version() == 1
    assert db.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert db.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_migrate_is_idempotent(tmp_path):
    p = tmp_path / "m.sqlite"
    Database(p).close()
    d = Database(p)
    assert d.migrate() == 2
    assert d.conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 2
    assert d.migrate() == 1
    assert d.conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 1
    d.close()


def test_fk_rejects_fact_source_with_missing_turn(db):
    _seed(db)
    db.conn.execute(
        "INSERT INTO memories(memory_id, conversation_id, subject_entity_id, fact_text, "
        "recorded_session_ordinal, recorded_turn_index) VALUES (?, ?, ?, ?, ?, ?)",
        ("m1", "conv-1", "conv-1:e:caroline", "Caroline attended an LGBTQ support group.", 1, 0),
    )
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        db.conn.execute(
            "INSERT INTO fact_sources(memory_id, turn_uid, quote, role) VALUES (?, ?, ?, ?)",
            ("m1", "locomo:conv-1:s01:t999", "nope", "ASSERTS"),
        )


def test_fts5_returns_inserted_episode(db):
    uid = _seed(db)
    rows = db.search_episodes_fts("support group")
    assert [r["turn_uid"] for r in rows] == [uid]
    # porter stemming: "groups" matches "group"
    assert db.search_episodes_fts("groups", conversation_id="conv-1")[0]["turn_uid"] == uid
    assert db.search_episodes_fts("hiking") == []


def test_episodes_are_immutable(db):
    uid = _seed(db)
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        db.conn.execute("UPDATE episodes SET text = 'x' WHERE turn_uid = ?", (uid,))
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        db.conn.execute("DELETE FROM episodes WHERE turn_uid = ?", (uid,))
    row = dict(db.conn.execute("SELECT * FROM episodes WHERE turn_uid = ?", (uid,)).fetchone())
    assert db.append_episode(row) is False  # identical re-append is a no-op
    with pytest.raises(ImmutableEpisodeError):
        db.append_episode({**row, "text": "changed"})


def test_session_transaction_rolls_back_on_error(db):
    _seed(db)
    with pytest.raises(RuntimeError), db.session_transaction("conv-1:s01") as tx:
        tx.conn.execute(
            "INSERT INTO memories(memory_id, conversation_id, subject_entity_id, fact_text, "
            "recorded_session_ordinal, recorded_turn_index) VALUES ('m2','conv-1',"
            "'conv-1:e:caroline','x',1,0)"
        )
        raise RuntimeError("boom")
    assert db.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
    ingested = db.conn.execute("SELECT ingested_at FROM sessions").fetchone()[0]
    assert ingested is None


def test_session_transaction_marks_ingested_and_reingest_is_flagged(db):
    _seed(db)
    with db.session_transaction("conv-1:s01") as tx:
        assert tx.already_ingested is False
    with db.session_transaction("conv-1:s01") as tx:
        assert tx.already_ingested is True


def test_memories_fts_tracks_inserts(db):
    _seed(db)
    db.conn.execute(
        "INSERT INTO memories(memory_id, conversation_id, subject_entity_id, fact_text, "
        "recorded_session_ordinal, recorded_turn_index) VALUES (?, ?, ?, ?, ?, ?)",
        ("m1", "conv-1", "conv-1:e:caroline", "Caroline lives in Mumbai.", 1, 0),
    )
    hit = db.conn.execute(
        "SELECT m.memory_id FROM memories_fts JOIN memories m ON m.rowid = memories_fts.rowid "
        "WHERE memories_fts MATCH 'mumbai'"
    ).fetchall()
    assert [r[0] for r in hit] == ["m1"]


def test_enum_checks(db):
    _seed(db)
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        db.conn.execute(
            "INSERT INTO memories(memory_id, conversation_id, subject_entity_id, fact_text, "
            "status, recorded_session_ordinal, recorded_turn_index) "
            "VALUES ('m3','conv-1','conv-1:e:caroline','x','MAYBE',1,0)"
        )


def test_v1_database_upgrades_to_v2(tmp_path):
    import sqlite3

    from chronos.storage.db import MIGRATIONS, _split_sql

    p = tmp_path / "old.sqlite"
    conn = sqlite3.connect(p)
    conn.execute(
        "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT)"
    )
    for stmt in _split_sql(MIGRATIONS[0][2]):
        conn.execute(stmt)
    conn.execute("INSERT INTO schema_migrations(version, name) VALUES (1, 'initial_schema')")
    conn.commit()
    conn.close()
    d = Database(p)
    assert d.schema_version() == 2
    assert d.conn.execute(
        "SELECT name FROM sqlite_master WHERE name = 'episode_embeddings'"
    ).fetchone()
    d.close()
