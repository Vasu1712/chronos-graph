"""Ingesting the same session twice changes nothing (SPEC §5: re-ingest is a no-op)."""

from db_dump import dump

from chronos.config import REPO_ROOT
from chronos.ingestion.files import load_conversation_file
from chronos.ingestion.pipeline import IngestionPipeline
from chronos.ingestion.scripted import ScriptedLLM
from chronos.llm.embed import HashingEmbedder
from chronos.storage.db import Database

DEMO = REPO_ROOT / "data" / "demo_sessions" / "demo.json"


def test_ingesting_a_session_twice_changes_nothing(tmp_path):
    conv, data = load_conversation_file(DEMO)
    with Database(tmp_path / "i.sqlite") as db:
        llm = ScriptedLLM(conv, data["scripted"])
        pipe = IngestionPipeline(db, llm, HashingEmbedder())
        pipe.ingest_conversation(conv)
        before = dump(db.conn, keep_timestamps=True)
        calls = (llm.extraction_calls, llm.classification_calls)

        results = pipe.ingest_conversation(conv)  # every session again
        assert all(r.skipped and r.episodes_added == 0 for r in results)
        r = pipe.ingest_session(conv.conversation_id, conv.sessions[1])  # one session again
        assert r.skipped
        assert dump(db.conn, keep_timestamps=True) == before  # byte-identical, timestamps included
        assert (llm.extraction_calls, llm.classification_calls) == calls  # no model calls


def test_fresh_process_reingest_is_noop(tmp_path):
    conv, data = load_conversation_file(DEMO)
    path = tmp_path / "j.sqlite"
    with Database(path) as db:
        IngestionPipeline(
            db, ScriptedLLM(conv, data["scripted"]), HashingEmbedder()
        ).ingest_conversation(conv)
        before = dump(db.conn, keep_timestamps=True)
    with Database(path) as db:  # "restart": new connection and pipeline
        llm = ScriptedLLM(conv, data["scripted"])
        IngestionPipeline(db, llm, HashingEmbedder()).ingest_conversation(conv)
        assert dump(db.conn, keep_timestamps=True) == before
        assert llm.extraction_calls == 0
