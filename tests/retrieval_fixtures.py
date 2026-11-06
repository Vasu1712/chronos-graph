"""Shared fixture: the Alice/Bob demo ingested with scripted outputs and the hashing embedder."""

from chronos.config import REPO_ROOT
from chronos.ingestion.files import load_conversation_file
from chronos.ingestion.pipeline import IngestionPipeline
from chronos.ingestion.scripted import ScriptedLLM
from chronos.llm.embed import HashingEmbedder
from chronos.storage.db import Database

DEMO = REPO_ROOT / "data" / "demo_sessions" / "demo.json"
CONV = "demo-alice-bob"
QDATE = "2023-11-10T12:00:00"


def ingested_demo(path):
    conv, data = load_conversation_file(DEMO)
    db = Database(path)
    IngestionPipeline(
        db, ScriptedLLM(conv, data["scripted"]), HashingEmbedder()
    ).ingest_conversation(conv)
    return db


def mem_id(db, contains):
    rows = db.conn.execute(
        "SELECT memory_id FROM memories WHERE fact_text LIKE ?", (f"%{contains}%",)
    ).fetchall()
    assert len(rows) == 1, (contains, rows)
    return rows[0][0]
