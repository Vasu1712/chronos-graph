import csv
import subprocess
import sys

from chronos.config import REPO_ROOT
from chronos.ingestion.files import load_conversation_file
from chronos.ingestion.pipeline import IngestionPipeline
from chronos.ingestion.scripted import ScriptedLLM
from chronos.llm.embed import HashingEmbedder
from chronos.storage.db import Database


def test_audit_csv_columns_and_blank_labels(tmp_path):
    conv, data = load_conversation_file(REPO_ROOT / "data" / "demo_sessions" / "demo.json")
    db_path = tmp_path / "a.sqlite"
    with Database(db_path) as db:
        IngestionPipeline(
            db, ScriptedLLM(conv, data["scripted"]), HashingEmbedder()
        ).ingest_conversation(conv)
    out = tmp_path / "audit.csv"
    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "audit_facts.py"), "--db", str(db_path),
         "--conversation", conv.conversation_id, "--n", "5", "--out", str(out)],
        capture_output=True, text=True,
    )  # fmt: skip
    assert res.returncode == 0, res.stderr
    rows = list(csv.DictReader(out.open()))
    assert list(rows[0]) == [
        "memory_id",
        "quote",
        "fact_text",
        "attribute_key",
        "status",
        "interval",
        "correct?",
        "note",
    ]
    assert len(rows) == 5
    assert all(r["correct?"] == "" and r["note"] == "" for r in rows)  # never pre-labelled
    assert all("ASSERTS" in r["quote"] or "CORRECTS" in r["quote"] for r in rows)
