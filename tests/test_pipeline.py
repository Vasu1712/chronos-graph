import json

import pytest
from typer.testing import CliRunner

from chronos.cli import app
from chronos.config import REPO_ROOT
from chronos.ingestion.files import load_conversation_file
from chronos.ingestion.pipeline import IngestionPipeline
from chronos.ingestion.scripted import ScriptedLLM
from chronos.llm.embed import HashingEmbedder
from chronos.storage.db import Database

DEMO = REPO_ROOT / "data" / "demo_sessions" / "demo.json"


def test_cli_ingest_scripted_probe(tmp_path):
    probe = REPO_ROOT / "data" / "synthetic_probes" / "late_mention_01.json"
    res = CliRunner().invoke(
        app,
        [
            "ingest",
            str(probe),
            "--db",
            str(tmp_path / "x.sqlite"),
            "--scripted",
            "--embedder",
            "hash",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "ACTIVE     POS residence        Priya lives in Bengaluru." in res.output
    assert "Priya lived in Delhi in 2019.  [2019-01-01 (YEAR) -> 2019-12-31 (YEAR)]" in res.output
    again = CliRunner().invoke(
        app,
        [
            "ingest",
            str(probe),
            "--db",
            str(tmp_path / "x.sqlite"),
            "--scripted",
            "--embedder",
            "hash",
        ],
    )
    assert "already ingested (no-op)" in again.output


def test_pipeline_writes_sources_entities_and_embeddings(tmp_path):
    conv, data = load_conversation_file(DEMO)
    with Database(tmp_path / "d.sqlite") as db:
        IngestionPipeline(
            db, ScriptedLLM(conv, data["scripted"]), HashingEmbedder()
        ).ingest_conversation(conv)
        c = db.conn
        assert c.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == sum(
            len(s.turns) for s in conv.sessions
        )
        assert c.execute("SELECT COUNT(*) FROM memories WHERE embedding IS NULL").fetchone()[0] == 0
        # every memory has at least one source pointing at a real episode
        orphan = c.execute(
            "SELECT COUNT(*) FROM memories m WHERE NOT EXISTS (SELECT 1 FROM fact_sources s WHERE s.memory_id = m.memory_id)"
        ).fetchone()[0]
        assert orphan == 0
        # 1-hop structure: Sam is a MENTION on the running fact, and an entity with a relation
        sam = c.execute(
            "SELECT entity_id, descriptor FROM entities WHERE canonical_name = 'Sam'"
        ).fetchone()
        assert json.loads(sam["descriptor"])["relation"] == "brother"
        roles = {
            r[0]
            for r in c.execute(
                "SELECT role FROM fact_entities WHERE entity_id = ?", (sam["entity_id"],)
            )
        }
        assert roles == {"MENTION"}
        # quotes are verbatim spans of the episode text
        for q, text, a, b in c.execute(
            "SELECT s.quote, e.text, s.quote_start, s.quote_end FROM fact_sources s JOIN episodes e USING(turn_uid)"
        ):
            assert text[a:b] == q


class ExplodingLLM:
    def structured(self, messages, response_model):
        raise RuntimeError("model crashed")


def test_tier1_commits_even_if_tier2_fails(tmp_path):
    conv, data = load_conversation_file(DEMO)

    class FailOnWrite(ScriptedLLM):
        armed = False

        def structured(self, messages, response_model):
            if self.armed and response_model.__name__ == "RelationOut":
                raise RuntimeError("classifier crashed")
            return super().structured(messages, response_model)

    with Database(tmp_path / "f.sqlite") as db:
        llm = FailOnWrite(conv, data["scripted"])
        pipe = IngestionPipeline(db, llm, HashingEmbedder())
        pipe.register_conversation(conv.conversation_id, conv.dataset, conv.speakers)
        pipe.ingest_session(conv.conversation_id, conv.sessions[0])
        llm.armed = True
        with pytest.raises(RuntimeError, match="classifier crashed"):
            pipe.ingest_session(conv.conversation_id, conv.sessions[1])
        c = db.conn
        assert (
            c.execute("SELECT COUNT(*) FROM episodes WHERE session_ordinal = 2").fetchone()[0] == 3
        )
        assert (
            c.execute(
                "SELECT COUNT(*) FROM memories WHERE recorded_session_ordinal = 2"
            ).fetchone()[0]
            == 0
        )
        assert (
            c.execute("SELECT ingested_at FROM sessions WHERE session_ordinal = 2").fetchone()[0]
            is None
        )


def test_extraction_failure_is_counted_and_episodes_kept(tmp_path):
    conv, _ = load_conversation_file(DEMO)
    with Database(tmp_path / "g.sqlite") as db:
        pipe = IngestionPipeline(db, ExplodingLLM(), HashingEmbedder())
        pipe.register_conversation(conv.conversation_id, conv.dataset, conv.speakers)
        r = pipe.ingest_session(conv.conversation_id, conv.sessions[0])
        assert r.extraction.chunks_failed == 1 and r.decisions == []
        assert db.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 4
