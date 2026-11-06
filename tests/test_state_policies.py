"""State policies (ablation B): replaying recorded classifier decisions under another policy."""

import json

import pytest
from db_dump import dump

from chronos.config import REPO_ROOT
from chronos.ingestion.files import conversation_from_dict, load_conversation_file
from chronos.ingestion.pipeline import IngestionPipeline, recorded_decisions
from chronos.ingestion.scripted import ScriptedLLM
from chronos.llm.embed import HashingEmbedder
from chronos.storage.db import Database

PROBES = REPO_ROOT / "data" / "synthetic_probes"
DEMO = REPO_ROOT / "data" / "demo_sessions" / "demo.json"


class NoLLM:
    """A replay must never ask a model anything."""

    def structured(self, messages, model):
        raise AssertionError(f"unexpected LLM call for {model.__name__}")


def ingest(path, *, policy="bitemporal", recorded=None, llm=None, probe=None):
    data = json.loads(probe.read_text()) if probe else None
    conv = conversation_from_dict(data) if data else load_conversation_file(DEMO)[0]
    scripted = (data or json.loads(DEMO.read_text()))["scripted"]
    db = Database(path)
    llm = llm or ScriptedLLM(conv, scripted)
    IngestionPipeline(
        db, llm, HashingEmbedder(), policy=policy, recorded=recorded
    ).ingest_conversation(conv)
    return db, conv


def statuses(db):
    return {
        r[0]: (r[1], r[2])
        for r in db.conn.execute("SELECT fact_text, status, valid_to FROM memories")
    }


def test_resolver_log_records_classifier_and_applied_relation(tmp_path):
    db, conv = ingest(tmp_path / "full.sqlite")
    rows = {r["memory_id"]: dict(r) for r in db.conn.execute("SELECT * FROM resolver_log")}
    n_memories = db.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert len(rows) >= n_memories and all(
        r["conversation_id"] == conv.conversation_id for r in rows.values()
    )
    upd = [
        r
        for r in rows.values()
        if r["llm_relation"] == "UPDATES" and r["applied_relation"] == "UPDATES"
    ]
    assert upd and all(r["llm_target_id"] and r["applied_target_id"] for r in upd)
    first = db.conn.execute(
        "SELECT * FROM resolver_log ORDER BY session_ordinal, turn_index LIMIT 1"
    ).fetchone()
    assert (
        first["llm_relation"] is None and first["applied_relation"] == "NEW"
    )  # no candidates, no LLM call


def test_replaying_the_same_policy_reproduces_the_database_without_any_llm_call(tmp_path):
    a, conv = ingest(tmp_path / "a.sqlite")
    rec = recorded_decisions(a.conn)
    scripted_extraction = ScriptedLLM(conv, json.loads(DEMO.read_text())["scripted"])

    class ExtractionOnly:
        def structured(self, messages, model):
            if model.__name__ == "RelationOut":
                raise AssertionError("replay asked the classifier")
            return scripted_extraction.structured(messages, model)

    b, _ = ingest(tmp_path / "b.sqlite", recorded=rec, llm=ExtractionOnly())
    assert dump(a.conn) == dump(b.conn)
    assert any(v[0] for v in rec.values())  # the demo does contain classifier decisions


def current(db):
    return sorted(
        r[0]
        for r in db.conn.execute(
            "SELECT fact_text FROM memories WHERE attribute_key IS NOT NULL AND status = 'ACTIVE' "
            "AND valid_to IS NULL AND polarity = 'POS'"
        )
    )


def replay(tmp_path, probe, policy, rec):
    path = PROBES / f"{probe}.json"
    return ingest(
        tmp_path / f"{policy}.sqlite",
        policy=policy,
        recorded=rec,
        llm=NoLLMExtraction(path),
        probe=path,
    )[0]


def test_arrival_order_loses_when_event_time_disagrees_with_arrival(tmp_path):
    full, _ = ingest(tmp_path / "full.sqlite", probe=PROBES / "out_of_order_02.json")
    rec = recorded_decisions(full.conn)
    assert current(full) == [
        "Owen works at Fairway Logistics."
    ]  # told later, but it started in 2020
    arrival = replay(tmp_path, "out_of_order_02", "arrival_order", rec)
    assert current(arrival) == ["Owen started working at Summit Foods."]  # last mentioned wins


def test_rule_7_protects_the_current_fact_only_with_event_time(tmp_path):
    """The live classifier often answers UPDATES for 'back in 2019 I lived in Delhi'."""
    full, _ = ingest(tmp_path / "full.sqlite", probe=PROBES / "late_mention_01.json")
    rec = recorded_decisions(full.conn)
    delhi = next(k for k, v in rec.items() if v[0] == "HISTORICAL")
    rec[delhi] = ("UPDATES", rec[delhi][1])  # what the classifier said in the live run
    bitemporal = replay(tmp_path, "late_mention_01", "bitemporal", rec)
    arrival = replay(tmp_path, "late_mention_01", "arrival_order", rec)
    assert current(bitemporal) == ["Priya lives in Bengaluru."]
    assert (
        current(arrival) == []
    )  # an ended state replaced the current one: no current residence left
    bengaluru = arrival.conn.execute(
        "SELECT status FROM memories WHERE fact_text LIKE '%Bengaluru%'"
    ).fetchone()
    assert bengaluru[0] == "SUPERSEDED"


@pytest.mark.parametrize("probe", ["update_01", "out_of_order_02"])
def test_append_only_never_supersedes_so_old_values_stay_current(tmp_path, probe):
    full, _ = ingest(tmp_path / "full.sqlite", probe=PROBES / f"{probe}.json")
    app = replay(tmp_path, probe, "append_only", recorded_decisions(full.conn))
    assert len(current(full)) == 1 and len(current(app)) == 2
    assert not app.conn.execute(
        "SELECT 1 FROM memories WHERE status IN ('SUPERSEDED', 'CONFLICTED')"
    ).fetchone()


class NoLLMExtraction:
    """Extraction from the scripted outputs; the classifier is forbidden (decisions are replayed)."""

    def __init__(self, probe_path):
        data = json.loads(probe_path.read_text())
        self.inner = ScriptedLLM(conversation_from_dict(data), data["scripted"])

    def structured(self, messages, model):
        if model.__name__ == "RelationOut":
            raise AssertionError("replay asked the classifier")
        return self.inner.structured(messages, model)


def test_append_only_keeps_corrections_and_duplicates_but_not_updates(tmp_path):
    full, conv = ingest(tmp_path / "full.sqlite", probe=PROBES / "correction_01.json")
    app, _ = ingest(tmp_path / "app.sqlite", policy="append_only", recorded=recorded_decisions(full.conn),
                    llm=NoLLMExtraction(PROBES / "correction_01.json"), probe=PROBES / "correction_01.json")  # fmt: skip
    assert statuses(full) == statuses(app)  # a correction is a retraction, which append-only keeps
    assert {v[0] for v in statuses(app).values()} == {"RETRACTED", "ACTIVE"}


def test_replay_with_missing_record_inserts_new(tmp_path):
    path = PROBES / "update_01.json"
    full, _ = ingest(tmp_path / "full.sqlite", probe=path)
    arr, _ = ingest(
        tmp_path / "arr.sqlite",
        policy="arrival_order",
        recorded={},
        llm=NoLLMExtraction(path),
        probe=path,
    )
    assert all(
        v[0] == "ACTIVE" for v in statuses(arr).values()
    )  # no recorded UPDATES -> nothing superseded
