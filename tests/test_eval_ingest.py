from dataclasses import replace

import pytest

from chronos.config import REPO_ROOT
from chronos.ingestion.files import load_conversation_file
from chronos.ingestion.scripted import ScriptedLLM
from chronos.llm.embed import HashingEmbedder
from chronos.storage.db import Database
from eval.common import ManifestData, ingestion_config_for, read_jsonl
from eval.ingest_all import ingest_manifest

DEMO = REPO_ROOT / "data" / "demo_sessions" / "demo.json"


def manifest_data(conv):
    return ManifestData(
        name="unit", path=DEMO, manifest={}, conversations={conv.conversation_id: conv}
    )


def test_ingestion_is_resumable_and_logs_every_session(tmp_path):
    conv, data = load_conversation_file(DEMO)
    llm = ScriptedLLM(conv, data["scripted"])
    db_path, log = tmp_path / "m.sqlite", tmp_path / "log.jsonl"
    partial = replace(conv, sessions=conv.sessions[:1])  # "stopped after session 1"
    first = ingest_manifest(
        manifest_data(partial),
        db_path,
        llm,
        HashingEmbedder(),
        log_path=log,
        progress=lambda m: None,
    )
    assert first["sessions_ingested"] == 1 and first["sessions_skipped"] == 0

    resumed = ingest_manifest(
        manifest_data(conv), db_path, llm, HashingEmbedder(), log_path=log, progress=lambda m: None
    )
    assert (resumed["sessions_ingested"], resumed["sessions_skipped"]) == (2, 1)
    before = len(read_jsonl(log))
    again = ingest_manifest(
        manifest_data(conv), db_path, llm, HashingEmbedder(), log_path=log, progress=lambda m: None
    )
    assert (
        again["sessions_ingested"] == 0 and len(read_jsonl(log)) == before
    )  # a finished run is a no-op

    recs = read_jsonl(log)
    sessions = [r for r in recs if r["type"] == "session"]
    assert [r["session_ordinal"] for r in sessions] == [1, 2, 3] and all(
        r["seconds"] >= 0 for r in sessions
    )
    assert all(
        {"calls", "prompt_tokens", "completion_tokens", "llm_seconds"} <= set(r["llm"])
        for r in sessions
    )
    summaries = [r for r in recs if r["type"] == "conversation"]
    assert summaries[-1]["memories"] > 0 and summaries[-1]["memories_by_status"]["SUPERSEDED"] >= 1
    assert summaries[-1]["ingest_seconds"] == pytest.approx(sum(r["seconds"] for r in sessions))
    assert summaries[-1]["entity_name_duplicates"] == []


def test_replayed_policy_ingest_goes_to_its_own_db_and_log(tmp_path):
    from chronos.ingestion.pipeline import recorded_decisions

    conv, data = load_conversation_file(DEMO)
    llm = ScriptedLLM(conv, data["scripted"])
    full = tmp_path / "full.sqlite"
    ingest_manifest(
        manifest_data(conv),
        full,
        llm,
        HashingEmbedder(),
        log_path=tmp_path / "a.jsonl",
        progress=lambda m: None,
    )
    rec = recorded_decisions(Database(full).conn)
    app = tmp_path / "app.sqlite"
    ingest_manifest(manifest_data(conv), app, llm, HashingEmbedder(), log_path=tmp_path / "b.jsonl",
                    policy="append_only", recorded=rec, variant="append_only", progress=lambda m: None)  # fmt: skip
    conn = Database(app).conn
    assert not conn.execute("SELECT 1 FROM memories WHERE status = 'SUPERSEDED'").fetchone()
    assert all(r.get("variant") == "append_only" for r in read_jsonl(tmp_path / "b.jsonl"))


def test_longmemeval_conversations_extract_user_turns_only():
    conv, _ = load_conversation_file(DEMO)
    assert ingestion_config_for(conv).target_speakers is None
    lme = replace(conv, dataset="longmemeval")
    cfg = ingestion_config_for(lme)
    assert cfg.target_speakers == ("user",) and cfg.max_prompt_turn_chars == 1500


def test_sigterm_still_records_the_interrupted_segment(tmp_path):
    """`pkill` sends SIGTERM: the step must be logged as interrupted, not vanish."""
    import signal
    import subprocess
    import sys
    import textwrap
    import time

    script = textwrap.dedent(f"""
        import time, pathlib
        import eval.common as c
        c.ARTIFACTS = pathlib.Path({str(tmp_path)!r})
        c.stop_on_sigterm()
        with c.timed_step("m", "ingest") as info:
            info["note"] = "x"
            print("ready", flush=True)
            time.sleep(60)
    """)
    p = subprocess.Popen(
        [sys.executable, "-c", script], cwd=REPO_ROOT, stdout=subprocess.PIPE, text=True
    )
    assert p.stdout.readline().strip() == "ready"
    time.sleep(0.3)
    p.send_signal(signal.SIGTERM)
    p.wait(timeout=20)
    [rec] = read_jsonl(tmp_path / "step_timings" / "m.jsonl")
    assert (
        rec["step"] == "ingest"
        and rec["status"] == "interrupted"
        and rec["note"] == "x"
        and rec["seconds"] >= 0.3
    )
