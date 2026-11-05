"""kill -9 mid-ingestion, restart, re-run: no partial session, intact episodes, same final DB."""

import os
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path

from db_dump import dump

WORKER = Path(__file__).with_name("restart_worker.py")


def run_worker(db: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(WORKER), str(db), *args], capture_output=True, text=True, timeout=120
    )


def test_kill_mid_session_then_restart_equals_uninterrupted(tmp_path):
    killed_db, clean_db = tmp_path / "killed.sqlite", tmp_path / "clean.sqlite"

    proc = subprocess.Popen(
        [sys.executable, str(WORKER), str(killed_db), "2"], stdout=subprocess.PIPE, text=True
    )
    lines = []
    for line in proc.stdout:
        lines.append(line.strip())
        if line.startswith("IN_TX"):
            break
    assert lines[-1].startswith("IN_TX 2"), lines
    assert int(lines[-1].split()[2]) >= 1  # the transaction had already written a memory
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait(timeout=10)
    assert proc.returncode == -signal.SIGKILL
    assert "DONE 1" in lines and "DONE 2" not in lines

    # --- state after the crash
    conn = sqlite3.connect(killed_db)
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    ingested = dict(conn.execute("SELECT session_ordinal, ingested_at IS NOT NULL FROM sessions"))
    assert ingested == {1: 1, 2: 0}  # session 1 committed, session 2 not
    assert (
        conn.execute("SELECT COUNT(*) FROM memories WHERE recorded_session_ordinal = 2").fetchone()[
            0
        ]
        == 0
    )
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM fact_sources s JOIN episodes e USING(turn_uid) WHERE e.session_ordinal = 2"
        ).fetchone()[0]
        == 0
    )
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM entities WHERE canonical_name IN ('Denver', 'Brightwave')"
        ).fetchone()[0]
        == 0
    )
    eps = dict(
        conn.execute("SELECT session_ordinal, COUNT(*) FROM episodes GROUP BY session_ordinal")
    )
    assert eps == {1: 4, 2: 3}  # Tier 1 of session 2 had committed before the kill
    conn.close()

    # --- restart and re-run to completion
    resumed = run_worker(killed_db)
    assert resumed.returncode == 0, resumed.stderr
    clean = run_worker(clean_db)
    assert clean.returncode == 0, clean.stderr

    a, b = sqlite3.connect(killed_db), sqlite3.connect(clean_db)
    assert dump(a) == dump(b)
    assert dict(a.execute("SELECT session_ordinal, ingested_at IS NOT NULL FROM sessions")) == {
        1: 1,
        2: 1,
        3: 1,
    }
