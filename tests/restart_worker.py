"""Subprocess for test_persistence_restart: ingest the demo conversation into a database.

usage: restart_worker.py DB_PATH [PAUSE_AT_SESSION]
With PAUSE_AT_SESSION, the worker blocks inside that session's Tier 2 transaction (at its second
classifier call, i.e. after at least one memory was written) and prints "IN_TX <n> <writes>".
"""

import sys
import time

from chronos.config import REPO_ROOT
from chronos.ingestion.files import load_conversation_file
from chronos.ingestion.pipeline import IngestionPipeline
from chronos.ingestion.scripted import ScriptedLLM
from chronos.llm.embed import HashingEmbedder
from chronos.storage.db import Database


def main() -> None:
    db_path = sys.argv[1]
    pause_at = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    conv, data = load_conversation_file(REPO_ROOT / "data" / "demo_sessions" / "demo.json")
    db = Database(db_path)

    class PausingLLM(ScriptedLLM):
        session = 0
        classify_in_session = 0

        def structured(self, messages, response_model):
            if response_model.__name__ == "RelationOut" and self.session == pause_at:
                self.classify_in_session += 1
                if self.classify_in_session == 2:
                    writes = db.conn.execute(
                        "SELECT COUNT(*) FROM memories WHERE recorded_session_ordinal = ?",
                        (pause_at,),
                    ).fetchone()[0]
                    print(f"IN_TX {pause_at} {writes}", flush=True)
                    time.sleep(60)
            return super().structured(messages, response_model)

    llm = PausingLLM(conv, data["scripted"])
    pipe = IngestionPipeline(db, llm, HashingEmbedder())
    pipe.register_conversation(conv.conversation_id, conv.dataset, conv.speakers)
    for s in conv.sessions:
        llm.session = s.ordinal
        pipe.ingest_session(conv.conversation_id, s)
        print(f"DONE {s.ordinal}", flush=True)
    db.close()


if __name__ == "__main__":
    main()
