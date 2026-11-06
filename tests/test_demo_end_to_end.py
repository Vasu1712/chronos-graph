"""Live end-to-end demo (Part 3 acceptance): real Qwen2.5-7B via Ollama, bge, DeBERTa NLI.

Each assertion is its own test so a failure names exactly what is wrong. Latency per question and
per step is measured on the demo questions and written to artifacts/demo_latency.json.
"""

import json
import urllib.request
from datetime import date

import pytest

from chronos.config import REPO_ROOT

pytestmark = pytest.mark.live

STORY = REPO_ROOT / "data" / "demo_sessions" / "user_story.json"
CONV = "demo-meera"
LATENCY_QUESTIONS = [
    "Where do I live now?",
    "Where did I live before Bengaluru?",
    "Where did I live in 2019?",
    "What does our auth service use now?",
    "Who is my manager?",
    "What have I been reading lately?",
    "What does my sister do?",
]


def _ollama_up() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    if not _ollama_up():
        pytest.skip("Ollama is not running")
    from chronos.agent.memory_agent import MemoryAgent
    from chronos.config import Paths
    from chronos.ingestion.files import load_conversation_file
    from chronos.ingestion.pipeline import IngestionPipeline
    from chronos.llm.cache import ResponseCache
    from chronos.llm.embed import Embedder
    from chronos.llm.local_client import LocalLLM
    from chronos.llm.nli import NLI
    from chronos.storage.db import Database

    db = Database(tmp_path_factory.mktemp("demo") / "demo.sqlite")
    llm, emb = LocalLLM(cache=ResponseCache(Paths().llm_cache)), Embedder()
    conv, _ = load_conversation_file(STORY)
    IngestionPipeline(db, llm, emb).ingest_conversation(conv)
    agent = MemoryAgent(db, llm, emb, NLI())
    yield db, agent
    db.close()


def memories(db, contains, exclude=None, key=None):
    rows = db.conn.execute(
        "SELECT * FROM memories WHERE conversation_id = ? ORDER BY rowid", (CONV,)
    ).fetchall()
    out = [r for r in rows if contains.lower() in r["fact_text"].lower()]
    if exclude:
        out = [r for r in out if exclude.lower() not in r["fact_text"].lower()]
    if key:
        out = [r for r in out if r["attribute_key"] == key]
    return out


def table(db):
    return "\n".join(
        f"{r['status']:<10} {r['attribute_key'] or '-':<12} {r['fact_text']} "
        f"[{r['valid_from']} ({r['valid_from_precision']}) -> {r['valid_to']} ({r['valid_to_precision']})]"
        for r in db.conn.execute(
            "SELECT * FROM memories WHERE conversation_id = ? ORDER BY rowid", (CONV,)
        )
    )


def current_residence(db):
    return [r for r in memories(db, "", key="residence") if r["status"] == "ACTIVE" and r["valid_to"] is None
            and r["polarity"] == "POS"]  # fmt: skip


def test_current_residence_is_bengaluru_from_about_may_5(demo):
    db, _ = demo
    cur = current_residence(db)
    assert [("bengaluru" in r["fact_text"].lower()) for r in cur] == [True], table(db)
    vf = cur[0]["valid_from"]
    assert vf and abs((date.fromisoformat(vf) - date(2026, 5, 5)).days) <= 3, table(db)


def test_pune_is_superseded(demo):
    db, _ = demo
    pune = memories(db, "Pune")
    assert pune and all(r["status"] == "SUPERSEDED" for r in pune), table(db)


def test_delhi_has_closed_2019_interval_and_bengaluru_stays_current(demo):
    db, _ = demo
    delhi = memories(db, "Delhi")
    assert any(r["valid_from"] == "2019-01-01" and r["valid_to"] == "2019-12-31" for r in delhi), (
        table(db)
    )
    assert [("bengaluru" in r["fact_text"].lower()) for r in current_residence(db)] == [True], (
        table(db)
    )


def test_priya_is_retracted(demo):
    db, _ = demo
    priya = memories(db, "Priya", exclude="Priyanka")
    assert priya and all(r["status"] == "RETRACTED" for r in priya), table(db)


def test_sister_question_abstains(demo):
    db, agent = demo
    ans = agent.answer(CONV, "What does my sister do?")
    assert ans.abstained, (ans.final_answer, [c.text for c in ans.claims])


def test_latency_per_question_and_step(demo, stats):
    """Measured with an *uncached* LLM client so every question pays real model time."""
    from chronos.agent.memory_agent import MemoryAgent
    from chronos.llm.local_client import LocalLLM

    db, cached = demo
    agent = MemoryAgent(db, LocalLLM(cache=None), cached.retriever.embedder, cached.verifier.nli)
    agent.answer(
        CONV, "warm-up: is the model loaded?"
    )  # exclude the one-off model load from the timings
    steps = ("analyze", "retrieve", "pack", "gate", "generate", "verify", "regenerate", "total")
    rows = []
    for q in LATENCY_QUESTIONS:
        ans = agent.answer(CONV, q)
        rows.append({"question": q, "answer": ans.final_answer, "abstained": ans.abstained,
                     "abstained_at": ans.abstained_at, "llm_calls": ans.llm_calls,
                     "timings": {s: ans.timings.get(s, 0.0) for s in steps}})  # fmt: skip
    mean = {s: sum(r["timings"][s] for r in rows) / len(rows) for s in steps}
    out = REPO_ROOT / "artifacts" / "demo_latency.json"
    out.write_text(
        json.dumps({"llm_cache": "disabled", "questions": rows, "mean_seconds": mean}, indent=1)
        + "\n"
    )
    stats.append("demo latency (mean s): " + "  ".join(f"{s} {v:.2f}" for s, v in mean.items()))
    for r in rows:
        stats.append(f"  {r['timings']['total']:.2f}s  {r['question']} -> {r['answer']}"
                     + (f" [abstained at {r['abstained_at']}]" if r["abstained"] else ""))  # fmt: skip
    assert len(rows) == len(LATENCY_QUESTIONS)
