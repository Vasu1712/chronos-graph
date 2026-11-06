import re

import pytest
from retrieval_fixtures import CONV, QDATE, ingested_demo, mem_id

from chronos.config import RetrievalConfig
from chronos.llm.embed import HashingEmbedder
from chronos.retrieval.packer import Packer, qwen_token_count
from chronos.retrieval.query import analyze
from chronos.retrieval.retriever import Retriever

WORDS = lambda s: len(s.split())  # noqa: E731 - cheap stand-in tokenizer for unit tests
TURN_LINE = re.compile(r"^\s+\[([^ |]+) \| (\d{4}-\d{2}-\d{2}) \| ([^\]]+)\] ")


@pytest.fixture
def db(tmp_path):
    d = ingested_demo(tmp_path / "p.sqlite")
    yield d
    d.close()


def packed(db, q, budget=1500, counter=WORDS, as_of=None, config=None):
    a = analyze(q, conn=db.conn, conversation_id=CONV, question_date=QDATE, as_of=as_of)
    r = Retriever(db, HashingEmbedder(), config).retrieve(a, CONV)
    return Packer(db, config, count_tokens=counter).pack(r, budget)


def test_turn_prefix_grouping_and_exact_id_lists(db):
    p = packed(db, "Where does Alice live now?")
    lines = p.text.splitlines()
    turn_lines = [ln for ln in lines if TURN_LINE.match(ln)]
    assert turn_lines and all(TURN_LINE.match(ln).group(3) in ("Alice", "Bob") for ln in turn_lines)
    printed = [TURN_LINE.match(ln).group(1) for ln in turn_lines]
    assert printed == p.turn_uids and len(set(printed)) == len(printed)  # exact list, deduplicated
    shown_mems = re.findall(r"^- \(memory ([^)]+)\)", p.text, re.MULTILINE)
    assert set(shown_mems) <= set(p.memory_ids)
    assert lines[0] == "## Alice"
    assert mem_id(db, "Milwaukee") in p.memory_ids and mem_id(db, "Chicago") not in p.memory_ids


def test_memory_line_shows_status_interval_precision(db):
    p = packed(db, "Where does Alice live now?")
    line = next(
        ln for ln in p.text.splitlines() if "Milwaukee." in ln and ln.startswith("- (memory ")
    )
    assert (
        "| ACTIVE, residence | from 2023-10-01 (MONTH) to now | sources: synthetic:demo-alice-bob:s03:t002"
        in line
    )


def test_source_turn_window(db):
    p = packed(db, "Where does Alice live now?")
    # Milwaukee's source is s03:t002; the +-1 window adds t001 and t003
    for t in ("s03:t001", "s03:t002", "s03:t003"):
        assert f"synthetic:demo-alice-bob:{t}" in p.turn_uids
    cfg = RetrievalConfig(episode_window=0)
    a = analyze(
        "Where does Alice live now?", conn=db.conn, conversation_id=CONV, question_date=QDATE
    )
    r = Retriever(db, HashingEmbedder(), cfg).retrieve(a, CONV)
    narrow = Packer(db, cfg, count_tokens=WORDS).pack(r)
    sources = {u for (u,) in db.conn.execute("SELECT turn_uid FROM fact_sources")}
    assert set(narrow.turn_uids) <= sources | set(r.turn_uids)  # no window turns without a reason
    mil = mem_id(db, "Milwaukee")
    one = Packer(db, RetrievalConfig(episode_window=1), count_tokens=WORDS)._memory_block(
        mil, a.intent
    )
    zero = Packer(db, cfg, count_tokens=WORDS)._memory_block(mil, a.intent)
    assert [t.turn_uid[-8:] for t in zero.turns] == ["s03:t002"]
    assert [t.turn_uid[-8:] for t in one.turns] == ["s03:t001", "s03:t002", "s03:t003"]


def test_history_intent_shows_chain(db):
    p = packed(db, "Where did Alice live before Milwaukee?")
    hist = [ln for ln in p.text.splitlines() if ln.strip().startswith("history:")]
    assert any(
        "Chicago" in h and "Milwaukee" in h and h.index("Chicago") < h.index("Milwaukee")
        for h in hist
    )
    assert mem_id(db, "Chicago") in p.memory_ids
    cur = packed(db, "Where does Alice live now?")
    assert "history:" not in cur.text  # CURRENT does not show chains


def test_budget_is_respected_and_truncation_flagged(db):
    full = packed(db, "Tell me about Alice", budget=10_000)
    small = packed(db, "Tell me about Alice", budget=60)
    assert small.tokens <= 60 < full.tokens
    assert small.truncated and not full.truncated
    assert set(small.memory_ids) <= set(full.memory_ids)
    nothing = packed(db, "Tell me about Alice", budget=1)
    assert nothing.text == "" and nothing.memory_ids == [] and nothing.turn_uids == []


def test_real_qwen_tokenizer_budget(db):
    try:
        qwen_token_count("hello")
    except Exception as exc:  # pragma: no cover - offline without the tokenizer files
        pytest.skip(f"Qwen tokenizer unavailable: {exc}")
    p = packed(db, "Where does Alice live now?", budget=300, counter=None)
    assert 0 < p.tokens <= 300 and qwen_token_count(p.text) == p.tokens
