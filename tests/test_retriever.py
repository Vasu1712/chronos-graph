import pytest
from retrieval_fixtures import CONV, QDATE, ingested_demo, mem_id

from chronos.config import RetrievalConfig
from chronos.llm.embed import HashingEmbedder
from chronos.retrieval.query import analyze
from chronos.retrieval.retriever import LISTS, Candidate, Retriever, interleave_kinds, rrf


@pytest.fixture
def db(tmp_path):
    d = ingested_demo(tmp_path / "r.sqlite")
    yield d
    d.close()


def run(db, q, config=None, as_of=None):
    a = analyze(q, conn=db.conn, conversation_id=CONV, question_date=QDATE, as_of=as_of)
    return Retriever(db, HashingEmbedder(), config).retrieve(a, CONV)


def test_rrf_formula_and_tie_break():
    fused = rrf(
        {"a": [("memory", "x"), ("memory", "y")], "b": [("memory", "y"), ("episode", "z")]}, k=60
    )
    assert [c.id for c in fused] == ["y", "x", "z"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0].ranks == {"a": 2, "b": 1}
    assert fused[1].score == pytest.approx(1 / 61) and fused[2].score == pytest.approx(1 / 62)


def test_interleave_alternates_kinds_keeping_each_kinds_order():
    """Three of five lists rank memories, so plain RRF buries turns under memories."""
    fused = [Candidate("memory", m) for m in "abcd"] + [Candidate("episode", e) for e in "xy"]
    assert [c.id for c in interleave_kinds(fused)] == ["a", "x", "b", "y", "c", "d"]
    assert interleave_kinds([]) == []
    eps_only = [Candidate("episode", "x"), Candidate("episode", "y")]
    assert [c.id for c in interleave_kinds(eps_only)] == ["x", "y"]


def test_balanced_ranking_puts_a_turn_second_and_plain_rrf_is_kept_as_an_option(db):
    q = "Where does Alice live now?"
    balanced = run(db, q)
    assert [c.kind for c in balanced.fused[:2]] == ["memory", "episode"]
    plain = run(db, q, RetrievalConfig(balance_kinds=False))
    assert sorted((c.kind, c.id) for c in plain.fused) == sorted(
        (c.kind, c.id) for c in balanced.fused
    )
    assert [c.score for c in plain.fused] == sorted((c.score for c in plain.fused), reverse=True)


def test_all_lists_present_and_bounded(db):
    r = run(db, "Where does Alice live now?")
    assert set(r.lists) == set(LISTS)
    assert all(len(v) <= 50 for v in r.lists.values())
    assert r.lists["entity_memories"]  # Alice is a speaker entity linked to her memories
    assert {c.kind for c in r.fused} == {"memory", "episode"}


def test_current_filter(db):
    r = run(db, "Where does Alice live now?")
    assert mem_id(db, "Milwaukee") in r.memory_ids
    for old in ("Chicago", "Detroit", "every morning"):  # superseded, closed historical, NEG-closed
        assert mem_id(db, old) not in r.memory_ids
        assert mem_id(db, old) in r.filtered_out


def test_as_of_filter_uses_interval_overlap(db):
    r = run(db, "Where did Alice live in March 2023?")
    assert r.analysis.intent == "AS_OF"
    assert mem_id(db, "Chicago") in r.memory_ids  # 2023-02-01 (SESSION) -> 2023-10-01
    assert mem_id(db, "Milwaukee") not in r.memory_ids  # starts 2023-10-01
    assert mem_id(db, "Detroit") not in r.memory_ids  # 2019 only
    r19 = run(db, "Where did Alice live in 2019?")
    assert mem_id(db, "Detroit") in r19.memory_ids and mem_id(db, "Milwaukee") not in r19.memory_ids


def test_explicit_as_of_point_on_superseding_boundary(db):
    r = run(db, "Where does Alice live?", as_of="2023-10-01")
    assert mem_id(db, "Milwaukee") in r.memory_ids
    assert mem_id(db, "Chicago") not in r.memory_ids  # valid_to is exclusive


def test_history_keeps_versions(db):
    r = run(db, "Where did Alice live before Milwaukee?")
    assert r.analysis.intent == "HISTORY"
    assert {mem_id(db, "Chicago"), mem_id(db, "Milwaukee"), mem_id(db, "Detroit")} <= set(
        r.memory_ids
    )
    assert r.filtered_out == []


def test_flags_switch_lists_off(db):
    only_bm25_ep = RetrievalConfig(use_dense_memories=False, use_bm25_memories=False, use_dense_episodes=False,
                                   use_entity_expansion=False)  # fmt: skip
    r = run(db, "Where does Alice live now?", only_bm25_ep)
    assert set(r.lists) == {"bm25_episodes"} and r.memory_ids == []
    no_filter = run(db, "Where does Alice live now?", RetrievalConfig(use_temporal_filter=False))
    assert mem_id(db, "Chicago") in no_filter.memory_ids and no_filter.filtered_out == []


def test_entity_expansion_reaches_mentions(db):
    r = run(db, "What does Sam like doing?", RetrievalConfig(use_temporal_filter=False))
    assert [e.name for e in r.analysis.entities] == ["Sam"]
    assert mem_id(db, "brother Sam") in r.lists["entity_memories"]


def test_episode_vectors_backfilled_and_index_refreshes(db):
    n = db.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    assert (
        db.conn.execute("SELECT COUNT(*) FROM episode_embeddings").fetchone()[0] == n
    )  # written at ingest
    db.conn.execute("DELETE FROM episode_embeddings")
    ret = Retriever(db, HashingEmbedder())
    a = analyze("Where does Alice live?", conn=db.conn, conversation_id=CONV, question_date=QDATE)
    assert ret.retrieve(a, CONV).lists["dense_episodes"]
    assert (
        db.conn.execute("SELECT COUNT(*) FROM episode_embeddings").fetchone()[0] == n
    )  # backfilled, persisted
    before = len(ret._memory_index(CONV).ids)
    db.conn.execute("UPDATE memories SET embedding = embedding")  # no change -> same index object
    assert len(ret._memory_index(CONV).ids) == before


def test_first_person_question_expands_through_the_resolved_speaker(db):
    a = analyze(
        "Where do I live now?",
        conn=db.conn,
        conversation_id=CONV,
        question_date=QDATE,
        asker="Alice",
    )
    assert a.entities == [] and a.subject_speaker_name == "Alice"
    r = Retriever(db, HashingEmbedder()).retrieve(a, CONV)
    assert mem_id(db, "Milwaukee") in r.lists["entity_memories"]
