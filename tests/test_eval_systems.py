"""Systems B1-B4, ChronosGraph and the benchmark runner, offline (fake LLM and NLI)."""

import re

import pytest
from retrieval_fixtures import CONV, ingested_demo

import eval.common as common
from chronos.agent.answerability import GateOut
from chronos.agent.generator import GenerationOut
from chronos.config import REPO_ROOT
from chronos.ingestion.files import load_conversation_file
from chronos.llm.embed import HashingEmbedder
from chronos.llm.local_client import CallRecord
from chronos.llm.nli import NLIScore
from eval.common import ManifestData, read_jsonl
from eval.loaders.types import QAItem
from eval.run_benchmark import RunSpec, plan, run_system
from eval.systems import Runtime, build_system
from eval.systems import baselines as baselines_mod
from eval.systems.common import Skipped

TURN = re.compile(r"\[([^ |\]]+) \| [0-9-]+ \| [^\]]+\] (.*)")
CONV_ID = CONV  # "demo-alice-bob"


class FakeLLM:
    """Gate: ANSWERABLE. Generator: cites the first context line mentioning the keyword, else abstains."""

    def __init__(self, keyword="Milwaukee"):
        self.keyword, self.records, self.prompts = keyword, [], []

    def structured(self, messages, model):
        self.records.append(CallRecord(model.__name__, 100, 20, 1.5, 0.001, False))
        user = messages[-1]["content"]
        self.prompts.append(user)
        if model is GateOut:
            return GateOut(
                reason="r", verdict="ANSWERABLE", missing_detail=None, supporting_turn_uids=[]
            )
        hits = [
            m.group(1) for m in TURN.finditer(user) if self.keyword.lower() in m.group(2).lower()
        ]
        if not hits:
            return GenerationOut(
                final_answer="I don't know.", claims=[], abstained=True, abstention_reason="none"
            )
        return GenerationOut(final_answer="Milwaukee", abstained=False, abstention_reason=None,
                             claims=[{"text": "Alice moved to Milwaukee.", "cited_turn_uids": [hits[0]], "cited_memory_ids": []}])  # fmt: skip


class YesNLI:
    def score(self, pairs):
        return [NLIScore(0.9, 0.05, 0.05) for _ in pairs]


def qa(i, question, answer, *, dataset="locomo", unanswerable=False, evidence=()):
    return QAItem(dataset, f"q{i}", CONV_ID, question, answer, "single-hop", tuple(evidence),
                  (3,) if evidence else (), unanswerable)  # fmt: skip


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ARTIFACTS", tmp_path / "artifacts")
    db = ingested_demo(tmp_path / "m.sqlite")
    conv, _ = load_conversation_file(REPO_ROOT / "data" / "demo_sessions" / "demo.json")
    uid = "synthetic:demo-alice-bob:s03:t002"
    data = ManifestData("unit", tmp_path / "m.json", {}, {CONV_ID: conv}, [
        qa(1, "Where does Alice live now?", "Milwaukee", evidence=[uid]),
        qa(2, "Where did Alice work first?", "Lumen Health"),
        qa(3, "What is Bob's job?", "n/a", unanswerable=True),
    ])  # fmt: skip
    llm = FakeLLM()
    rt = Runtime(
        data=data, llm=llm, embedder=HashingEmbedder(), nli=YesNLI(), open_db=lambda variant: db
    )
    yield rt, data, llm
    db.close()


@pytest.mark.parametrize(
    "name",
    ["chronos", "B1_sliding_window", "B3_chunk_rag", "B4_turns_plus_notes", "B5_append_only"],
)
def test_every_system_returns_a_complete_record(env, name):
    rt, data, llm = env
    rec = build_system(name, rt).answer(data.qas[0])
    assert (
        rec["system"] == name and rec["question_id"] == "q1" and rec["gold_answer"] == "Milwaukee"
    )
    assert {
        "final_answer",
        "abstained",
        "claims",
        "verifier",
        "context",
        "llm_calls",
        "timings",
        "latency_s",
    } <= set(rec)
    ctx = rec["context"]
    assert ctx["tokens"] <= 1500 and rec["gold_evidence_turn_uids"] == [
        "synthetic:demo-alice-bob:s03:t002"
    ]
    assert all(
        re.match(r"\[synthetic:", ln) or ln.startswith(("##", "-", " "))
        for ln in ctx["text"].splitlines()
        if ln
    )
    assert rec["llm_calls"] and rec["llm_calls"][0]["prompt_tokens"] == 100
    assert rec["latency_s"] == pytest.approx(
        rec["timings"]["total"] - 0.001 * len(rec["llm_calls"]) + 1.5 * len(rec["llm_calls"]),
        abs=0.05,
    )
    if not rec[
        "abstained"
    ]:  # whatever it cites passed the shared verifier: exists, was in context, entailed
        assert all(u in ctx["turn_uids"] for c in rec["claims"] for u in c["cited_turn_uids"])


def test_sliding_window_keeps_the_most_recent_turns_in_order(env):
    rt, data, _ = env
    sw = build_system("B1_sliding_window", rt)
    big = sw.build_context(data.qas[0], None)
    assert (
        big.turn_uids[-1] == "synthetic:demo-alice-bob:s03:t004"
    )  # the very last turn is always in
    assert big.turn_uids == sorted(big.turn_uids, key=lambda u: u.split(":")[-2:])  # chronological
    small = build_system("B1_sliding_window", rt, budget=60).build_context(data.qas[0], None)
    assert 0 < len(small.turn_uids) < len(big.turn_uids) and small.tokens <= 60
    assert small.turn_uids == big.turn_uids[-len(small.turn_uids) :]


def test_chunk_rag_uses_three_turn_chunks(env):
    rt, data, _ = env
    b3 = build_system("B3_chunk_rag", rt)
    chunks, vecs = b3.index(CONV_ID)
    assert vecs.shape[0] == len(chunks) and all(1 <= len(c) <= 3 for c in chunks)
    assert all(len({t["session_ordinal"] for t in c}) == 1 for c in chunks)  # never spans a session
    assert sum(len(c) for c in chunks) == len([t for t in baselines_mod.load_turns(b3.db, CONV_ID)])
    ctx = b3.build_context(data.qas[0], None)
    assert "Milwaukee" in ctx.text and ctx.tokens <= 1500


def test_notes_baseline_mixes_turns_and_notes_without_state(env):
    rt, data, _ = env
    ctx = build_system("B4_turns_plus_notes", rt).build_context(data.qas[0], None)
    notes = [ln for ln in ctx.text.splitlines() if "NOTE:" in ln]
    assert notes and any("Milwaukee" in n for n in notes)
    assert (
        any("Chicago" in n for n in notes) or "Chicago" in ctx.text
    )  # the superseded fact is still a note
    assert not re.search(r"SUPERSEDED|ACTIVE|history:", ctx.text)  # no status, no versions
    assert all(
        re.match(r"\[synthetic:demo-alice-bob:s\d\d:t\d{3} \| \d{4}-\d\d-\d\d \| \w+\] ", ln)
        for ln in ctx.text.splitlines()
    )


def test_full_context_asserts_the_window_and_skips_longmemeval(env, monkeypatch):
    rt, data, llm = env
    b2 = build_system("B2_full_context", rt)
    ctx = b2.build_context(data.qas[0], "2023-11-05")
    assert len(ctx.turn_uids) == 11 and ctx.budget is None  # the whole demo history
    monkeypatch.setattr(baselines_mod, "NUM_CTX", 300)
    with pytest.raises(Skipped, match="context_overflow"):
        b2.build_context(data.qas[0], "2023-11-05")
    monkeypatch.setattr(baselines_mod, "NUM_CTX", 32768)
    with pytest.raises(Skipped, match="LoCoMo only"):
        b2.build_context(qa(9, "q?", "a", dataset="longmemeval"), None)


def test_long_allowed_id_lists_are_compacted_for_big_contexts(env):
    from chronos.agent.generator import Generator

    msg = Generator.messages("q", "ctx", [f"t{i}" for i in range(500)])[1]["content"]
    assert "(500 turns)" in msg and "t499" not in msg
    assert "t1, t2" in Generator.messages("q", "ctx", ["t1", "t2"])[1]["content"]


def test_run_system_is_resumable_and_records_skips(env, tmp_path):
    rt, data, llm = env
    spec = RunSpec("B2_full_context", "B2_full_context")
    data.qas.append(qa(4, "lme?", "a", dataset="longmemeval"))
    first = run_system(spec, rt, data, progress=lambda m: None)
    assert (first["answered"], first["skipped"], first["errors"]) == (3, 1, 0)
    lines = read_jsonl(common.raw_output_path("unit", "B2_full_context"))
    assert len(lines) == 4 and [r for r in lines if r.get("skipped")][0]["question_id"] == "q4"
    calls = len(llm.records)
    second = run_system(spec, rt, data, progress=lambda m: None)
    assert second["already_done"] == 4 and second["answered"] == 0 and len(llm.records) == calls


def test_plan_orders_main_systems_then_ablation_variants_and_adds_alias_targets():
    from eval.ablations import load_ablations, resolve_selection

    specs = plan(["B3_chunk_rag"], resolve_selection("B,C", load_ablations()))
    ids = [s.system_id for s in specs]
    assert ids[:3] == ["B3_chunk_rag", "B5_append_only", "chronos"]  # B's aliases are pulled in
    assert "B_state/arrival_order" in ids and "C_gate/off" in ids and "C_gate/speaker_only" in ids
    assert "B_state/bitemporal" not in ids and "C_gate/full" not in ids  # aliases are never re-run
    arrival = next(s for s in specs if s.system_id == "B_state/arrival_order")
    assert (
        arrival.spec["db"] == "arrival_order"
        and arrival.select(qa(1, "q", "a", dataset="locomo")) is False
    )


def test_ablation_variant_overrides_reach_the_system(env):
    rt, data, _ = env
    gate_off = build_system("chronos", rt, agent={"answerability": "off"}, system_id="C_gate/off")
    assert gate_off.id == "C_gate/off" and gate_off.agent.agent_config.answerability == "off"
    small = build_system("chronos", rt, budget=500, retrieval={"use_bm25_episodes": False})
    cfg = small.agent.retrieval_config
    assert (
        cfg.token_budget == 500
        and cfg.use_bm25_episodes is False
        and cfg.use_dense_episodes is True
    )
    rec = small.answer(data.qas[0])
    assert rec["context"]["tokens"] <= 500
