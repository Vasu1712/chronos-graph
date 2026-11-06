"""MemoryAgent end-to-end with fake LLM/NLI, and the rich CLI commands (offline)."""

import re

import pytest
from retrieval_fixtures import CONV, ingested_demo, mem_id
from typer.testing import CliRunner

import chronos.cli as cli
from chronos.agent.answerability import GateOut
from chronos.agent.generator import GenerationOut
from chronos.agent.memory_agent import STEPS, MemoryAgent
from chronos.agent.verifier import AnswerLog
from chronos.config import AgentConfig
from chronos.llm.embed import HashingEmbedder
from chronos.llm.nli import NLIScore

TURN = re.compile(r"\[([^ |\]]+) \| [0-9-]+ \| [^\]]+\] (.*)")


class FakeLLM:
    """Gate: ANSWERABLE. Generator: cites the first context turn mentioning the keyword."""

    def __init__(self, keyword="Milwaukee", claim="Alice moved to Milwaukee."):
        self.keyword, self.claim, self.seen = keyword, claim, []

    def structured(self, messages, model):
        self.seen.append(model.__name__)
        user = messages[-1]["content"]
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
        return GenerationOut(final_answer=self.claim, abstained=False, abstention_reason=None,
                             claims=[{"text": self.claim, "cited_turn_uids": [hits[0]], "cited_memory_ids": []}])  # fmt: skip


class FakeNLI:
    def score(self, pairs):
        return [
            NLIScore(0.9, 0.05, 0.05) if "Milwaukee" in p else NLIScore(0.1, 0.2, 0.7)
            for p, _ in pairs
        ]


@pytest.fixture
def db(tmp_path):
    d = ingested_demo(tmp_path / "agent.sqlite")
    yield d
    d.close()


def agent(db, llm=None, **cfg):
    return MemoryAgent(db, llm or FakeLLM(), HashingEmbedder(), FakeNLI(), agent=AgentConfig(**cfg),
                       count_tokens=lambda s: len(s.split()))  # fmt: skip


def test_answer_runs_all_steps_and_logs(db):
    ans = agent(db).answer(CONV, "Where does Alice live now?")
    assert not ans.abstained and ans.final_answer == "Alice moved to Milwaukee."
    assert ans.claims[0].cited_turn_uids == ["synthetic:demo-alice-bob:s03:t002"]
    assert set(STEPS) <= set(ans.timings) and ans.timings["total"] >= ans.timings["retrieve"]
    assert ans.analysis.question_date.startswith("2023-11-05")  # defaults to the last session
    row = AnswerLog(db.conn).get(ans.query_id)
    assert row["retrieved_turn_uids"] == ans.packed.turn_uids
    assert row["response_json"]["claims"][0]["text"] == "Alice moved to Milwaukee."
    assert row["verifier_json"]["rounds"][0]["claims"][0]["nli_entailment"] == pytest.approx(0.9)


def test_gate_abstention_skips_generation(db):
    llm = FakeLLM()
    ans = agent(db, llm, answerability="speaker").answer(CONV, "Where does Bob live?")
    assert (
        ans.abstained and ans.abstained_at == "gate" and "speaker mismatch" in ans.abstention_reason
    )
    assert "GenerationOut" not in llm.seen


def test_unsupported_answer_abstains_after_one_retry(db):
    llm = FakeLLM(keyword="Milwaukee", claim="Alice owns a sailboat.")

    class StrictNLI:
        def score(self, pairs):
            return [NLIScore(0.1, 0.1, 0.8) for _ in pairs]

    a = MemoryAgent(db, llm, HashingEmbedder(), StrictNLI(), count_tokens=lambda s: len(s.split()))
    ans = a.answer(CONV, "Where does Alice live now?")
    assert (
        ans.abstained and ans.abstained_at == "verifier" and ans.abstention_reason == "unsupported"
    )
    assert llm.seen.count("GenerationOut") == 2


@pytest.fixture
def cli_env(db, monkeypatch):
    monkeypatch.setattr(cli, "_build_agent", lambda database, **kw: agent(database))
    return db


def run_cli(*args):
    res = CliRunner().invoke(cli.app, list(args), env={"COLUMNS": "250"})
    assert res.exit_code == 0, res.output
    return res.output


def test_cli_ask_prints_cited_turns(cli_env):
    out = run_cli(
        "ask", "Where does Alice live now?", "--db", str(cli_env.path), "--embedder", "hash"
    )
    assert "Alice moved to Milwaukee." in out
    assert "[synthetic:demo-alice-bob:s03:t002 | 2023-11-05 | Alice] I don't run anymore" in out
    assert "latency (s)" in out and "intent CURRENT, about Alice" in out


def test_cli_inspect_memory_shows_chain_quotes_turns_and_nli(cli_env):
    run_cli("ask", "Where does Alice live now?", "--db", str(cli_env.path), "--embedder", "hash")
    out = run_cli("inspect", mem_id(cli_env, "Milwaukee"), "--db", str(cli_env.path))
    chain = out[out.index("version chain") : out.index("quotes")]
    assert chain.index("Alice lives in Chicago.") < chain.index("Alice moved to Milwaukee.")
    assert "SUPERSEDED" in chain and "ACTIVE" in chain
    assert '"we moved to Milwaukee last month"' in out and "ASSERTS" in out
    assert "> [synthetic:demo-alice-bob:s03:t002" in out
    assert "NLI verdicts in answer_log" in out and "0.900" in out


def test_cli_inspect_query(cli_env):
    out = run_cli(
        "ask",
        "Where does Alice live now?",
        "--db",
        str(cli_env.path),
        "--embedder",
        "hash",
        "--json",
    )
    qid = re.search(r'"query_id": "([^"]+)"', out).group(1)
    q = run_cli("inspect", qid, "--db", str(cli_env.path))
    assert "verifier round 1" in q and "Alice moved to Milwaukee." in q and "0.900" in q


def test_cli_history_and_stats(cli_env):
    h = run_cli("history", "Alice", "--attribute", "residence", "--db", str(cli_env.path))
    assert all(city in h for city in ("Chicago", "Milwaukee", "Detroit"))
    assert "Lumen" not in h  # employer facts are excluded by --attribute
    s = run_cli("stats", "--db", str(cli_env.path))
    assert "episodes" in s and "memories ACTIVE" in s and "memories SUPERSEDED" in s


def test_cli_rejects_mismatched_embedder(db, monkeypatch):
    monkeypatch.setattr(cli, "_build_agent", lambda database, **kw: MemoryAgent(
        database, FakeLLM(), type("E", (), {"name": "BAAI/bge-small-en-v1.5"})(), FakeNLI()))  # fmt: skip
    res = CliRunner().invoke(cli.app, ["ask", "Where?", "--db", str(db.path)])
    assert res.exit_code != 0 and "embedded with" in res.output
