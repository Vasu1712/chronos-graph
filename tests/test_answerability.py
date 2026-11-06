import pytest
from retrieval_fixtures import CONV, QDATE, ingested_demo

from chronos.agent.answerability import AnswerabilityGate, GateOut
from chronos.llm.embed import HashingEmbedder
from chronos.retrieval.packer import Packer
from chronos.retrieval.query import analyze
from chronos.retrieval.retriever import Retriever


class FakeGateLLM:
    def __init__(self, verdict="ANSWERABLE", supporting=()):
        self.verdict, self.supporting, self.calls = verdict, list(supporting), []

    def structured(self, messages, response_model):
        assert response_model is GateOut
        self.calls.append(messages)
        return GateOut(reason="fake", verdict=self.verdict, missing_detail=None if self.verdict == "ANSWERABLE" else "x",
                       supporting_turn_uids=self.supporting)  # fmt: skip


@pytest.fixture
def db(tmp_path):
    d = ingested_demo(tmp_path / "a.sqlite")
    yield d
    d.close()


def ctx(db, q):
    a = analyze(q, conn=db.conn, conversation_id=CONV, question_date=QDATE)
    p = Packer(db, count_tokens=lambda s: len(s.split())).pack(
        Retriever(db, HashingEmbedder()).retrieve(a, CONV)
    )
    return a, p


def test_speaker_mismatch_abstains_in_speaker_mode(db):
    a, p = ctx(db, "Where does Bob live?")  # only Alice's residence is known
    g = AnswerabilityGate(db, None, "speaker").check(a, p)
    assert g.speaker_mismatch and not g.contrary_evidence
    assert g.abstain and "speaker mismatch" in g.reason and "Alice" in g.mismatch_detail
    assert not g.llm_called


def test_no_mismatch_when_question_matches_speaker(db):
    a, p = ctx(db, "Where does Alice live?")
    g = AnswerabilityGate(db, None, "speaker").check(a, p)
    assert not g.speaker_mismatch and not g.abstain


def test_question_naming_both_speakers_is_not_a_mismatch(db):
    a, p = ctx(
        db, "Where do Alice and Bob live?"
    )  # Alice's residence answers a question about both
    g = AnswerabilityGate(db, None, "speaker").check(a, p)
    assert not g.speaker_mismatch and not g.abstain


def test_contrary_turn_evidence_prevents_abstention(db):
    a, p = ctx(
        db, "Is Bob still running?"
    )  # Alice's running memories; Bob asked "Still running?" in s01
    g = AnswerabilityGate(db, None, "speaker").check(a, p)
    assert g.speaker_mismatch and g.contrary_evidence and not g.abstain


@pytest.mark.parametrize(
    "verdict, abstain", [("ANSWERABLE", False), ("PARTIAL", False), ("NOT_ANSWERABLE", True)]
)
def test_full_mode_llm_verdicts(db, verdict, abstain):
    a, p = ctx(db, "Where does Alice live?")
    llm = FakeGateLLM(verdict, supporting=[p.turn_uids[0], "made-up:turn"])
    g = AnswerabilityGate(db, llm, "full").check(a, p)
    assert g.llm_called and len(llm.calls) == 1 and g.verdict == verdict and g.abstain is abstain
    assert g.supporting_turn_uids == [p.turn_uids[0]]  # ids outside the context are dropped
    user = llm.calls[0][1]["content"]
    assert (
        user.startswith("QUESTION (about Alice): Where does Alice live?")
        and "EVIDENCE:\n## Alice" in user
    )


def test_full_mode_mismatch_overridden_only_by_support_from_the_right_speaker(db):
    a, p = ctx(db, "Where does Bob live?")
    alice_turn = next(u for u in p.turn_uids if db.conn.execute(
        "SELECT speaker FROM episodes WHERE turn_uid = ?", (u,)).fetchone()[0] == "Alice")  # fmt: skip
    g = AnswerabilityGate(db, FakeGateLLM("ANSWERABLE", [alice_turn]), "full").check(a, p)
    assert (
        g.abstain and "speaker mismatch" in g.reason
    )  # an ANSWERABLE verdict citing Alice is not enough


def test_off_mode_never_abstains_or_calls(db):
    a, p = ctx(db, "Where does Bob live?")
    llm = FakeGateLLM("NOT_ANSWERABLE")
    g = AnswerabilityGate(db, llm, "off").check(a, p)
    assert not g.abstain and llm.calls == []


def test_empty_context_abstains_without_llm(db):
    from chronos.retrieval.packer import PackedContext

    a, _ = ctx(db, "Where does Alice live?")
    llm = FakeGateLLM()
    g = AnswerabilityGate(db, llm, "full").check(a, PackedContext("", [], [], 0, 1500, True))
    assert g.abstain and llm.calls == []


@pytest.mark.parametrize("term, text, hit", [
    ("running", "Alice no longer runs.", True), ("lives", "Alice lived in Detroit.", True),
    ("painting", "Melanie paints landscapes.", True), ("job", "Alice joined Brightwave.", False),
    ("car", "Alice bought a bike.", False),
])  # fmt: skip
def test_term_hit_stemming(term, text, hit):
    from chronos.agent.answerability import _term_hit

    assert _term_hit([term], text) is hit
