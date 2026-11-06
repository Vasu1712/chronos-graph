"""Verifier: fake turn rejected; real-but-unretrieved rejected; retrieved-but-unsupportive rejected;
supported kept; one regeneration with feedback; abstain when nothing survives; answer_log records."""

import re

import pytest
from retrieval_fixtures import CONV, ingested_demo

from chronos.agent.generator import Claim, GenerationOut, Generator
from chronos.agent.verifier import AnswerLog, Verifier, verify_and_repair
from chronos.llm.nli import NLIScore

T = "synthetic:demo-alice-bob:"
MILWAUKEE_TURN = (
    T + "s03:t002"
)  # "I don't run anymore, my knee gave out. Also we moved to Milwaukee last month."
CHICAGO_TURN = T + "s01:t002"  # "I live in Chicago now and work at Lumen Health as a designer."
BOB_TURN = T + "s02:t002"  # "Fun! Any news?"


class OverlapNLI:
    """Entailed iff every content word of the hypothesis (minus names) appears in the premise."""

    STOP = {"alice", "the", "a", "in", "to", "at", "as", "is", "and", "of", "her", "she"}

    def __init__(self):
        self.pairs = []

    def score(self, pairs):
        self.pairs += pairs
        out = []
        for premise, hyp in pairs:
            words = [w for w in re.findall(r"[a-z]+", hyp.lower()) if w not in self.STOP]
            ok = all(w in premise.lower() for w in words)
            out.append(NLIScore(0.95, 0.03, 0.02) if ok else NLIScore(0.05, 0.25, 0.70))
        return out


class ScriptedGenerator(Generator):
    def __init__(self, *outs):
        self.outs, self.calls = list(outs), []

    def generate(self, question, context, allowed, *, feedback=None, **hints):
        self.calls.append(feedback)
        return self.outs.pop(0)


def gen(*claims, answer="answer", abstained=False):
    return GenerationOut(final_answer=answer, claims=[Claim(text=t, cited_turn_uids=list(u), cited_memory_ids=[])
                                                      for t, u in claims], abstained=abstained,
                         abstention_reason="no evidence" if abstained else None)  # fmt: skip


@pytest.fixture
def env(tmp_path):
    db = ingested_demo(tmp_path / "v.sqlite")
    AnswerLog(db.conn).open("q1", CONV, "Where does Alice live?", [], [MILWAUKEE_TURN, BOB_TURN])
    yield db, Verifier(db.conn, OverlapNLI(), threshold=0.5)
    db.close()


def check(env, *claims):
    db, ver = env
    return ver.check("q1", gen(*claims)).claims


def test_fake_turn_id_is_rejected(env):
    [c] = check(env, ("Alice moved to Milwaukee.", ["synthetic:demo-alice-bob:s09:t999"]))
    assert not c.supported and c.dropped == [
        {"turn_uid": "synthetic:demo-alice-bob:s09:t999", "reason": "not_found"}
    ]
    assert c.nli_entailment is None


def test_real_but_unretrieved_turn_is_rejected(env):
    [c] = check(
        env, ("Alice lives in Chicago.", [CHICAGO_TURN])
    )  # exists, supports it, but not in context
    assert not c.supported and c.dropped == [{"turn_uid": CHICAGO_TURN, "reason": "not_in_context"}]


def test_retrieved_turn_that_does_not_support_is_rejected(env):
    [c] = check(env, ("Alice moved to Milwaukee.", [BOB_TURN]))
    assert not c.supported and c.nli_entailment < 0.5 and c.dropped[0]["reason"] == "not_entailed"


def test_supported_claim_is_kept_and_bad_citations_dropped(env):
    [c] = check(env, ("Alice moved to Milwaukee.", [MILWAUKEE_TURN, "fake:turn", CHICAGO_TURN]))
    assert c.supported and c.kept_turn_uids == [MILWAUKEE_TURN] and c.nli_entailment >= 0.5
    assert {d["reason"] for d in c.dropped} == {"not_found", "not_in_context"}
    db, ver = env
    premise = ver.nli.pairs[-1][0]
    assert premise.startswith("On 2023-11-05, Alice said: I don't run anymore")


def test_repair_regenerates_once_with_feedback_then_keeps_supported(env):
    db, ver = env
    first = gen(
        ("Alice moved to Milwaukee.", [MILWAUKEE_TURN]), ("Alice lives in Chicago.", [CHICAGO_TURN])
    )
    second = gen(
        ("Alice moved to Milwaukee.", [MILWAUKEE_TURN]), ("Alice owns a boat.", [BOB_TURN])
    )
    g = ScriptedGenerator(second)
    final, rec = verify_and_repair(query_id="q1", question="q", context="c", allowed_turn_uids=[], generator=g,
                                   verifier=ver, first=first)  # fmt: skip
    assert (
        len(g.calls) == 1
        and '"Alice lives in Chicago."' in g.calls[0]
        and "not_in_context" in g.calls[0]
    )
    assert [c.text for c in final.claims] == [
        "Alice moved to Milwaukee."
    ]  # unsupported removed after retry
    assert final.final_answer == "Alice moved to Milwaukee." and not final.abstained
    assert len(rec["rounds"]) == 2 and rec["rounds"][1]["claims"][1]["supported"] is False


def test_all_unsupported_after_retry_abstains(env):
    db, ver = env
    bad = gen(("Alice owns a boat.", [BOB_TURN]))
    final, rec = verify_and_repair(query_id="q1", question="q", context="c", allowed_turn_uids=[],
                                   generator=ScriptedGenerator(bad), verifier=ver, first=bad)  # fmt: skip
    assert final.abstained and final.abstention_reason == "unsupported" and final.claims == []


def test_answer_without_claims_is_treated_as_unsupported(env):
    db, ver = env
    g = ScriptedGenerator(gen(("Alice moved to Milwaukee.", [MILWAUKEE_TURN])))
    final, _ = verify_and_repair(query_id="q1", question="q", context="c", allowed_turn_uids=[], generator=g,
                                 verifier=ver, first=gen(answer="Milwaukee"))  # fmt: skip
    assert "no claims" in g.calls[0] and final.claims[0].text == "Alice moved to Milwaukee."


def test_model_abstention_passes_through_without_verification(env):
    db, ver = env
    g = ScriptedGenerator()
    final, rec = verify_and_repair(query_id="q1", question="q", context="c", allowed_turn_uids=[], generator=g,
                                   verifier=ver, first=gen(abstained=True))  # fmt: skip
    assert final.abstained and final.abstention_reason == "no evidence" and g.calls == []


def test_answer_log_round_trip_with_nli_scores(env):
    db, ver = env
    first = gen(("Alice moved to Milwaukee.", [MILWAUKEE_TURN]))
    final, rec = verify_and_repair(query_id="q1", question="q", context="c", allowed_turn_uids=[],
                                   generator=ScriptedGenerator(), verifier=ver, first=first)  # fmt: skip
    ver.log.finish("q1", final.model_dump(), rec)
    row = AnswerLog(db.conn).get("q1")
    assert row["retrieved_turn_uids"] == [MILWAUKEE_TURN, BOB_TURN]
    claim = row["verifier_json"]["rounds"][0]["claims"][0]
    assert claim["nli_entailment"] == pytest.approx(0.95) and claim["supported"] is True
    assert row["response_json"]["claims"][0]["cited_turn_uids"] == [MILWAUKEE_TURN]


def test_malformed_citations_are_normalized_not_guessed(env):
    db, ver = env
    AnswerLog(db.conn).open("q2", CONV, "q", ["demo-alice-bob:m:abc"], [MILWAUKEE_TURN, BOB_TURN])
    line = "[" + MILWAUKEE_TURN + " | 2023-11-05 | Alice] I don't run anymore"
    pasted_text = (
        "[demo-alice-bob:m:abc | 2023-11-05 | Alice] Fun! Any news?"  # memory id + exact turn text
    )
    claims = gen(("Alice moved to Milwaukee.", [line, "demo-alice-bob:m:abc"]), ("Alice moved to Milwaukee.", [pasted_text]),
                 ("Alice moved to Milwaukee.", ["[nothing | x | y] no such text"]))  # fmt: skip
    a, b, c = ver.check("q2", claims).claims
    assert (
        a.kept_turn_uids == [MILWAUKEE_TURN]
        and a.kept_memory_ids == ["demo-alice-bob:m:abc"]
        and a.supported
    )
    assert {n["how"] for n in a.normalized} == {"extracted turn id", "memory id cited as a turn"}
    # the memory id is moved out first, so the exact-text rule never fires for b; b keeps only the memory id
    assert (
        b.kept_turn_uids == [] and b.kept_memory_ids == ["demo-alice-bob:m:abc"] and not b.supported
    )
    assert c.kept_turn_uids == [] and c.dropped[0]["reason"] == "not_found" and c.normalized == []


def test_text_match_repairs_a_line_without_a_valid_id(env):
    db, ver = env
    [c] = ver.check(
        "q1", gen(("Bob asked for news.", ["[garbled | 2023-06-20 | Bob] Fun! Any news?"]))
    ).claims
    assert (
        c.kept_turn_uids == [BOB_TURN]
        and c.normalized[0]["how"] == "matched turn by its exact text"
    )


def test_generator_prompt_carries_hints_and_allowed_ids():
    from chronos.agent.generator import Generator

    msgs = Generator(None).messages("Where did I live in 2019?", "CTX", ["t1", "t2"], question_date="2026-07-15T08:45",
                                    feedback="fix it", asker="Meera", question_type="AS_OF 2019-01-01..2019-12-31")  # fmt: skip
    user = msgs[1]["content"]
    assert user.splitlines()[:4] == [
        "QUESTION: Where did I live in 2019?", "QUESTION TYPE: AS_OF 2019-01-01..2019-12-31",
        'ASKED BY: Meera ("I", "me", "my" in the question mean Meera)', "QUESTION DATE: 2026-07-15",
    ]  # fmt: skip
    assert "ALLOWED TURN IDS: t1, t2" in user and user.endswith("fix it")
    bare = Generator(None).messages("q", "CTX", [])[1]["content"]  # baseline usage: no hints
    assert (
        "QUESTION TYPE" not in bare
        and "ASKED BY" not in bare
        and "ALLOWED TURN IDS: (none)" in bare
    )


def test_repair_round_abstention_keeps_round_one_supported_claims(env):
    """quick_v1: the repair round often answered "I don't know" after one claim failed NLI, which
    threw away claims that had already been verified."""
    db, ver = env
    first = gen(
        ("Alice moved to Milwaukee.", [MILWAUKEE_TURN]),
        ("Alice works at Lumen Health.", [MILWAUKEE_TURN]),
    )
    g = ScriptedGenerator(gen(abstained=True))
    final, record = verify_and_repair(
        query_id="q1", question="Where does Alice live?", context="", generator=g, verifier=ver,
        allowed_turn_uids=[MILWAUKEE_TURN, BOB_TURN], first=first,
    )  # fmt: skip
    assert not final.abstained
    assert [c.text for c in final.claims] == ["Alice moved to Milwaukee."]
    assert record["kept_round1_after_abstention"]


def test_long_turn_is_scored_in_windows(env, monkeypatch):
    """A supporting sentence past the NLI model's 512 tokens still counts."""
    db, ver = env
    filler = " ".join(f"Sentence number {i} is about something unrelated." for i in range(60))
    long_text = filler + " We moved to Milwaukee last month."
    monkeypatch.setattr(ver, "_parts", lambda uids: [("On 2024-03-01, Alice said:", long_text)])
    premises = ver.premises([MILWAUKEE_TURN])
    assert len(premises) > 1
    assert any("Milwaukee" in p and len(p) < 1500 for p in premises[1:])

    class FirstCharsNLI(OverlapNLI):  # reads only the first ~2,000 characters, like the real model
        def score(self, pairs):
            return super().score([(p[:2000], h) for p, h in pairs])

    ver.nli = FirstCharsNLI()
    [c] = check(env, ("Alice moved to Milwaukee.", [MILWAUKEE_TURN]))
    assert c.supported


def test_premise_resolves_relative_dates(env):
    db, ver = env
    assert "last month [= " in ver.premise([MILWAUKEE_TURN])


class ScriptedLLM:
    """Returns scripted GenerationOut values and records the prompts it was given."""

    def __init__(self, *outs):
        self.outs, self.prompts = list(outs), []

    def structured(self, messages, response_model):
        self.prompts.append(messages[1]["content"])
        return self.outs.pop(0)


def test_decline_gets_one_forced_answer_retry():
    """eval-50: the model declined 86% of answerable questions in every system, even with the gold
    turn in context. A decline with no claims is retried once with an instruction to answer."""
    llm = ScriptedLLM(gen(abstained=True), gen(("Alice lives in Milwaukee.", ["t1"])))
    out = Generator(llm).generate(
        "Where does Alice live?", "[t1 | d | Alice] I live in Milwaukee", ["t1"]
    )
    assert not out.abstained and [c.text for c in out.claims] == ["Alice lives in Milwaukee."]
    assert len(llm.prompts) == 2 and "INSTRUCTION: Your previous attempt declined" in llm.prompts[1]
    assert "INSTRUCTION" not in llm.prompts[0]


def test_decline_after_forced_retry_stays_a_decline():
    llm = ScriptedLLM(gen(abstained=True), gen(abstained=True))
    g = Generator(llm)
    out = g.generate("q", "[t1 | d | Alice] unrelated", ["t1"])
    assert out.abstained and g.forced_retries == 1 and len(llm.prompts) == 2


def test_abstained_flag_with_claims_is_overridden_and_empty_context_not_retried():
    llm = ScriptedLLM(gen(("Alice lives in Milwaukee.", ["t1"]), abstained=True))
    out = Generator(llm).generate("q", "[t1 | d | Alice] x", ["t1"])
    assert not out.abstained and len(llm.prompts) == 1
    llm = ScriptedLLM(gen(abstained=True))
    out = Generator(llm).generate("q", "", [])
    assert out.abstained and len(llm.prompts) == 1  # nothing was retrieved: nothing to force
