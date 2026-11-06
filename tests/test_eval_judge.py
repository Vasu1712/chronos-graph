import csv
import json

import pytest
from fakes import FakeChatClient
from retrieval_fixtures import ingested_demo

import eval.common as common
from chronos.config import JudgeProviderConfig, ProviderLimits
from chronos.llm.judge_pool import JudgePool, route_index
from eval import judge
from eval.common import append_jsonl, read_jsonl
from eval.prompts.lme_judge import TEMPLATES, batch_rubric, prompt_kind

LIMITS = ProviderLimits(1000, 10**9, 10**9)


def cfgs():
    return [
        JudgeProviderConfig("groq", "h", "m", "X", LIMITS),
        JudgeProviderConfig("cerebras", "h", "m", "Y", LIMITS),
    ]


def qid(idx, n=0):
    return [q for q in (f"q{i}" for i in range(200)) if route_index(q) == idx][n]


def make_pool(*, labeller=None, both=True):
    def responder(label_for):
        def respond(kwargs):
            user = kwargs["messages"][-1]["content"]
            items = json.loads(user.rsplit("\n\n", 1)[1])
            return json.dumps(
                {
                    "verdicts": [
                        {"item_id": it["item_id"], "rationale": "r", "label": label_for(it)}
                        for it in items
                    ]
                }
            )

        return respond

    fn = labeller or (lambda it: "CORRECT")
    clients = {"groq": FakeChatClient(responder(fn))}
    if both:
        clients["cerebras"] = FakeChatClient(responder(fn))
    return JudgePool(None, providers=cfgs(), clients=clients, available_only=True), clients


def raw(
    system,
    q,
    *,
    answer="Paris",
    gold="Paris",
    abstained=False,
    unanswerable=False,
    category="single-hop",
    dataset="locomo",
):
    return {
        "system": system, "question_id": q, "dataset": dataset, "category": category, "question": "Where?",
        "gold_answer": gold, "gold_unanswerable": unanswerable, "abstained": abstained, "final_answer": answer,
        "conversation_id": "c", "claims": [], "verifier": {"rounds": []}, "context": {"turn_uids": [], "tokens": 1},
    }  # fmt: skip


@pytest.fixture
def art(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ARTIFACTS", tmp_path)
    return tmp_path


def seed(manifest="m"):
    p = common.raw_output_path(manifest, "S")
    rows = [
        raw("S", "exact", answer="The Paris.", gold="paris"),  # normalized exact match
        raw("S", "judge1", answer="It was Paris, I think", gold="Paris"),
        raw("S", "judge2", answer="London", gold="Paris"),
        raw("S", "unans_ok", unanswerable=True, abstained=True, answer="I don't know."),
        raw("S", "unans_bad", unanswerable=True, answer="Berlin"),
        raw("S", "abst_answerable", abstained=True, answer="I don't know."),
    ]
    for r in rows:
        append_jsonl(p, r)
    append_jsonl(p, {"system": "S", "question_id": "skipme", "skipped": "context_overflow"})


def test_routing_flag_exact_judge(art):
    seed()
    pool, clients = make_pool(
        labeller=lambda it: "CORRECT" if "Paris" in it["response"] else "WRONG"
    )
    counts = judge.score_manifest("m", pool, progress=lambda m: None)
    assert counts == {"flag": 3, "exact": 1, "judge": 2, "skipped": 1, "pending_no_judge": 0}
    j = {r["question_id"]: r for r in read_jsonl(common.judgments_path("m"))}
    assert (
        j["unans_ok"]["correct"],
        j["unans_bad"]["correct"],
        j["abst_answerable"]["correct"],
    ) == (True, False, False)
    assert j["exact"]["correct"] and j["exact"]["method"] == "exact"
    assert (j["judge1"]["correct"], j["judge2"]["correct"]) == (True, False)
    assert j["judge1"]["method"] == "judge" and j["judge1"]["provider"] in ("groq", "cerebras")
    assert "skipme" not in j  # a skipped answer is never scored
    api_calls = sum(len(c.requests) for c in clients.values())
    assert api_calls >= 1
    # nothing flag/exact-scored reached the judge
    sent = [
        it["item_id"]
        for c in clients.values()
        for r in c.requests
        for it in json.loads(r["messages"][-1]["content"].rsplit("\n\n", 1)[1])
    ]
    assert sorted(sent) == ["S|judge1", "S|judge2"]


def test_rerun_scores_nothing_twice_but_rescores_a_changed_answer(art):
    seed()
    pool, clients = make_pool()
    judge.score_manifest("m", pool, progress=lambda m: None)
    n = len(read_jsonl(common.judgments_path("m")))
    again = judge.score_manifest("m", pool, progress=lambda m: None)
    assert (
        again["flag"] == again["exact"] == again["judge"] == 0
        and len(read_jsonl(common.judgments_path("m"))) == n
    )
    append_jsonl(
        common.raw_output_path("m", "S"),
        raw("S", "judge2", answer="Paris, definitely", gold="Paris"),
    )
    changed = judge.score_manifest("m", pool, progress=lambda m: None)
    assert changed["judge"] == 1


def test_longmemeval_questions_use_the_released_prompt_for_their_type(art):
    p = common.raw_output_path("m", "S")
    append_jsonl(
        p,
        raw(
            "S",
            "ku",
            answer="3 dogs now",
            gold="3",
            category="knowledge-update",
            dataset="longmemeval",
        ),
    )
    append_jsonl(
        p, raw("S", "lo", answer="3 dogs now", gold="3", category="temporal", dataset="locomo")
    )
    pool, clients = make_pool()
    judge.score_manifest("m", pool, progress=lambda m: None)
    systems = {r["messages"][0]["content"] for c in clients.values() for r in c.requests}
    assert systems == {batch_rubric("knowledge-update"), batch_rubric("default")}
    assert prompt_kind("longmemeval", "single-session-preference") == "preference"
    assert prompt_kind("locomo", "multi-hop") == "default"


def test_released_templates_are_the_vendored_originals():
    assert set(TEMPLATES) == {
        "default",
        "temporal-reasoning",
        "knowledge-update",
        "preference",
        "abstention",
    }
    assert "do not penalize off-by-one errors" in TEMPLATES["temporal-reasoning"]
    assert "updated answer is the required answer" in TEMPLATES["knowledge-update"]
    assert all(
        t.count("{}") == 3 and t.rstrip().endswith("Answer yes or no only.")
        for t in TEMPLATES.values()
    )


def test_without_a_provider_the_judge_work_is_left_pending_not_guessed(art):
    seed()
    counts = judge.score_manifest("m", None, progress=lambda m: None)
    assert counts["pending_no_judge"] == 2 and counts["flag"] == 3
    assert {r["question_id"] for r in read_jsonl(common.judgments_path("m"))} == {
        "exact",
        "unans_ok",
        "unans_bad",
        "abst_answerable",
    }


def test_calibration_reports_kappa_between_providers_or_says_it_cannot(art, tmp_path):
    p = common.raw_output_path("m", "S")
    for i in range(30):
        append_jsonl(p, raw("S", f"cq{i}", answer=f"maybe {i}", gold="Paris"))
    # groq says CORRECT for even items, cerebras agrees except on 5 of 30
    flips = {f"S|cq{i}" for i in range(0, 10, 2)}

    def label(provider):
        def f(it):
            n = int(it["item_id"].split("cq")[1])
            ok = n % 2 == 0
            return (
                "CORRECT"
                if (ok != (provider == "cerebras" and it["item_id"] in flips))
                else "WRONG"
            )

        return f

    def responder(provider):
        return lambda kw: json.dumps({"verdicts": [{"item_id": it["item_id"], "rationale": "r", "label": label(provider)(it)}
                                                    for it in json.loads(kw["messages"][-1]["content"].rsplit("\n\n", 1)[1])]})  # fmt: skip

    pool = JudgePool(None, providers=cfgs(), clients={"groq": FakeChatClient(responder("groq")),
                                                      "cerebras": FakeChatClient(responder("cerebras"))}, available_only=True)  # fmt: skip
    res = judge.calibrate(pool, "m", n=100, out=tmp_path / "cal.json")
    assert (
        res["status"] == "measured"
        and res["n"] == 30
        and res["agreement"] == pytest.approx(25 / 30)
    )
    assert 0.5 < res["kappa"] < 0.8 and json.loads((tmp_path / "cal.json").read_text())["n"] == 30
    one, _ = make_pool(both=False)
    res1 = judge.calibrate(one, "m", out=tmp_path / "cal1.json")
    assert (
        res1["status"] == "not_measured"
        and "needs two providers" in res1["reason"]
        and "cerebras" in res1["reason"]
    )


def test_audit_sheet_is_a_blank_sample_and_never_overwrites_hand_labels(art, tmp_path):
    p = common.raw_output_path("m", "S")
    for i in range(130):
        append_jsonl(p, raw("S", f"aq{i}", answer=f"guess {i}", gold="Paris"))
    pool, _ = make_pool()
    judge.score_manifest("m", pool, progress=lambda m: None)
    out = tmp_path / "audit.csv"
    assert judge.write_audit("m", out=out) == out
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 100 and len({r["question_id"] for r in rows}) == 100
    assert list(rows[0]) == judge.AUDIT_COLUMNS and all(
        r["human_correct?"] == "" and r["human_note"] == "" for r in rows
    )
    assert {r["judge_label"] for r in rows} == {"CORRECT"} and all(
        r["judge_provider"] for r in rows
    )
    # the user labels one row; a re-run must not touch that file
    rows[0]["human_correct?"] = "yes"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=judge.AUDIT_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    again = judge.write_audit("m", out=out)
    assert (
        again.name == "audit.new.csv"
        and list(csv.DictReader(out.open()))[0]["human_correct?"] == "yes"
    )


def test_claim_recheck_compares_nli_with_the_judge_on_sampled_claims(art, tmp_path):
    db = ingested_demo(tmp_path / "m.sqlite")
    uid = "synthetic:demo-alice-bob:s03:t002"
    p = common.raw_output_path("m", "S")
    for i in range(6):
        r = raw("S", f"cl{i}")
        r["verifier"] = {"rounds": [{"claims": [{"text": "Alice moved to Milwaukee.", "kept_turn_uids": [uid],
                                                  "nli_entailment": 0.9, "supported": True}]}]}  # fmt: skip
        append_jsonl(p, r)
    pool, clients = make_pool(
        labeller=lambda it: (
            "WRONG" if it["item_id"].endswith("|0") and "cl3" in it["item_id"] else "CORRECT"
        )
    )
    out = tmp_path / "recheck.jsonl"
    res = judge.recheck_claims(pool, "m", db, n=4, out=out)
    rows = read_jsonl(out)
    assert res["n"] == 4 and res["candidates"] == 6 and len(rows) == 4
    assert all(
        "On 2023-11-05, Alice said: I don't run anymore" in json.dumps(c.requests)
        for c in clients.values()
        if c.requests
    )
    assert {"nli_supported", "judge_supported", "provider", "claim", "turns"} <= set(rows[0])
    judge.recheck_claims(pool, "m", db, n=4, out=out)  # resumable: nothing re-judged
    assert len(read_jsonl(out)) == 4
    db.close()
