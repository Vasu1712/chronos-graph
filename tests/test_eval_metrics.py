"""Metrics on hand-built outputs whose correct values are computed by hand in the comments."""

import pytest

import eval.common as common
from eval.common import append_jsonl
from eval.metrics import (
    abstention_cells,
    answer_rows,
    citation_cells,
    compare_systems,
    ingest_cost,
    load_rows,
    prf,
    session_of,
    summarize_system,
)

U = "locomo:c:s01:t001"
V = "locomo:c:s02:t004"
W = "locomo:c:s03:t002"


def raw(qid, *, system="S", answer="x", abstained=False, unanswerable=False, gold=(), cited=(), context=(),
        first_round=(), calls=None, latency=2.0, category="single-hop", gold_answer="g"):  # fmt: skip
    claims = (
        [{"text": "c", "cited_turn_uids": list(cited), "cited_memory_ids": []}] if cited else []
    )
    rounds = (
        [{"round": 1, "claims": [{"supported": s} for s in first_round]}] if first_round else []
    )
    return {
        "system": system, "question_id": qid, "dataset": "locomo", "category": category, "conversation_id": "c",
        "gold_unanswerable": unanswerable, "abstained": abstained, "final_answer": answer, "gold_answer": gold_answer,
        "gold_evidence_turn_uids": list(gold), "claims": claims, "verifier": {"rounds": rounds},
        "context": {"turn_uids": list(context), "tokens": 100},
        "llm_calls": calls if calls is not None else [
            {"purpose": "GateOut", "prompt_tokens": 300, "completion_tokens": 10, "latency_s": 1.0, "wall_s": 1.0},
            {"purpose": "GenerationOut", "prompt_tokens": 1000, "completion_tokens": 50, "latency_s": 3.0, "wall_s": 3.0},
            {"purpose": "GenerationOut", "prompt_tokens": 1100, "completion_tokens": 40, "latency_s": 3.0, "wall_s": 3.0},
        ],
        "latency_s": latency,
    }  # fmt: skip


def judged(system, qid, correct, method="judge"):
    return {
        "type": "judgment",
        "system": system,
        "question_id": qid,
        "correct": correct,
        "method": method,
    }


def test_prf_and_session_parsing():
    assert prf({"a", "b"}, {"a", "c", "d"}) == (0.5, pytest.approx(1 / 3), pytest.approx(0.4))
    assert prf(set(), {"a"}) == (0.0, 0.0, 0.0) and prf({"a"}, {"a"}) == (1.0, 1.0, 1.0)
    assert session_of(U) == 1 and session_of(V) == 2 and session_of("garbage") is None


def test_citation_metrics_turn_and_session_level():
    # gold = {U, V}; cited {U, W}: turn P = 1/2, R = 1/2; sessions cited {1, 3} vs gold {1, 2}: P = R = 1/2
    r = raw("q1", gold=[U, V], cited=[U, W], context=[U, W, V])
    [row] = answer_rows([r], [judged("S", "q1", True)])
    assert (row["cit_p"], row["cit_r"], row["cit_f1"]) == (0.5, 0.5, 0.5)
    assert (row["sess_p"], row["sess_r"]) == (0.5, 0.5) and row["gold_in_context"] == 1.0
    # an answer that cites nothing scores 0; unanswerable questions and questions without evidence are excluded
    rows = answer_rows(
        [raw("q2", gold=[U], cited=[]), raw("q3", unanswerable=True, gold=[U]), raw("q4", gold=[])],
        [],
    )
    assert rows[0]["cit_f1"] == 0.0 and rows[1]["cit_f1"] is None and rows[2]["cit_f1"] is None


def test_citation_validity_support_rate_and_context_recall():
    rows = answer_rows([
        raw("a", gold=[U], cited=[U, "locomo:c:s09:t009"], context=[U], first_round=[True, False, True]),
        raw("b", gold=[V], cited=[V], context=[V, W], first_round=[True]),
    ], [])  # fmt: skip
    cells = citation_cells(rows)
    assert (
        cells["validity"]["value"] == pytest.approx(2 / 3) and cells["validity"]["n"] == 3
    )  # 3 cited, 2 valid
    assert (
        cells["support_rate"]["value"] == pytest.approx(3 / 4) and cells["support_rate"]["n"] == 4
    )
    assert cells["cit_f1"]["n"] == 2 and cells["gold_in_context"]["mean"] == 1.0


def test_abstention_metrics():
    # 4 unanswerable (3 abstained), 6 answerable (1 falsely abstained)
    rows = []
    for i in range(4):
        rows.append(raw(f"u{i}", unanswerable=True, abstained=i < 3))
    for i in range(6):
        rows.append(raw(f"a{i}", abstained=i == 0))
    cells = abstention_cells(answer_rows(rows, []))
    # TP = 3, predicted = 4, relevant = 4 -> P = R = 0.75, F1 = 2*3/(4+4) = 0.75; false abstention = 1/6
    assert cells["precision"]["value"] == pytest.approx(0.75) and cells["recall"][
        "value"
    ] == pytest.approx(0.75)
    assert cells["f1"]["value"] == pytest.approx(0.75)
    assert cells["false_abstention_rate"]["value"] == pytest.approx(1 / 6)
    assert (cells["n_unanswerable"], cells["n_answerable"]) == (4, 6)
    assert cells["recall"]["lo"] <= 0.75 <= cells["recall"]["hi"]


def test_tokens_latency_and_f1():
    rows = answer_rows([raw(f"q{i}", answer="Milwaukee Wisconsin", gold_answer="Milwaukee", latency=float(i + 1)) for i in range(10)],
                       [judged("S", f"q{i}", i < 7) for i in range(10)])  # fmt: skip
    s = summarize_system(rows)
    assert s["accuracy"]["n"] == 10 and s["accuracy"]["mean"] == pytest.approx(0.7)
    assert s["token_f1"]["mean"] == pytest.approx(2 * 0.5 * 1 / 1.5)  # P = 1/2, R = 1
    assert s["prompt_tokens_answer"]["mean"] == 1000  # the first GenerationOut call alone
    assert s["prompt_tokens_all"]["mean"] == 2400  # gate + both generation calls
    assert s["latency_s"]["p50"] == pytest.approx(5.5) and s["latency_s"]["p95"] == pytest.approx(
        9.55
    )


def test_unjudged_answers_are_not_counted_in_accuracy():
    rows = answer_rows([raw("q1"), raw("q2")], [judged("S", "q1", True)])
    assert summarize_system(rows)["accuracy"]["n"] == 1


def test_compare_systems_pairs_on_common_questions_only():
    a = [raw(f"q{i}", system="A") for i in range(40)]
    b = [raw(f"q{i}", system="B") for i in range(30)]  # B did not answer q30..q39
    js = [judged("A", f"q{i}", True) for i in range(40)] + [
        judged("B", f"q{i}", i % 3 == 0) for i in range(30)
    ]
    cmp = compare_systems(answer_rows(a + b, js), "A", "B")
    assert cmp["n"] == 30 and cmp["acc_a"] == 1.0 and cmp["acc_b"] == pytest.approx(10 / 30)
    assert (
        cmp["verdict"] == "first better"
        and cmp["mcnemar"]["only_a"] == 20
        and cmp["mcnemar"]["only_b"] == 0
    )
    assert cmp["mcnemar"]["p"] < 1e-5
    assert compare_systems(answer_rows(a, []), "A", "B") is None


def test_load_rows_takes_the_latest_record_and_separates_skips(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ARTIFACTS", tmp_path)
    p = common.raw_output_path("m", "S")
    append_jsonl(p, raw("q1", answer="old"))
    append_jsonl(p, raw("q1", answer="new"))
    append_jsonl(p, {"system": "S", "question_id": "q2", "skipped": "context_overflow"})
    append_jsonl(common.judgments_path("m"), judged("S", "q1", True))
    rows, skips = load_rows("m")
    assert [r["question_id"] for r in rows] == ["q1"] and rows[0]["correct"] is True
    assert skips[0]["skipped"] == "context_overflow"


def test_ingest_cost_aggregates_sessions_per_conversation_and_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ARTIFACTS", tmp_path)
    log = common.ingest_log_path("m")
    for conv, ds, secs in [
        ("c1", "locomo", 10.0),
        ("c1", "locomo", 20.0),
        ("c2", "longmemeval", 5.0),
    ]:
        append_jsonl(log, {"type": "session", "dataset": ds, "conversation_id": conv, "turns": 4, "seconds": secs,
                           "llm": {"prompt_tokens": 100, "completion_tokens": 10, "calls": 2, "llm_seconds": 3.0}})  # fmt: skip
    cost = ingest_cost("m")
    assert (
        cost["conversations"]["c1"]["seconds"] == 30.0
        and cost["conversations"]["c1"]["sessions"] == 2
    )
    assert (
        cost["by_dataset"]["locomo"]["prompt_tokens"] == 200
        and cost["by_dataset"]["longmemeval"]["conversations"] == 1
    )


def test_ingest_step_time_is_never_below_the_sum_of_session_times(tmp_path, monkeypatch):
    import eval.build_report as br

    monkeypatch.setattr(common, "ARTIFACTS", tmp_path)
    for secs in (30.0, 40.0):
        append_jsonl(common.ingest_log_path("m"), {"type": "session", "dataset": "locomo", "conversation_id": "c", "turns": 5, "seconds": secs,
                                                   "llm": {"prompt_tokens": 1, "completion_tokens": 1, "calls": 1, "llm_seconds": 1.0}})  # fmt: skip
    assert br.steps_with_ingest("m")["ingest"] == 70.0  # the killed segment logged nothing
    append_jsonl(
        common.step_timings_path("m"), {"step": "ingest", "seconds": 100.0, "status": "interrupted"}
    )
    assert (
        br.steps_with_ingest("m")["ingest"] == 100.0
    )  # a recorded segment is longer (startup, model load)
