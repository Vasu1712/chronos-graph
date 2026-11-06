import pytest

import eval.common as common
from chronos.ingestion.types import Conversation, Session, Turn
from eval.common import ManifestData, append_jsonl
from eval.estimate import estimate, fit_latency_model, human, predict
from eval.loaders.types import QAItem


def conv(cid, dataset, n_sessions, turns_per_session):
    sessions = tuple(
        Session(f"{cid}:s{s:02d}", s, "2023-01-01T00:00:00",
                tuple(Turn(f"{dataset}:{cid}:s{s:02d}:t{t:03d}", s, t, "A", "hello there", "x") for t in range(1, turns_per_session + 1)), "")
        for s in range(1, n_sessions + 1)
    )  # fmt: skip
    return Conversation(dataset, cid, ("A", "B"), sessions)


def test_latency_model_recovers_a_linear_cost():
    calls = [{"prompt_tokens": p, "completion_tokens": c, "latency_s": 0.5 + 0.001 * p + 0.04 * c}
             for p in range(500, 3000, 250) for c in (20, 60, 120)]  # fmt: skip
    m = fit_latency_model(calls)
    assert (
        m["a"] == pytest.approx(0.5, abs=1e-6)
        and m["b"] == pytest.approx(0.001)
        and m["c"] == pytest.approx(0.04)
    )
    assert m["r2"] == pytest.approx(1.0) and predict(m, 20000, 100) == pytest.approx(24.5)
    assert fit_latency_model(calls[:5]) is None  # too few points to fit


def test_projection_scales_measured_unit_costs(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ARTIFACTS", tmp_path)
    # base run: LoCoMo at 2 s/turn (10 turns per session), LongMemEval at 30 s/session
    for _ in range(4):
        append_jsonl(common.ingest_log_path("base"), {"type": "session", "dataset": "locomo", "conversation_id": "c1", "turns": 10, "seconds": 20.0,
                                                      "llm": {"prompt_tokens": 1, "completion_tokens": 1, "calls": 1, "llm_seconds": 1.0}})  # fmt: skip
    for _ in range(3):
        append_jsonl(common.ingest_log_path("base"), {"type": "session", "dataset": "longmemeval", "conversation_id": "q1", "turns": 8, "seconds": 30.0,
                                                      "llm": {"prompt_tokens": 1, "completion_tokens": 1, "calls": 1, "llm_seconds": 1.0}})  # fmt: skip
    calls = [{"purpose": "GenerationOut", "prompt_tokens": p, "completion_tokens": 50, "latency_s": 1 + 0.002 * p, "wall_s": 0.0}
             for p in range(800, 2400, 100)]  # fmt: skip
    for i in range(10):
        for s, lat in (("chronos", 12.0), ("B3_chunk_rag", 8.0)):
            append_jsonl(
                common.raw_output_path("base", s),
                {"system": s, "question_id": f"q{i}", "latency_s": lat, "llm_calls": calls[:2]},
            )
    append_jsonl(
        common.step_timings_path("base"),
        {"step": "judge", "status": "ok", "seconds": 50.0, "items": 10, "judge_tokens": 5000},
    )
    target = ManifestData("target", tmp_path / "t.json", {}, {
        "c9": conv("c9", "locomo", 5, 20), "lq": conv("lq", "longmemeval", 12, 6),
    }, [QAItem("locomo", f"t{i}", "c9", "?", "a", "single-hop") for i in range(30)]
       + [QAItem("longmemeval", f"l{i}", "lq", "?", "a", "knowledge-update") for i in range(10)])  # fmt: skip
    est = estimate("base", "target", target_data=target, only_providers=["groq"])
    ing = est["steps"]["ingest"]
    assert ing["parts"]["locomo"] == pytest.approx(2.0 * 100) and ing["parts"][
        "longmemeval"
    ] == pytest.approx(30.0 * 12)
    by = est["steps"]["answer_main"]["by_system"]
    assert by["chronos"] == pytest.approx(12.0 * 40) and by["B3_chunk_rag"] == pytest.approx(
        8.0 * 40
    )
    assert by["B5_append_only"] == by["chronos"] and by["B1_sliding_window"] == by["B3_chunk_rag"]
    assert (
        est["questions_per_system"]["B2_full_context"] == 30
    )  # the small LoCoMo history fits the window
    assert est["per_question_seconds"]["B2_full_context"] > 0
    j = est["steps"]["judge"]
    assert j["judge_share_of_answers"] == pytest.approx(10 / 20)
    assert j["floor_days_from_daily_token_limit"] == pytest.approx(j["tokens"] / 200_000)
    assert est["total_seconds"] == pytest.approx(
        sum(v["seconds"] for v in est["steps"].values() if v.get("seconds"))
    )


def test_human_durations():
    assert (human(30), human(600), human(7200), human(None)) == (
        "30 s",
        "10 min",
        "2.0 h",
        "not estimable",
    )
