"""make report: tables, figures and documents rebuilt from artifacts alone."""

import json
import socket
from pathlib import Path

import pytest

import eval.build_report as br
import eval.common as common
from eval import report_tables as T
from eval.common import append_jsonl

MEM_CONTEXT = (
    "## Alice\n"
    "- (memory c:m:1) Alice moved to Pune. | ACTIVE, residence | from 2023-01-01 (DAY) to now | sources: t1\n"
    "- (memory c:m:2) Alice lives in Delhi. | ACTIVE, job_title | from 2023-02-01 (DAY) to now | sources: t2\n"
    "- (memory c:m:3) Alice is visiting Goa for a week. | ACTIVE, residence | from 2023-03-01 (DAY) to now | sources: t3\n"
    "  [synthetic:c:s01:t001 | 2023-01-01 | Alice] hello"
)


def rec(
    system,
    i,
    *,
    dataset="locomo",
    category="single-hop",
    unanswerable=False,
    abstained=False,
    ok=True,
    ctx="",
):
    uid = "locomo:c:s01:t001"
    return {
        "system": system, "manifest": "quick_v1", "dataset": dataset, "question_id": f"q{i}", "conversation_id": "c",
        "category": category, "question": f"question {i}?", "gold_answer": "Paris", "gold_unanswerable": unanswerable,
        "gold_evidence_turn_uids": [] if unanswerable else [uid], "gold_evidence_sessions": [1],
        "final_answer": "I don't know." if abstained else ("Paris" if ok else "London"), "abstained": abstained,
        "abstained_at": "gate" if abstained else None, "abstention_reason": "gate said so" if abstained else None,
        "claims": [] if abstained else [{"text": "c", "cited_turn_uids": [uid], "cited_memory_ids": []}],
        "verifier": {"rounds": [] if abstained else [{"round": 1, "claims": [{"text": "c", "supported": True, "nli_entailment": 0.9,
                                                                            "kept_turn_uids": [uid]}]}]},  # fmt: skip
        "gate": None, "analysis": None,
        "context": {"tokens": 120, "budget": 1500, "truncated": False, "turn_uids": [uid], "memory_ids": [], "text": ctx},
        "llm_calls": [{"purpose": "GenerationOut", "prompt_tokens": 900, "completion_tokens": 40, "latency_s": 4.0, "wall_s": 0.01, "cached": True}],
        "timings": {"total": 0.05}, "latency_s": 4.5,
    }  # fmt: skip


def seed(tmp: Path, monkeypatch):
    monkeypatch.setattr(common, "ARTIFACTS", tmp / "artifacts")
    name = "quick_v1"
    n = 24

    def questions():  # (index, kwargs shared by every system)
        for i in range(n):
            if i < 4:
                yield i, {"category": "adversarial", "unanswerable": True}
            elif i < 8:
                yield i, {"dataset": "longmemeval", "category": "knowledge-update"}
            else:
                yield i, {"category": "multi-hop" if i % 2 else "single-hop"}

    systems = {  # system -> (correct on answerable?, abstains on unanswerable?)
        "chronos": (lambda i: i % 6 != 0, True),
        "B3_chunk_rag": (lambda i: i % 2 == 0, False),
        "B4_turns_plus_notes": (
            lambda i: i % 6 != 0,
            True,
        ),  # identical to chronos: no detectable difference
        "A_retrieval/dense_only": (lambda i: i % 3 != 0, True),
        "C_gate/off": (lambda i: i % 6 != 0, False),
        "D_budget/tokens_500": (lambda i: i % 4 != 0, True),
        "D_budget/tokens_1500": (lambda i: i % 6 != 0, True),
    }
    judgments = []
    for s, (good, abstains) in systems.items():
        for i, kw in questions():
            unans = kw.get("unanswerable", False)
            ok = good(i)
            r = rec(
                s,
                i,
                ok=ok,
                abstained=unans and abstains,
                ctx=MEM_CONTEXT if s == "chronos" else "",
                **kw,
            )
            if s == "chronos" and i == 9:  # an over-abstention on an answerable question
                r |= {
                    "abstained": True,
                    "final_answer": "I don't know.",
                    "abstained_at": "model",
                    "claims": [],
                }
            append_jsonl(common.raw_output_path(name, s), r)
            correct = r["abstained"] if unans else (not r["abstained"] and ok)
            judgments.append({"type": "judgment", "system": s, "question_id": f"q{i}", "category": kw.get("category", "single-hop"),
                              "dataset": kw.get("dataset", "locomo"), "correct": correct, "method": "judge" if not unans else "flag",
                              "provider": None if unans else "groq", "response_sha": "x"})  # fmt: skip
    append_jsonl(common.raw_output_path(name, "B2_full_context"), {"system": "B2_full_context", "question_id": "q8", "skipped": "context_overflow: prompt is 40000 tokens"} | {k: 0 for k in ()})  # fmt: skip
    for j in judgments:
        append_jsonl(common.judgments_path(name), j)
    for _ in range(3):
        append_jsonl(common.ingest_log_path(name), {"type": "session", "dataset": "locomo", "conversation_id": "conv-50", "turns": 20, "seconds": 60.0,
                                                    "llm": {"prompt_tokens": 9000, "completion_tokens": 600, "calls": 10, "llm_seconds": 50.0}})  # fmt: skip
    append_jsonl(common.ingest_log_path(name), {"type": "conversation", "conversation_id": "conv-50", "dataset": "locomo", "sessions": 3, "turns": 60, "ingest_seconds": 180,
                                                "entity_name_duplicates": [{"name": "alex", "type": "PERSON", "count": 2, "descriptors": ["brother", "coworker"]}]})  # fmt: skip
    append_jsonl(
        common.step_timings_path(name), {"step": "ingest", "seconds": 600.0, "status": "ok"}
    )
    append_jsonl(
        common.step_timings_path(name), {"step": "answer:chronos", "seconds": 120.0, "status": "ok"}
    )
    return name


@pytest.fixture
def built(tmp_path, monkeypatch):
    seed(tmp_path, monkeypatch)
    monkeypatch.setattr(br, "FIGURES", tmp_path / "figures")

    def no_network(*a, **k):
        raise AssertionError("make report must not touch the network")

    monkeypatch.setattr(socket, "socket", no_network)
    out = tmp_path / "docs_out"
    out.mkdir()
    summary = br.build(out_dir=out, manifests=["quick_v1"], with_estimate=False)
    return summary, out, tmp_path


def test_report_writes_all_documents_summary_and_figures_without_network(built):
    summary, out, tmp = built
    for doc in ("README.md", "EVAL.md", "DESIGN.md"):
        text = (out / doc).read_text()
        assert text.strip() and "{{" not in text, doc
    assert (
        json.loads((tmp / "artifacts" / "summary_metrics.json").read_text())["manifests"][
            "quick_v1"
        ]["name"]
        == "quick_v1"
    )
    assert (tmp / "figures" / "accuracy_by_category_quick_v1.png").stat().st_size > 2000
    assert "docs/figures/accuracy_by_category_quick_v1.png" in (out / "EVAL.md").read_text()


def test_every_number_in_the_documents_is_in_the_summary(built):
    summary, out, _ = built
    m = summary["manifests"]["quick_v1"]
    ev = (out / "EVAL.md").read_text()
    acc = m["systems"]["chronos"]["accuracy"]
    assert T.cell(acc) in ev  # the exact string, CI and n included
    assert f"n={acc['n']}" in ev
    for s, v in m["systems"].items():
        assert T.name(s) in ev and f"n={v['accuracy']['n']}" in ev
    assert (
        acc["n"] == 24 and m["questions"]["answers_scored"] == 24 * 3 + 24 * 4
    )  # scored answers over all runs


def test_intervals_whose_ci_crosses_zero_are_called_no_detectable_difference(built):
    summary, out, _ = built
    pw = {p["b"]: p for p in summary["manifests"]["quick_v1"]["pairwise"]}
    assert (
        pw["B4_turns_plus_notes"]["verdict"] == "no detectable difference"
        and abs(pw["B4_turns_plus_notes"]["mean_diff"]) < 0.1  # one injected over-abstention apart
    )
    assert pw["B3_chunk_rag"]["mean_diff"] > 0
    ev = (out / "EVAL.md").read_text()
    row = next(ln for ln in ev.splitlines() if "ChronosGraph vs B4 turns + notes" in ln)
    assert "no detectable difference" in row and "McNemar" not in row
    if pw["B3_chunk_rag"]["lo"] > 0:
        assert "ChronosGraph better" in next(
            ln for ln in ev.splitlines() if "ChronosGraph vs B3 chunk RAG" in ln
        )


def test_ablation_tables_use_the_reference_and_alias_the_full_system(built):
    summary, out, _ = built
    abl = summary["manifests"]["quick_v1"]["ablations"]
    assert [v["name"] for v in abl["retrieval"]["variants"]] == [
        "dense_only",
        "plus_episodic",
    ]  # the others were not run
    ref = next(v for v in abl["retrieval"]["variants"] if v["is_reference"])
    assert ref["system"] == "chronos" and ref["vs_reference"] is None
    assert {v["name"] for v in abl["gate"]["variants"]} == {"off", "full"}
    off = next(v for v in abl["gate"]["variants"] if v["name"] == "off")
    assert (
        off["gate"]["locomo_adversarial_abstained"]["mean"] == 0.0
        and off["gate"]["locomo_adversarial_abstained"]["n"] == 4
    )
    assert [v["budget"] for v in abl["budget"]["variants"]] == [500, 1500]
    assert (built[2] / "figures" / "accuracy_vs_budget_quick_v1.png").exists()
    ev = (out / "EVAL.md").read_text()
    assert "### A. Retrieval" in ev and "dense_only" in ev and "reference" in ev


def test_failure_modes_are_mined_from_the_raw_outputs(built):
    fm = built[0]["manifests"]["quick_v1"]["failure_modes"]
    assert fm["cross_attribute_contradiction"]["count"] >= 1
    assert any(
        "Pune" in f for e in fm["cross_attribute_contradiction"]["examples"] for f in e["facts"]
    )
    assert (
        fm["attribute_mismatch"]["count"] >= 1
        and fm["attribute_mismatch"]["examples"][0]["key"] == "job_title"
    )
    assert fm["temporary_stay"]["count"] >= 1
    assert (
        fm["over_abstention"]["count"] >= 1
        and fm["over_abstention"]["examples"][0]["abstained_at"] == "model"
    )
    assert fm["entity_split"]["count"] == 1 and fm["entity_split"]["examples"][0]["name"] == "alex"
    assert fm["nli_error"]["count"] == 0
    design = (built[1] / "DESIGN.md").read_text()
    assert (
        "#### cross attribute contradiction" in design
        and "_No instance in the evaluated sample._" in design
    )


def test_missing_measurements_say_so_instead_of_inventing_numbers(built):
    ev = (built[1] / "EVAL.md").read_text()
    assert "Not measured: Cohen" not in ev  # wording check below instead
    assert "Not measured: the calibration step has not run" in ev
    assert "Not measured: the claim re-check needs a judge provider" in ev
    assert "**not measured**" in ev  # the status banner
    assert "B2 full context | context_overflow | 1" in ev
    readme = (built[1] / "README.md").read_text()
    assert "Not measured: no basis run to project from" in readme or "Not measured" in readme


def test_documents_reproduce_exactly_from_the_same_artifacts(built, tmp_path, monkeypatch):
    summary, out, tmp = built
    first = {d: (out / d).read_text() for d in ("README.md", "EVAL.md", "DESIGN.md")}
    again = tmp / "again"
    again.mkdir()
    br.build(out_dir=again, manifests=["quick_v1"], with_estimate=False)
    assert {d: (again / d).read_text() for d in first} == first


def test_unknown_placeholders_fail_loudly():
    with pytest.raises(KeyError, match="nope"):
        br.fill("a {{nope}} b", {"x": "1"})
    assert br.fill("a {{ x }} b", {"x": "1"}) == "a 1 b"


def test_spec_sections_are_extracted_by_heading():
    assert "Closed single-valued attribute keys" in br.spec_section(
        "3."
    ) or "Ingestion Engine" in br.spec_section("3.")
    assert "CORRECTS" in br.spec_section("5.")
    with pytest.raises(KeyError):
        br.spec_section("99.")
