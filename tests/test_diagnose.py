import json

from eval import common, diagnose


def row(system, qid, *, abstained, stage=None, unanswerable=False, shown=True):
    return {
        "system": system, "question_id": qid, "question": f"q {qid}", "gold_answer": "a",
        "gold_unanswerable": unanswerable, "gold_evidence_turn_uids": ["x:s01:t001"],
        "abstained": abstained, "abstained_at": stage, "abstention_reason": "no info",
        "context": {"text": "[x:s01:t001 | 2023-01-01 | A] hi" if shown else "other"},
        "verifier": {"forced_answer_retries": 1},
    }  # fmt: skip


def test_diagnose_splits_generator_faults_from_retrieval_misses(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(common, "ARTIFACTS", tmp_path)
    d = tmp_path / "raw_outputs" / "m"
    d.mkdir(parents=True)
    rows = [
        row("chronos", "1", abstained=True, stage="model", shown=True),  # generator's fault
        row("chronos", "2", abstained=True, stage="model", shown=False),  # retrieval miss
        row("chronos", "3", abstained=False, shown=True),
        row("chronos", "4", abstained=True, stage="gate", unanswerable=True),  # not counted
        {"system": "chronos", "question_id": "5", "skipped": "x"},
    ]
    (d / "chronos.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    assert diagnose.main(["--manifest", "m"]) == 0
    out = capsys.readouterr().out
    assert "66.7% (2/3)" in out  # declined: 2 of 3 answerable
    assert "66.7% (2/3)" in out and "50.0% (1/2)" in out  # shown 2/3, declined|shown 1/2
    assert "q 1" in out and "q 2" not in out  # example list has only declines with gold shown
