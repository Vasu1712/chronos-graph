"""Metrics over raw outputs + judgments. Pure functions; no model, database or network.

Per-answer rows (`answer_rows`) join one raw output with its judgment. Aggregators then compute each
metric with a bootstrap CI over questions and always carry `n`.

Definitions
  accuracy                 judged correct / answers scored. Unanswerable questions are correct iff the
                           system abstained; answerable ones need a correct answer (flag, exact, judge).
  token F1                 SQuAD-style F1 of the answer text vs the reference (secondary); 0 if abstained.
  abstention P / R / F1    positive = "system abstained"; relevant = unanswerable question.
  false abstention rate    abstained / answerable questions.
  citation P / R / F1      cited turns vs gold evidence turns (LoCoMo dia_id via source_ref, LongMemEval
                           has_answer turns), over answerable questions that have gold evidence; an
                           answer that cites nothing scores 0. Also at session level.
  Citation Validity        cited turns that exist and were in the packed context. TRUE BY CONSTRUCTION: the
                           verifier drops every other citation before the answer is returned.
  Citation Support Rate    share of generated claims whose NLI entailment passed the threshold in the
                           FIRST verification round (before repair).
  prompt tokens            Qwen-tokenized chat prompt: the first generation call alone, and every call.
  latency                  per question, with each LLM call at the latency it was recorded with.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from eval.common import (
    ingest_log_path,
    judgments_path,
    raw_output_path,
    read_jsonl,
    step_timings_path,
)
from eval.stats import bootstrap_ci, bootstrap_stat, mcnemar_exact, paired_bootstrap
from eval.textnorm import token_f1

_SESSION_OF_UID = re.compile(r":s(\d+):t\d+$")


def session_of(turn_uid: str) -> int | None:
    m = _SESSION_OF_UID.search(turn_uid)
    return int(m.group(1)) if m else None


# ------------------------------------------------------------------------------ rows
def load_outputs(manifest: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(answers, skips) from artifacts/raw_outputs/<manifest>/*.jsonl; the last record per
    (system, question) wins, so a re-run replaces an earlier answer."""
    base = raw_output_path(manifest, "x").parent
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(base.glob("*.jsonl")):
        for r in read_jsonl(path):
            latest[(r["system"], r["question_id"])] = r
    answers = [r for r in latest.values() if not r.get("skipped")]
    skips = [r for r in latest.values() if r.get("skipped")]
    return answers, skips


def prf(pred: set, gold: set) -> tuple[float, float, float]:
    hit = len(pred & gold)
    p = hit / len(pred) if pred else 0.0
    r = hit / len(gold) if gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def answer_rows(
    answers: Iterable[dict[str, Any]], judgments: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    judged = {(j["system"], j["question_id"]): j for j in judgments if j.get("type") == "judgment"}
    rows = []
    for r in answers:
        j = judged.get((r["system"], r["question_id"]))
        cited = {u for c in r["claims"] for u in c["cited_turn_uids"]}
        gold = set(r["gold_evidence_turn_uids"])
        in_ctx = set(r["context"]["turn_uids"])
        has_cit = (not r["gold_unanswerable"]) and bool(gold)
        cp, cr, cf = prf(cited, gold) if has_cit else (None, None, None)
        sp, sr, sf = (
            prf({s for u in cited if (s := session_of(u))}, {s for u in gold if (s := session_of(u))})
            if has_cit else (None, None, None)
        )  # fmt: skip
        rounds = (r.get("verifier") or {}).get("rounds") or []
        first = rounds[0]["claims"] if rounds and not rounds[0].get("abstained_by_model") else []
        calls = r.get("llm_calls", [])
        gen = [c for c in calls if c["purpose"] == "GenerationOut"]
        rows.append({
            "system": r["system"], "question_id": r["question_id"], "dataset": r["dataset"],
            "category": r["category"], "conversation_id": r["conversation_id"],
            "unanswerable": bool(r["gold_unanswerable"]), "abstained": bool(r["abstained"]),
            "correct": (None if j is None else bool(j["correct"])), "method": (j or {}).get("method"),
            "provider": (j or {}).get("provider"),
            "f1": 0.0 if r["abstained"] or r["gold_unanswerable"] else token_f1(r["final_answer"], str(r["gold_answer"])),
            "cit_p": cp, "cit_r": cr, "cit_f1": cf, "sess_p": sp, "sess_r": sr, "sess_f1": sf,
            "cited": len(cited), "cited_valid": len(cited & in_ctx),
            "claims_first": len(first), "claims_first_pass": sum(bool(c["supported"]) for c in first),
            "prompt_tokens_answer": gen[0]["prompt_tokens"] if gen else None,
            "prompt_tokens_all": sum(c["prompt_tokens"] for c in calls) if calls else None,
            "completion_tokens_all": sum(c["completion_tokens"] for c in calls) if calls else None,
            "latency_s": r.get("latency_s"), "context_tokens": r["context"]["tokens"],
            "gold_in_context": (len(gold & in_ctx) / len(gold)) if has_cit else None,
        })  # fmt: skip
    return rows


def load_rows(manifest: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    answers, skips = load_outputs(manifest)
    return answer_rows(answers, read_jsonl(judgments_path(manifest))), skips


# ------------------------------------------------------------------------------ aggregation
def ci_cell(values: list[float]) -> dict[str, Any]:
    m, lo, hi = bootstrap_ci(values)
    return {"n": len(values), "mean": m, "lo": lo, "hi": hi}


def ratio_cell(
    rows: list[dict[str, Any]], num: Callable[[dict], float], den: Callable[[dict], float]
) -> dict[str, Any]:
    n = np.array([num(r) for r in rows], dtype=float)
    d = np.array([den(r) for r in rows], dtype=float)

    def stat(idx: np.ndarray) -> float:
        dd = d[idx].sum()
        return float(n[idx].sum() / dd) if dd else float("nan")

    point, lo, hi = bootstrap_stat(len(rows), stat)
    return {"n": int(d.sum()), "n_questions": len(rows), "value": point, "lo": lo, "hi": hi}


def accuracy_cell(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return ci_cell([float(r["correct"]) for r in rows if r["correct"] is not None])


def abstention_cells(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = lambda r: float(r["abstained"] and r["unanswerable"])  # noqa: E731
    pred = lambda r: float(r["abstained"])  # noqa: E731
    rel = lambda r: float(r["unanswerable"])  # noqa: E731
    answerable = lambda r: float(not r["unanswerable"])  # noqa: E731
    fa = lambda r: float(r["abstained"] and not r["unanswerable"])  # noqa: E731
    p = ratio_cell(rows, tp, pred)
    rc = ratio_cell(rows, tp, rel)

    def f1_stat_rows(idx: np.ndarray) -> float:
        a = sum(tp(rows[i]) for i in idx)
        b = sum(pred(rows[i]) for i in idx)
        c = sum(rel(rows[i]) for i in idx)
        return 2 * a / (b + c) if (b + c) else float("nan")  # F1 = 2TP / (predicted + relevant)

    f1, lo, hi = bootstrap_stat(len(rows), f1_stat_rows)
    return {
        "precision": p, "recall": rc, "f1": {"value": f1, "lo": lo, "hi": hi, "n_questions": len(rows)},
        "false_abstention_rate": ratio_cell(rows, fa, answerable),
        "n_unanswerable": int(sum(rel(r) for r in rows)), "n_answerable": int(sum(answerable(r) for r in rows)),
    }  # fmt: skip


def citation_cells(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ev = [r for r in rows if r["cit_f1"] is not None]
    cells = {
        k: ci_cell([r[k] for r in ev])
        for k in ("cit_p", "cit_r", "cit_f1", "sess_p", "sess_r", "sess_f1")
    }
    cells["validity"] = ratio_cell(
        [r for r in rows if r["cited"]], lambda r: r["cited_valid"], lambda r: r["cited"]
    )
    cells["support_rate"] = ratio_cell(
        [r for r in rows if r["claims_first"]],
        lambda r: r["claims_first_pass"],
        lambda r: r["claims_first"],
    )
    cells["gold_in_context"] = ci_cell([r["gold_in_context"] for r in ev])
    return cells


def percentile_cell(values: list[float]) -> dict[str, Any]:
    v = np.array([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "p50": float(np.percentile(v, 50)),
        "p95": float(np.percentile(v, 95)),
    }


def summarize_system(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_answers": len(rows),
        "accuracy": accuracy_cell(rows),
        "token_f1": ci_cell([r["f1"] for r in rows if not r["unanswerable"]]),
        "abstention": abstention_cells(rows),
        "citation": citation_cells(rows),
        "prompt_tokens_answer": percentile_cell([r["prompt_tokens_answer"] for r in rows]),
        "prompt_tokens_all": percentile_cell([r["prompt_tokens_all"] for r in rows]),
        "latency_s": percentile_cell([r["latency_s"] for r in rows]),
        "context_tokens": percentile_cell([r["context_tokens"] for r in rows]),
    }


def group_rows(
    rows: list[dict[str, Any]], key: Callable[[dict], Any]
) -> dict[Any, list[dict[str, Any]]]:
    out: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[key(r)].append(r)
    return dict(out)


def compare_systems(
    rows: list[dict[str, Any]], a: str, b: str, *, subset: Callable[[dict], bool] | None = None
) -> dict[str, Any] | None:
    """Accuracy of system a minus system b on the questions both answered and were judged."""
    ra = {
        r["question_id"]: r
        for r in rows
        if r["system"] == a and r["correct"] is not None and (subset is None or subset(r))
    }
    rb = {
        r["question_id"]: r
        for r in rows
        if r["system"] == b and r["correct"] is not None and (subset is None or subset(r))
    }
    common = sorted(set(ra) & set(rb))
    if not common:
        return None
    xa = [float(ra[q]["correct"]) for q in common]
    xb = [float(rb[q]["correct"]) for q in common]
    res = paired_bootstrap(xa, xb)
    res["mcnemar"] = mcnemar_exact(np.array(xa, bool), np.array(xb, bool))
    res |= {"a": a, "b": b, "acc_a": float(np.mean(xa)), "acc_b": float(np.mean(xb))}
    return res


# ------------------------------------------------------------------------------ ingest cost
def ingest_cost(manifest: str, variant: str | None = None) -> dict[str, Any]:
    """Per-conversation ingest wall time and LLM tokens, from artifacts/ingest_logs."""
    sessions = [
        r for r in read_jsonl(ingest_log_path(manifest, variant)) if r.get("type") == "session"
    ]
    by_conv: dict[str, dict[str, Any]] = {}
    for s in sessions:
        c = by_conv.setdefault(s["conversation_id"], {"dataset": s["dataset"], "sessions": 0, "turns": 0, "seconds": 0.0,
                                                      "prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0, "llm_seconds": 0.0})  # fmt: skip
        c["sessions"] += 1
        c["turns"] += s["turns"]
        c["seconds"] += s["seconds"]
        c["prompt_tokens"] += s["llm"]["prompt_tokens"]
        c["completion_tokens"] += s["llm"]["completion_tokens"]
        c["llm_calls"] += s["llm"]["calls"]
        c["llm_seconds"] += s["llm"]["llm_seconds"]
    by_dataset: dict[str, dict[str, Any]] = {}
    for conv in by_conv.values():
        d = by_dataset.setdefault(conv["dataset"], {"conversations": 0, "sessions": 0, "turns": 0, "seconds": 0.0, "prompt_tokens": 0,
                                                    "completion_tokens": 0, "llm_calls": 0, "per_conversation_seconds": []})  # fmt: skip
        d["conversations"] += 1
        for k in (
            "sessions",
            "turns",
            "seconds",
            "prompt_tokens",
            "completion_tokens",
            "llm_calls",
        ):
            d[k] += conv[k]
        d["per_conversation_seconds"].append(conv["seconds"])
    return {"conversations": by_conv, "by_dataset": by_dataset}


def step_seconds(manifest: str) -> dict[str, float]:
    """Cumulative wall time per pipeline step over every recorded segment (a resumed or interrupted run
    adds its segments up), from artifacts/step_timings."""
    out: dict[str, float] = {}
    for r in read_jsonl(step_timings_path(manifest)):
        out[r["step"]] = out.get(r["step"], 0.0) + r["seconds"]
    return out
