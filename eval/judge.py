"""Score raw outputs: flags, exact matches, then the pooled gpt-oss-120b judge for the rest.

  uv run python -m eval.judge --manifest eval/manifests/quick_v1.json

Per (system, question), in this order:
  unanswerable question (LoCoMo cat. 5, LongMemEval _abs)  correct iff the system abstained   [flag]
  answerable question, system abstained                    wrong                                [flag]
  normalized answer == normalized reference                correct                              [exact]
  everything else                                          LongMemEval's released prompt, 8 per request,
                                                           routed by sha256(question_id)         [judge]

Outputs (all under artifacts/): judgments/<manifest>.jsonl (one line per scored answer, provider
recorded), judge_calibration.json (Cohen's kappa between providers on the same items),
judge_audit.csv (100 random verdicts for hand checking, never pre-filled) and
claim_recheck.jsonl (gpt-oss-120b re-check of sampled NLI verdicts). Re-running skips work already
done; the judge cache makes repeated verdicts free. Needs GROQ_API_KEY and/or CEREBRAS_API_KEY.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

from chronos.llm.cache import ResponseCache
from chronos.llm.judge_pool import JudgeItem, JudgePool
from eval.common import (
    append_jsonl,
    artifacts,
    judge_cache_path,
    judgments_path,
    load_manifest_data,
    memory_db_path,
    now_iso,
    raw_output_path,
    read_jsonl,
    stop_on_sigterm,
    timed_step,
)
from eval.prompts.lme_judge import batch_rubric, prompt_kind
from eval.stats import cohens_kappa
from eval.textnorm import exact_match

SEED = 42
AUDIT_N = 100
CALIBRATION_N = 100
RECHECK_N = 200

CLAIM_RUBRIC = """You check whether evidence from a conversation supports a claim.
Each item has a claim (response) and the evidence (reference_answer): the turns the claim cites,
each written "On <date>, <speaker> said: <text>". Answer CORRECT if the claim is stated in the
evidence or follows directly from it, and WRONG if the evidence does not state it, only mentions the
topic, or contradicts it. Use only the evidence, never outside knowledge. Judge each item independently."""


def response_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def load_raw(manifest: str) -> list[dict[str, Any]]:
    base = raw_output_path(manifest, "x").parent
    out: list[dict[str, Any]] = []
    for path in sorted(base.glob("*.jsonl")):
        out += read_jsonl(path)
    return out


def route(rec: dict[str, Any]) -> str:
    """How an answer is scored: flag | exact | judge."""
    if rec["gold_unanswerable"] or rec["abstained"]:
        return "flag"
    return "exact" if exact_match(rec["final_answer"], str(rec["gold_answer"])) else "judge"


def flag_judgment(rec: dict[str, Any]) -> dict[str, Any]:
    if rec["gold_unanswerable"]:
        ok = bool(rec["abstained"])
        why = "abstained on an unanswerable question" if ok else "answered an unanswerable question"
    else:
        ok, why = False, "abstained on an answerable question"
    return {"correct": ok, "method": "flag", "rationale": why}


def base_judgment(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "judgment", "system": rec["system"], "question_id": rec["question_id"],
        "category": rec["category"], "dataset": rec["dataset"], "response_sha": response_sha(rec["final_answer"]),
        "gold_unanswerable": rec["gold_unanswerable"], "abstained": rec["abstained"],
    }  # fmt: skip


def judge_item(rec: dict[str, Any]) -> JudgeItem:
    kind = prompt_kind(rec["dataset"], rec["category"])
    return JudgeItem(
        item_id=f"{rec['system']}|{rec['question_id']}", question_id=rec["question_id"],
        question=rec["question"], reference=str(rec["gold_answer"]), response=rec["final_answer"],
        rubric=batch_rubric(kind),
    )  # fmt: skip


def score_manifest(
    manifest: str, pool: JudgePool | None, *, progress=print, slice_size: int = 32
) -> dict[str, int]:
    """Write judgments for every raw output that lacks a current one. Returns counts per method."""
    path = judgments_path(manifest)
    done = {
        (j["system"], j["question_id"]): j for j in read_jsonl(path) if j.get("type") == "judgment"
    }
    counts = {"flag": 0, "exact": 0, "judge": 0, "skipped": 0, "pending_no_judge": 0}
    to_judge: list[dict[str, Any]] = []
    for rec in load_raw(manifest):
        if rec.get("skipped"):
            counts["skipped"] += 1
            continue
        prev = done.get((rec["system"], rec["question_id"]))
        if prev and prev["response_sha"] == response_sha(rec["final_answer"]):
            continue
        how = route(rec)
        if how == "flag":
            append_jsonl(path, base_judgment(rec) | flag_judgment(rec) | {"at": now_iso()})
            counts["flag"] += 1
        elif how == "exact":
            append_jsonl(
                path,
                base_judgment(rec)
                | {
                    "correct": True,
                    "method": "exact",
                    "rationale": "normalized exact match",
                    "at": now_iso(),
                },
            )
            counts["exact"] += 1
        else:
            to_judge.append(rec)
    if to_judge and pool is None:
        counts["pending_no_judge"] = len(to_judge)
        progress(f"{len(to_judge)} answers need the judge but no judge provider is configured")
        return counts
    for i in range(0, len(to_judge), slice_size):
        part = to_judge[i : i + slice_size]
        verdicts = pool.judge([judge_item(r) for r in part])
        for rec, v in zip(part, verdicts, strict=True):
            append_jsonl(path, base_judgment(rec) | {
                "correct": v.label == "CORRECT", "method": "judge", "provider": v.provider,
                "judge_model": v.model, "rationale": v.rationale, "cached": v.cached,
                "prompt_kind": prompt_kind(rec["dataset"], rec["category"]), "at": now_iso(),
            })  # fmt: skip
        counts["judge"] += len(part)
        progress(f"  judged {counts['judge']}/{len(to_judge)}")
    return counts


# ------------------------------------------------------------------------------ calibration
def calibrate(
    pool: JudgePool, manifest: str, *, n: int = CALIBRATION_N, out: Path | None = None
) -> dict[str, Any]:
    """Judge the same sampled items with every provider and report Cohen's kappa."""
    out = out or artifacts() / "judge_calibration.json"
    names = [p.config.name for p in pool.providers]
    if len(names) < 2:
        res = {"status": "not_measured", "reason": f"needs two providers; only {names} available "
               f"(not configured: {pool.unavailable})", "n": 0}  # fmt: skip
    else:
        recs = [r for r in load_raw(manifest) if not r.get("skipped") and route(r) == "judge"]
        sample = random.Random(f"{SEED}:calibration").sample(recs, min(n, len(recs)))
        items = [judge_item(r) for r in sample]
        labels = {
            name: [v.label == "CORRECT" for v in pool.judge(items, provider=name)] for name in names
        }
        a, b = labels[names[0]], labels[names[1]]
        res = {
            "status": "measured", "providers": names[:2], "n": len(sample), "kappa": cohens_kappa(a, b),
            "agreement": sum(x == y for x, y in zip(a, b, strict=True)) / max(1, len(a)),
            "positive_rate": {k: sum(v) / max(1, len(v)) for k, v in labels.items()},
        }  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    return res


# ------------------------------------------------------------------------------ hand audit
AUDIT_COLUMNS = ["system", "question_id", "category", "question", "gold_answer", "system_answer",
                 "judge_label", "judge_provider", "judge_rationale", "human_correct?", "human_note"]  # fmt: skip


def write_audit(manifest: str, *, n: int = AUDIT_N, out: Path | None = None) -> Path | None:
    """100 random judge verdicts for the user to check by hand. Never overwrites a labelled file."""
    out = out or artifacts() / "judge_audit.csv"
    if out.exists() and any(
        (r.get("human_correct?") or "").strip() for r in csv.DictReader(out.open(encoding="utf-8"))
    ):
        # the existing file holds hand labels: leave it alone and write beside it
        out = out.with_name(f"{out.stem}.new{out.suffix}")
    raw = {(r["system"], r["question_id"]): r for r in load_raw(manifest) if not r.get("skipped")}
    judged = [j for j in read_jsonl(judgments_path(manifest)) if j.get("method") == "judge"]
    if not judged:
        return None
    sample = random.Random(f"{SEED}:audit").sample(judged, min(n, len(judged)))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS)
        w.writeheader()
        for j in sample:
            r = raw[(j["system"], j["question_id"])]
            w.writerow({
                "system": j["system"], "question_id": j["question_id"], "category": j["category"],
                "question": r["question"], "gold_answer": r["gold_answer"], "system_answer": r["final_answer"],
                "judge_label": "CORRECT" if j["correct"] else "WRONG", "judge_provider": j.get("provider"),
                "judge_rationale": j.get("rationale"), "human_correct?": "", "human_note": "",
            })  # fmt: skip
    return out


# ------------------------------------------------------------------------------ claim re-check
def claim_candidates(manifest: str) -> list[dict[str, Any]]:
    """Every claim that went through NLI in the final verifier round of an answer."""
    out = []
    for r in load_raw(manifest):
        if r.get("skipped") or not r.get("verifier"):
            continue
        rounds = r["verifier"].get("rounds") or []
        if not rounds:
            continue
        for i, c in enumerate(rounds[-1].get("claims", [])):
            if c.get("nli_entailment") is not None and c.get("kept_turn_uids"):
                out.append({"system": r["system"], "question_id": r["question_id"], "conversation_id": r["conversation_id"],
                            "claim_index": i, "claim": c["text"], "turns": c["kept_turn_uids"],
                            "nli_entailment": c["nli_entailment"], "nli_supported": bool(c["supported"])})  # fmt: skip
    return out


def premise_for(db, turn_uids: list[str]) -> str:
    rows = {
        r[0]: r
        for r in db.conn.execute(
            f"SELECT turn_uid, session_ts, speaker, text FROM episodes WHERE turn_uid IN ({','.join('?' * len(turn_uids))})",
            turn_uids,
        )
    }
    return " ".join(
        f"On {(rows[u][1] or '')[:10]}, {rows[u][2]} said: {rows[u][3]}"
        for u in turn_uids
        if u in rows
    )


def recheck_claims(
    pool: JudgePool, manifest: str, db, *, n: int = RECHECK_N, out: Path | None = None
) -> dict[str, Any]:
    out = out or artifacts() / "claim_recheck" / f"{manifest}.jsonl"
    cands = claim_candidates(manifest)
    sample = random.Random(f"{SEED}:recheck").sample(cands, min(n, len(cands)))
    have = {(r["system"], r["question_id"], r["claim_index"]) for r in read_jsonl(out)}
    todo = [c for c in sample if (c["system"], c["question_id"], c["claim_index"]) not in have]
    items = [
        JudgeItem(
            item_id=f"{c['system']}|{c['question_id']}|{c['claim_index']}",
            question_id=c["question_id"],
            question="Is the claim supported by the evidence?",
            reference=premise_for(db, c["turns"]),
            response=c["claim"],
            rubric=CLAIM_RUBRIC,
        )  # fmt: skip
        for c in todo
    ]
    for i in range(0, len(items), 32):
        batch = items[i : i + 32]
        for c, item, v in zip(todo[i : i + 32], batch, pool.judge(batch), strict=True):
            append_jsonl(
                out,
                c
                | {
                    "judge_supported": v.label == "CORRECT",
                    "provider": v.provider,
                    "rationale": v.rationale,
                    "premise": item.reference,
                },
            )
    rows = read_jsonl(out)
    return {"n": len(rows), "candidates": len(cands)}


def build_pool(*, available_only: bool = True) -> JudgePool:
    return JudgePool(ResponseCache(judge_cache_path()), available_only=available_only)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--skip-calibration", action="store_true")
    ap.add_argument("--skip-recheck", action="store_true")
    args = ap.parse_args(argv)
    stop_on_sigterm()
    name = load_manifest_data(args.manifest, use_cache=True).name
    try:
        pool: JudgePool | None = build_pool()
    except RuntimeError as exc:
        print(f"judge unavailable: {exc}")
        pool = None
    if pool is not None and pool.unavailable:
        print(
            f"note: no API key for {pool.unavailable}; all questions are routed to {[p.config.name for p in pool.providers]}"
        )
    with timed_step(name, "judge") as info:
        counts = score_manifest(name, pool, progress=lambda m: print(m, flush=True))
        info |= {"items": counts["judge"], "scored": counts, "judge_tokens": sum(p.tokens for p in pool.providers) if pool else 0,
                 "judge_calls": sum(p.calls for p in pool.providers) if pool else 0}  # fmt: skip
    print("scored:", counts)
    if pool is None:
        return 1
    if not args.skip_calibration:
        with timed_step(name, "judge_calibration"):
            print("calibration:", calibrate(pool, name))
    audit = write_audit(name)
    print("audit sheet:", audit or "nothing judged yet")
    if not args.skip_recheck:
        from chronos.storage.db import Database

        with timed_step(name, "claim_recheck"):
            db = Database(memory_db_path(name))
            print("claim re-check:", recheck_claims(pool, name, db))
    return 0


if __name__ == "__main__":
    sys.exit(main())
