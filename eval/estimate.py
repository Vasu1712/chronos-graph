"""Project the runtime of a larger run from the per-unit costs measured on a smaller one.

  measured (artifacts/step_timings, ingest_logs, raw_outputs)  ->  projected seconds per step

Method, so every number is reproducible and every assumption visible:
  ingest      LoCoMo: seconds per turn; LongMemEval: seconds per session (both from the measured run's
              logs) times the target manifest's turns / sessions.
  answer      systems measured in the base run use their own mean latency per question; the others are
              predicted from a least-squares model fitted on all recorded LLM calls,
              latency = a + b * prompt_tokens + c * completion_tokens.
              B1, B3, B4 reuse B3's mean (same prompt size and one generation call); B5 reuses ChronosGraph's.
              B2 (whole history in the prompt) uses the fitted model at its mean prompt size, as an
              upper bound: it ignores Ollama's prompt-prefix reuse across questions of one conversation.
  ablations   (variants x questions in the filter) x ChronosGraph's mean latency.
  judge       measured seconds per judged item from the judge step; items = answers that need the judge,
              at the share observed in the base run; plus calibration and claim re-check items.
              The daily token limits of the available providers set a floor in days.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from chronos.config import judge_providers
from chronos.llm.tokens import count_tokens
from eval.ablations import Ablation, load_ablations
from eval.common import (
    ManifestData,
    ingest_log_path,
    load_manifest_data,
    read_jsonl,
    step_timings_path,
)
from eval.metrics import load_outputs
from eval.systems import MAIN_SYSTEMS

REPLAY_VARIANTS = ("append_only", "arrival_order")
CALIBRATION_ITEMS = 100
RECHECK_ITEMS = 200


def fit_latency_model(calls: list[dict[str, Any]]) -> dict[str, Any] | None:
    pts = [
        (c["prompt_tokens"], c["completion_tokens"], c["latency_s"])
        for c in calls
        if c.get("latency_s") is not None
    ]
    if len(pts) < 12:
        return None
    x = np.array([[1.0, p, q] for p, q, _ in pts])
    y = np.array([t for _, _, t in pts])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    ss_res, ss_tot = float(((y - pred) ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    return {
        "a": float(coef[0]),
        "b": float(coef[1]),
        "c": float(coef[2]),
        "r2": 1 - ss_res / ss_tot if ss_tot else 0.0,
        "n_calls": len(pts),
    }


def predict(model: dict[str, Any], prompt_tokens: float, completion_tokens: float) -> float:
    return max(0.0, model["a"] + model["b"] * prompt_tokens + model["c"] * completion_tokens)


def _count_questions(ab: Ablation, data: ManifestData) -> int:
    return sum(ab.selects(q) for q in data.qas)


def estimate(
    base: str,
    target: str,
    target_data: ManifestData | None = None,
    *,
    only_providers: list[str] | None = None,
) -> dict[str, Any]:
    """Seconds per step for `target`, projected from the measured run `base`."""
    notes: list[str] = []
    answers, _ = load_outputs(base)
    sessions = [r for r in read_jsonl(ingest_log_path(base)) if r.get("type") == "session"]
    out: dict[str, Any] = {"base": base, "target": target, "assumptions": [], "steps": {}}
    if target_data is None:
        target_data = load_manifest_data(f"eval/manifests/{target}.json")

    # ---- ingest
    ingest = {"seconds": None}
    secs = {"locomo": [0.0, 0, 0], "longmemeval": [0.0, 0, 0]}  # seconds, turns, sessions
    for s in sessions:
        d = secs.setdefault(s["dataset"], [0.0, 0, 0])
        d[0] += s["seconds"]
        d[1] += s["turns"]
        d[2] += 1
    total = 0.0
    parts = {}
    lo_turns = sum(
        len(s.turns)
        for c in target_data.conversations.values()
        if c.dataset == "locomo"
        for s in c.sessions
    )
    lme_sessions = sum(
        len(c.sessions) for c in target_data.conversations.values() if c.dataset == "longmemeval"
    )
    if lo_turns:
        if secs["locomo"][1]:
            parts["locomo"] = secs["locomo"][0] / secs["locomo"][1] * lo_turns
        else:
            notes.append("no LoCoMo ingest measured: LoCoMo ingest not projected")
    if lme_sessions:
        if secs["longmemeval"][2]:
            parts["longmemeval"] = secs["longmemeval"][0] / secs["longmemeval"][2] * lme_sessions
        else:
            notes.append("no LongMemEval ingest measured: LongMemEval ingest not projected")
    total = sum(parts.values())
    ingest |= {"seconds": total or None, "parts": parts, "locomo_turns": lo_turns, "longmemeval_sessions": lme_sessions,
               "rates": {"locomo_s_per_turn": secs["locomo"][0] / secs["locomo"][1] if secs["locomo"][1] else None,
                         "longmemeval_s_per_session": secs["longmemeval"][0] / secs["longmemeval"][2] if secs["longmemeval"][2] else None}}  # fmt: skip
    out["steps"]["ingest"] = ingest

    # ---- answering
    by_system: dict[str, list[dict[str, Any]]] = {}
    for r in answers:
        by_system.setdefault(r["system"], []).append(r)
    mean_latency = {
        s: float(np.mean([r["latency_s"] for r in rs]))
        for s, rs in by_system.items()
        if "/" not in s and rs
    }
    all_calls = [c for rs in by_system.values() for r in rs for c in r["llm_calls"]]
    model = fit_latency_model(all_calls)
    out["latency_model"] = model
    n_target = len(target_data.qas)
    n_locomo_fit = None
    answer: dict[str, float | None] = {}
    chronos_lat = mean_latency.get("chronos")
    b3_lat = mean_latency.get("B3_chunk_rag")
    for sys_id in MAIN_SYSTEMS:
        if sys_id in mean_latency:
            lat, n = mean_latency[sys_id], n_target
        elif sys_id == "B5_append_only" and chronos_lat:
            lat, n = chronos_lat, n_target
        elif sys_id in ("B1_sliding_window", "B4_turns_plus_notes") and (b3_lat or chronos_lat):
            lat, n = (b3_lat or chronos_lat), n_target
        elif sys_id == "B2_full_context" and model:
            fit = [c for c in target_data.conversations.values() if c.dataset == "locomo"]
            toks = []
            for c in fit:
                text = "\n".join(
                    f"[{t.turn_uid} | {s.timestamp[:10]} | {t.speaker}] {t.display_text}"
                    for s in c.sessions
                    for t in s.turns
                )
                n_tok = count_tokens(text)
                if n_tok + 900 <= 32768 - 1024:
                    toks.append((c.conversation_id, n_tok))
            fits = {cid for cid, _ in toks}
            n = sum(q.conversation_id in fits for q in target_data.qas)
            comp = float(
                np.mean(
                    [c["completion_tokens"] for c in all_calls if c["purpose"] == "GenerationOut"]
                    or [100]
                )
            )
            lat = predict(model, float(np.mean([t for _, t in toks])) + 900, comp) if toks else None
            n_locomo_fit = len(fits)
        else:
            lat, n = None, 0
        answer[sys_id] = (lat * n) if lat is not None else None
        out.setdefault("per_question_seconds", {})[sys_id] = lat
        out.setdefault("questions_per_system", {})[sys_id] = n
    out["steps"]["answer_main"] = {
        "by_system": answer,
        "seconds": sum(v for v in answer.values() if v),
    }
    out["locomo_conversations_fitting_32k"] = n_locomo_fit

    # ---- ablations
    abl_seconds = 0.0
    abl_detail: dict[str, Any] = {}
    for ab in load_ablations().values():
        nq = _count_questions(ab, target_data)
        runs = sum(1 for v in ab.variants if not v.alias_of)
        sec = nq * runs * chronos_lat if chronos_lat else None
        abl_detail[ab.id] = {"questions": nq, "runs": runs, "seconds": sec}
        abl_seconds += sec or 0.0
    out["steps"]["ablations"] = {"seconds": abl_seconds or None, "by_ablation": abl_detail}

    # ---- replays and synthetic probes
    replay = sum(
        v["seconds"]
        for v in read_jsonl(step_timings_path(base))
        if str(v.get("step", "")).startswith("replay:") and v.get("status") == "ok"
    )
    base_convs = len({s["conversation_id"] for s in sessions})
    if replay and base_convs:
        out["steps"]["replays"] = {"seconds": replay / base_convs * len(target_data.conversations)}

    # ---- judge
    judge_step = next(
        (
            r
            for r in reversed(read_jsonl(step_timings_path(base)))
            if r["step"] == "judge" and r.get("status") == "ok" and r.get("items")
        ),
        None,
    )
    if judge_step:
        n_answers_base = sum(len(rs) for s, rs in by_system.items() if "/" not in s)
        share = judge_step["items"] / n_answers_base if n_answers_base else 0.0
        answers_target = sum(out["questions_per_system"].values()) + sum(
            d["questions"] * d["runs"] for d in abl_detail.values()
        )  # fmt: skip
        items = share * answers_target + 2 * CALIBRATION_ITEMS + RECHECK_ITEMS
        per_item = judge_step["seconds"] / judge_step["items"]
        tokens = judge_step.get("judge_tokens", 0) / judge_step["items"] * items
        tpd = sum(
            p.limits.tpd
            for p in judge_providers()
            if (only_providers is None or p.name in only_providers)
        )
        out["steps"]["judge"] = {
            "seconds": per_item * items, "items": items, "judge_share_of_answers": share, "tokens": tokens,
            "floor_days_from_daily_token_limit": (tokens / tpd) if tpd else None, "providers": only_providers,
        }  # fmt: skip
    else:
        notes.append("the judge step has not been measured: judge time not projected")

    seconds = [v["seconds"] for v in out["steps"].values() if v.get("seconds")]
    out["total_seconds"] = sum(seconds)
    out["notes"] = notes
    return out


def human(seconds: float | None) -> str:
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return "not estimable"
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"
