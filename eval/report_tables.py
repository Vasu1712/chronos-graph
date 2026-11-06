"""Markdown renderers for the report. They read only the summary dict (artifacts/summary_metrics.json),
so every number in the docs is a number in that file. Every cell carries its n."""

from __future__ import annotations

import math
from typing import Any

from eval.estimate import human

SYSTEM_NAMES = {
    "chronos": "ChronosGraph",
    "B1_sliding_window": "B1 sliding window",
    "B2_full_context": "B2 full context",
    "B3_chunk_rag": "B3 chunk RAG",
    "B4_turns_plus_notes": "B4 turns + notes",
    "B5_append_only": "B5 append-only",
    "B6_mem0": "B6 mem0",
}
SYSTEM_DESCRIPTIONS = {
    "chronos": "Full system: bi-temporal state, hybrid retrieval, answerability gate, entailment-checked citations.",
    "B1_sliding_window": "Most recent turns that fit the budget.",
    "B2_full_context": "The whole history in the prompt (LoCoMo conversations that fit the window only).",
    "B3_chunk_rag": "Three-turn chunks, bge-small embeddings, top chunks until the budget is full.",
    "B4_turns_plus_notes": "Dense + BM25 over raw turns and over dated fact notes (the extracted memories with no status, "
    "versions or temporal filter). The strongest simple baseline.",
    "B5_append_only": "ChronosGraph over a store where nothing is ever superseded: every non-retracted memory stays current.",
    "B6_mem0": "mem0 (open source) as an external reference point. Not run: its memories carry no turn provenance, so the "
    "shared turn-citation verifier would drop every claim; a fair run needs a verifier exemption the other systems do not get.",
}
VERDICT_TEXT = {"first better": "{a} better", "second better": "{b} better", "no detectable difference": "no detectable difference",
                "not measurable": "not measurable"}  # fmt: skip


def name(system: str) -> str:
    if "/" in system:
        head, _, tail = system.partition("/")
        return f"{head.split('_', 1)[-1]}: {tail}"
    return SYSTEM_NAMES.get(system, system)


def _ok(x: Any) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def pct(x: Any) -> str:
    return f"{100 * x:.1f}%" if _ok(x) else "–"


def cell(c: dict[str, Any] | None, *, value: str = "mean", n: str = "n") -> str:
    """'71.4% [62.1, 79.3] n=70' from a CI cell ({n, mean|value, lo, hi})."""
    if not c or not _ok(c.get(value)):
        return f"– n={c.get(n, 0) if c else 0}"
    return f"{pct(c[value])} [{100 * c['lo']:.1f}, {100 * c['hi']:.1f}] n={c[n]}"


def plain(c: dict[str, Any] | None, key: str = "mean", digits: int = 2) -> str:
    if not c or not _ok(c.get(key)):
        return "–"
    return f"{c[key]:.{digits}f} (n={c['n']})"


def num(x: Any, digits: int = 0) -> str:
    return f"{x:,.{digits}f}" if _ok(x) else "–"


def table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_no data_"
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def verdict(cmp: dict[str, Any], a: str, b: str) -> str:
    return VERDICT_TEXT[cmp["verdict"]].format(a=name(a), b=name(b))


def not_measured(why: str) -> str:
    return f"_Not measured: {why}._"


# ------------------------------------------------------------------------------ main tables
def manifest_table(m: dict[str, Any]) -> str:
    rows = [[k, str(n)] for k, n in sorted(m["questions"]["by_dataset_category"].items())]
    rows.append(["**total**", str(m["questions"]["total"])])
    return table(["dataset/category", "n questions"], rows)


def main_results(m: dict[str, Any]) -> str:
    rows = []
    for s, v in m["systems"].items():
        a = v["abstention"]
        rows.append([
            name(s), cell(v["accuracy"]), plain(v["token_f1"], "mean", 3),
            cell(a["precision"], value="value"), cell(a["recall"], value="value"), cell(a["f1"], value="value", n="n_questions"),
            cell(a["false_abstention_rate"], value="value"),
        ])  # fmt: skip
    return table(
        [
            "system",
            "accuracy [95% CI]",
            "token F1 (n)",
            "abstention precision",
            "abstention recall",
            "abstention F1",
            "false abstention",
        ],
        rows,
    )


def category_results(m: dict[str, Any]) -> str:
    cats = sorted({c for v in m["by_category"].values() for c in v})
    rows = []
    for s, v in m["by_category"].items():
        rows.append(
            [name(s)]
            + [cell(v.get(c)) if c in v else "–" for c in cats]
            + [cell(m["systems"][s]["accuracy"])]
        )
    return table(["system"] + cats + ["overall"], rows)


def pairwise_results(m: dict[str, Any]) -> str:
    rows = []
    for c in m["pairwise"]:
        mc = c["mcnemar"]
        rows.append([
            f"{name(c['a'])} vs {name(c['b'])}", str(c["n"]), pct(c["acc_a"]), pct(c["acc_b"]),
            f"{100 * c['mean_diff']:+.1f} pts [{100 * c['lo']:+.1f}, {100 * c['hi']:+.1f}]",
            f"{mc['only_a']} / {mc['only_b']}", f"{mc['p']:.3f}", verdict(c, c["a"], c["b"]),
        ])  # fmt: skip
    return table(
        [
            "pair",
            "n paired",
            "acc A",
            "acc B",
            "difference [95% CI]",
            "only A / only B correct",
            "McNemar p",
            "verdict",
        ],
        rows,
    )


def citation_results(m: dict[str, Any]) -> str:
    rows = []
    for s, v in m["systems"].items():
        c = v["citation"]
        rows.append([
            name(s), cell(c["cit_p"]), cell(c["cit_r"]), cell(c["cit_f1"]), cell(c["sess_f1"]),
            cell(c["validity"], value="value"), cell(c["support_rate"], value="value"), cell(c["gold_in_context"]),
        ])  # fmt: skip
    return table(
        ["system", "turn precision", "turn recall", "turn F1", "session F1", "citation validity (true by construction)",
         "citation support rate (NLI, first round)", "gold evidence in context"], rows)  # fmt: skip


def cost_results(m: dict[str, Any]) -> str:
    rows = []
    for s, v in m["systems"].items():
        rows.append([
            name(s), plain(v["prompt_tokens_answer"], "mean", 0), plain(v["prompt_tokens_all"], "mean", 0),
            plain(v["context_tokens"], "mean", 0), plain(v["latency_s"], "p50", 1), plain(v["latency_s"], "p95", 1),
        ])  # fmt: skip
    return table(
        [
            "system",
            "prompt tokens, answer call",
            "prompt tokens, all calls",
            "context tokens",
            "latency p50 (s)",
            "latency p95 (s)",
        ],
        rows,
    )


def ingest_results(m: dict[str, Any]) -> str:
    rows = []
    for ds, d in sorted(m["ingest"]["by_dataset"].items()):
        n = d["conversations"]
        secs = d["per_conversation_seconds"]
        rows.append([
            ds, str(n), str(d["sessions"]), str(d["turns"]), f"{human(sum(secs) / n)} (n={n})",
            f"{num(d['prompt_tokens'] / n)} / {num(d['completion_tokens'] / n)} (n={n})", f"{num(d['llm_calls'] / n, 1)} (n={n})",
        ])  # fmt: skip
    return table(
        [
            "dataset",
            "conversations",
            "sessions",
            "turns",
            "wall time per conversation",
            "LLM tokens per conversation (prompt / completion)",
            "LLM calls per conversation",
        ],
        rows,
    )


def skips_results(m: dict[str, Any]) -> str:
    if not m["skips"]:
        return "_No system skipped a question._"
    rows = [
        [name(s), reason, str(n)] for s, d in m["skips"].items() for reason, n in sorted(d.items())
    ]
    return table(["system", "skipped because", "n questions"], rows)


# ------------------------------------------------------------------------------ ablations
def _ablation_head(ab: dict[str, Any]) -> str:
    return f"{ab['title']}. {ab['question']} Questions: {ab['population']}."


def ablation_table(ab: dict[str, Any], *, extra: str = "") -> str:
    rows = []
    for v in ab["variants"]:
        d = v["vs_reference"]
        diff = "reference" if v["is_reference"] else (
            "–" if d is None else f"{100 * d['mean_diff']:+.1f} pts [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}] (n={d['n']}): {verdict(d, v['system'], ab['reference_system'])}"
        )  # fmt: skip
        rows.append(
            [
                v["name"],
                cell(v["accuracy"]),
                cell(v["gold_in_context"]),
                plain(v["context_tokens"], "mean", 0),
                diff,
            ]
        )
    return _ablation_head(ab) + "\n\n" + table(
        ["variant", "accuracy [95% CI]", "gold evidence in context", "context tokens", "vs reference"], rows) + extra  # fmt: skip


def gate_table(ab: dict[str, Any]) -> str:
    rows = []
    for v in ab["variants"]:
        g = v["gate"]
        rows.append([v["name"], cell(g["locomo_adversarial_abstained"]), cell(g["longmemeval_abs_abstained"]),
                     cell(g["false_abstention"]), cell(v["accuracy"])])  # fmt: skip
    return _ablation_head(ab) + "\n\n" + table(
        ["gate", "abstained on LoCoMo adversarial", "abstained on LongMemEval _abs", "false abstention on answerable", "accuracy (all questions)"], rows)  # fmt: skip


def probe_table(p: dict[str, Any] | None) -> str:
    if not p:
        return not_measured(
            "run `make eval-full` (or `python -m eval.state_probes`) to produce the synthetic probe results"
        )
    pols = ["append_only", "arrival_order", "bitemporal"]
    cases = sorted({c for pol in pols for c in p["per_policy_case"].get(pol, {})})
    rows = []
    for c in cases:
        row = [c]
        for pol in pols:
            d = p["per_policy_case"][pol][c]
            row.append(
                f"{d['passed']}/{d['total']} strict, {d['current_ok']}/{d['total']} current value"
            )
        rows.append(row)
    rows.append(["**all**"] + [f"{sum(d['passed'] for d in p['per_policy_case'][pol].values())}/{sum(d['total'] for d in p['per_policy_case'][pol].values())} strict, "
                               f"{sum(d['current_ok'] for d in p['per_policy_case'][pol].values())}/{sum(d['total'] for d in p['per_policy_case'][pol].values())} current value" for pol in pols])  # fmt: skip
    return (
        f"**Synthetic** (hand-written probes, extractor mode `{p['mode']}`, not a benchmark result). strict = every expected memory with its interval; current value = which value is current.\n\n"
        + table(["case"] + pols, rows)
    )


# ------------------------------------------------------------------------------ judge, failures
def judge_results(m: dict[str, Any]) -> str:
    j = m["judge"]
    methods = table(
        ["scored by", "n answers"], [[k, str(v)] for k, v in sorted(j["methods"].items())]
    )
    prov = (
        table(
            ["judge provider", "n judged answers"],
            [[k, str(v)] for k, v in sorted(j["providers"].items())],
        )
        if j["providers"]
        else "_No answer needed the judge yet._"
    )
    cal = j["calibration"]
    if cal and cal.get("status") == "measured":
        agree = f"Cohen's kappa between {cal['providers'][0]} and {cal['providers'][1]} on the same {cal['n']} items: **{cal['kappa']:.2f}** (raw agreement {pct(cal['agreement'])}, n={cal['n']})."
    else:
        agree = not_measured((cal or {}).get("reason", "the calibration step has not run"))
    audit = j["audit"]
    audit_text = (
        f"Hand-audit sheet: `artifacts/judge_audit.csv`, {audit['rows']} sampled verdicts, {audit['labelled']} labelled so far."
        if audit["exists"]
        else "Hand-audit sheet not written yet (it needs judged answers)."
    )
    return "\n\n".join([methods, prov, agree, audit_text])


def recheck_results(m: dict[str, Any]) -> str:
    r = m["recheck"]
    if not r or not r["n"]:
        return not_measured(
            "the claim re-check needs a judge provider (set GROQ_API_KEY) and answers with NLI-scored claims"
        )
    rows = [["NLI supported, judge supports", str(r["both_support"])], ["NLI supported, judge does not", str(r["nli_only"])],
            ["NLI rejected, judge supports", str(r["judge_only"])], ["NLI rejected, judge rejects", str(r["both_reject"])]]  # fmt: skip
    k = "n/a" if r["kappa"] is None else f"{r['kappa']:.2f}"
    return (f"Agreement between the NLI verdict and a gpt-oss-120b re-check on {r['n']} sampled claims: {pct(r['agreement'])} "
            f"(95% CI [{100 * r['lo']:.1f}, {100 * r['hi']:.1f}], n={r['n']}), kappa {k}.\n\n" + table(["outcome", "n claims"], rows))  # fmt: skip


def failure_results(fm: dict[str, Any], descriptions: dict[str, str]) -> str:
    out = []
    for mode, d in fm.items():
        out.append(
            f"#### {mode.replace('_', ' ')}\n\nRule: {descriptions[mode]}. Found **{d['count']}** among {d['population'] or 'n/a'}."
        )
        if not d["examples"]:
            out.append("_No instance in the evaluated sample._")
        for e in d["examples"]:
            bits = []
            if e.get("question"):
                bits.append(f"question `{e['question_id']}`: {e['question']!r}")
            for k in (
                "facts",
                "fact",
                "key",
                "name",
                "entities",
                "descriptors",
                "claim",
                "premise",
                "kind",
                "abstained_at",
                "reason",
                "gold_in_context",
                "nli_entailment",
            ):
                if k in e:
                    bits.append(f"{k}: {e[k]!r}")
            if e.get("gold") or e.get("answer"):
                bits.append(f"gold {e['gold']!r}, answered {e['answer']!r}")
            out.append("- " + "; ".join(bits))
    return "\n\n".join(out)


# ------------------------------------------------------------------------------ runtime
def runtime_table(m: dict[str, Any]) -> str:
    steps = m["steps"]
    if not steps:
        return not_measured("no step timings recorded yet; run `make eval-quick`")
    rows = [[k, human(v)] for k, v in sorted(steps.items())]
    rows.append(["**total**", human(sum(steps.values()))])
    return table(["step", "wall time"], rows)


def projection_table(est: dict[str, Any] | None) -> str:
    if not est:
        return not_measured("no basis run to project from")
    rows = []
    s = est["steps"]
    rows.append(["ingest", human(s["ingest"]["seconds"])])
    for sys_id, sec in s["answer_main"]["by_system"].items():
        n = est["questions_per_system"].get(sys_id)
        rows.append([f"answer: {name(sys_id)} (n={n})", human(sec)])
    for aid, d in s["ablations"]["by_ablation"].items():
        rows.append(
            [f"ablation {aid} ({d['runs']} extra runs x n={d['questions']})", human(d["seconds"])]
        )
    if "replays" in s:
        rows.append(["state-policy replays", human(s["replays"]["seconds"])])
    if "judge" in s:
        j = s["judge"]
        floor = j["floor_days_from_daily_token_limit"]
        rows.append(
            [
                f"judge (about {num(j['items'])} items)",
                human(j["seconds"])
                + (
                    f"; the daily token limit allows no less than {floor:.1f} days" if floor else ""
                ),
            ]
        )
    rows.append(["**total (sequential)**", human(est["total_seconds"])])
    return table(["step", f"projected for {est['target']}"], rows)
