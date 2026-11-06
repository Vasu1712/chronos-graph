"""Failure-mode analysis mined from artifacts (raw outputs, judgments, ingest logs, claim re-check).

Seven modes. Each is detected by a fixed rule, counted over a stated population, and illustrated by
up to two examples chosen deterministically (sorted by question id). Heuristics are deliberately
simple and described in `MODES`; they find candidates for a human to read, not verdicts.
"""

from __future__ import annotations

import re
from typing import Any

from eval.common import ingest_log_path, read_jsonl
from eval.metrics import load_outputs

MODES: dict[str, str] = {
    "cross_attribute_contradiction": "two current (ACTIVE, open) memories of one subject in the same context assert different places, under different attribute keys",
    "attribute_mismatch": "a memory's attribute key has no lexical support in its own fact text (a key chosen for the wrong reason)",
    "temporary_stay": "a current residence memory whose text is a trip, visit or short stay",
    "temporal_resolution": "answerable, answered, judged wrong, with all gold evidence turns in the context, on a question about time",
    "entity_split": "one name, several non-speaker entities in one conversation (under-merging)",
    "over_abstention": "an answerable question the system declined",
    "nli_error": "the NLI verdict on a sampled claim disagrees with the gpt-oss-120b re-check",
}
EXAMPLES_PER_MODE = 2

_MEM_LINE = re.compile(
    r"^- \(memory (?P<id>[^)]+)\) (?P<text>.*?) \| (?P<status>ACTIVE|SUPERSEDED|RETRACTED|CONFLICTED)"
    r"(?:, (?P<key>[a-z_]+(?::[a-z0-9_]+)?))?(?:, NEGATIVE)? \| (?P<interval>.*?) \| sources:"
)
_PLACE = re.compile(
    r"\b(?:lives?|lived|living|based|moved|relocated|settled)\s+(?:in|to|into)\s+([A-Z][\w'-]+(?:\s[A-Z][\w'-]+)*)"
)
_TEMPORARY = re.compile(
    r"\b(visiting|visit|trip|vacation|holiday|this week|for a (?:wedding|conference|week|few days)|temporar|stay(?:ing)? (?:at|with|in))\b",
    re.I,
)
_TIME_Q = re.compile(
    r"\b(when|how long|how many (?:days|weeks|months|years)|what (?:date|day|month|year)|first|before|after|until|since)\b",
    re.I,
)
_CUES: dict[str, re.Pattern[str]] = {
    "residence": re.compile(
        r"\b(live|lives|lived|living|based|moved|move|relocated|settled|home|resid)\w*", re.I
    ),
    "hometown": re.compile(r"\b(grew up|born|hometown|raised|from)\b", re.I),
    "employer": re.compile(
        r"\b(work|works|worked|working|joined|employ|job|company|hired)\w*", re.I
    ),
    "job_title": re.compile(
        r"\b(work|works|worked|as a|is a|role|job|title|promoted|became|position)\w*", re.I
    ),
    "school": re.compile(
        r"\b(study|studies|studied|studying|school|college|universit|student|enrolled|attend|degree)\w*",
        re.I,
    ),
    "partner": re.compile(
        r"\b(partner|girlfriend|boyfriend|wife|husband|spouse|fianc|engaged|married|dating)\w*",
        re.I,
    ),
    "relationship_status": re.compile(
        r"\b(single|married|engaged|divorced|dating|relationship|widow)\w*", re.I
    ),
    "diet": re.compile(
        r"\b(vegan|vegetarian|diet|keto|gluten|pescatarian|meat|eats?|eating)\b", re.I
    ),
}


def memory_lines(context_text: str) -> list[dict[str, str]]:
    """Memory lines of a packed ChronosGraph context, with the subject heading they sit under."""
    out, subject = [], ""
    for ln in context_text.splitlines():
        if ln.startswith("## "):
            subject = ln[3:].strip()
        elif m := _MEM_LINE.match(ln):
            out.append({"subject": subject, **m.groupdict()})
    return out


def key_supported(key: str, text: str) -> bool:
    base, _, slot = key.partition(":")
    if base in _CUES:
        return bool(_CUES[base].search(text))
    words = [w for w in re.split(r"[_\s]+", slot) if len(w) > 2]  # favorite:<x> / primary_tool:<x>
    stems = [w[:5].lower() for w in words]
    return any(s in text.lower() for s in stems) if stems else True


def _current(line: dict[str, str]) -> bool:
    return line["status"] == "ACTIVE" and line["interval"].endswith("to now")


def _example(row: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"system": row["system"], "question_id": row["question_id"], "question": row["question"],
            "gold": str(row["gold_answer"]), "answer": row["final_answer"], **extra}  # fmt: skip


def analyze_failures(
    manifest: str, judgments: list[dict[str, Any]], system: str = "chronos"
) -> dict[str, Any]:
    answers, _ = load_outputs(manifest)
    mine = sorted((r for r in answers if r["system"] == system), key=lambda r: r["question_id"])
    verdict = {
        (j["system"], j["question_id"]): j["correct"]
        for j in judgments
        if j.get("type") == "judgment"
    }
    result: dict[str, dict[str, Any]] = {
        k: {"count": 0, "population": "", "examples": []} for k in MODES
    }

    def add(mode: str, example: dict[str, Any]) -> None:
        result[mode]["count"] += 1
        if len(result[mode]["examples"]) < EXAMPLES_PER_MODE:
            result[mode]["examples"].append(example)

    seen_mem: set[str] = set()
    for r in mine:
        lines = memory_lines(r["context"]["text"])
        cur = [ln for ln in lines if _current(ln)]
        by_subject: dict[str, list[dict[str, str]]] = {}
        for ln in cur:
            by_subject.setdefault(ln["subject"], []).append(ln)
        for subj, ls in by_subject.items():
            places = {}
            for ln in ls:
                if m := _PLACE.search(ln["text"]):
                    places.setdefault(m.group(1), ln)
            keys = {ln.get("key") for ln in ls if _PLACE.search(ln["text"])}
            if len(places) > 1 and len(keys) > 1 and (r["question_id"], subj) not in seen_mem:
                seen_mem.add((r["question_id"], subj))
                add(
                    "cross_attribute_contradiction",
                    _example(
                        r,
                        facts=[
                            f"{ln['text']} [{ln.get('key') or 'no key'}]" for ln in places.values()
                        ],
                    ),
                )
        for ln in lines:
            if ln["id"] in seen_mem:
                continue
            seen_mem.add(ln["id"])
            if ln.get("key") and not key_supported(ln["key"], ln["text"]):
                add("attribute_mismatch", _example(r, fact=ln["text"], key=ln["key"]))
            if ln.get("key") == "residence" and _current(ln) and _TEMPORARY.search(ln["text"]):
                add("temporary_stay", _example(r, fact=ln["text"]))
        correct = verdict.get((r["system"], r["question_id"]))
        if r["gold_unanswerable"] or correct is None:
            continue
        if r["abstained"] and not correct:
            add("over_abstention", _example(r, abstained_at=r["abstained_at"], reason=r["abstention_reason"],
                                            gold_in_context=set(r["gold_evidence_turn_uids"]) <= set(r["context"]["turn_uids"]) and bool(r["gold_evidence_turn_uids"])))  # fmt: skip
        elif (
            not correct
            and r["gold_evidence_turn_uids"]
            and set(r["gold_evidence_turn_uids"]) <= set(r["context"]["turn_uids"])
            and (
                _TIME_Q.search(r["question"]) or r["category"] in ("temporal", "temporal-reasoning")
            )
        ):
            add("temporal_resolution", _example(r))
    result["over_abstention"]["population"] = (
        f"{sum(not r['gold_unanswerable'] for r in mine)} answerable questions"
    )
    result["temporal_resolution"]["population"] = (
        f"{sum(not r['gold_unanswerable'] and not r['abstained'] for r in mine)} answered questions"
    )
    result["attribute_mismatch"]["population"] = (
        f"{len(seen_mem)} distinct memories shown in contexts"
    )
    result["cross_attribute_contradiction"]["population"] = f"{len(mine)} contexts"
    result["temporary_stay"]["population"] = f"{len(mine)} contexts"

    convs = [r for r in read_jsonl(ingest_log_path(manifest)) if r.get("type") == "conversation"]
    latest = {c["conversation_id"]: c for c in convs}
    result["entity_split"]["population"] = f"{len(latest)} conversations"
    for cid, c in sorted(latest.items()):
        for dup in c.get("entity_name_duplicates", []):
            add("entity_split", {"system": "ingestion", "question_id": cid, "question": "", "gold": "", "answer": "",
                                 "name": dup["name"], "type": dup["type"], "entities": dup["count"], "descriptors": dup["descriptors"]})  # fmt: skip

    from eval.common import artifacts

    rc = read_jsonl(artifacts() / "claim_recheck" / f"{manifest}.jsonl")
    result["nli_error"]["population"] = f"{len(rc)} re-checked claims"
    for row in sorted(rc, key=lambda x: (x["question_id"], x["claim_index"])):
        if row["nli_supported"] != row["judge_supported"]:
            add("nli_error", {"system": row["system"], "question_id": row["question_id"], "question": "", "gold": "", "answer": "",
                              "claim": row["claim"], "premise": row.get("premise", ""), "nli_entailment": row["nli_entailment"],
                              "kind": "NLI passed, judge disagrees" if row["nli_supported"] else "NLI failed, judge disagrees"})  # fmt: skip
    return result
