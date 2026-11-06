"""Claim verifier with one repair round, recorded in answer_log (SPEC §6 step 9).

Citations are first normalized deterministically (models sometimes paste a whole context line):
  a string that is or starts with a turn id -> that turn; a memory id -> moved to the memory
  citations; a "[...] text" line whose text equals exactly one allowed turn -> that turn. Every
  repair is recorded in `normalized`; nothing is guessed beyond these rules.

Per claim, per cited turn:
  1. the turn_uid exists in episodes                     else dropped: "not_found"
  2. it is in this query's allowed list (answer_log)     else dropped: "not_in_context"
Then NLI: premise = the surviving cited turns ("On <date>, <speaker> said: <text>"), hypothesis =
the claim; supported iff P(entailment) >= threshold. A claim with no surviving citation, or whose
entailment is below threshold, is unsupported.

Repair: if any claim is unsupported (or a non-abstained answer has no claims), regenerate once
with feedback naming the unsupported claims; after that, unsupported claims are removed. No claims
left means abstain with reason "unsupported". If claims were removed, final_answer becomes the
concatenation of the surviving claims so no unsupported text reaches the user.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any

from chronos.agent.generator import Claim, GenerationOut, Generator


@dataclass
class ClaimCheck:
    text: str
    cited_turn_uids: list[str]
    cited_memory_ids: list[str]
    kept_turn_uids: list[str] = field(default_factory=list)
    kept_memory_ids: list[str] = field(default_factory=list)
    dropped: list[dict[str, str]] = field(default_factory=list)
    normalized: list[dict[str, str]] = field(default_factory=list)
    nli_entailment: float | None = None
    nli_contradiction: float | None = None
    nli_label: str | None = None
    supported: bool = False


@dataclass
class VerifyReport:
    claims: list[ClaimCheck]

    @property
    def unsupported(self) -> list[ClaimCheck]:
        return [c for c in self.claims if not c.supported]


class AnswerLog:
    """Thin wrapper over the answer_log table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def open(
        self,
        query_id: str,
        conversation_id: str,
        query: str,
        memory_ids: list[str],
        turn_uids: list[str],
    ) -> None:
        self.conn.execute(
            "INSERT INTO answer_log(query_id, conversation_id, query, retrieved_memory_ids, retrieved_turn_uids) "
            "VALUES (?, ?, ?, ?, ?)",
            (query_id, conversation_id, query, json.dumps(memory_ids), json.dumps(turn_uids)),
        )

    def allowed(self, query_id: str) -> tuple[list[str], list[str]]:
        row = self.conn.execute(
            "SELECT retrieved_turn_uids, retrieved_memory_ids FROM answer_log WHERE query_id = ?",
            (query_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"no answer_log row for {query_id}")
        return json.loads(row[0]), json.loads(row[1])

    def finish(self, query_id: str, response: dict[str, Any], verifier: dict[str, Any]) -> None:
        self.conn.execute(
            "UPDATE answer_log SET response_json = ?, verifier_json = ? WHERE query_id = ?",
            (json.dumps(response, default=str), json.dumps(verifier, default=str), query_id),
        )

    def get(self, query_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM answer_log WHERE query_id = ?", (query_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        for k in ("retrieved_memory_ids", "retrieved_turn_uids", "response_json", "verifier_json"):
            d[k] = json.loads(d[k]) if d[k] else None
        return d


class Verifier:
    def __init__(self, conn: sqlite3.Connection, nli: Any, threshold: float = 0.5) -> None:
        self.conn = conn
        self.nli = nli
        self.threshold = threshold
        self.log = AnswerLog(conn)

    def premise(self, turn_uids: list[str]) -> str:
        rows = {
            r[0]: r
            for r in self.conn.execute(
                f"SELECT turn_uid, session_ts, speaker, text FROM episodes WHERE turn_uid IN ({','.join('?' * len(turn_uids))})",
                turn_uids,
            )
        }
        return " ".join(
            f"On {(rows[u][1] or '')[:10]}, {rows[u][2]} said: {rows[u][3]}"
            for u in turn_uids
            if u in rows
        )

    def _exists(self, uids: list[str]) -> set[str]:
        if not uids:
            return set()
        return {
            r[0]
            for r in self.conn.execute(
                f"SELECT turn_uid FROM episodes WHERE turn_uid IN ({','.join('?' * len(uids))})",
                uids,
            )
        }

    def _turn_texts(self, uids: list[str]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for i in range(0, len(uids), 500):
            chunk = uids[i : i + 500]
            for uid, text in self.conn.execute(
                f"SELECT turn_uid, text FROM episodes WHERE turn_uid IN ({','.join('?' * len(chunk))})",
                chunk,
            ):
                out.setdefault(text.strip(), []).append(uid)
        return out

    def normalize(
        self, raw: str, allowed_t: set[str], allowed_m: set[str], by_text: dict[str, list[str]]
    ) -> tuple[str, str, str | None]:
        """-> (kind "turn" | "memory", value, how-or-None)."""
        s = raw.strip()
        if s in allowed_t or self._exists([s]):
            return "turn", s, None
        token = re.match(r"^\[?\s*\(?(?:memory\s+)?([^\s|\])]+)", s)
        tok = token.group(1) if token else s
        if tok in allowed_t or (tok != s and self._exists([tok])):
            return "turn", tok, "extracted turn id"
        if tok in allowed_m:
            return "memory", tok, "memory id cited as a turn"
        if "] " in s:
            uids = by_text.get(s.split("] ", 1)[1].strip(), [])
            if len(uids) == 1:
                return "turn", uids[0], "matched turn by its exact text"
        return "turn", s, None

    def check(self, query_id: str, gen: GenerationOut) -> VerifyReport:
        allowed_turns, allowed_mems = self.log.allowed(query_id)
        allowed_t, allowed_m = set(allowed_turns), set(allowed_mems)
        by_text = self._turn_texts(allowed_turns)
        checks: list[ClaimCheck] = []
        for c in gen.claims:
            ch = ClaimCheck(c.text, list(c.cited_turn_uids), list(c.cited_memory_ids))
            turns: list[str] = []
            mems = list(c.cited_memory_ids)
            for raw in c.cited_turn_uids:
                kind, value, how = self.normalize(raw, allowed_t, allowed_m, by_text)
                if how:
                    ch.normalized.append({"raw": raw, "to": value, "how": how})
                (turns if kind == "turn" else mems).append(value)
            exists = self._exists(list(dict.fromkeys(turns)))
            for uid in dict.fromkeys(turns):
                if uid not in exists:
                    ch.dropped.append({"turn_uid": uid, "reason": "not_found"})
                elif uid not in allowed_t:
                    ch.dropped.append({"turn_uid": uid, "reason": "not_in_context"})
                else:
                    ch.kept_turn_uids.append(uid)
            ch.kept_memory_ids = [m for m in dict.fromkeys(mems) if m in allowed_m]
            checks.append(ch)

        pairs = [(self.premise(ch.kept_turn_uids), ch.text) for ch in checks if ch.kept_turn_uids]
        scores = iter(self.nli.score(pairs) if pairs else [])
        for ch in checks:
            if not ch.kept_turn_uids:
                continue
            s = next(scores)
            ch.nli_entailment, ch.nli_contradiction, ch.nli_label = (
                s.entailment,
                s.contradiction,
                s.label,
            )
            ch.supported = s.entailment >= self.threshold
            if not ch.supported:
                ch.dropped += [{"turn_uid": u, "reason": "not_entailed"} for u in ch.kept_turn_uids]
        return VerifyReport(checks)


def _feedback(report: VerifyReport, no_claims: bool) -> str:
    if no_claims:
        return ("Your answer had no claims. Every factual statement must be a claim citing turn ids from "
                "ALLOWED TURN IDS, or abstain if the context does not answer the question.")  # fmt: skip
    lines = ["These claims were not supported by the turns they cite:"]
    for c in report.unsupported:
        why = ", ".join(sorted({d["reason"] for d in c.dropped})) or "no citations"
        lines.append(f'- "{c.text}" ({why})')
    lines.append("Cite only turns from ALLOWED TURN IDS whose text states the claim, rephrase claims to say only "
                 "what those turns say, or drop them. Abstain if nothing in the context supports an answer.")  # fmt: skip
    return "\n".join(lines)


def _finalize(gen: GenerationOut, report: VerifyReport) -> GenerationOut:
    kept = [c for c in report.claims if c.supported]
    claims = [
        Claim(text=c.text, cited_turn_uids=c.kept_turn_uids, cited_memory_ids=c.kept_memory_ids)
        for c in kept
    ]
    if not claims:
        return GenerationOut(
            final_answer="I don't know.", claims=[], abstained=True, abstention_reason="unsupported"
        )
    answer = (
        gen.final_answer if len(kept) == len(report.claims) else " ".join(c.text for c in claims)
    )
    return GenerationOut(
        final_answer=answer, claims=claims, abstained=False, abstention_reason=None
    )


def verify_and_repair(
    *,
    query_id: str,
    question: str,
    context: str,
    allowed_turn_uids: list[str],
    generator: Generator,
    verifier: Verifier,
    first: GenerationOut,
    question_date: str | None = None,
    regenerate_once: bool = True,
    asker: str | None = None,
    question_type: str | None = None,
) -> tuple[GenerationOut, dict[str, Any]]:
    """Verify `first`, regenerate once if needed, and return (final answer, verifier record)."""
    record: dict[str, Any] = {"threshold": verifier.threshold, "rounds": []}
    gen = first
    for round_no in (1, 2):
        if gen.abstained:
            record["rounds"].append({"round": round_no, "abstained_by_model": True, "claims": []})
            return GenerationOut(final_answer="I don't know.", claims=[], abstained=True,
                                 abstention_reason=gen.abstention_reason or "model abstained"), record  # fmt: skip
        report = verifier.check(query_id, gen)
        no_claims = not report.claims
        record["rounds"].append(
            {
                "round": round_no,
                "claims": [asdict(c) for c in report.claims],
                "no_claims": no_claims,
            }
        )
        if report.claims and not report.unsupported:
            return _finalize(gen, report), record
        if round_no == 1 and regenerate_once:
            feedback = _feedback(report, no_claims)
            record["feedback"] = feedback
            gen = generator.generate(
                question,
                context,
                allowed_turn_uids,
                question_date=question_date,
                feedback=feedback,
                asker=asker,
                question_type=question_type,
            )
            continue
        return _finalize(gen, report), record
    raise AssertionError("unreachable")  # pragma: no cover
