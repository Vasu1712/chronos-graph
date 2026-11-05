from collections import Counter

import pytest

from eval.loaders.locomo import CATEGORY_MAP, normalize_evidence, parse_locomo_datetime
from eval.loaders.longmemeval import parse_lme_datetime, reduce_haystack, session_token_counts

# ------------------------------------------------------------------------------ unit (no data)


def test_parsers():
    assert parse_locomo_datetime("1:56 pm on 8 May, 2023") == "2023-05-08T13:56:00"
    assert parse_locomo_datetime("11:01 am on 17 December, 2022") == "2022-12-17T11:01:00"
    assert parse_lme_datetime("2023/05/30 (Tue) 23:40") == "2023-05-30T23:40:00"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("D3:7", [(3, 7)]),
        ("D8:6; D9:17", [(8, 6), (9, 17)]),
        ("D9:1 D4:4 D4:6", [(9, 1), (4, 4), (4, 6)]),
        ("D:11:26", [(11, 26)]),
        ("D30:05", [(30, 5)]),
        ("D", []),
    ],
)
def test_normalize_evidence(raw, expected):
    assert normalize_evidence(raw) == expected


# ------------------------------------------------------------------------------ LoCoMo (data)


def test_locomo_categories_and_evidence(locomo_data, stats):
    convs, qas, report = locomo_data
    counts = Counter(q.category for q in qas)
    stats.append(
        f"LoCoMo: {report.conversations} conversations, {report.sessions} sessions, "
        f"{report.turns} turns, {report.questions} QA"
    )
    for cid in sorted(CATEGORY_MAP):
        name = CATEGORY_MAP[cid]
        stats.append(f"  category {cid} {name:<12} {counts[name]:>4}")
    stats.append(
        f"  evidence: {report.evidence_ids} raw ids, {report.evidence_resolved} resolved, "
        f"{len(report.evidence_repaired)} repaired, {len(report.evidence_malformed)} malformed, "
        f"{report.questions_without_evidence} QA with no evidence"
    )
    for qid, raw, why in report.evidence_malformed:
        stats.append(f"    malformed {qid}: {raw!r} ({why})")
    for qid, raw, fixed in report.evidence_repaired:
        stats.append(f"    repaired  {qid}: {raw!r} -> {fixed}")
    if report.dateless_sessions_skipped:
        by_conv = Counter(c for c, _ in report.dateless_sessions_skipped)
        stats.append(f"  dated-but-empty sessions skipped: {dict(by_conv)}")

    assert set(counts) == set(CATEGORY_MAP.values())
    assert all(counts[c] > 0 for c in CATEGORY_MAP.values())
    assert sum(counts.values()) == report.questions == len(qas)

    # every resolved gold evidence id maps to exactly one turn_uid in its conversation
    turn_index: dict[str, list[str]] = {}
    for conv in convs:
        for t in conv.turns():
            turn_index.setdefault(t.turn_uid, []).append(conv.conversation_id)
    assert all(len(v) == 1 for v in turn_index.values()), "turn_uid collision"
    for q in qas:
        for uid in q.evidence_turn_uids:
            assert turn_index.get(uid) == [q.conversation_id], (q.question_id, uid)

    # malformed ids are counted, not fatal, and rare
    assert len(report.evidence_malformed) <= 0.01 * report.evidence_ids


def test_locomo_adversarial_reference_and_turn_uids(locomo_data):
    convs, qas, _ = locomo_data
    adv = [q for q in qas if q.category == "adversarial"]
    assert adv and all(q.is_abstention for q in adv)
    assert sum(1 for q in adv if q.answer) >= 0.95 * len(adv)  # taken from adversarial_answer
    first = convs[0].sessions[2].turns[6]
    assert first.source_ref == "D3:7"
    assert first.turn_uid == f"locomo:{convs[0].conversation_id}:s03:t007"
    assert all(s.timestamp for c in convs for s in c.sessions)


# ------------------------------------------------------------------------------ LongMemEval (data)


def test_longmemeval_counts(lme_data, stats):
    types = Counter(q.question_type for q in lme_data)
    abs_types = Counter(q.question_type for q in lme_data if q.qa.is_abstention)
    stats.append(f"LongMemEval-S: {len(lme_data)} questions")
    for t, n in sorted(types.items()):
        stats.append(f"  {t:<26} {n:>4}  (abstention: {abs_types.get(t, 0)})")
    assert types["knowledge-update"] == 78
    assert sum(abs_types.values()) == sum("_abs" in q.question_id for q in lme_data) == 30
    for q in lme_data:
        answer = q.answer_ordinals()
        assert answer, q.question_id  # even abstention questions have answer sessions
        has_answer = {t.session_ordinal for t in q.history.turns() if t.has_answer}
        assert has_answer <= answer, q.question_id
        assert {t.turn_uid for t in q.history.turns() if t.has_answer} == set(
            q.qa.evidence_turn_uids
        )


def _sample(lme_data):
    """All knowledge-update + abstention questions plus every 5th other one (keeps tests fast)."""
    return [
        q
        for i, q in enumerate(lme_data)
        if q.question_type == "knowledge-update" or q.qa.is_abstention or i % 5 == 0
    ]


def test_reduce_haystack_keeps_evidence_and_budget(lme_data, stats):
    sample = _sample(lme_data)
    kept_sessions, kept_tokens = [], []
    for q in sample:
        r = reduce_haystack(q, max_tokens=24000, seed=42)
        ordinals = {s.ordinal for s in r.history.sessions}
        assert q.answer_ordinals() <= ordinals, q.question_id
        counts = session_token_counts(q)
        tokens = sum(counts[o] for o in ordinals)
        assert tokens <= 24000 or ordinals == q.answer_ordinals()
        assert [s.ordinal for s in r.history.sessions] == sorted(ordinals)  # haystack order kept
        kept_sessions.append(len(ordinals))
        kept_tokens.append(tokens)
    stats.append(
        f"  reduce_haystack(24000, seed=42) on {len(sample)} questions: "
        f"sessions kept {min(kept_sessions)}-{max(kept_sessions)} "
        f"(mean {sum(kept_sessions) / len(sample):.1f}), "
        f"tokens {min(kept_tokens)}-{max(kept_tokens)} (Qwen2.5 tokenizer)"
    )


def test_reduce_haystack_deterministic(lme_data):
    qs = [q for q in lme_data if q.question_type == "knowledge-update"][:10]
    a = [tuple(s.ordinal for s in reduce_haystack(q, seed=42).history.sessions) for q in qs]
    b = [tuple(s.ordinal for s in reduce_haystack(q, seed=42).history.sessions) for q in qs]
    c = [tuple(s.ordinal for s in reduce_haystack(q, seed=7).history.sessions) for q in qs]
    assert a == b
    assert a != c


def test_reduce_haystack_with_fake_counter(lme_data):
    q = lme_data[0]
    r = reduce_haystack(q, max_tokens=0, counter=lambda texts: [100] * len(texts))
    assert {s.ordinal for s in r.history.sessions} == q.answer_ordinals()
