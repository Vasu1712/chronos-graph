import pytest
from fakes import FakeChatClient, FakeClock, completion, judge_responder, rate_limit_error

from chronos.config import JudgeProviderConfig, ProviderLimits
from chronos.llm.cache import ResponseCache
from chronos.llm.judge_pool import JudgeItem, JudgePool, ProviderLimiter, route_index

GROQ_DEFAULT = ProviderLimits(30, 8000, 200_000)
CEREBRAS_DEFAULT = ProviderLimits(5, 90_000, 1_000_000)


def providers(groq=GROQ_DEFAULT, cerebras=CEREBRAS_DEFAULT):
    return [
        JudgeProviderConfig("groq", "http://g", "openai/gpt-oss-120b", "X", groq),
        JudgeProviderConfig("cerebras", "http://c", "gpt-oss-120b", "Y", cerebras),
    ]


def qid_for(provider_index: int, n: int = 1) -> list[str]:
    out, i = [], 0
    while len(out) < n:
        q = f"q{i}"
        if route_index(q) == provider_index:
            out.append(q)
        i += 1
    return out


def item(qid: str, system: str = "sysA") -> JudgeItem:
    return JudgeItem(
        item_id=f"{system}:{qid}",
        question_id=qid,
        question="Where does Caroline live?",
        reference="Mumbai",
        response="She lives in Mumbai.",
    )


def make_pool(tmp_path=None, groq=None, cerebras=None, clock=None, **kw):
    groq = groq or FakeChatClient(judge_responder())
    cerebras = cerebras or FakeChatClient(judge_responder())
    clock = clock or FakeClock()
    cache = ResponseCache(tmp_path / "c.sqlite") if tmp_path else None
    pool = JudgePool(
        cache,
        providers=kw.pop("providers", providers()),
        clients={"groq": groq, "cerebras": cerebras},
        clock=clock,
        sleep=clock.sleep,
        **kw,
    )
    return pool, groq, cerebras, clock


def test_routing_is_deterministic_and_uses_both_providers():
    assert route_index("abc") == route_index("abc")
    buckets = {route_index(f"q{i}") for i in range(50)}
    assert buckets == {0, 1}


def test_same_question_same_provider_across_systems():
    pool, groq, cerebras, _ = make_pool()
    [q] = qid_for(1)
    verdicts = pool.judge([item(q, "sysA"), item(q, "sysB"), item(q, "sysC")])
    assert {v.provider for v in verdicts} == {"cerebras"}
    assert groq.requests == []
    assert len(cerebras.requests) == 1  # all three batched together


def test_batches_of_at_most_eight_and_order_preserved():
    pool, groq, _, _ = make_pool()
    items = [item(q) for q in qid_for(0, 10)]
    verdicts = pool.judge(items)
    assert [v.item_id for v in verdicts] == [i.item_id for i in items]
    assert len(groq.requests) == 2
    assert all(v.provider == "groq" and v.label == "CORRECT" for v in verdicts)
    req = groq.requests[0]
    assert req["reasoning_effort"] == "low"
    assert req["model"] == "openai/gpt-oss-120b"
    assert req["response_format"]["type"] == "json_schema"


def test_bare_json_array_is_accepted():
    pool, *_ = make_pool(groq=FakeChatClient(judge_responder(bare=True)))
    [v] = pool.judge([item(qid_for(0)[0])])
    assert v.label == "CORRECT"


def test_dropped_items_are_rejudged_individually():
    qs = qid_for(0, 3)
    dropped = f"sysA:{qs[1]}"
    calls = {"n": 0}

    def respond(kw):
        calls["n"] += 1
        return judge_responder(drop={dropped} if calls["n"] == 1 else None)(kw)

    pool, groq, *_ = make_pool(groq=FakeChatClient(respond))
    verdicts = pool.judge([item(q) for q in qs])
    assert len(verdicts) == 3
    assert len(groq.requests) == 2


def test_cache_prevents_repeat_calls(tmp_path):
    pool, groq, cerebras, _ = make_pool(tmp_path)
    items = [item(q) for q in qid_for(0, 2) + qid_for(1, 2)]
    first = pool.judge(items)
    n = len(groq.requests) + len(cerebras.requests)
    second = pool.judge(items)
    assert len(groq.requests) + len(cerebras.requests) == n
    assert all(v.cached for v in second)
    assert [v.provider for v in first] == [v.provider for v in second]


def test_rpm_bucket_makes_requests_wait():
    clock = FakeClock()
    lim = ProviderLimiter(
        "p", ProviderLimits(rpm=2, tpm=10**9, tpd=10**9), clock=clock, sleep=clock.sleep
    )
    assert lim.acquire(10) == 0
    assert lim.acquire(10) == 0
    waited = lim.acquire(10)
    assert waited == pytest.approx(30.0)  # 1 token at 2/60 per second


def test_daily_quota_exhausted_waits_instead_of_switching():
    clock = FakeClock()
    tight = ProviderLimits(rpm=1000, tpm=10**9, tpd=3000)
    heavy = FakeChatClient(lambda kw: completion(judge_responder()(kw), total_tokens=2_900))
    pool, groq, cerebras, clock = make_pool(
        groq=heavy, providers=providers(groq=tight), clock=clock
    )
    q1, q2 = qid_for(0, 2)
    pool.judge([item(q1)])
    pool.judge([item(q2)])
    assert cerebras.requests == []  # never re-routed
    assert len(groq.requests) == 2
    assert sum(clock.slept) > 3600  # had to wait for the daily bucket to refill


def test_429_blocks_for_retry_after_then_retries_same_provider():
    replies = [rate_limit_error(retry_after=7)]

    def respond(kw):
        return replies.pop(0) if replies else judge_responder()(kw)

    pool, groq, cerebras, clock = make_pool(groq=FakeChatClient(respond))
    [v] = pool.judge([item(qid_for(0)[0])])
    assert v.provider == "groq"
    assert len(groq.requests) == 2 and cerebras.requests == []
    assert sum(clock.slept) >= 7


def test_usage_reconciliation_debits_actual_tokens():
    clock = FakeClock()
    big = FakeChatClient(lambda kw: completion(judge_responder()(kw), total_tokens=5000))
    pool, *_ = make_pool(groq=big, clock=clock)
    pool.judge([item(qid_for(0)[0])])
    tpm = pool.providers[0].limiter.buckets["tpm"]
    assert tpm.level == pytest.approx(8000 - 5000)


def test_limiter_state_persists_across_restarts(tmp_path):
    clock = FakeClock()
    path = tmp_path / "lim.sqlite"
    limits = ProviderLimits(rpm=100, tpm=10**9, tpd=10_000)
    a = ProviderLimiter("groq", limits, clock=clock, sleep=clock.sleep, state_path=path)
    a.acquire(9_000)
    b = ProviderLimiter("groq", limits, clock=clock, sleep=clock.sleep, state_path=path)
    assert b.buckets["tpd"].level == pytest.approx(1_000)


def test_duplicate_item_ids_rejected():
    pool, *_ = make_pool()
    q = qid_for(0)[0]
    with pytest.raises(ValueError):
        pool.judge([item(q), item(q)])


def test_items_with_different_rubrics_never_share_a_batch():
    pool, groq, _, _ = make_pool()
    qs = qid_for(0, 4)
    items = [
        item(q).model_copy(update={"rubric": "RUBRIC A" if i % 2 else None})
        for i, q in enumerate(qs)
    ]
    pool.judge(items)
    systems = [r["messages"][0]["content"] for r in groq.requests]
    assert len(groq.requests) == 2 and sorted(systems)[0] == "RUBRIC A" and systems[0] != systems[1]
    assert set(systems) == {"RUBRIC A", pool.rubric}  # one batch per rubric


def test_rubric_is_part_of_the_cache_key(tmp_path):
    pool, groq, _, _ = make_pool(tmp_path)
    q = qid_for(0)[0]
    pool.judge([item(q)])
    pool.judge([item(q).model_copy(update={"rubric": "other"})])
    assert len(groq.requests) == 2  # same item, different instructions: not a cache hit


def test_available_only_drops_providers_without_keys_and_routes_everything_to_the_rest(monkeypatch):
    monkeypatch.delenv("X", raising=False)
    monkeypatch.setenv("Y", "key")
    groq = FakeChatClient(judge_responder())
    pool = JudgePool(providers=providers(), clients={"cerebras": groq}, available_only=True)
    assert [p.config.name for p in pool.providers] == ["cerebras"] and pool.unavailable == ["groq"]
    verdicts = pool.judge([item(q) for q in qid_for(0, 2) + qid_for(1, 2)])
    assert {v.provider for v in verdicts} == {"cerebras"}
    monkeypatch.delenv("Y")
    with pytest.raises(RuntimeError, match="no judge provider"):
        JudgePool(providers=providers(), available_only=True)


def test_forced_provider_judges_the_same_items_on_each_provider():
    pool, groq, cerebras, _ = make_pool()
    [q] = qid_for(0, 1)
    a = pool.judge([item(q)], provider="cerebras")
    b = pool.judge([item(q)], provider="groq")
    assert (
        a[0].provider == "cerebras"
        and b[0].provider == "groq"
        and len(cerebras.requests) == len(groq.requests) == 1
    )
    with pytest.raises(ValueError, match="not available"):
        pool.judge([item(q)], provider="nope")
