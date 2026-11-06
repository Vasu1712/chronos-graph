import json

import pytest
from fakes import FakeChatClient, connection_error, scripted
from pydantic import BaseModel

from chronos.config import LocalLLMConfig
from chronos.llm.cache import ResponseCache, cache_key
from chronos.llm.local_client import LocalLLM, StructuredOutputError
from chronos.llm.schema_utils import json_schema_for


class Inner(BaseModel):
    text: str


class Facts(BaseModel):
    facts: list[Inner]
    count: int


GOOD = json.dumps({"facts": [{"text": "Caroline lives in Mumbai."}], "count": 1})
MSGS = [{"role": "user", "content": "extract"}]


@pytest.fixture
def cache(tmp_path):
    c = ResponseCache(tmp_path / "cache.sqlite")
    yield c
    c.close()


def test_structured_call_shape_and_cache(cache):
    client = scripted(GOOD)
    llm = LocalLLM(LocalLLMConfig(), cache=cache, client=client)
    out = llm.structured(MSGS, Facts)
    assert out.count == 1 and out.facts[0].text.startswith("Caroline")

    req = client.requests[0]
    assert req["model"] == LocalLLMConfig().served_model
    assert req["temperature"] == 0
    assert req["extra_body"]["options"]["num_ctx"] == 32768
    assert req["response_format"]["type"] == "json_schema"
    schema = req["response_format"]["json_schema"]["schema"]
    assert "$defs" not in schema and "$ref" not in json.dumps(schema)

    # second identical call is served from cache
    again = llm.structured(MSGS, Facts)
    assert again == out
    assert len(client.requests) == 1
    assert cache.hits == 1


def test_invalid_json_retries_once_with_validation_error(cache):
    client = scripted('{"facts": "oops"}', GOOD)
    llm = LocalLLM(cache=cache, client=client)
    out = llm.structured(MSGS, Facts)
    assert out.count == 1
    assert len(client.requests) == 2
    feedback = client.requests[1]["messages"][-1]["content"]
    assert "Validation error" in feedback and "facts" in feedback


def test_invalid_twice_raises_and_is_not_cached(cache):
    client = scripted("not json", "{}")
    llm = LocalLLM(cache=cache, client=client)
    with pytest.raises(StructuredOutputError):
        llm.structured(MSGS, Facts)
    assert len(cache) == 0


def test_transient_errors_are_retried():
    client = scripted(connection_error(), connection_error(), GOOD)
    slept = []
    llm = LocalLLM(client=client, sleep=slept.append)
    assert llm.structured(MSGS, Facts).count == 1
    assert slept == [1, 2]


def test_complete_text_is_cached(cache):
    client = FakeChatClient(lambda kw: "hello")
    llm = LocalLLM(cache=cache, client=client)
    assert llm.complete(MSGS) == "hello"
    assert llm.complete(MSGS) == "hello"
    assert len(client.requests) == 1
    assert "response_format" not in client.requests[0]


def test_cache_key_depends_on_every_component():
    base = dict(
        provider="ollama",
        model="m",
        quantization="q4",
        prompt=MSGS,
        schema=json_schema_for(Facts),
        params={"temperature": 0},
    )
    k = cache_key(**base)
    for field, val in [
        ("provider", "groq"),
        ("model", "m2"),
        ("quantization", "q8"),
        ("prompt", [{"role": "user", "content": "x"}]),
        ("schema", {"type": "object"}),
        ("params", {"temperature": 0.5}),
    ]:
        assert cache_key(**{**base, field: val}) != k, field
    # dict ordering does not matter
    assert cache_key(**{**base, "params": {"temperature": 0}}) == k


def test_call_records_track_tokens_latency_and_replay_from_cache(cache):
    from chronos.llm.local_client import CacheMiss

    client = scripted(GOOD)
    llm = LocalLLM(cache=cache, client=client, prompt_counter=lambda msgs: 123)
    llm.structured(MSGS, Facts)
    [rec] = llm.records
    assert (rec.purpose, rec.prompt_tokens, rec.cached, rec.attempts) == ("Facts", 123, False, 1)
    assert rec.completion_tokens > 0 and rec.latency_s is not None and rec.latency_s >= 0

    again = LocalLLM(cache=cache, client=None, prompt_counter=lambda msgs: 999, cache_only=True)
    again.structured(MSGS, Facts)  # served from cache: original usage is replayed, not recounted
    [hit] = again.records
    assert (
        hit.cached and hit.prompt_tokens == 123 and hit.completion_tokens == rec.completion_tokens
    )
    assert hit.latency_s == pytest.approx(rec.latency_s) and hit.wall_s < 0.5

    with pytest.raises(CacheMiss):
        again.structured([{"role": "user", "content": "never seen"}], Facts)


def test_retry_records_both_attempts_as_one_call(cache):
    client = scripted('{"facts": "oops"}', GOOD)
    llm = LocalLLM(cache=cache, client=client, prompt_counter=lambda msgs: 100)
    llm.structured(MSGS, Facts)
    [rec] = llm.records
    assert rec.attempts == 2 and rec.prompt_tokens == 200
