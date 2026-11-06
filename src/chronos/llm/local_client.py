"""Local system LLM: Qwen2.5-7B-Instruct served by Ollama's OpenAI-compatible endpoint.

Every structured call sends the Pydantic model's JSON schema as `response_format=json_schema`,
which Ollama maps onto its native JSON-schema `format` (grammar-constrained decoding).

Context size: Ollama's /v1 endpoint may ignore per-request options, so `num_ctx` is sent in
`extra_body.options` *and* should be baked into the served model with
`scripts/setup_ollama.sh` (Modelfile `PARAMETER num_ctx 32768`). `verify_context()` checks it.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, TypeVar

import httpx
import openai
from pydantic import BaseModel, ValidationError

from chronos.config import LocalLLMConfig
from chronos.llm.cache import ResponseCache, cache_key
from chronos.llm.schema_utils import json_schema_for
from chronos.llm.tokens import count_chat_tokens, count_tokens

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

Message = dict[str, str]

_TRANSIENT = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


class StructuredOutputError(RuntimeError):
    """The model produced invalid output even after the validation-feedback retry."""


class CacheMiss(RuntimeError):
    """Raised in cache_only mode when a call would need the model (replays must never ingest anew)."""


@dataclass
class CallRecord:
    """One logical LLM call. `latency_s` is the original model latency, replayed from the cache on
    a hit; `wall_s` is what this run actually waited."""

    purpose: str  # response model name, or "text"
    prompt_tokens: int
    completion_tokens: int
    latency_s: float | None
    wall_s: float
    cached: bool
    attempts: int = 1

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalLLM:
    provider = "ollama"

    def __init__(
        self,
        config: LocalLLMConfig | None = None,
        cache: ResponseCache | None = None,
        client: Any | None = None,
        *,
        sleep=time.sleep,
        cache_only: bool = False,
        prompt_counter: Callable[[list[dict[str, str]]], int] | None = None,
    ) -> None:
        self.config = config or LocalLLMConfig()
        self.cache = cache
        self.client = client or openai.OpenAI(
            base_url=self.config.base_url,
            api_key="ollama",  # required by the SDK, ignored by Ollama
            timeout=self.config.timeout_s,
            max_retries=0,  # retries are handled here so they are visible and bounded
        )
        self._sleep = sleep
        self.calls = 0  # network calls actually made (cache misses)
        self.cache_only = cache_only
        self._prompt_counter = prompt_counter or count_chat_tokens
        self.records: list[CallRecord] = []  # every logical call, in order; slice by index to scope

    # ------------------------------------------------------------------ params
    @property
    def params(self) -> dict[str, Any]:
        return {"temperature": self.config.temperature, "num_ctx": self.config.num_ctx, "seed": 0}

    def _request_kwargs(self, messages: list[Message], schema: dict | None, name: str) -> dict:
        kwargs: dict[str, Any] = {
            "model": self.config.served_model,
            "messages": messages,
            "temperature": self.config.temperature,
            "seed": 0,
            "extra_body": {
                "options": {
                    "num_ctx": self.config.num_ctx,
                    "temperature": self.config.temperature,
                    "seed": 0,
                }
            },
        }
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": True},
            }
        return kwargs

    def _key(self, messages: list[Message], schema: dict | None) -> str:
        return cache_key(
            provider=self.provider,
            model=self.config.model,
            quantization=self.config.quantization,
            prompt=messages,
            schema=schema,
            params=self.params,
        )

    # ------------------------------------------------------------------ transport
    def _call(self, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """One chat completion -> (content, usage). Usage counts prompt tokens with our own tokenizer
        (Ollama's reported prompt count excludes tokens served from its prompt cache)."""
        if self.cache_only:
            raise CacheMiss("cache_only LLM asked to call the model")
        last: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                self.calls += 1
                started = time.perf_counter()
                resp = self.client.chat.completions.create(**kwargs)
                latency = time.perf_counter() - started
                content = resp.choices[0].message.content
                if content is None:
                    raise StructuredOutputError("empty completion")
                reported = getattr(getattr(resp, "usage", None), "completion_tokens", None)
                usage = {
                    "prompt_tokens": self._prompt_counter(kwargs["messages"]),
                    "completion_tokens": int(reported) if reported else count_tokens(content),
                    "latency_s": latency,
                }
                return content, usage
            except _TRANSIENT as exc:
                last = exc
                delay = min(2**attempt, 30)
                log.warning("ollama transient error (%s); retry %d in %ss", exc, attempt + 1, delay)
                self._sleep(delay)
        raise RuntimeError(f"Ollama request failed after retries: {last}") from last

    # ------------------------------------------------------------------ public API
    def complete(self, messages: list[Message]) -> str:
        """Free-text completion (cached)."""
        key = self._key(messages, None)
        started = time.perf_counter()
        if self.cache is not None and (hit := self.cache.get_entry(key)) is not None:
            self._record_hit("text", messages, hit[1], started)
            return hit[0]
        text, usage = self._call(self._request_kwargs(messages, None, "text"))
        self._record_miss("text", [usage], started)
        if self.cache is not None:
            self.cache.put(key, text, provider=self.provider, model=self.config.model,
                           meta=self.records[-1].as_dict() | {"cached": False})  # fmt: skip
        return text

    def structured(self, messages: list[Message], response_model: type[T]) -> T:
        """Schema-constrained call validated with Pydantic.

        On invalid JSON / schema mismatch, retry once with the validation error appended.
        """
        schema = json_schema_for(response_model)
        name = response_model.__name__
        key = self._key(messages, schema)
        started = time.perf_counter()
        if self.cache is not None and (hit := self.cache.get_entry(key)) is not None:
            self._record_hit(name, messages, hit[1], started)
            return response_model.model_validate(hit[0])

        usages: list[dict[str, Any]] = []
        attempt_messages = list(messages)
        raw, usage = self._call(self._request_kwargs(attempt_messages, schema, name))
        usages.append(usage)
        try:
            obj = _parse(raw, response_model)
        except (ValidationError, ValueError) as err:
            log.info("invalid structured output for %s; retrying with feedback", name)
            attempt_messages = [
                *messages,
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "Your previous reply did not validate against the required JSON schema.\n"
                        f"Validation error:\n{_short_error(err)}\n"
                        "Reply again with only a JSON object that satisfies the schema."
                    ),
                },
            ]
            raw, usage = self._call(self._request_kwargs(attempt_messages, schema, name))
            usages.append(usage)
            try:
                obj = _parse(raw, response_model)
            except (ValidationError, ValueError) as err2:
                raise StructuredOutputError(
                    f"{name}: invalid output after feedback retry: {_short_error(err2)}"
                ) from err2

        self._record_miss(name, usages, started)
        if self.cache is not None:
            self.cache.put(
                key,
                obj.model_dump(mode="json"),
                provider=self.provider,
                model=self.config.model,
                meta=self.records[-1].as_dict() | {"cached": False},
            )
        return obj

    # ------------------------------------------------------------------ usage records
    def _record_hit(
        self, purpose: str, messages: list[Message], meta: dict[str, Any], t0: float
    ) -> None:
        """A cache hit replays the original call's tokens and latency (entries from before usage
        tracking have no meta: prompt tokens are recounted, latency is unknown)."""
        prompt = meta.get("prompt_tokens")
        self.records.append(CallRecord(
            purpose=purpose,
            prompt_tokens=int(prompt) if prompt is not None else self._prompt_counter(messages),
            completion_tokens=int(meta.get("completion_tokens") or 0),
            latency_s=meta.get("latency_s"),
            wall_s=time.perf_counter() - t0,
            cached=True,
            attempts=int(meta.get("attempts", 1)),
        ))  # fmt: skip

    def _record_miss(self, purpose: str, usages: list[dict[str, Any]], t0: float) -> None:
        self.records.append(CallRecord(
            purpose=purpose,
            prompt_tokens=sum(u["prompt_tokens"] for u in usages),
            completion_tokens=sum(u["completion_tokens"] for u in usages),
            latency_s=sum(u["latency_s"] for u in usages),
            wall_s=time.perf_counter() - t0,
            cached=False,
            attempts=len(usages),
        ))  # fmt: skip

    def verify_context(self) -> int | None:
        """Return the served model's num_ctx via Ollama's native /api/show (None if unset)."""
        native = self.config.base_url.rstrip("/").removesuffix("/v1")
        r = httpx.post(f"{native}/api/show", json={"model": self.config.served_model}, timeout=10)
        r.raise_for_status()
        for line in (r.json().get("parameters") or "").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "num_ctx":
                return int(parts[1])
        return None


def _parse(raw: str, model: type[T]) -> T:
    return model.model_validate(json.loads(raw))


def _short_error(err: Exception, limit: int = 1500) -> str:
    return str(err)[:limit]
