"""Evaluation judge: gpt-oss-120b pooled across Groq and Cerebras free tiers.

- Routing: provider = providers[sha256(question_id) mod 2], so every system's answer to the same
  question is judged by the same provider. If that provider is out of quota we *wait*; we never
  switch, since that would mix judges within a question.
- Rate limiting: client-side token buckets per provider for RPM, TPM and TPD. Bucket state is
  persisted in SQLite so a restart does not reset the daily budget. Server 429s also block the
  provider for the Retry-After interval.
- Batching: up to 8 items per request; the model returns a JSON array of verdicts.
- Caching: per item, through ResponseCache. Every verdict records its provider.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import openai
from pydantic import BaseModel, ConfigDict, ValidationError

from chronos.config import JudgeProviderConfig, ProviderLimits, judge_providers
from chronos.llm.cache import ResponseCache, cache_key
from chronos.llm.schema_utils import json_schema_for

log = logging.getLogger(__name__)

MAX_BATCH = 8

DEFAULT_RUBRIC = """You are grading answers produced by a conversational memory system.
For each item you get a question, a reference answer and the system's response.
Label CORRECT if the response contains the reference answer's key information or is
semantically equivalent to it (paraphrases, equivalent dates and partial-but-sufficient
answers count). Label WRONG if it is incorrect, contradicts the reference, misses the key
information, or is only a refusal when an answer was expected. If the reference says the
question is unanswerable, label CORRECT only if the response abstains or says it does not know.
Judge each item independently. Return one verdict per item_id, in any order."""


# ---------------------------------------------------------------------------- data types
class JudgeItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str
    question_id: str
    question: str
    reference: str
    response: str
    rubric: str | None = None  # per-item grading instructions; None = the pool's default rubric


class _VerdictOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    rationale: str
    label: Literal["CORRECT", "WRONG"]


class _BatchOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdicts: list[_VerdictOut]


@dataclass(frozen=True)
class Verdict:
    item_id: str
    question_id: str
    label: str
    rationale: str
    provider: str
    model: str
    cached: bool = False


def route_index(question_id: str, n: int = 2) -> int:
    return int(hashlib.sha256(question_id.encode("utf-8")).hexdigest(), 16) % n


# ---------------------------------------------------------------------------- rate limiting
@dataclass
class TokenBucket:
    """Classic token bucket. Level may go negative after reconciling an under-estimate."""

    capacity: float
    refill_per_s: float
    level: float
    last: float

    def refill(self, now: float) -> None:
        if now > self.last:
            self.level = min(self.capacity, self.level + (now - self.last) * self.refill_per_s)
            self.last = now

    def wait_time(self, cost: float, now: float) -> float:
        self.refill(now)
        # A request larger than the whole bucket waits for a full bucket, then overdraws.
        need = min(cost, self.capacity)
        if self.level >= need:
            return 0.0
        return (need - self.level) / self.refill_per_s


class ProviderLimiter:
    """RPM / TPM / TPD token buckets for one provider, optionally persisted."""

    def __init__(
        self,
        name: str,
        limits: ProviderLimits,
        *,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        state_path: str | Path | None = None,
    ) -> None:
        self.name = name
        self.clock = clock
        self.sleep = sleep
        self._lock = threading.Lock()
        self._blocked_until = 0.0
        now = clock()
        self.buckets = {
            "rpm": TokenBucket(limits.rpm, limits.rpm / 60.0, limits.rpm, now),
            "tpm": TokenBucket(limits.tpm, limits.tpm / 60.0, limits.tpm, now),
            "tpd": TokenBucket(limits.tpd, limits.tpd / 86_400.0, limits.tpd, now),
        }
        self._db: sqlite3.Connection | None = None
        if state_path is not None:
            self._db = sqlite3.connect(
                str(state_path), check_same_thread=False, isolation_level=None
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS judge_limiter_state ("
                "provider TEXT, bucket TEXT, level REAL, last REAL, PRIMARY KEY(provider, bucket))"
            )
            for b, level, last in self._db.execute(
                "SELECT bucket, level, last FROM judge_limiter_state WHERE provider = ?", (name,)
            ):
                if b in self.buckets:
                    bucket = self.buckets[b]
                    bucket.level, bucket.last = min(level, bucket.capacity), last

    def _persist(self) -> None:
        if self._db is None:
            return
        self._db.executemany(
            "INSERT INTO judge_limiter_state(provider, bucket, level, last) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(provider, bucket) DO UPDATE SET level = excluded.level, last = excluded.last",
            [(self.name, k, b.level, b.last) for k, b in self.buckets.items()],
        )

    def acquire(self, est_tokens: int) -> float:
        """Block until one request of `est_tokens` fits every bucket. Returns seconds waited."""
        waited = 0.0
        while True:
            with self._lock:
                now = self.clock()
                wait = max(
                    self._blocked_until - now,
                    self.buckets["rpm"].wait_time(1, now),
                    self.buckets["tpm"].wait_time(est_tokens, now),
                    self.buckets["tpd"].wait_time(est_tokens, now),
                )
                if wait <= 0:
                    self.buckets["rpm"].level -= 1
                    self.buckets["tpm"].level -= est_tokens
                    self.buckets["tpd"].level -= est_tokens
                    self._persist()
                    return waited
            if wait > 60:
                log.warning(
                    "%s: quota exhausted, waiting %.0fs (not switching provider)", self.name, wait
                )
            self.sleep(wait)
            waited += wait

    def reconcile(self, est_tokens: int, actual_tokens: int) -> None:
        delta = actual_tokens - est_tokens
        with self._lock:
            self.buckets["tpm"].level -= delta
            self.buckets["tpd"].level -= delta
            self._persist()

    def block_for(self, seconds: float) -> None:
        with self._lock:
            self._blocked_until = max(self._blocked_until, self.clock() + seconds)


# ---------------------------------------------------------------------------- the pool
@dataclass
class _Provider:
    config: JudgeProviderConfig
    client: Any
    limiter: ProviderLimiter
    calls: int = 0
    tokens: int = 0
    cache_hits: int = 0
    errors: list[str] = field(default_factory=list)


class JudgePool:
    def __init__(
        self,
        cache: ResponseCache | None = None,
        *,
        providers: Sequence[JudgeProviderConfig] | None = None,
        clients: dict[str, Any] | None = None,
        rubric: str = DEFAULT_RUBRIC,
        batch_size: int = MAX_BATCH,
        reasoning_effort: str = "low",
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        limiter_state_path: str | Path | None = None,
        max_rate_limit_waits: int = 1000,
        available_only: bool = False,
    ) -> None:
        if not 1 <= batch_size <= MAX_BATCH:
            raise ValueError(f"batch_size must be in 1..{MAX_BATCH}")
        self.cache = cache
        self.rubric = rubric
        self.batch_size = batch_size
        self.reasoning_effort = reasoning_effort
        self.sleep = sleep
        self.max_rate_limit_waits = max_rate_limit_waits
        self.schema = json_schema_for(_BatchOut)
        clients = clients or {}
        self.providers: list[_Provider] = []
        self.unavailable: list[str] = []  # providers dropped because their API key is not set
        for cfg in providers or judge_providers():
            client = clients.get(cfg.name)
            if client is None:
                if not cfg.api_key:
                    if available_only:
                        self.unavailable.append(cfg.name)
                        continue
                    raise RuntimeError(f"{cfg.api_key_env} is not set")
                client = openai.OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, max_retries=0)
            limiter = ProviderLimiter(
                cfg.name, cfg.limits, clock=clock, sleep=sleep, state_path=limiter_state_path
            )
            self.providers.append(_Provider(cfg, client, limiter))
        if not self.providers:
            raise RuntimeError(
                "no judge provider is available: set GROQ_API_KEY and/or CEREBRAS_API_KEY"
            )

    # ------------------------------------------------------------------ routing
    def route(self, question_id: str) -> str:
        return self.providers[route_index(question_id, len(self.providers))].config.name

    @property
    def params(self) -> dict[str, Any]:
        return {"reasoning_effort": self.reasoning_effort, "temperature": 0}

    def _item_key(self, p: _Provider, item: JudgeItem) -> str:
        return cache_key(
            provider=p.config.name,
            model=p.config.model,
            quantization=None,
            prompt={
                "rubric": item.rubric or self.rubric,
                "question_id": item.question_id,
                "question": item.question,
                "reference": item.reference,
                "response": item.response,
            },
            schema=self.schema,
            params=self.params,
        )

    # ------------------------------------------------------------------ public API
    def judge(self, items: Sequence[JudgeItem], *, provider: str | None = None) -> list[Verdict]:
        """Judge items, preserving input order. item_ids must be unique.

        Items are routed by sha256(question_id) over the available providers; `provider` forces
        every item to one provider (used to compare providers on the same items).
        """
        ids = [it.item_id for it in items]
        if len(set(ids)) != len(ids):
            raise ValueError("item_id values must be unique")
        by_provider: dict[int, list[JudgeItem]] = {}
        if provider is not None:
            names = [p.config.name for p in self.providers]
            if provider not in names:
                raise ValueError(f"provider {provider!r} is not available (have {names})")
            by_provider[names.index(provider)] = list(items)
        else:
            for it in items:
                idx = route_index(it.question_id, len(self.providers))
                by_provider.setdefault(idx, []).append(it)

        results: dict[str, Verdict] = {}
        with ThreadPoolExecutor(max_workers=max(1, len(by_provider))) as pool:
            futures = [
                pool.submit(self._judge_provider, self.providers[idx], group)
                for idx, group in by_provider.items()
            ]
            for fut in futures:
                results.update(fut.result())
        return [results[i] for i in ids]

    # ------------------------------------------------------------------ internals
    def _judge_provider(self, p: _Provider, items: list[JudgeItem]) -> dict[str, Verdict]:
        out: dict[str, Verdict] = {}
        pending: list[JudgeItem] = []
        for it in items:
            hit = self.cache.get(self._item_key(p, it)) if self.cache is not None else None
            if hit is not None:
                p.cache_hits += 1
                out[it.item_id] = Verdict(
                    it.item_id, it.question_id, hit["label"], hit["rationale"],
                    p.config.name, p.config.model, cached=True,
                )  # fmt: skip
            else:
                pending.append(it)

        groups: dict[str, list[JudgeItem]] = {}
        for it in pending:  # one batch never mixes grading instructions
            groups.setdefault(it.rubric or self.rubric, []).append(it)
        chunks = [
            (rubric, group[i : i + self.batch_size])
            for rubric, group in groups.items()
            for i in range(0, len(group), self.batch_size)
        ]
        for rubric, chunk in chunks:
            got = self._judge_chunk(p, chunk, rubric)
            missing = [it for it in chunk if it.item_id not in got]
            for it in missing:  # model dropped some items: re-ask one at a time
                got.update(self._judge_chunk(p, [it], rubric))
            for it in chunk:
                if it.item_id not in got:
                    raise RuntimeError(f"{p.config.name}: no verdict for {it.item_id}")
                v = got[it.item_id]
                out[it.item_id] = Verdict(
                    it.item_id, it.question_id, v.label, v.rationale, p.config.name, p.config.model
                )
                if self.cache is not None:
                    self.cache.put(
                        self._item_key(p, it),
                        {"label": v.label, "rationale": v.rationale},
                        provider=p.config.name,
                        model=p.config.model,
                        meta={"question_id": it.question_id, "item_id": it.item_id},
                    )
        return out

    def _messages(self, chunk: list[JudgeItem], rubric: str | None = None) -> list[dict[str, str]]:
        payload = [
            {
                "item_id": it.item_id,
                "question": it.question,
                "reference_answer": it.reference,
                "response": it.response,
            }
            for it in chunk
        ]
        return [
            {"role": "system", "content": rubric or self.rubric},
            {
                "role": "user",
                "content": (
                    f"Grade these {len(chunk)} items. Reply with JSON "
                    '{"verdicts": [{"item_id", "rationale", "label"}, ...]} '
                    "with exactly one verdict per item_id; keep each rationale to one sentence.\n\n"
                    + json.dumps(payload, ensure_ascii=False, indent=1)
                ),
            },
        ]

    def _judge_chunk(
        self, p: _Provider, chunk: list[JudgeItem], rubric: str | None = None
    ) -> dict[str, _VerdictOut]:
        messages = self._messages(chunk, rubric)
        max_out = 400 + 250 * len(chunk)
        est = sum(len(m["content"]) for m in messages) // 4 + max_out
        kwargs = {
            "model": p.config.model,
            "messages": messages,
            "reasoning_effort": self.reasoning_effort,
            "temperature": 0,
            "max_completion_tokens": max_out,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "verdicts", "schema": self.schema, "strict": True},
            },
        }
        waits = 0
        while True:
            p.limiter.acquire(est)
            try:
                resp = p.client.chat.completions.create(**kwargs)
            except openai.RateLimitError as exc:
                waits += 1
                if waits > self.max_rate_limit_waits:
                    raise
                retry_after = _retry_after(exc) or 60.0
                log.warning("%s 429; blocking %.0fs", p.config.name, retry_after)
                p.limiter.block_for(retry_after)
                continue
            except (
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.InternalServerError,
            ) as exc:
                waits += 1
                if waits > 5:
                    raise
                p.errors.append(str(exc))
                self.sleep(min(2**waits, 60))
                continue
            break

        p.calls += 1
        used = getattr(getattr(resp, "usage", None), "total_tokens", None)
        if used:
            p.tokens += int(used)
            p.limiter.reconcile(est, int(used))
        content = resp.choices[0].message.content or ""
        wanted = {it.item_id for it in chunk}
        try:
            verdicts = parse_verdicts(content)
        except (ValueError, ValidationError) as exc:
            p.errors.append(f"unparseable judge output: {exc}")
            return {}
        return {v.item_id: v for v in verdicts if v.item_id in wanted}


def parse_verdicts(content: str) -> list[_VerdictOut]:
    """Accept {"verdicts": [...]} or a bare JSON array."""
    data = json.loads(content)
    if isinstance(data, list):
        data = {"verdicts": data}
    return _BatchOut.model_validate(data).verdicts


def _retry_after(exc: openai.RateLimitError) -> float | None:
    try:
        val = exc.response.headers.get("retry-after")
        return float(val) if val is not None else None
    except (AttributeError, ValueError):
        return None
