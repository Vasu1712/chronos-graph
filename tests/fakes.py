"""Fake OpenAI-style clients and a fake clock for offline tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import httpx
import openai


def completion(content: str, total_tokens: int = 100) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(total_tokens=total_tokens),
    )


class FakeChatClient:
    """Mimics `client.chat.completions.create(**kwargs)`.

    `responder(kwargs) -> str | Exception | SimpleNamespace` decides each reply.
    """

    def __init__(self, responder: Callable[[dict[str, Any]], Any]) -> None:
        self.responder = responder
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        out = self.responder(kwargs)
        if isinstance(out, BaseException):
            raise out
        if isinstance(out, str):
            return completion(out)
        return out


def scripted(*replies: Any) -> FakeChatClient:
    queue = list(replies)
    return FakeChatClient(lambda _kw: queue.pop(0))


def rate_limit_error(retry_after: float | None = 7) -> openai.RateLimitError:
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    resp = httpx.Response(429, headers=headers, request=httpx.Request("POST", "http://fake"))
    return openai.RateLimitError("rate limited", response=resp, body=None)


def connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=httpx.Request("POST", "http://fake"))


def judge_responder(label: str = "CORRECT", drop: set[str] | None = None, bare: bool = False):
    """Answer every item in the batch prompt with `label` (optionally dropping some ids)."""

    def respond(kwargs: dict[str, Any]) -> str:
        user = kwargs["messages"][-1]["content"]
        items = json.loads(user.rsplit("\n\n", 1)[1])  # payload follows the blank line
        verdicts = [
            {"item_id": it["item_id"], "rationale": "ok", "label": label}
            for it in items
            if it["item_id"] not in (drop or set())
        ]
        return json.dumps(verdicts if bare else {"verdicts": verdicts})

    return respond


class FakeClock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.now += s


class FakeEmbedder:
    """Deterministic bag-of-words embedder: shared words -> high cosine similarity."""

    dim = 256

    def _vec(self, text: str):
        import hashlib
        import re

        import numpy as np

        v = np.zeros(self.dim, dtype=np.float32)
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_passages(self, texts):
        import numpy as np

        return (
            np.stack([self._vec(t) for t in texts])
            if texts
            else np.zeros((0, self.dim), np.float32)
        )

    encode_queries = encode_passages
    encode = encode_passages
