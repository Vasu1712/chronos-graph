"""Token counting with the Qwen2.5 tokenizer (the system LLM's own), shared by packers and clients."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

TOKENIZER_REPO = "Qwen/Qwen2.5-7B-Instruct"


@lru_cache(maxsize=1)
def qwen_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_REPO)


def count_tokens(text: str) -> int:
    try:
        return len(qwen_tokenizer()(text, add_special_tokens=False).input_ids)
    except Exception:  # tokenizer files unavailable offline: a coarse, documented fallback
        return max(1, len(text) // 4)


def count_chat_tokens(messages: Sequence[dict[str, str]]) -> int:
    """Prompt tokens of a chat request after applying Qwen's chat template (+ generation prompt)."""
    try:
        tok = qwen_tokenizer()
        text = tok.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)
        return len(tok(text, add_special_tokens=False).input_ids)
    except Exception:
        return max(1, sum(len(m["content"]) for m in messages) // 4 + 4 * len(messages))
