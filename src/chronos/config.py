"""Central configuration. Values come from the environment (and `.env`), with SPEC defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env", override=False)


def _env(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val if val not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


@dataclass(frozen=True)
class LocalLLMConfig:
    base_url: str = field(
        default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    )
    # Canonical model tag: used in cache keys, logs and reports.
    model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M"))
    # Tag actually requested: an alias of `model` with num_ctx=32768 baked in by
    # scripts/setup_ollama.sh, because Ollama's /v1 endpoint may ignore per-request num_ctx.
    served_model: str = field(
        default_factory=lambda: _env("OLLAMA_SERVED_MODEL", "chronos-qwen2.5-7b-32k")
    )
    quantization: str = "q4_K_M"
    num_ctx: int = 32768
    temperature: float = 0.0
    max_retries: int = 3
    timeout_s: float = 600.0


@dataclass(frozen=True)
class ProviderLimits:
    rpm: int
    tpm: int
    tpd: int


@dataclass(frozen=True)
class JudgeProviderConfig:
    name: str
    base_url: str
    model: str
    api_key_env: str
    limits: ProviderLimits

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) or None


def judge_providers() -> list[JudgeProviderConfig]:
    """Ordered list; index is the routing bucket (sha256(question_id) mod 2)."""
    return [
        JudgeProviderConfig(
            name="groq",
            base_url="https://api.groq.com/openai/v1",
            model="openai/gpt-oss-120b",
            api_key_env="GROQ_API_KEY",
            limits=ProviderLimits(
                rpm=_env_int("GROQ_RPM", 30),
                tpm=_env_int("GROQ_TPM", 8000),
                tpd=_env_int("GROQ_TPD", 200_000),
            ),
        ),
        JudgeProviderConfig(
            name="cerebras",
            base_url="https://api.cerebras.ai/v1",
            model="gpt-oss-120b",
            api_key_env="CEREBRAS_API_KEY",
            limits=ProviderLimits(
                rpm=_env_int("CEREBRAS_RPM", 5),
                tpm=_env_int("CEREBRAS_TPM", 90_000),
                tpd=_env_int("CEREBRAS_TPD", 1_000_000),
            ),
        ),
    ]


@dataclass(frozen=True)
class ModelConfig:
    embed_model: str = "BAAI/bge-small-en-v1.5"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    device: str = field(default_factory=lambda: _env("CHRONOS_DEVICE", "mps"))
    embed_batch_size: int = 64
    nli_batch_size: int = 16


@dataclass(frozen=True)
class Paths:
    db: Path = field(
        default_factory=lambda: REPO_ROOT / _env("CHRONOS_DB_PATH", "artifacts/chronos.sqlite")
    )
    llm_cache: Path = field(
        default_factory=lambda: (
            REPO_ROOT / _env("CHRONOS_LLM_CACHE_PATH", "artifacts/llm_cache.sqlite")
        )
    )
    data_raw: Path = REPO_ROOT / "data" / "raw"
    manifests: Path = REPO_ROOT / "eval" / "manifests"
    ontology: Path = REPO_ROOT / "src" / "chronos" / "ontology"


@dataclass(frozen=True)
class RetrievalConfig:
    rrf_k: int = 60
    candidate_top_k: int = 5
    token_budget: int = 1500
    episode_window: int = 1


SEED = 42
