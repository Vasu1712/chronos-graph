"""Optional live smoke tests: `make test-live` (or `uv run pytest -m live`)."""

import os

import pytest
from pydantic import BaseModel

pytestmark = pytest.mark.live


class Capital(BaseModel):
    city: str


def test_live_ollama_structured_and_context():
    from chronos.llm.local_client import LocalLLM

    llm = LocalLLM()
    assert llm.verify_context() == 32768, "run scripts/setup_ollama.sh"
    out = llm.structured(
        [{"role": "user", "content": "What is the capital of France? Answer as JSON."}], Capital
    )
    assert "paris" in out.city.lower()


@pytest.mark.parametrize("provider", ["groq", "cerebras"])
def test_live_judge_provider(provider):
    from chronos.config import judge_providers
    from chronos.llm.judge_pool import JudgeItem, JudgePool, route_index

    cfg = next(p for p in judge_providers() if p.name == provider)
    if not os.environ.get(cfg.api_key_env):
        pytest.skip(f"{cfg.api_key_env} not set")
    idx = [p.name for p in judge_providers()].index(provider)
    qid = next(f"live{i}" for i in range(100) if route_index(f"live{i}") == idx)
    pool = JudgePool(providers=[*judge_providers()])
    [v] = pool.judge(
        [JudgeItem(item_id="x", question_id=qid, question="2+2?", reference="4", response="four")]
    )
    assert v.provider == provider and v.label == "CORRECT"


def test_live_embedder():
    from chronos.llm.embed import Embedder

    v = Embedder().encode_passages(["Caroline lives in Mumbai.", "Melanie paints."])
    assert v.shape == (2, 384)


def test_live_nli():
    from chronos.llm.nli import NLI

    nli = NLI()
    assert nli.entails("Caroline moved to Mumbai in 2022.", "Caroline lives in Mumbai.")
    assert not nli.entails("Caroline moved to Mumbai in 2022.", "Caroline lives in Delhi.")
