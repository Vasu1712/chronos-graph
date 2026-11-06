from chronos.config import LocalLLMConfig, judge_providers


def test_local_llm_defaults_match_spec():
    cfg = LocalLLMConfig()
    assert cfg.model == "qwen2.5:7b-instruct-q4_K_M"
    assert cfg.num_ctx == 32768
    assert cfg.temperature == 0.0


def test_judge_providers_order_and_model():
    names = [p.name for p in judge_providers()]
    assert names == ["groq", "cerebras"]
    assert all("gpt-oss-120b" in p.model for p in judge_providers())
