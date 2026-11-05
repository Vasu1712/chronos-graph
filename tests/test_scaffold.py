from chronos.config import LocalLLMConfig, judge_providers
import yaml

from chronos.config import LocalLLMConfig, Paths, judge_providers


def test_local_llm_defaults_match_spec():
    cfg = LocalLLMConfig()
    assert cfg.model == "qwen2.5:7b-instruct-q4_K_M"
    assert cfg.num_ctx == 32768
    assert cfg.temperature == 0.0


def test_judge_providers_order_and_model():
    names = [p.name for p in judge_providers()]
    assert names == ["groq", "cerebras"]
    assert all("gpt-oss-120b" in p.model for p in judge_providers())


def test_attribute_ontology_keys():
    data = yaml.safe_load((Paths().ontology / "attributes.yaml").read_text())
    expected = {
        "residence", "hometown", "employer", "job_title", "school",
        "relationship_status", "partner", "diet",
        "favorite:<category>", "primary_tool:<purpose>",
    }  # fmt: skip
    assert set(data) == expected
    for key, spec in data.items():
        assert spec["description"], key
        assert len(spec["examples"]) >= 2, key
