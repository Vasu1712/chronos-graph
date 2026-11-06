import pytest

from chronos.config import AgentConfig, RetrievalConfig
from eval.ablations import LETTERS, load_ablations, resolve_selection
from eval.loaders.types import QAItem


@pytest.fixture(scope="module")
def abl():
    return load_ablations()


def test_four_ablations_with_the_variants_the_brief_names(abl):
    assert set(abl) == set(LETTERS.values())
    names = {k: [v.name for v in a.variants] for k, a in abl.items()}
    assert names["retrieval"] == [
        "dense_only",
        "bm25_only",
        "dense_plus_bm25",
        "plus_1hop",
        "plus_episodic",
    ]
    assert names["state"] == ["append_only", "arrival_order", "bitemporal"]
    assert names["gate"] == [
        "off",
        "speaker_only",
        "full",
    ]  # 'off' must stay a string, not a YAML boolean
    assert [v.spec.get("budget") for v in abl["budget"].variants] == [500, 1000, None, 3000]


def test_every_override_is_a_real_config_field(abl):
    r, a = RetrievalConfig(), AgentConfig()
    for ab in abl.values():
        for v in ab.variants:
            for k in v.spec.get("retrieval", {}):
                assert hasattr(r, k), (v.system_id, k)
            for k, val in v.spec.get("agent", {}).items():
                assert hasattr(a, k) and isinstance(val, str) == isinstance(getattr(a, k), str), (
                    v.system_id,
                    k,
                )
            assert (v.system is None) != (v.alias_of is None), v.system_id  # exactly one of the two


def test_aliases_point_at_main_systems_and_state_uses_replayed_dbs(abl):
    assert abl["state"].aliases == {"B5_append_only", "chronos"} and abl["state"].db_variants == {
        "arrival_order"
    }
    assert abl["state"].probes is True


def qa(dataset, category):
    return QAItem(dataset, "q", "c", "?", "a", category)


def test_question_filters(abl):
    assert abl["retrieval"].selects(qa("locomo", "multi-hop")) and not abl["retrieval"].selects(
        qa("locomo", "temporal")
    )
    assert not abl["retrieval"].selects(qa("longmemeval", "multi-hop"))
    assert abl["state"].selects(qa("longmemeval", "knowledge-update")) and not abl["state"].selects(
        qa("locomo", "multi-hop")
    )
    assert abl["gate"].selects(qa("locomo", "adversarial")) and abl["budget"].selects(
        qa("longmemeval", "anything")
    )


def test_selection_parsing(abl):
    assert [a.id for a in resolve_selection("A,C", abl)] == ["A_retrieval", "C_gate"]
    assert [a.id for a in resolve_selection("all", abl)] == [
        "A_retrieval",
        "B_state",
        "C_gate",
        "D_budget",
    ]
    assert resolve_selection("", abl) == []
    with pytest.raises(SystemExit):
        resolve_selection("Z", abl)


def test_every_ablation_names_an_existing_reference_variant(abl):
    for ab in abl.values():
        assert ab.reference in [v.name for v in ab.variants], ab.id
