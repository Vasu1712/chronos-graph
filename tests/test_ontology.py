import pytest
import yaml

from chronos.ontology import ONTOLOGY_PATH, OntologyError, load_ontology, parse_ontology

SPEC_KEYS = {
    "residence", "hometown", "employer", "job_title", "school",
    "relationship_status", "partner", "diet",
    "favorite:<category>", "primary_tool:<purpose>",
}  # fmt: skip


def test_ontology_has_exactly_spec_keys_and_validates():
    onto = load_ontology()
    assert set(onto.specs) == SPEC_KEYS
    for spec in onto:
        assert spec.description and "\n" not in spec.description
        assert 3 <= len(spec.paraphrases) <= 5
        assert spec.not_examples
    res = onto.spec_for("residence")
    assert {"lives in", "based in", "moved to", "settled in"} <= set(res.paraphrases)
    assert any("wedding" in n for n in res.not_examples)
    assert any("visiting" in n for n in res.not_examples)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("residence", "residence"),
        (" Residence ", "residence"),
        ("favorite:book", "favorite:book"),
        ("Favorite: Board Game", "favorite:board_game"),
        ("primary_tool:note-taking", "primary_tool:note_taking"),
        ("favorite", None),  # bare template
        ("favorite:<category>", None),  # unfilled template
        ("favorite:category", None),  # slot name, not a value
        ("residence:city", None),  # non-template with a slot
        ("hobby", None),  # unknown
        ("", None),
        (None, None),
    ],
)
def test_canonicalize(raw, expected):
    assert load_ontology().canonicalize(raw) == expected


def test_prompt_rendering_is_deterministic_and_complete():
    onto = load_ontology()
    text = onto.render_for_prompt()
    assert text == load_ontology().render_for_prompt()
    for key in SPEC_KEYS:
        assert f"- {key}:" in text
    assert "NOT this key" in text


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda d: d["residence"].pop("not"), "missing 'not'"),
        (lambda d: d["residence"].update(paraphrases=["a", "b"]), "3-5 paraphrases"),
        (lambda d: d["residence"].update(paraphrases=list("abcdef")), "3-5 paraphrases"),
        (lambda d: d["residence"].update(description="two\nlines"), "one non-empty line"),
        (lambda d: d.update({"Bad Key": d["residence"]}), "bad key"),
        (lambda d: d.update({"favorite:category": d["diet"]}), "bad key"),
    ],
)
def test_validation_rejects_bad_files(mutate, msg):
    data = yaml.safe_load(ONTOLOGY_PATH.read_text())
    mutate(data)
    with pytest.raises(OntologyError, match=msg):
        parse_ontology(data)
