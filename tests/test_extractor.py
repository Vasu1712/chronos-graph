import pytest

from chronos.ingestion.extractor import (
    FEW_SHOT,
    ExtractionOutput,
    Extractor,
    match_quote,
)
from chronos.ingestion.types import Session, Turn


def turn(i, speaker, text, conv="c1", s=1):
    return Turn(f"t:{conv}:s{s:02d}:t{i:03d}", s, i, speaker, text, f"D{s}:{i}")


def session(texts, ts="2023-05-17T10:00:00"):
    speakers = ["Priya", "Rahul"]
    turns = tuple(turn(i, speakers[(i - 1) % 2], t) for i, t in enumerate(texts, start=1))
    return Session("c1:s01", 1, ts, turns, "session_1")


def fact(
    turn_index, quote, text="Priya lives in Bengaluru.", key="residence", tense="CURRENT", **kw
):
    return {
        "turn_index": turn_index,
        "subject": kw.get("subject", "Priya"),
        "fact_text": text,
        "attribute_key": key,
        "polarity": kw.get("polarity", "POS"),
        "mode": "EXPLICIT",
        "is_event": kw.get("is_event", False),
        "entities_mentioned": [],
        "time": {"kind": "NONE", "value": None, "offset": None, "unit": None, "tense": tense},
        "is_correction": kw.get("corr", False),
        "correction": kw.get("correction"),
        "quote": quote,
    }


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def structured(self, messages, response_model):
        self.calls.append(messages)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return response_model.model_validate({"facts": r})


@pytest.mark.parametrize(
    "quote, text, expected",
    [
        ("lives in Bengaluru", "I  lives in\nBengaluru now", (3, 21, False)),  # newline in turn
        ("live in Bengaluru", "I live in Bengalru now", "fuzzy"),  # typo in turn
        ("live in Bengaluru", "I live in Bengaluru now", (2, 19, False)),
        ("live  in\nBengaluru", "I live in Bengaluru now", (2, 19, False)),  # whitespace normalized
        ("live in  Bengaluru", "I live in\n  Bengaluru now", (2, 21, False)),  # spans original text
        (
            '"live in Bengaluru"',
            "I live in Bengaluru now",
            (2, 19, False),
        ),  # wrapping quotes stripped
        ("I live in Bengaluru now!", "I live in Bengaluru now", (0, 23, True)),  # longer than turn
        ("I am living in Bengaluru", "Honestly I'm living in Bengaluru these days", "fuzzy"),
        ("moved to Mumbai", "I live in Bengaluru now", None),
        ("", "anything", None),
    ],
)
def test_match_quote(quote, text, expected):
    got = match_quote(quote, text)
    if expected == "fuzzy":
        assert got is not None and got[2] is True
        assert "Bengal" in text[got[0] : got[1]]
    else:
        assert got == expected


def test_few_shot_examples_validate_and_cover_required_cases():
    assert len(FEW_SHOT) == 4
    text = "\n".join(FEW_SHOT)
    assert "moved to Porto" in text  # move
    assert '"is_correction": true' in text  # correction
    assert '"tense": "PAST_ENDED"' in text and '"value": "2015"' in text  # past fact
    assert '"is_event": true' in text  # point event


def test_chunking_and_static_prompt_prefix():
    texts = [f"turn number {i}" for i in range(1, 21)]
    llm = FakeLLM([[], [], []])
    ex = Extractor(llm)
    facts, stats = ex.extract_session(
        session(texts), ["Priya", "Rahul"], ["Priya (PERSON, speaker)"]
    )
    assert stats.chunks == 3 and facts == []
    systems = {c[0]["content"] for c in llm.calls}
    assert len(systems) == 1  # identical static prefix for every chunk
    assert "residence:" in next(iter(systems)) and "in Pune this week for a wedding" in next(
        iter(systems)
    )
    second = llm.calls[1][1]["content"]
    ctx, tgt = second.split("TARGET turns:")
    assert "[t5]" in ctx and "[t8]" in ctx and "[t4]" not in ctx  # previous 4 turns as context
    assert "[t9]" in tgt and "[t16]" in tgt and "[t17]" not in tgt  # up to 8 targets
    assert "Session date: 2023-05-17 (Wednesday)" in second
    assert "Speakers: Priya, Rahul" in second


def test_validation_keeps_fixes_and_drops():
    texts = ["I live in Bengaluru now.", "Cool!", "I'm in Pune this week for a wedding."]
    replies = [
        [
            fact(1, "live in   Bengaluru"),  # exact after whitespace normalization
            fact(1, "live in Bengaluru", text="Priya lives in Bengaluru."),  # duplicate
            fact(
                3,
                "in Pune this wek for a wedding",
                text="Priya is in Pune for a wedding.",
                key=None,
            ),  # fuzzy
            fact(3, "I moved to Pune", text="Priya moved to Pune."),  # quote not in turn
            fact(7, "anything", text="Bad turn."),  # not a target turn
            fact(
                1, "live in Bengaluru", text="Priya's city is Bengaluru.", key="city"
            ),  # unknown key
            fact(
                3, "I'm in Pune", text="Priya plans to move to Pune.", tense="FUTURE_PLAN"
            ),  # key dropped
            fact(1, "live", text="  ", key=None),  # empty fact_text
        ]
    ]
    ex = Extractor(FakeLLM(replies))
    facts, stats = ex.extract_session(session(texts), ["Priya", "Rahul"])
    assert [f.fact_text for f in facts] == [
        "Priya lives in Bengaluru.",
        "Priya is in Pune for a wedding.",
        "Priya's city is Bengaluru.",
        "Priya plans to move to Pune.",
    ]
    first, fuzzy, unknown_key, plan = facts
    assert first.quote == "live in Bengaluru" and (first.quote_start, first.quote_end) == (2, 19)
    assert fuzzy.quote_fuzzy and fuzzy.quote in texts[2]
    assert unknown_key.attribute_key is None
    assert plan.attribute_key is None and plan.time.tense == "FUTURE_PLAN"
    assert [f.ordinal_in_turn for f in facts] == [0, 0, 1, 1]
    assert stats.raw_facts == 8 and stats.kept == 4
    assert stats.dropped == {"duplicate": 1, "quote_not_found": 1, "bad_turn_index": 1, "empty": 1}
    assert stats.fuzzy_quotes == 1 and stats.attribute_key_nulled == 1


def test_failed_chunk_is_counted_not_fatal():
    texts = [f"t{i}" for i in range(1, 10)]
    ex = Extractor(
        FakeLLM([RuntimeError("bad json"), [fact(9, "t9", text="Priya said t9.", key=None)]])
    )
    facts, stats = ex.extract_session(session(texts), ["Priya", "Rahul"])
    assert stats.chunks == 2 and stats.chunks_failed == 1
    assert len(facts) == 1


def test_output_schema_has_no_refs():
    from chronos.llm.schema_utils import json_schema_for

    schema = json_schema_for(ExtractionOutput)
    assert "$ref" not in str(schema)
    props = schema["properties"]["facts"]["items"]["properties"]
    assert {"turn_index", "subject", "fact_text", "attribute_key", "polarity", "mode",
            "entities_mentioned", "time", "is_correction", "correction", "quote", "is_event"} == set(props)  # fmt: skip


def test_wrong_turn_index_is_repaired_only_when_unambiguous():
    texts = [
        "I live in Pune now.",
        "Pune, nice!",
        "Pune, sorry, I meant Mumbai.",
        "Mumbai is great.",
    ]
    replies = [[
        fact(1, "sorry, I meant Mumbai", text="Priya lives in Mumbai."),  # really in t3: reassigned
        fact(2, "Mumbai", text="Mumbai is mentioned."),  # in t3 and t4: ambiguous -> dropped
        fact(9, "nowhere in this chunk", text="Ghost."),  # bad index, no match -> dropped
    ]]  # fmt: skip
    facts, stats = Extractor(FakeLLM(replies)).extract_session(session(texts), ["Priya", "Rahul"])
    assert [(f.fact_text, f.turn.turn_index) for f in facts] == [("Priya lives in Mumbai.", 3)]
    assert stats.turn_reassigned == 1
    assert stats.dropped == {"ambiguous_turn": 1, "bad_turn_index": 1}


def test_correction_values_and_wrong_span():
    texts = ["My manager is Priya, sorry, I meant Priyanka."]
    replies = [[fact(1, "sorry, I meant Priyanka", text="Priya's manager is Priyanka.", key=None, corr=True,
                     correction={"wrong": "Priya", "right": "Priyanka"})]]  # fmt: skip
    [f], _ = Extractor(FakeLLM(replies)).extract_session(session(texts), ["Priya", "Rahul"])
    assert (f.correction_wrong, f.correction_right) == ("Priya", "Priyanka")
    assert (
        texts[0][f.wrong_start : f.wrong_end] == "Priya" and f.wrong_start == 14
    )  # word match, not inside "Priyanka"


def test_tense_rule_and_correction_rule_in_prompt():
    from chronos.ingestion.extractor import system_prompt
    from chronos.ontology import load_ontology

    sp = system_prompt(load_ontology())
    assert "A state that began in the past and still holds is CURRENT" in sp
    assert "`correction`" in sp and '"wrong": "Riverside High"' in sp


@pytest.mark.parametrize(
    "quote, kind, value, offset, unit, expected",
    [
        ("We packed up and moved to Bengaluru last week", "RELATIVE", None, -1, "WEEK", True),
        ("I joined Quanta in February", "ABSOLUTE", "2023-02", None, None, True),
        ("I moved to Austin", "NONE", None, None, None, True),
        (
            "In 2015 I moved to Delhi",
            "ABSOLUTE",
            "2015",
            None,
            None,
            False,
        ),  # late mention stays historical
        ("Back in 2019 I lived in Delhi", "ABSOLUTE", "2019", None, None, False),  # no onset verb
        (
            "I moved to Pune last month but left in a week",
            "RELATIVE",
            None,
            -1,
            "MONTH",
            False,
        ),  # ending cue
        (
            "Before that, on May 2nd we moved to Hyderabad",
            "ABSOLUTE",
            "2023-05-02",
            None,
            None,
            False,
        ),
    ],
)
def test_recent_onset_rule(quote, kind, value, offset, unit, expected):
    from chronos.ingestion.extractor import is_recent_onset
    from chronos.ingestion.temporal import TimeExpr

    t = TimeExpr(kind=kind, value=value, offset=offset, unit=unit, tense="PAST_ENDED")
    assert is_recent_onset(t, quote, "2023-05-17T10:00:00") is expected
    assert is_recent_onset(t.model_copy(update={"tense": "CURRENT"}), quote, "2023-05-17") is False


def test_recent_onset_applied_only_to_attribute_facts():
    texts = ["We moved to Bengaluru last week!", "I moved my desk to the window."]
    past = {"kind": "RELATIVE", "value": None, "offset": -1, "unit": "WEEK", "tense": "PAST_ENDED"}
    a = fact(1, "We moved to Bengaluru last week", text="Priya moved to Bengaluru.")
    b = fact(2, "I moved my desk", text="Priya moved her desk.", key=None)
    a["time"], b["time"] = past, dict(past)
    facts, stats = Extractor(FakeLLM([[a, b]])).extract_session(session(texts), ["Priya", "Rahul"])
    assert [f.time.tense for f in facts] == [
        "CURRENT",
        "PAST_ENDED",
    ] and stats.tense_onset_fixed == 1
