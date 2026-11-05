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
        "is_correction": False,
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
            "entities_mentioned", "time", "is_correction", "quote", "is_event"} == set(props)  # fmt: skip
