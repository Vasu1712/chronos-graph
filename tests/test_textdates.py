"""Relative date words resolved in code (chronos.textdates).

The LoCoMo cases are real quick_v1 failures: the gold answer is the resolved date, and the 7B
generator answered with the session date instead.
"""

import pytest

from chronos import textdates


@pytest.mark.parametrize(
    ("text", "when", "expected"),
    [
        # LoCoMo conv-50, gold "8 June, 2023"
        ("Met with the creative team for my album yesterday.", "2023-06-09",
         "yesterday [= 8 June 2023]"),
        # gold "21 August, 2023"
        ("yesterday my friends and I recorded a podcast", "2023-08-22", "yesterday [= 21 August 2023]"),
        # gold "on the Friday before 22 August, 2023" (22 Aug 2023 is a Tuesday)
        ("Last Friday I had a card-night", "2023-08-22",
         "Last Friday [= the Friday before 22 August 2023 (18 August 2023)]"),
        # gold "On a week before 16 May, 2023"
        ("I had an incident last week", "2023-05-16", "last week [= the week before 16 May 2023]"),
        # 26 March 2023 is a Sunday: "last weekend" is the previous one, 18-19 March
        ("my favorite last weekend", "2023-03-26",
         "last weekend [= the weekend before 26 March 2023 (18 March 2023)]"),
        ("going to Tokyo next month", "2023-10-19", "next month [= November 2023]"),
        ("It all started last year", "2023-10-19", "last year [= 2022]"),
        ("three days ago", "2023-10-19", "three days ago [= 16 October 2023]"),
        ("last night was a blast", "2023-09-15T00:13:00", "last night [= 14 September 2023]"),
    ],
)  # fmt: skip
def test_annotate(text, when, expected):
    assert expected in textdates.annotate(text, when)


def test_vague_and_missing_dates_untouched():
    assert (
        textdates.annotate("a while ago I lived there", "2023-01-01") == "a while ago I lived there"
    )
    assert textdates.annotate("yesterday", None) == "yesterday"
    assert textdates.annotate("yesterday", "") == "yesterday"
    assert textdates.annotate("yesterday", "not a date") == "yesterday"


def test_annotate_is_idempotent_and_strippable():
    once = textdates.annotate("I saw it yesterday and last week", "2023-06-09")
    assert textdates.annotate(once, "2023-06-09") == once
    assert textdates.strip(once) == "I saw it yesterday and last week"


def test_time_expr_for_extractor_fallback():
    r = textdates.time_expr("Met the team yesterday", "2023-06-09")
    assert (r.kind, r.offset, r.unit) == ("RELATIVE", -1, "DAY")
    r = textdates.time_expr("Last Friday I had a card-night", "2023-08-22")
    assert (r.kind, r.value) == ("ABSOLUTE", "2023-08-18")
    assert textdates.time_expr("no dates here", "2023-08-22") is None
