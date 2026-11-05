"""Table-driven tests for the temporal resolver (SPEC §3 step 2)."""

import pytest

from chronos.ingestion.temporal import (
    Cmp,
    Precision,
    TimeExpr,
    TimePoint,
    compare,
    describe,
    resolve,
)

D, W, M, Y = Precision.DAY, Precision.WEEK, Precision.MONTH, Precision.YEAR
S, B, U = Precision.SESSION, Precision.BEFORE, Precision.UNKNOWN


def rel(offset, unit, tense="CURRENT"):
    return TimeExpr(kind="RELATIVE", offset=offset, unit=unit, tense=tense)


def ab(value, tense="CURRENT"):
    return TimeExpr(kind="ABSOLUTE", value=value, tense=tense)


NONE = TimeExpr()
USED_TO = TimeExpr(tense="PAST_ENDED")

# (id, expr, session_ts, is_event, expected (valid_from, p, valid_to, p, event_date, p))
RESOLVE_CASES = [
    # --- the rules named in the brief
    (
        "last week",
        rel(-1, "WEEK"),
        "2023-05-17T10:00:00",
        False,
        ("2023-05-10", W, None, None, None, None),
    ),
    (
        "last month",
        rel(-1, "MONTH"),
        "2023-05-17T10:00:00",
        False,
        ("2023-04-01", M, None, None, None, None),
    ),
    (
        "in 2019 (past)",
        ab("2019", "PAST_ENDED"),
        "2023-05-17T10:00:00",
        False,
        ("2019-01-01", Y, "2019-12-31", Y, None, None),
    ),
    (
        "no time words",
        NONE,
        "2023-05-17T10:00:00",
        False,
        ("2023-05-17", S, None, None, None, None),
    ),
    ("used to", USED_TO, "2023-05-17T10:00:00", False, (None, U, "2023-05-17", B, None, None)),
    (
        "vague current",
        TimeExpr(kind="RELATIVE", value="a while ago"),
        "2023-05-17",
        False,
        (None, U, None, None, None, None),
    ),
    (
        "vague past",
        TimeExpr(kind="RELATIVE", value="a while ago", tense="PAST_ENDED"),
        "2023-05-17",
        False,
        (None, U, "2023-05-17", B, None, None),
    ),
    # --- month boundaries
    (
        "last month in January",
        rel(-1, "MONTH"),
        "2023-01-15T09:00:00",
        False,
        ("2022-12-01", M, None, None, None, None),
    ),
    (
        "2 months ago from Mar 31",
        rel(-2, "MONTH"),
        "2023-03-31",
        False,
        ("2023-01-01", M, None, None, None, None),
    ),
    (
        "next month in December",
        rel(1, "MONTH", "FUTURE_PLAN"),
        "2023-12-10",
        False,
        ("2024-01-01", M, None, None, None, None),
    ),
    (
        "last month closed (Feb leap)",
        rel(-1, "MONTH", "PAST_ENDED"),
        "2024-03-05",
        False,
        ("2024-02-01", M, "2024-02-29", M, None, None),
    ),
    (
        "absolute month closed",
        ab("2023-02", "PAST_ENDED"),
        "2023-06-01",
        False,
        ("2023-02-01", M, "2023-02-28", M, None, None),
    ),
    (
        "yesterday across month (leap)",
        rel(-1, "DAY"),
        "2024-03-01",
        True,
        (None, None, None, None, "2024-02-29", D),
    ),
    # --- year boundaries
    (
        "last week across new year",
        rel(-1, "WEEK"),
        "2023-01-03",
        False,
        ("2022-12-27", W, None, None, None, None),
    ),
    (
        "last year on Jan 1",
        rel(-1, "YEAR", "PAST_ENDED"),
        "2023-01-01",
        False,
        ("2022-01-01", Y, "2022-12-31", Y, None, None),
    ),
    (
        "yesterday on Jan 1",
        rel(-1, "DAY"),
        "2023-01-01T08:00:00",
        True,
        (None, None, None, None, "2022-12-31", D),
    ),
    (
        "tomorrow on Dec 31",
        rel(1, "DAY", "FUTURE_PLAN"),
        "2023-12-31",
        True,
        (None, None, None, None, "2024-01-01", D),
    ),
    (
        "13 months ago",
        rel(-13, "MONTH"),
        "2023-01-20",
        False,
        ("2021-12-01", M, None, None, None, None),
    ),
    # --- other shapes
    (
        "since 2020 (current)",
        ab("2020"),
        "2023-05-17",
        False,
        ("2020-01-01", Y, None, None, None, None),
    ),
    (
        "absolute day event",
        ab("2023-05-03"),
        "2023-05-17",
        True,
        (None, None, None, None, "2023-05-03", D),
    ),
    ("event no time", NONE, "2023-05-17T22:10:00", True, (None, None, None, None, "2023-05-17", S)),
    (
        "event last week",
        rel(-1, "WEEK"),
        "2023-05-17",
        True,
        (None, None, None, None, "2023-05-10", W),
    ),
    (
        "event vague",
        TimeExpr(kind="ABSOLUTE", value="some time"),
        "2023-05-17",
        True,
        (None, None, None, None, None, U),
    ),
    (
        "3 days ago",
        rel(-3, "DAY", "PAST_ENDED"),
        "2023-05-02",
        False,
        ("2023-04-29", D, "2023-04-29", D, None, None),
    ),
    (
        "2 weeks ago closed",
        rel(-2, "WEEK", "PAST_ENDED"),
        "2023-05-17",
        False,
        ("2023-05-03", W, "2023-05-09", W, None, None),
    ),
    (
        "future plan no time",
        TimeExpr(tense="FUTURE_PLAN"),
        "2023-05-17",
        False,
        ("2023-05-17", S, None, None, None, None),
    ),
    (
        "invalid absolute date",
        ab("2023-02-30"),
        "2023-05-17",
        False,
        (None, U, None, None, None, None),
    ),
    (
        "relative missing unit",
        TimeExpr(kind="RELATIVE", offset=-2),
        "2023-05-17",
        False,
        (None, U, None, None, None, None),
    ),
    (
        "date-only session_ts",
        rel(-1, "YEAR"),
        "2020-02-29",
        False,
        ("2019-01-01", Y, None, None, None, None),
    ),
]


@pytest.mark.parametrize(
    "case_id, expr, ts, is_event, expected", RESOLVE_CASES, ids=[c[0] for c in RESOLVE_CASES]
)
def test_resolve(case_id, expr, ts, is_event, expected):
    rt = resolve(expr, ts, is_event=is_event)
    got = (
        rt.valid_from, rt.valid_from_precision, rt.valid_to, rt.valid_to_precision,
        rt.event_date, rt.event_date_precision,
    )  # fmt: skip
    assert got == expected


def P(d, p):
    return TimePoint(d, p)


COMPARE_CASES = [
    ("day later", P("2023-05-02", D), P("2023-05-01", D), Cmp.LATER),
    ("day earlier", P("2023-05-01", D), P("2023-05-02", D), Cmp.EARLIER),
    ("same day", P("2023-05-01", D), P("2023-05-01", D), Cmp.INCOMPARABLE),
    ("day inside month", P("2023-05-20", D), P("2023-05-01", M), Cmp.INCOMPARABLE),
    ("month boundary", P("2023-05-31", D), P("2023-06-01", M), Cmp.EARLIER),
    ("year beats day", P("2024-01-01", Y), P("2023-12-31", D), Cmp.LATER),
    ("same year", P("2019-06-01", M), P("2019-01-01", Y), Cmp.INCOMPARABLE),
    ("week same iso week", P("2023-05-10", W), P("2023-05-12", D), Cmp.INCOMPARABLE),
    ("week across year", P("2023-01-02", W), P("2022-12-31", D), Cmp.LATER),
    ("week vs month", P("2023-04-28", W), P("2023-05-01", M), Cmp.EARLIER),
    ("session incomparable", P("2023-05-17", S), P("2019-01-01", Y), Cmp.INCOMPARABLE),
    ("both session", P("2023-06-01", S), P("2023-05-01", S), Cmp.INCOMPARABLE),
    ("before incomparable", P("2023-05-17", B), P("2023-01-01", D), Cmp.INCOMPARABLE),
    ("unknown", P(None, U), P("2023-01-01", D), Cmp.INCOMPARABLE),
    ("missing", P("2023-01-01", D), P(None, None), Cmp.INCOMPARABLE),
]


@pytest.mark.parametrize(
    "case_id, a, b, expected", COMPARE_CASES, ids=[c[0] for c in COMPARE_CASES]
)
def test_compare(case_id, a, b, expected):
    assert compare(a, b) == expected
    # antisymmetry
    flipped = {Cmp.LATER: Cmp.EARLIER, Cmp.EARLIER: Cmp.LATER}.get(expected, Cmp.INCOMPARABLE)
    assert compare(b, a) == flipped


def test_describe():
    assert (
        describe(resolve(ab("2019", "PAST_ENDED"), "2023-01-01"))
        == "from 2019-01-01 (YEAR) to 2019-12-31 (YEAR)"
    )
    assert describe(resolve(NONE, "2023-01-01")) == "from 2023-01-01 (SESSION) to now"
    assert describe(resolve(USED_TO, "2023-01-01")) == "from unknown to 2023-01-01 (BEFORE)"
    assert describe(resolve(NONE, "2023-01-01", is_event=True)) == "event on 2023-01-01 (SESSION)"
