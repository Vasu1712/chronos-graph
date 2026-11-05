"""Temporal resolver: normalized time expression + session_ts -> event-time interval.

Pure Python, no LLM. The extractor only *normalizes* a time expression into a TimeExpr; this
module computes dates and precisions (SPEC §3 step 2).

Spans (lo..hi) for a resolved expression, anchored at d0 = date(session_ts):
  ABSOLUTE "YYYY"        -> Jan 1 .. Dec 31                YEAR
  ABSOLUTE "YYYY-MM"     -> first .. last day of month      MONTH
  ABSOLUTE "YYYY-MM-DD"  -> that day                        DAY
  RELATIVE n DAY         -> d0 + n days                     DAY
  RELATIVE n WEEK        -> d0 + 7n days .. +6 days         WEEK   ("last week" = d0 - 7 days)
  RELATIVE n MONTH       -> calendar month shifted by n     MONTH  ("last month" = previous month)
  RELATIVE n YEAR        -> calendar year shifted by n      YEAR
  anything unparseable / missing offset or unit -> vague

How a span becomes an interval depends on tense (state facts) or is_event (point events):
  state CURRENT      span: valid_from = lo           none: valid_from = d0 SESSION   vague: UNKNOWN
  state PAST_ENDED   span: [lo, hi] closed           none/vague ("used to"): valid_to = d0 BEFORE
  state FUTURE_PLAN  span: valid_from = lo           none: valid_from = d0 SESSION   vague: UNKNOWN
  event              span: event_date = lo           none: event_date = d0 SESSION   vague: UNKNOWN
Dates are stored as "YYYY-MM-DD".
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict


class Precision(StrEnum):
    DAY = "DAY"
    WEEK = "WEEK"
    MONTH = "MONTH"
    YEAR = "YEAR"
    SESSION = "SESSION"
    BEFORE = "BEFORE"
    UNKNOWN = "UNKNOWN"


class Cmp(StrEnum):
    LATER = "LATER"
    EARLIER = "EARLIER"
    INCOMPARABLE = "INCOMPARABLE"


# Precisions that denote a calendar bucket and can be compared, finest first.
_RANK = {Precision.DAY: 0, Precision.WEEK: 1, Precision.MONTH: 2, Precision.YEAR: 3}

TimeKind = Literal["ABSOLUTE", "RELATIVE", "NONE"]
Unit = Literal["DAY", "WEEK", "MONTH", "YEAR"]
Tense = Literal["CURRENT", "PAST_ENDED", "FUTURE_PLAN"]


class TimeExpr(BaseModel):
    """Normalized time expression as produced by the extractor."""

    model_config = ConfigDict(extra="forbid")

    kind: TimeKind = "NONE"
    value: str | None = None  # ABSOLUTE: "2019", "2019-03" or "2019-03-14"
    offset: int | None = None  # RELATIVE: signed count of units (last week = -1)
    unit: Unit | None = None  # RELATIVE
    tense: Tense = "CURRENT"


@dataclass(frozen=True)
class TimePoint:
    date: str | None
    precision: Precision | None


@dataclass(frozen=True)
class ResolvedTime:
    valid_from: str | None = None
    valid_from_precision: Precision | None = None
    valid_to: str | None = None
    valid_to_precision: Precision | None = None
    event_date: str | None = None
    event_date_precision: Precision | None = None

    @property
    def start(self) -> TimePoint:
        return TimePoint(self.valid_from, self.valid_from_precision)

    @property
    def end(self) -> TimePoint:
        return TimePoint(self.valid_to, self.valid_to_precision)

    @property
    def is_closed(self) -> bool:
        return self.valid_to is not None


# ------------------------------------------------------------------------------ helpers
def session_date(session_ts: str | date) -> date:
    if isinstance(session_ts, date):
        return session_ts
    return (
        datetime.fromisoformat(session_ts).date()
        if "T" in session_ts
        else date.fromisoformat(session_ts)
    )


def _shift_month(d: date, n: int) -> tuple[int, int]:
    idx = d.year * 12 + (d.month - 1) + n
    return idx // 12, idx % 12 + 1


def _month_span(y: int, m: int) -> tuple[date, date]:
    return date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])


def span(expr: TimeExpr, session_ts: str | date) -> tuple[date, date, Precision] | None:
    """Calendar span of a time expression, or None if it is absent or vague."""
    d0 = session_date(session_ts)
    if expr.kind == "ABSOLUTE":
        v = (expr.value or "").strip()
        try:
            if m := re.fullmatch(r"(\d{4})", v):
                y = int(m.group(1))
                return date(y, 1, 1), date(y, 12, 31), Precision.YEAR
            if m := re.fullmatch(r"(\d{4})-(\d{1,2})", v):
                lo, hi = _month_span(int(m.group(1)), int(m.group(2)))
                return lo, hi, Precision.MONTH
            if m := re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", v):
                d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                return d, d, Precision.DAY
        except ValueError:
            return None
        return None
    if expr.kind == "RELATIVE":
        if expr.offset is None or expr.unit is None:
            return None
        n = expr.offset
        if expr.unit == "DAY":
            d = d0 + timedelta(days=n)
            return d, d, Precision.DAY
        if expr.unit == "WEEK":
            d = d0 + timedelta(days=7 * n)
            return d, d + timedelta(days=6), Precision.WEEK
        if expr.unit == "MONTH":
            lo, hi = _month_span(*_shift_month(d0, n))
            return lo, hi, Precision.MONTH
        if expr.unit == "YEAR":
            y = d0.year + n
            return date(y, 1, 1), date(y, 12, 31), Precision.YEAR
    return None


def is_vague(expr: TimeExpr) -> bool:
    return expr.kind != "NONE" and span(expr, date(2000, 1, 1)) is None


# ------------------------------------------------------------------------------ resolve
def resolve(
    expr: TimeExpr | None, session_ts: str | date, *, is_event: bool = False
) -> ResolvedTime:
    expr = expr or TimeExpr()
    d0 = session_date(session_ts).isoformat()
    sp = span(expr, session_ts)
    lo, hi, prec = (sp[0].isoformat(), sp[1].isoformat(), sp[2]) if sp else (None, None, None)
    vague = expr.kind != "NONE" and sp is None

    if is_event:
        if sp:
            return ResolvedTime(event_date=lo, event_date_precision=prec)
        if vague:
            return ResolvedTime(event_date=None, event_date_precision=Precision.UNKNOWN)
        return ResolvedTime(event_date=d0, event_date_precision=Precision.SESSION)

    if expr.tense == "PAST_ENDED":
        if sp:
            return ResolvedTime(lo, prec, hi, prec)
        # "used to" / "a while ago": ended at or before the session, start unknown
        return ResolvedTime(None, Precision.UNKNOWN, d0, Precision.BEFORE)

    # CURRENT and FUTURE_PLAN: open interval
    if sp:
        return ResolvedTime(lo, prec)
    if vague:
        return ResolvedTime(None, Precision.UNKNOWN)
    return ResolvedTime(d0, Precision.SESSION)


# ------------------------------------------------------------------------------ compare
def _bucket(d: date, p: Precision) -> tuple[int, ...]:
    if p == Precision.DAY:
        return (d.toordinal(),)
    if p == Precision.WEEK:
        iso = d.isocalendar()
        return (iso.year, iso.week)
    if p == Precision.MONTH:
        return (d.year, d.month)
    return (d.year,)


def compare(a: TimePoint, b: TimePoint) -> Cmp:
    """Is `a` LATER or EARLIER than `b`, compared at the coarser of the two precisions?

    SESSION, BEFORE, UNKNOWN or missing dates are INCOMPARABLE, as are values that fall in the
    same bucket at the coarser precision.
    """
    if a.date is None or b.date is None or a.precision not in _RANK or b.precision not in _RANK:
        return Cmp.INCOMPARABLE
    coarse = a.precision if _RANK[a.precision] >= _RANK[b.precision] else b.precision
    ka = _bucket(date.fromisoformat(a.date), coarse)
    kb = _bucket(date.fromisoformat(b.date), coarse)
    if ka > kb:
        return Cmp.LATER
    if ka < kb:
        return Cmp.EARLIER
    return Cmp.INCOMPARABLE


def describe(rt: ResolvedTime) -> str:
    """Human-readable interval for prompts and audit output."""

    def pt(d: str | None, p: Precision | None) -> str:
        if d is None:
            return "unknown" if p in (None, Precision.UNKNOWN) else f"? ({p})"
        return f"{d} ({p})"

    if rt.event_date is not None or rt.event_date_precision is not None:
        return f"event on {pt(rt.event_date, rt.event_date_precision)}"
    start = pt(rt.valid_from, rt.valid_from_precision)
    end = "now" if rt.valid_to is None else pt(rt.valid_to, rt.valid_to_precision)
    return f"from {start} to {end}"
