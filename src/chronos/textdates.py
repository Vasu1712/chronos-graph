"""Deterministic resolution of relative date words in conversation text.

A 7B model often fails to turn "yesterday" or "last Friday" into a date, even when the turn's
date is printed next to it (quick_v1: every LoCoMo temporal answer was off by the offset). This
module does it in code:

- `annotate(text, when)` appends the resolved date after each relative expression:
  "Met the team yesterday" on 2023-06-09 -> "Met the team yesterday [= 8 June 2023]".
  Used by every context formatter (ChronosGraph's packer and the baselines' `turn_line`), so all
  systems see the same annotated turns.
- `strip(text)` removes those annotations again (for matching a copied line to its turn).
- `time_expr(text)` returns the first relative expression as (kind, value, offset, unit) for the
  extractor's fallback when the model left a fact's time empty.

Only unambiguous expressions are handled; anything vague ("a while ago", "recently") is left alone.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

_NUM = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "couple of": 2, "a couple of": 2, "few": 3,
    "a few": 3,
}  # fmt: skip
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_NUM_RE = r"(\d{1,2}|a couple of|couple of|a few|few|an?|one|two|three|four|five|six|seven|eight|nine|ten)"

_PATTERN = re.compile(
    r"\b("
    r"the day before yesterday|day before yesterday"
    r"|yesterday|last night|today|tonight|this morning|this afternoon|this evening|tomorrow"
    r"|(?:this past|last|past) weekend|next weekend|this weekend"
    r"|(?:this past|last|past) (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"|(?:last|past|previous) week|next week|this week|a week ago"
    r"|last month|next month|this month|a month ago"
    r"|last year|next year|this year|a year ago"
    rf"|{_NUM_RE} (?:days?|weeks?|months?|years?) ago"
    r")\b(?! \[=)",
    re.IGNORECASE,
)
_ANNOT = re.compile(r" \[= [^\]]*\]")


def _d(when: str | date) -> date:
    if isinstance(when, date):
        return when
    s = str(when).strip()[:19]
    return datetime.fromisoformat(s).date() if "T" in s else date.fromisoformat(s[:10])


def _day(d: date) -> str:
    return f"{d.day} {calendar.month_name[d.month]} {d.year}"


def _month(y: int, m: int) -> str:
    return f"{calendar.month_name[m]} {y}"


def _shift_month(d: date, n: int) -> tuple[int, int]:
    idx = d.year * 12 + (d.month - 1) + n
    return idx // 12, idx % 12 + 1


def _num(word: str) -> int:
    w = word.lower()
    return int(w) if w.isdigit() else _NUM[w]


@dataclass(frozen=True)
class Resolved:
    label: str  # human-readable resolution for annotations
    kind: str  # ABSOLUTE | RELATIVE
    value: str | None = None  # ABSOLUTE date "YYYY-MM-DD"
    offset: int | None = None  # RELATIVE
    unit: str | None = None  # RELATIVE: DAY | WEEK | MONTH | YEAR


def resolve(expr: str, when: str | date) -> Resolved | None:
    """Resolve one matched expression against the turn date."""
    d0 = _d(when)
    e = re.sub(r"\s+", " ", expr.lower().strip())
    if e in ("the day before yesterday", "day before yesterday"):
        return Resolved(_day(d0 - timedelta(days=2)), "RELATIVE", offset=-2, unit="DAY")
    if e in ("yesterday", "last night"):
        return Resolved(_day(d0 - timedelta(days=1)), "RELATIVE", offset=-1, unit="DAY")
    if e in ("today", "tonight", "this morning", "this afternoon", "this evening"):
        return Resolved(_day(d0), "RELATIVE", offset=0, unit="DAY")
    if e == "tomorrow":
        return Resolved(_day(d0 + timedelta(days=1)), "RELATIVE", offset=1, unit="DAY")
    if e.endswith("weekend"):
        if e.startswith("next"):
            return Resolved(f"the weekend after {_day(d0)}", "RELATIVE", offset=1, unit="WEEK")
        if e.startswith("this weekend"):
            return Resolved(f"the weekend of {_day(d0)}", "RELATIVE", offset=0, unit="WEEK")
        # said on a weekend day, "last weekend" is the previous one
        back = d0.weekday() - 5 + 7 if d0.weekday() >= 5 else (d0.weekday() - 5) % 7
        sat = d0 - timedelta(days=back)
        return Resolved(
            f"the weekend before {_day(d0)} ({_day(sat)})", "ABSOLUTE", value=sat.isoformat()
        )
    for i, wd in enumerate(_WEEKDAYS):
        if e.endswith(wd):
            back = (d0.weekday() - i) % 7 or 7
            day = d0 - timedelta(days=back)
            return Resolved(
                f"the {wd.capitalize()} before {_day(d0)} ({_day(day)})",
                "ABSOLUTE",
                value=day.isoformat(),
            )
    if e in ("last week", "past week", "previous week", "a week ago"):
        return Resolved(f"the week before {_day(d0)}", "RELATIVE", offset=-1, unit="WEEK")
    if e == "next week":
        return Resolved(f"the week after {_day(d0)}", "RELATIVE", offset=1, unit="WEEK")
    if e == "this week":
        return Resolved(f"the week of {_day(d0)}", "RELATIVE", offset=0, unit="WEEK")
    for word, n in (("last month", -1), ("a month ago", -1), ("next month", 1), ("this month", 0)):
        if e == word:
            return Resolved(_month(*_shift_month(d0, n)), "RELATIVE", offset=n, unit="MONTH")
    for word, n in (("last year", -1), ("a year ago", -1), ("next year", 1), ("this year", 0)):
        if e == word:
            return Resolved(str(d0.year + n), "RELATIVE", offset=n, unit="YEAR")
    m = re.fullmatch(rf"{_NUM_RE} (day|week|month|year)s? ago", e)
    if m:
        n, unit = _num(m.group(1)), m.group(2)
        if unit == "day":
            return Resolved(_day(d0 - timedelta(days=n)), "RELATIVE", offset=-n, unit="DAY")
        if unit == "week":
            return Resolved(
                f"about {_day(d0 - timedelta(days=7 * n))}", "RELATIVE", offset=-n, unit="WEEK"
            )
        if unit == "month":
            return Resolved(_month(*_shift_month(d0, -n)), "RELATIVE", offset=-n, unit="MONTH")
        return Resolved(str(d0.year - n), "RELATIVE", offset=-n, unit="YEAR")
    return None


def annotate(text: str, when: str | date | None) -> str:
    """Append "[= <resolved date>]" after every relative date expression in `text`."""
    if not text or not when:
        return text
    try:
        _d(when)
    except ValueError:
        return text

    def sub(m: re.Match[str]) -> str:
        r = resolve(m.group(0), when)
        return f"{m.group(0)} [= {r.label}]" if r else m.group(0)

    return _PATTERN.sub(sub, text)


def strip(text: str) -> str:
    """Remove annotations added by `annotate`."""
    return _ANNOT.sub("", text)


def time_expr(text: str, when: str | date) -> Resolved | None:
    """First resolvable relative expression in `text`, or None."""
    for m in _PATTERN.finditer(text or ""):
        r = resolve(m.group(0), when)
        if r:
            return r
    return None
