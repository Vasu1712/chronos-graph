"""Generate data/synthetic_probes/*.json: 30 small conversations, 5 per case.

Cases: late_mention, in_session_correction, update_across_sessions, out_of_order_dates,
neg_polarity, temporary_vs_move.

Each probe holds:
  sessions   the conversation
  scripted   hand-written outputs of an *ideal* extractor/classifier (offline resolver tests)
  expected   final memories (status, interval), the set of current values per attribute,
             and values that must NOT be current
The expectations were written from SPEC.md before any live run.

Run: uv run python scripts/make_probes.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "synthetic_probes"


def F(s, t, subject, text, quote, key=None, tense="CURRENT", kind="NONE", value=None, offset=None,
      unit=None, pol="POS", event=False, corr=False, ents=()):  # fmt: skip
    return {
        "session": s, "turn_index": t, "subject": subject, "fact_text": text, "attribute_key": key,
        "polarity": pol, "mode": "EXPLICIT", "is_event": event,
        "entities_mentioned": [{"name": n, "entity_type": ty, "relation": r} for n, ty, r in ents],
        "time": {"kind": kind, "value": value, "offset": offset, "unit": unit, "tense": tense},
        "is_correction": corr, "quote": quote,
    }  # fmt: skip


def M(subject, contains, **kw):
    """Expected memory. kw: attribute_key, status (str or list), polarity, valid_from,
    valid_from_precision, valid_to ("*" = any non-null), valid_to_precision."""
    return {"subject": subject, "contains": contains, **kw}


def CUR(subject, key, contains):
    return {"subject": subject, "attribute_key": key, "contains": contains}


def ABSENT(subject, contains, key=None, polarity=None):
    out = {"subject": subject, "contains": contains}
    if key:
        out["attribute_key"] = key
    if polarity:
        out["polarity"] = polarity
    return out


def probe(pid, case, desc, speakers, sessions, facts, relations, memories, current=(), absent=()):
    return {
        "conversation_id": f"probe-{pid}",
        "probe_id": pid,
        "case": case,
        "description": desc,
        "speakers": list(speakers),
        "sessions": [
            {"ts": ts, "turns": [{"speaker": s, "text": t} for s, t in turns]}
            for ts, turns in sessions
        ],
        "scripted": {
            "extraction": facts,
            "relations": {k: {"relation": r, "target": tg} for k, (r, tg) in relations.items()},
        },
        "expected": {"memories": memories, "current": list(current), "absent": list(absent)},
    }


P = []
PLACE, ORG, PERSON = "PLACE", "ORG", "PERSON"

# ------------------------------------------------------------------ 1. late mention of an old fact
P.append(
    probe(
        "late_mention_01",
        "late_mention",
        "Delhi 2019 mentioned late must not replace Bengaluru.",
        ["Priya", "Rahul"],
        [
            (
                "2023-03-05T10:00:00",
                [
                    ("Rahul", "How is the new job treating you?"),
                    ("Priya", "Work has been hectic since I settled in Bengaluru."),
                ],
            ),
            (
                "2023-06-10T18:30:00",
                [
                    ("Rahul", "Did you ever live up north?"),
                    (
                        "Priya",
                        "Yes, back in 2019 I lived in Delhi for a year, the winters were brutal.",
                    ),
                ],
            ),
        ],
        [
            F(
                1,
                2,
                "Priya",
                "Priya lives in Bengaluru.",
                "since I settled in Bengaluru",
                key="residence",
                ents=[("Bengaluru", PLACE, None)],
            ),
            F(
                2,
                2,
                "Priya",
                "Priya lived in Delhi in 2019.",
                "back in 2019 I lived in Delhi for a year",
                key="residence",
                tense="PAST_ENDED",
                kind="ABSOLUTE",
                value="2019",
                ents=[("Delhi", PLACE, None)],
            ),
        ],
        {"Priya lived in Delhi in 2019.": ("HISTORICAL", "Bengaluru")},
        [
            M("Priya", "Bengaluru", attribute_key="residence", status="ACTIVE", valid_to=None),
            M(
                "Priya",
                "Delhi",
                attribute_key="residence",
                valid_from="2019-01-01",
                valid_from_precision="YEAR",
                valid_to="2019-12-31",
                valid_to_precision="YEAR",
            ),
        ],
        [CUR("Priya", "residence", ["Bengaluru"])],
    )
)
P.append(
    probe(
        "late_mention_02",
        "late_mention",
        "A 2018 internship must not replace the current employer.",
        ["Marco", "Jess"],
        [
            (
                "2023-02-14T09:00:00",
                [
                    ("Jess", "Still enjoying work?"),
                    ("Marco", "I've been at Novatek for a while now and I really like the team."),
                ],
            ),
            (
                "2023-09-01T20:00:00",
                [
                    ("Jess", "How did you get into logistics software?"),
                    (
                        "Marco",
                        "Funny thing, in 2018 I worked at Brightline as an intern before any of this.",
                    ),
                ],
            ),
        ],
        [
            F(
                1,
                2,
                "Marco",
                "Marco works at Novatek.",
                "I've been at Novatek for a while now",
                key="employer",
                ents=[("Novatek", ORG, None)],
            ),
            F(
                2,
                2,
                "Marco",
                "Marco worked at Brightline as an intern in 2018.",
                "in 2018 I worked at Brightline as an intern",
                key="employer",
                tense="PAST_ENDED",
                kind="ABSOLUTE",
                value="2018",
                ents=[("Brightline", ORG, None)],
            ),
        ],
        {"Marco worked at Brightline as an intern in 2018.": ("HISTORICAL", "Novatek")},
        [
            M("Marco", "Novatek", attribute_key="employer", status="ACTIVE", valid_to=None),
            M(
                "Marco",
                "Brightline",
                valid_from="2018-01-01",
                valid_to="2018-12-31",
                valid_to_precision="YEAR",
            ),
        ],
        [CUR("Marco", "employer", ["Novatek"])],
    )
)
P.append(
    probe(
        "late_mention_03",
        "late_mention",
        "A college from 2016 must not replace the current university.",
        ["Aisha", "Ben"],
        [
            (
                "2022-10-03T16:00:00",
                [
                    ("Aisha", "Classes at Westfield University are intense this semester."),
                    ("Ben", "Hang in there!"),
                ],
            ),
            (
                "2023-01-20T12:00:00",
                [
                    ("Ben", "Was Westfield always the plan?"),
                    (
                        "Aisha",
                        "Not at all, I studied at Lakeside College back in 2016 before transferring.",
                    ),
                ],
            ),
        ],
        [
            F(
                1,
                1,
                "Aisha",
                "Aisha attends Westfield University.",
                "Classes at Westfield University are intense this semester",
                key="school",
                ents=[("Westfield University", ORG, None)],
            ),
            F(
                2,
                2,
                "Aisha",
                "Aisha studied at Lakeside College in 2016.",
                "I studied at Lakeside College back in 2016",
                key="school",
                tense="PAST_ENDED",
                kind="ABSOLUTE",
                value="2016",
                ents=[("Lakeside College", ORG, None)],
            ),
        ],
        {"Aisha studied at Lakeside College in 2016.": ("HISTORICAL", "Westfield")},
        [
            M("Aisha", "Westfield", attribute_key="school", status="ACTIVE", valid_to=None),
            M("Aisha", "Lakeside", valid_from="2016-01-01", valid_to="2016-12-31"),
        ],
        [CUR("Aisha", "school", ["Westfield"])],
    )
)
P.append(
    probe(
        "late_mention_04",
        "late_mention",
        "A 2015 barista job must not replace the current nursing job.",
        ["Kenji", "Mara"],
        [
            (
                "2023-04-11T07:45:00",
                [
                    ("Mara", "Any plans this weekend?"),
                    ("Kenji", "As a nurse I barely get weekends off."),
                ],
            ),
            (
                "2023-07-30T21:10:00",
                [
                    ("Mara", "How do you stand all day?"),
                    (
                        "Kenji",
                        "In 2015 I was a barista, which is how I learned to stay on my feet all day.",
                    ),
                ],
            ),
        ],
        [
            F(
                1,
                2,
                "Kenji",
                "Kenji works as a nurse.",
                "As a nurse I barely get weekends off",
                key="job_title",
            ),
            F(
                2,
                2,
                "Kenji",
                "Kenji was a barista in 2015.",
                "In 2015 I was a barista",
                key="job_title",
                tense="PAST_ENDED",
                kind="ABSOLUTE",
                value="2015",
            ),
        ],
        {"Kenji was a barista in 2015.": ("HISTORICAL", "nurse")},
        [
            M("Kenji", "nurse", attribute_key="job_title", status="ACTIVE", valid_to=None),
            M("Kenji", "barista", valid_from="2015-01-01", valid_to="2015-12-31"),
        ],
        [CUR("Kenji", "job_title", ["nurse"])],
    )
)
P.append(
    probe(
        "late_mention_05",
        "late_mention",
        "Living in Montreal in March 2020 must not replace Toronto.",
        ["Sofia", "Liam"],
        [
            (
                "2023-05-02T11:00:00",
                [
                    ("Sofia", "I live in Toronto now, close to the lake."),
                    ("Liam", "Lucky you, the lake is gorgeous."),
                ],
            ),
            (
                "2023-08-19T15:20:00",
                [
                    ("Liam", "Where were you when the lockdowns started?"),
                    (
                        "Sofia",
                        "In March 2020 I was living in Montreal, right when everything shut down.",
                    ),
                ],
            ),
        ],
        [
            F(
                1,
                1,
                "Sofia",
                "Sofia lives in Toronto.",
                "I live in Toronto now",
                key="residence",
                ents=[("Toronto", PLACE, None)],
            ),
            F(
                2,
                2,
                "Sofia",
                "Sofia lived in Montreal in March 2020.",
                "In March 2020 I was living in Montreal",
                key="residence",
                tense="PAST_ENDED",
                kind="ABSOLUTE",
                value="2020-03",
                ents=[("Montreal", PLACE, None)],
            ),
        ],
        {"Sofia lived in Montreal in March 2020.": ("HISTORICAL", "Toronto")},
        [
            M("Sofia", "Toronto", attribute_key="residence", status="ACTIVE", valid_to=None),
            M(
                "Sofia",
                "Montreal",
                valid_from="2020-03-01",
                valid_from_precision="MONTH",
                valid_to="2020-03-31",
            ),
        ],
        [CUR("Sofia", "residence", ["Toronto"])],
    )
)

# ------------------------------------------------------------------ 2. in-session correction
P.append(
    probe(
        "correction_01",
        "in_session_correction",
        "Pune, sorry, I meant Mumbai.",
        ["Neha", "Sam"],
        [
            (
                "2023-04-04T19:00:00",
                [
                    ("Sam", "Where are you based these days?"),
                    ("Neha", "I live in Pune now."),
                    ("Sam", "Pune, nice!"),
                    ("Neha", "Pune, sorry, I meant Mumbai. I always mix them up when I'm tired."),
                ],
            )
        ],
        [
            F(
                1,
                2,
                "Neha",
                "Neha lives in Pune.",
                "I live in Pune now",
                key="residence",
                ents=[("Pune", PLACE, None)],
            ),
            F(
                1,
                4,
                "Neha",
                "Neha lives in Mumbai.",
                "sorry, I meant Mumbai",
                key="residence",
                corr=True,
                ents=[("Mumbai", PLACE, None)],
            ),
        ],
        {"Neha lives in Mumbai.": ("CORRECTS", "Pune")},
        [
            M("Neha", "Pune", attribute_key="residence", status="RETRACTED"),
            M("Neha", "Mumbai", attribute_key="residence", status="ACTIVE", valid_to=None),
        ],
        [CUR("Neha", "residence", ["Mumbai"])],
    )
)
P.append(
    probe(
        "correction_02",
        "in_session_correction",
        "Employer corrected in the same session.",
        ["Daniel", "Chloe"],
        [
            (
                "2023-03-21T08:30:00",
                [
                    ("Daniel", "I just started working at Pinecrest Bank."),
                    ("Chloe", "Congrats!"),
                    (
                        "Daniel",
                        "Wait, not Pinecrest, I meant Harbor Bank. Pinecrest was the other offer.",
                    ),
                ],
            )
        ],
        [
            F(
                1,
                1,
                "Daniel",
                "Daniel works at Pinecrest Bank.",
                "I just started working at Pinecrest Bank",
                key="employer",
                ents=[("Pinecrest Bank", ORG, None)],
            ),
            F(
                1,
                3,
                "Daniel",
                "Daniel works at Harbor Bank.",
                "not Pinecrest, I meant Harbor Bank",
                key="employer",
                corr=True,
                ents=[("Harbor Bank", ORG, None)],
            ),
        ],
        {"Daniel works at Harbor Bank.": ("CORRECTS", "Pinecrest")},
        [
            M("Daniel", "Pinecrest", attribute_key="employer", status="RETRACTED"),
            M("Daniel", "Harbor", attribute_key="employer", status="ACTIVE", valid_to=None),
        ],
        [CUR("Daniel", "employer", ["Harbor"])],
    )
)
P.append(
    probe(
        "correction_03",
        "in_session_correction",
        "Vegan corrected to vegetarian.",
        ["Ravi", "Ella"],
        [
            (
                "2023-05-30T13:00:00",
                [
                    ("Ravi", "I've been vegan for a year now."),
                    ("Ella", "Impressive."),
                    ("Ravi", "Actually no, vegetarian, I still eat cheese."),
                ],
            )
        ],
        [
            F(1, 1, "Ravi", "Ravi is vegan.", "I've been vegan for a year now", key="diet"),
            F(
                1,
                3,
                "Ravi",
                "Ravi is vegetarian.",
                "Actually no, vegetarian, I still eat cheese",
                key="diet",
                corr=True,
            ),
        ],
        {"Ravi is vegetarian.": ("CORRECTS", "vegan")},
        [
            M("Ravi", "vegan", attribute_key="diet", status="RETRACTED"),
            M("Ravi", "vegetarian", attribute_key="diet", status="ACTIVE", valid_to=None),
        ],
        [CUR("Ravi", "diet", ["vegetarian"])],
    )
)
P.append(
    probe(
        "correction_04",
        "in_session_correction",
        "Favorite book corrected.",
        ["Grace", "Tom"],
        [
            (
                "2023-02-08T20:15:00",
                [
                    ("Grace", "My favorite book is Rebecca."),
                    ("Tom", "Classic."),
                    ("Grace", "Sorry, I meant Jane Eyre, Rebecca is my sister's favorite."),
                ],
            )
        ],
        [
            F(
                1,
                1,
                "Grace",
                "Grace's favorite book is Rebecca.",
                "My favorite book is Rebecca",
                key="favorite:book",
            ),
            F(
                1,
                3,
                "Grace",
                "Grace's favorite book is Jane Eyre.",
                "Sorry, I meant Jane Eyre",
                key="favorite:book",
                corr=True,
            ),
        ],
        {"Grace's favorite book is Jane Eyre.": ("CORRECTS", "Rebecca")},
        [
            M("Grace", "Rebecca", attribute_key="favorite:book", status="RETRACTED"),
            M("Grace", "Jane Eyre", attribute_key="favorite:book", status="ACTIVE", valid_to=None),
        ],
        [CUR("Grace", "favorite:book", ["Jane Eyre"])],
    )
)
P.append(
    probe(
        "correction_05",
        "in_session_correction",
        "Partner's name corrected.",
        ["Leo", "Nina"],
        [
            (
                "2023-06-02T17:40:00",
                [
                    ("Leo", "My girlfriend Hannah loves hiking."),
                    ("Nina", "Hannah sounds fun."),
                    ("Leo", "Oops, I meant Anna. Hannah is my cousin."),
                ],
            )
        ],
        [
            F(
                1,
                1,
                "Leo",
                "Leo's girlfriend is Hannah.",
                "My girlfriend Hannah",
                key="partner",
                ents=[("Hannah", PERSON, "girlfriend")],
            ),
            F(
                1,
                3,
                "Leo",
                "Leo's girlfriend is Anna.",
                "Oops, I meant Anna",
                key="partner",
                corr=True,
                ents=[("Anna", PERSON, "girlfriend")],
            ),
        ],
        {"Leo's girlfriend is Anna.": ("CORRECTS", "Hannah")},
        [
            M("Leo", "Hannah", attribute_key="partner", status="RETRACTED"),
            M("Leo", "Anna", attribute_key="partner", status="ACTIVE", valid_to=None),
        ],
        [CUR("Leo", "partner", ["Anna"])],
    )
)

# ------------------------------------------------------------------ 3. update across sessions
P.append(
    probe(
        "update_01",
        "update_across_sessions",
        "Austin -> Seattle last month.",
        ["Jordan", "Kim"],
        [
            (
                "2023-01-15T10:00:00",
                [
                    ("Jordan", "Austin has been great, I love living here."),
                    ("Kim", "The tacos alone!"),
                ],
            ),
            (
                "2023-05-17T10:00:00",
                [
                    ("Jordan", "Big news, I moved to Seattle last month!"),
                    ("Kim", "No way, congrats!"),
                ],
            ),
        ],
        [
            F(
                1,
                1,
                "Jordan",
                "Jordan lives in Austin.",
                "Austin has been great, I love living here",
                key="residence",
                ents=[("Austin", PLACE, None)],
            ),
            F(
                2,
                1,
                "Jordan",
                "Jordan moved to Seattle.",
                "I moved to Seattle last month",
                key="residence",
                kind="RELATIVE",
                offset=-1,
                unit="MONTH",
                ents=[("Seattle", PLACE, None)],
            ),
        ],
        {"Jordan moved to Seattle.": ("UPDATES", "Austin")},
        [
            M(
                "Jordan",
                "Austin",
                attribute_key="residence",
                status="SUPERSEDED",
                valid_to="2023-04-01",
                valid_to_precision="MONTH",
            ),
            M(
                "Jordan",
                "Seattle",
                attribute_key="residence",
                status="ACTIVE",
                valid_from="2023-04-01",
                valid_from_precision="MONTH",
                valid_to=None,
            ),
        ],
        [CUR("Jordan", "residence", ["Seattle"])],
    )
)
P.append(
    probe(
        "update_02",
        "update_across_sessions",
        "Orbit Labs -> Quanta in February.",
        ["Hana", "Raj"],
        [
            (
                "2022-11-02T09:15:00",
                [
                    ("Raj", "What do you do again?"),
                    ("Hana", "I work at Orbit Labs as a data analyst."),
                ],
            ),
            (
                "2023-03-09T19:00:00",
                [
                    ("Hana", "I left Orbit Labs and joined Quanta in February."),
                    ("Raj", "Exciting!"),
                ],
            ),
        ],
        [
            F(
                1,
                2,
                "Hana",
                "Hana works at Orbit Labs.",
                "I work at Orbit Labs",
                key="employer",
                ents=[("Orbit Labs", ORG, None)],
            ),
            F(1, 2, "Hana", "Hana works as a data analyst.", "as a data analyst", key="job_title"),
            F(
                2,
                1,
                "Hana",
                "Hana joined Quanta.",
                "joined Quanta in February",
                key="employer",
                kind="ABSOLUTE",
                value="2023-02",
                ents=[("Quanta", ORG, None)],
            ),
        ],
        {"Hana joined Quanta.": ("UPDATES", "Orbit")},
        [
            M("Hana", "Orbit", attribute_key="employer", status="SUPERSEDED", valid_to="*"),
            M(
                "Hana",
                "Quanta",
                attribute_key="employer",
                status="ACTIVE",
                valid_from="2023-02-01",
                valid_from_precision="MONTH",
                valid_to=None,
            ),
        ],
        [CUR("Hana", "employer", ["Quanta"])],
    )
)
P.append(
    probe(
        "update_03",
        "update_across_sessions",
        "Single -> engaged last week.",
        ["Maya", "Eli"],
        [
            (
                "2022-08-20T14:00:00",
                [("Eli", "Seeing anyone?"), ("Maya", "I'm single and honestly enjoying it.")],
            ),
            (
                "2023-02-11T21:00:00",
                [("Maya", "I got engaged to Chris last weekend!"), ("Eli", "Congratulations!!")],
            ),
        ],
        [
            F(
                1,
                2,
                "Maya",
                "Maya is single.",
                "I'm single and honestly enjoying it",
                key="relationship_status",
            ),
            F(
                2,
                1,
                "Maya",
                "Maya is engaged.",
                "I got engaged to Chris last weekend",
                key="relationship_status",
                kind="RELATIVE",
                offset=-1,
                unit="WEEK",
            ),
            F(
                2,
                1,
                "Maya",
                "Maya is engaged to Chris.",
                "engaged to Chris",
                key="partner",
                kind="RELATIVE",
                offset=-1,
                unit="WEEK",
                ents=[("Chris", PERSON, "fiance")],
            ),
        ],
        {"Maya is engaged.": ("UPDATES", "single")},
        [
            M(
                "Maya",
                "single",
                attribute_key="relationship_status",
                status="SUPERSEDED",
                valid_to="2023-02-04",
                valid_to_precision="WEEK",
            ),
            M(
                "Maya",
                "engaged",
                attribute_key="relationship_status",
                status="ACTIVE",
                valid_from="2023-02-04",
                valid_to=None,
            ),
        ],
        [CUR("Maya", "relationship_status", ["engaged"])],
    )
)
P.append(
    probe(
        "update_04",
        "update_across_sessions",
        "Evernote -> Obsidian two weeks ago.",
        ["Felix", "Uma"],
        [
            (
                "2023-03-01T08:00:00",
                [("Felix", "I keep all my notes in Evernote."), ("Uma", "Same, it's handy.")],
            ),
            (
                "2023-06-20T08:00:00",
                [
                    ("Felix", "I switched from Evernote to Obsidian for notes two weeks ago."),
                    ("Uma", "How is it?"),
                ],
            ),
        ],
        [
            F(
                1,
                1,
                "Felix",
                "Felix uses Evernote for notes.",
                "I keep all my notes in Evernote",
                key="primary_tool:note_taking",
            ),
            F(
                2,
                1,
                "Felix",
                "Felix uses Obsidian for notes.",
                "I switched from Evernote to Obsidian for notes two weeks ago",
                key="primary_tool:note_taking",
                kind="RELATIVE",
                offset=-2,
                unit="WEEK",
            ),
        ],
        {"Felix uses Obsidian for notes.": ("UPDATES", "Evernote")},
        [
            M(
                "Felix",
                "Evernote",
                status="SUPERSEDED",
                valid_to="2023-06-06",
                valid_to_precision="WEEK",
            ),
            M(
                "Felix",
                "Obsidian",
                status="ACTIVE",
                valid_from="2023-06-06",
                valid_from_precision="WEEK",
                valid_to=None,
            ),
        ],
        [CUR("Felix", "primary_tool:note_taking", ["Obsidian"])],
    )
)
P.append(
    probe(
        "update_05",
        "update_across_sessions",
        "Leeds -> Bristol (October) -> Cardiff (May), three sessions.",
        ["Zoe", "Max"],
        [
            (
                "2022-05-10T12:00:00",
                [("Zoe", "I live in Leeds, it rains a lot."), ("Max", "Classic Yorkshire.")],
            ),
            (
                "2022-12-01T12:00:00",
                [("Zoe", "We moved to Bristol in October."), ("Max", "Nice harbour there.")],
            ),
            (
                "2023-07-15T12:00:00",
                [
                    ("Max", "Still in Bristol?"),
                    ("Zoe", "No, since May we've been living in Cardiff."),
                ],
            ),
        ],
        [
            F(
                1,
                1,
                "Zoe",
                "Zoe lives in Leeds.",
                "I live in Leeds",
                key="residence",
                ents=[("Leeds", PLACE, None)],
            ),
            F(
                2,
                1,
                "Zoe",
                "Zoe moved to Bristol.",
                "We moved to Bristol in October",
                key="residence",
                kind="ABSOLUTE",
                value="2022-10",
                ents=[("Bristol", PLACE, None)],
            ),
            F(
                3,
                2,
                "Zoe",
                "Zoe lives in Cardiff.",
                "since May we've been living in Cardiff",
                key="residence",
                kind="ABSOLUTE",
                value="2023-05",
                ents=[("Cardiff", PLACE, None)],
            ),
        ],
        {
            "Zoe moved to Bristol.": ("UPDATES", "Leeds"),
            "Zoe lives in Cardiff.": ("UPDATES", "Bristol"),
        },
        [
            M("Zoe", "Leeds", status="SUPERSEDED", valid_to="2022-10-01"),
            M(
                "Zoe",
                "Bristol",
                status="SUPERSEDED",
                valid_from="2022-10-01",
                valid_to="2023-05-01",
            ),
            M("Zoe", "Cardiff", status="ACTIVE", valid_from="2023-05-01", valid_to=None),
        ],
        [CUR("Zoe", "residence", ["Cardiff"])],
    )
)

# ------------------------------------------------------------------ 4. out-of-order dates
P.append(
    probe(
        "out_of_order_01",
        "out_of_order_dates",
        "Denver (March 2021) told after Austin (June 2022).",
        ["Ines", "Paul"],
        [
            (
                "2023-07-10T10:00:00",
                [
                    ("Ines", "I moved to Austin in June 2022 and still love it."),
                    ("Paul", "Austin suits you."),
                ],
            ),
            (
                "2023-09-02T10:00:00",
                [
                    ("Paul", "Was Austin your first move?"),
                    ("Ines", "No, I moved to Denver in March 2021, that was my first big move."),
                ],
            ),
        ],
        [
            F(
                1,
                1,
                "Ines",
                "Ines moved to Austin.",
                "I moved to Austin in June 2022",
                key="residence",
                kind="ABSOLUTE",
                value="2022-06",
                ents=[("Austin", PLACE, None)],
            ),
            F(
                2,
                2,
                "Ines",
                "Ines moved to Denver.",
                "I moved to Denver in March 2021",
                key="residence",
                kind="ABSOLUTE",
                value="2021-03",
                ents=[("Denver", PLACE, None)],
            ),
        ],
        {"Ines moved to Denver.": ("UPDATES", "Austin")},
        [
            M(
                "Ines",
                "Austin",
                attribute_key="residence",
                status="ACTIVE",
                valid_from="2022-06-01",
                valid_to=None,
            ),
            M(
                "Ines",
                "Denver",
                valid_from="2021-03-01",
                valid_from_precision="MONTH",
                valid_to="*",
            ),
        ],
        [CUR("Ines", "residence", ["Austin"])],
    )
)
P.append(
    probe(
        "out_of_order_02",
        "out_of_order_dates",
        "A 2020 job told after the 2023 one.",
        ["Owen", "Rita"],
        [
            (
                "2023-04-01T09:00:00",
                [
                    ("Owen", "I joined Fairway Logistics in January 2023."),
                    ("Rita", "How's the commute?"),
                ],
            ),
            (
                "2023-06-15T09:00:00",
                [
                    ("Rita", "What was your first job?"),
                    ("Owen", "I started at Summit Foods in 2020, my first real job."),
                ],
            ),
        ],
        [
            F(
                1,
                1,
                "Owen",
                "Owen works at Fairway Logistics.",
                "I joined Fairway Logistics in January 2023",
                key="employer",
                kind="ABSOLUTE",
                value="2023-01",
                ents=[("Fairway Logistics", ORG, None)],
            ),
            F(
                2,
                2,
                "Owen",
                "Owen started working at Summit Foods.",
                "I started at Summit Foods in 2020",
                key="employer",
                kind="ABSOLUTE",
                value="2020",
                ents=[("Summit Foods", ORG, None)],
            ),
        ],
        {"Owen started working at Summit Foods.": ("UPDATES", "Fairway")},
        [
            M(
                "Owen",
                "Fairway",
                attribute_key="employer",
                status="ACTIVE",
                valid_from="2023-01-01",
                valid_to=None,
            ),
            M("Owen", "Summit", valid_from="2020-01-01", valid_from_precision="YEAR", valid_to="*"),
        ],
        [CUR("Owen", "employer", ["Fairway"])],
    )
)
P.append(
    probe(
        "out_of_order_03",
        "out_of_order_dates",
        "Software engineer (2019) told after team lead (April).",
        ["Lucia", "Ken"],
        [
            (
                "2023-05-05T18:00:00",
                [("Lucia", "I became a team lead in April."), ("Ken", "Well deserved!")],
            ),
            (
                "2023-08-08T18:00:00",
                [
                    ("Ken", "How long have you been coding?"),
                    ("Lucia", "I became a software engineer in 2019 after my bootcamp."),
                ],
            ),
        ],
        [
            F(
                1,
                1,
                "Lucia",
                "Lucia works as a team lead.",
                "I became a team lead in April",
                key="job_title",
                kind="ABSOLUTE",
                value="2023-04",
            ),
            F(
                2,
                2,
                "Lucia",
                "Lucia works as a software engineer.",
                "I became a software engineer in 2019",
                key="job_title",
                kind="ABSOLUTE",
                value="2019",
            ),
        ],
        {"Lucia works as a software engineer.": ("UPDATES", "team lead")},
        [
            M(
                "Lucia",
                "team lead",
                attribute_key="job_title",
                status="ACTIVE",
                valid_from="2023-04-01",
                valid_to=None,
            ),
            M("Lucia", "software engineer", valid_from="2019-01-01", valid_to="*"),
        ],
        [CUR("Lucia", "job_title", ["team lead"])],
    )
)
P.append(
    probe(
        "out_of_order_04",
        "out_of_order_dates",
        "Day-precision moves told in reverse order.",
        ["Arun", "Bea"],
        [
            (
                "2023-10-20T09:30:00",
                [
                    ("Arun", "We moved into our place in Chennai on October 1st."),
                    ("Bea", "Housewarming soon?"),
                ],
            ),
            (
                "2023-11-05T09:30:00",
                [
                    ("Bea", "Where were you before Chennai?"),
                    ("Arun", "Before that, on May 2nd we moved to Hyderabad."),
                ],
            ),
        ],
        [
            F(
                1,
                1,
                "Arun",
                "Arun lives in Chennai.",
                "We moved into our place in Chennai on October 1st",
                key="residence",
                kind="ABSOLUTE",
                value="2023-10-01",
                ents=[("Chennai", PLACE, None)],
            ),
            F(
                2,
                2,
                "Arun",
                "Arun moved to Hyderabad.",
                "on May 2nd we moved to Hyderabad",
                key="residence",
                kind="ABSOLUTE",
                value="2023-05-02",
                ents=[("Hyderabad", PLACE, None)],
            ),
        ],
        {"Arun moved to Hyderabad.": ("UPDATES", "Chennai")},
        [
            M(
                "Arun",
                "Chennai",
                attribute_key="residence",
                status="ACTIVE",
                valid_from="2023-10-01",
                valid_from_precision="DAY",
                valid_to=None,
            ),
            M(
                "Arun",
                "Hyderabad",
                valid_from="2023-05-02",
                valid_from_precision="DAY",
                valid_to="2023-10-01",
            ),
        ],
        [CUR("Arun", "residence", ["Chennai"])],
    )
)
P.append(
    probe(
        "out_of_order_05",
        "out_of_order_dates",
        "A 2018 college told after the current master's.",
        ["Tara", "Vik"],
        [
            (
                "2023-09-12T15:00:00",
                [
                    ("Tara", "I started my master's at Kingsbridge University in September."),
                    ("Vik", "Good luck!"),
                ],
            ),
            (
                "2023-10-30T15:00:00",
                [
                    ("Vik", "Where did you do your bachelor's?"),
                    ("Tara", "I enrolled at Elmwood College in 2018 for my bachelor's."),
                ],
            ),
        ],
        [
            F(
                1,
                1,
                "Tara",
                "Tara studies at Kingsbridge University.",
                "I started my master's at Kingsbridge University in September",
                key="school",
                kind="ABSOLUTE",
                value="2023-09",
                ents=[("Kingsbridge University", ORG, None)],
            ),
            F(
                2,
                2,
                "Tara",
                "Tara studied at Elmwood College.",
                "I enrolled at Elmwood College in 2018",
                key="school",
                kind="ABSOLUTE",
                value="2018",
                ents=[("Elmwood College", ORG, None)],
            ),
        ],
        {"Tara studied at Elmwood College.": ("UPDATES", "Kingsbridge")},
        [
            M(
                "Tara",
                "Kingsbridge",
                attribute_key="school",
                status="ACTIVE",
                valid_from="2023-09-01",
                valid_to=None,
            ),
            M("Tara", "Elmwood", valid_from="2018-01-01", valid_to="*"),
        ],
        [CUR("Tara", "school", ["Kingsbridge"])],
    )
)

# ------------------------------------------------------------------ 5. NEG polarity
P.append(
    probe(
        "neg_01",
        "neg_polarity",
        "I don't play guitar anymore.",
        ["Arjun", "Mei"],
        [
            (
                "2023-02-02T19:00:00",
                [
                    ("Arjun", "I play guitar in a little band on weekends."),
                    ("Mei", "What kind of music?"),
                ],
            ),
            (
                "2023-06-18T19:00:00",
                [
                    ("Mei", "Still gigging?"),
                    ("Arjun", "I don't play guitar anymore, my wrist can't take it."),
                ],
            ),
        ],
        [
            F(
                1,
                1,
                "Arjun",
                "Arjun plays guitar in a band on weekends.",
                "I play guitar in a little band on weekends",
            ),
            F(
                2,
                2,
                "Arjun",
                "Arjun no longer plays guitar.",
                "I don't play guitar anymore",
                pol="NEG",
            ),
        ],
        {"Arjun no longer plays guitar.": ("UPDATES", "guitar")},
        [
            M(
                "Arjun",
                "guitar",
                polarity="POS",
                status="SUPERSEDED",
                valid_to="2023-06-18",
                valid_to_precision="SESSION",
            ),
            M("Arjun", "guitar", polarity="NEG", status="ACTIVE", valid_to=None),
        ],
        absent=[ABSENT("Arjun", "guitar", polarity="POS")],
    )
)
P.append(
    probe(
        "neg_02",
        "neg_polarity",
        "I don't smoke anymore.",
        ["Pedro", "Ann"],
        [
            (
                "2022-12-01T22:00:00",
                [
                    ("Pedro", "I smoke maybe half a pack a day, I know, I know."),
                    ("Ann", "Pedro..."),
                ],
            ),
            (
                "2023-04-14T22:00:00",
                [
                    ("Pedro", "I don't smoke anymore, three months clean!"),
                    ("Ann", "So proud of you!"),
                ],
            ),
        ],
        [
            F(
                1,
                1,
                "Pedro",
                "Pedro smokes about half a pack a day.",
                "I smoke maybe half a pack a day",
            ),
            F(2, 1, "Pedro", "Pedro no longer smokes.", "I don't smoke anymore", pol="NEG"),
        ],
        {"Pedro no longer smokes.": ("UPDATES", "smokes")},
        [
            M("Pedro", "smoke", polarity="POS", status="SUPERSEDED", valid_to="*"),
            M("Pedro", "smoke", polarity="NEG", status="ACTIVE", valid_to=None),
        ],
        absent=[ABSENT("Pedro", "smoke", polarity="POS")],
    )
)
P.append(
    probe(
        "neg_03",
        "neg_polarity",
        "I don't work at Delta Print anymore.",
        ["Yusuf", "Clara"],
        [
            (
                "2023-01-09T07:00:00",
                [
                    ("Yusuf", "I work at Delta Print, mostly night shifts."),
                    ("Clara", "That sounds exhausting."),
                ],
            ),
            (
                "2023-05-22T07:00:00",
                [
                    ("Yusuf", "I don't work at Delta Print anymore, I'm taking some time off."),
                    ("Clara", "Good for you."),
                ],
            ),
        ],
        [
            F(
                1,
                1,
                "Yusuf",
                "Yusuf works at Delta Print.",
                "I work at Delta Print",
                key="employer",
                ents=[("Delta Print", ORG, None)],
            ),
            F(
                2,
                1,
                "Yusuf",
                "Yusuf no longer works at Delta Print.",
                "I don't work at Delta Print anymore",
                key="employer",
                pol="NEG",
                ents=[("Delta Print", ORG, None)],
            ),
        ],
        {"Yusuf no longer works at Delta Print.": ("UPDATES", "Delta")},
        [
            M("Yusuf", "Delta", polarity="POS", status="SUPERSEDED", valid_to="*"),
            M("Yusuf", "Delta", polarity="NEG", status="ACTIVE"),
        ],
        absent=[ABSENT("Yusuf", "Delta", polarity="POS")],
    )
)
P.append(
    probe(
        "neg_04",
        "neg_polarity",
        "I don't have a cat anymore.",
        ["Lily", "Dev"],
        [
            (
                "2023-03-03T12:00:00",
                [
                    ("Lily", "My cat Mochi sleeps on my keyboard all day."),
                    ("Dev", "Cats love warm keyboards."),
                ],
            ),
            (
                "2023-09-09T12:00:00",
                [
                    ("Dev", "How's Mochi?"),
                    ("Lily", "I don't have a cat anymore, Mochi went to live with my parents."),
                ],
            ),
        ],
        [
            F(
                1,
                1,
                "Lily",
                "Lily has a cat named Mochi.",
                "My cat Mochi",
                ents=[("Mochi", "OTHER", None)],
            ),
            F(2, 2, "Lily", "Lily no longer has a cat.", "I don't have a cat anymore", pol="NEG"),
        ],
        {"Lily no longer has a cat.": ("UPDATES", "cat")},
        [
            M("Lily", "cat", polarity="POS", status="SUPERSEDED", valid_to="*"),
            M("Lily", "cat", polarity="NEG", status="ACTIVE"),
        ],
        absent=[ABSENT("Lily", "cat", polarity="POS")],
    )
)
P.append(
    probe(
        "neg_05",
        "neg_polarity",
        "I no longer drink coffee.",
        ["Sven", "Ivy"],
        [
            (
                "2023-01-20T06:30:00",
                [
                    ("Sven", "I drink three espressos every morning."),
                    ("Ivy", "That's a lot of caffeine."),
                ],
            ),
            (
                "2023-05-01T06:30:00",
                [
                    ("Sven", "I no longer drink coffee at all, doctor's orders."),
                    ("Ivy", "Tea instead?"),
                ],
            ),
        ],
        [
            F(
                1,
                1,
                "Sven",
                "Sven drinks three espressos every morning.",
                "I drink three espressos every morning",
            ),
            F(
                2,
                1,
                "Sven",
                "Sven no longer drinks coffee.",
                "I no longer drink coffee at all",
                pol="NEG",
            ),
        ],
        {"Sven no longer drinks coffee.": ("UPDATES", "espresso")},
        [
            M("Sven", "espresso", polarity="POS", status="SUPERSEDED", valid_to="*"),
            M("Sven", "coffee", polarity="NEG", status="ACTIVE"),
        ],
        absent=[ABSENT("Sven", "espresso", polarity="POS")],
    )
)

# ------------------------------------------------------------------ 6. temporary stay vs move
P.append(
    probe(
        "temporary_01",
        "temporary_vs_move",
        "In Pune this week for a wedding is not a move.",
        ["Kavya", "Nick"],
        [
            (
                "2023-04-02T10:00:00",
                [
                    ("Kavya", "I live in Bengaluru, near the office."),
                    ("Nick", "Short commute, nice."),
                ],
            ),
            (
                "2023-06-12T10:00:00",
                [
                    ("Kavya", "I'm in Pune this week for my cousin's wedding."),
                    ("Nick", "Have fun!"),
                ],
            ),
        ],
        [
            F(
                1,
                1,
                "Kavya",
                "Kavya lives in Bengaluru.",
                "I live in Bengaluru",
                key="residence",
                ents=[("Bengaluru", PLACE, None)],
            ),
            F(
                2,
                1,
                "Kavya",
                "Kavya is in Pune this week for a cousin's wedding.",
                "I'm in Pune this week for my cousin's wedding",
                kind="RELATIVE",
                offset=0,
                unit="WEEK",
                ents=[("Pune", PLACE, None)],
            ),
        ],
        {},
        [M("Kavya", "Bengaluru", attribute_key="residence", status="ACTIVE", valid_to=None)],
        [CUR("Kavya", "residence", ["Bengaluru"])],
        [ABSENT("Kavya", "Pune", key="residence")],
    )
)
P.append(
    probe(
        "temporary_02",
        "temporary_vs_move",
        "A work trip to Berlin is not a move.",
        ["Hugo", "Asha"],
        [
            (
                "2023-02-20T10:00:00",
                [("Hugo", "I live in Lyon with my partner."), ("Asha", "Lovely city.")],
            ),
            (
                "2023-03-15T10:00:00",
                [
                    ("Hugo", "I'm on a work trip in Berlin until Friday."),
                    ("Asha", "Try the currywurst."),
                ],
            ),
        ],
        [
            F(
                1,
                1,
                "Hugo",
                "Hugo lives in Lyon.",
                "I live in Lyon",
                key="residence",
                ents=[("Lyon", PLACE, None)],
            ),
            F(
                2,
                1,
                "Hugo",
                "Hugo is on a work trip in Berlin.",
                "I'm on a work trip in Berlin until Friday",
                ents=[("Berlin", PLACE, None)],
            ),
        ],
        {},
        [M("Hugo", "Lyon", attribute_key="residence", status="ACTIVE", valid_to=None)],
        [CUR("Hugo", "residence", ["Lyon"])],
        [ABSENT("Hugo", "Berlin", key="residence")],
    )
)
P.append(
    probe(
        "temporary_03",
        "temporary_vs_move",
        "Control: relocating to Valencia for good IS a move.",
        ["Rosa", "Theo"],
        [
            (
                "2023-01-05T10:00:00",
                [("Rosa", "I live in Madrid, right in the centre."), ("Theo", "Busy area!")],
            ),
            (
                "2023-07-07T10:00:00",
                [
                    (
                        "Rosa",
                        "We relocated to Valencia for good last month, the flat is finally ours.",
                    ),
                    ("Theo", "Beach life!"),
                ],
            ),
        ],
        [
            F(
                1,
                1,
                "Rosa",
                "Rosa lives in Madrid.",
                "I live in Madrid",
                key="residence",
                ents=[("Madrid", PLACE, None)],
            ),
            F(
                2,
                1,
                "Rosa",
                "Rosa relocated to Valencia.",
                "We relocated to Valencia for good last month",
                key="residence",
                kind="RELATIVE",
                offset=-1,
                unit="MONTH",
                ents=[("Valencia", PLACE, None)],
            ),
        ],
        {"Rosa relocated to Valencia.": ("UPDATES", "Madrid")},
        [
            M(
                "Rosa",
                "Madrid",
                attribute_key="residence",
                status="SUPERSEDED",
                valid_to="2023-06-01",
            ),
            M(
                "Rosa",
                "Valencia",
                attribute_key="residence",
                status="ACTIVE",
                valid_from="2023-06-01",
                valid_from_precision="MONTH",
                valid_to=None,
            ),
        ],
        [CUR("Rosa", "residence", ["Valencia"])],
    )
)
P.append(
    probe(
        "temporary_04",
        "temporary_vs_move",
        "Visiting parents in Cairo is not a move.",
        ["Ahmed", "Joy"],
        [
            (
                "2023-05-10T10:00:00",
                [("Ahmed", "I live in Dubai and work near the marina."), ("Joy", "Must be hot.")],
            ),
            (
                "2023-08-03T10:00:00",
                [
                    ("Ahmed", "I'm visiting my parents in Cairo for two weeks."),
                    ("Joy", "Say hi to them!"),
                ],
            ),
        ],
        [
            F(
                1,
                1,
                "Ahmed",
                "Ahmed lives in Dubai.",
                "I live in Dubai",
                key="residence",
                ents=[("Dubai", PLACE, None)],
            ),
            F(
                2,
                1,
                "Ahmed",
                "Ahmed is visiting his parents in Cairo for two weeks.",
                "I'm visiting my parents in Cairo for two weeks",
                ents=[("Cairo", PLACE, None)],
            ),
        ],
        {},
        [M("Ahmed", "Dubai", attribute_key="residence", status="ACTIVE", valid_to=None)],
        [CUR("Ahmed", "residence", ["Dubai"])],
        [ABSENT("Ahmed", "Cairo", key="residence")],
    )
)
P.append(
    probe(
        "temporary_05",
        "temporary_vs_move",
        "Control: settling in Denver with a lease IS a move.",
        ["Mira", "Sean"],
        [
            (
                "2023-03-12T10:00:00",
                [("Mira", "I'm based in Boston for now."), ("Sean", "Cold winters!")],
            ),
            (
                "2023-10-01T10:00:00",
                [
                    ("Mira", "I settled in Denver in September, signed a two-year lease."),
                    ("Sean", "Mountains!"),
                ],
            ),
        ],
        [
            F(
                1,
                1,
                "Mira",
                "Mira lives in Boston.",
                "I'm based in Boston for now",
                key="residence",
                ents=[("Boston", PLACE, None)],
            ),
            F(
                2,
                1,
                "Mira",
                "Mira lives in Denver.",
                "I settled in Denver in September",
                key="residence",
                kind="ABSOLUTE",
                value="2023-09",
                ents=[("Denver", PLACE, None)],
            ),
        ],
        {"Mira lives in Denver.": ("UPDATES", "Boston")},
        [
            M(
                "Mira",
                "Boston",
                attribute_key="residence",
                status="SUPERSEDED",
                valid_to="2023-09-01",
            ),
            M(
                "Mira",
                "Denver",
                attribute_key="residence",
                status="ACTIVE",
                valid_from="2023-09-01",
                valid_to=None,
            ),
        ],
        [CUR("Mira", "residence", ["Denver"])],
    )
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.json"):
        old.unlink()
    for p in P:
        (OUT / f"{p['probe_id']}.json").write_text(
            json.dumps(p, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    cases: dict[str, int] = {}
    for p in P:
        cases[p["case"]] = cases.get(p["case"], 0) + 1
    print(f"wrote {len(P)} probes to {OUT}: {cases}")


if __name__ == "__main__":
    main()
