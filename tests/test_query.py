"""Query analysis: 32 hand-written questions, LoCoMo-style and first-person demo questions."""

import pytest

from chronos.ingestion.entities import EntityResolver
from chronos.retrieval.query import Intent, analyze, parse_time
from chronos.storage.db import Database

QDATE = "2023-08-15T12:00:00"
C, A, H, T, OP = Intent.CURRENT, Intent.AS_OF, Intent.HISTORY, Intent.TEMPORAL_ORDER, Intent.OPEN


@pytest.fixture(scope="module")
def conn(tmp_path_factory):
    db = Database(tmp_path_factory.mktemp("q") / "q.sqlite")
    db.add_conversation("c1", "locomo")
    er = EntityResolver(db.conn, "c1")
    er.register_speakers(["Caroline", "Melanie"])
    alex = er.resolve("Alex", speaker="Melanie", relation="friend")
    db.conn.execute(
        "INSERT INTO entity_aliases(entity_id, alias, turn_uid) VALUES (?, 'Alexander', NULL)",
        (alex,),
    )
    er.resolve("Mumbai", speaker="Caroline", entity_type="PLACE")
    yield db.conn
    db.close()


# (question, intent, subject speaker or None, attribute or "-" (don't care), time (lo, hi, precision) or None)
CASES = [
    ("Where does Caroline live now?", C, "Caroline", "residence", None),
    ("Where does Caroline currently work?", C, "Caroline", "employer", None),
    ("Is Melanie still painting?", C, "Melanie", "-", None),
    ("What is Caroline's favorite book?", C, "Caroline", "favorite:book", None),
    ("Where do I live?", C, "Caroline", "residence", None),
    ("What does my sister do?", C, "Caroline", "job_title", None),
    ("What is Melanie's diet these days?", C, "Melanie", "diet", None),
    ("Where are you based?", C, "Melanie", "residence", None),
    ("What books has Alexander read?", C, None, "-", None),
    (
        "Where did Caroline live in May 2023?",
        A,
        "Caroline",
        "residence",
        ("2023-05-01", "2023-05-31", "MONTH"),
    ),
    ("What was Melanie's job in 2019?", A, "Melanie", "-", ("2019-01-01", "2019-12-31", "YEAR")),
    ("Where did Caroline live back when she started painting?", A, "Caroline", "residence", None),
    ("What was Caroline doing at that time?", A, "Caroline", "-", None),
    (
        "Where did Caroline live last year?",
        A,
        "Caroline",
        "residence",
        ("2022-01-01", "2022-12-31", "YEAR"),
    ),
    ("What did Melanie do 2 weeks ago?", A, "Melanie", "-", ("2023-08-01", "2023-08-07", "WEEK")),
    (
        "What did Caroline say on 3 May 2023?",
        A,
        "Caroline",
        "-",
        ("2023-05-03", "2023-05-03", "DAY"),
    ),
    (
        "What was Caroline working on in March?",
        A,
        "Caroline",
        "-",
        ("2023-03-01", "2023-03-31", "MONTH"),
    ),
    ("Where did Caroline live before Mumbai?", H, "Caroline", "residence", None),
    ("What did Caroline used to do for work?", H, "Caroline", "-", None),
    ("Where did Caroline previously work?", H, "Caroline", "employer", None),
    ("What was the first car Melanie owned?", H, "Melanie", "-", None),
    ("Has Caroline ever been to Paris?", H, "Caroline", "-", None),
    ("When did Caroline go to the LGBTQ support group?", T, "Caroline", "-", None),
    ("How long has Melanie been married?", T, "Melanie", "relationship_status", None),
    (
        "Which came first, Caroline's adoption meeting or the pride parade?",
        T,
        "Caroline",
        "-",
        None,
    ),
    (
        "How many days passed between Melanie's camping trip and the concert?",
        T,
        "Melanie",
        "-",
        None,
    ),
    ("In what year did Caroline move from Sweden?", T, "Caroline", "residence", None),
    ("What did Melanie paint recently?", OP, "Melanie", "-", None),
    (
        "Would Caroline be considered an ally to the transgender community?",
        OP,
        "Caroline",
        "-",
        None,
    ),
    ("Why did Melanie choose the pottery class?", OP, "Melanie", "-", None),
    ("What did Alex recommend to Melanie?", OP, "Melanie", "-", None),
    ("May I ask what Caroline's hometown was?", OP, "Caroline", "hometown", None),
]


@pytest.mark.parametrize("q, intent, speaker, attr, time", CASES, ids=[c[0] for c in CASES])
def test_query_analysis(conn, q, intent, speaker, attr, time):
    a = analyze(q, conn=conn, conversation_id="c1", question_date=QDATE)
    assert a.intent == intent, a.cues
    assert a.subject_speaker_name == speaker
    if attr != "-":
        assert a.attribute_key == attr
    if time is None:
        assert a.time is None, a.time
    else:
        assert (a.time.lo, a.time.hi, a.time.precision) == time


def test_entity_linking_exact_alias_and_place(conn):
    a = analyze(
        "Did Alexander ever visit Mumbai with Melanie?",
        conn=conn,
        conversation_id="c1",
        question_date=QDATE,
    )
    names = [e.name for e in a.entities]
    assert names == [
        "Alex",
        "Mumbai",
        "Melanie",
    ]  # ordered by position; alias resolves to canonical
    assert a.entities[0].matched == "Alexander"
    assert a.subject_speaker_name == "Melanie"


def test_possessive_and_case_insensitive(conn):
    a = analyze(
        "what is CAROLINE'S favourite movie", conn=conn, conversation_id="c1", question_date=QDATE
    )
    assert a.subject_speaker_name == "Caroline" and a.attribute_key == "favorite:movie"


def test_explicit_as_of_overrides(conn):
    a = analyze(
        "Where does Caroline live?",
        conn=conn,
        conversation_id="c1",
        question_date=QDATE,
        as_of="2023-01-10",
    )
    assert a.intent == A and (a.time.lo, a.time.hi) == ("2023-01-10", "2023-01-10")


def test_topic_terms_exclude_stopwords_and_names(conn):
    a = analyze(
        "What did Melanie paint at the pottery class?",
        conn=conn,
        conversation_id="c1",
        question_date=QDATE,
    )
    assert a.topic_terms == ["paint", "pottery", "class"]


def test_month_without_year_uses_most_recent_occurrence():
    assert parse_time("in December", "2023-08-15").lo == "2022-12-01"
    assert parse_time("in June", "2023-08-15").lo == "2023-06-01"
    assert parse_time("May I help?", "2023-08-15") is None
