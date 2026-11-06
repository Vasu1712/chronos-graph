"""One test per bi-temporal write rule (SPEC §5), with hand-built facts and a scripted classifier."""

import pytest
from fakes import FakeEmbedder

from chronos.ingestion.resolver import NewFact, RelationOut, StateResolver
from chronos.ingestion.temporal import TimeExpr, resolve
from chronos.storage.db import Database, make_turn_uid

CONV = "c1"
SUBJ = "c1:e:priya"
SESSION_TS = {1: "2023-01-10T10:00:00", 2: "2023-05-17T10:00:00", 3: "2023-08-01T10:00:00"}
EMB = FakeEmbedder()


class ScriptedClassifier:
    def __init__(self):
        self.queue: list[tuple[str, str | None]] = []
        self.calls = []

    def say(self, relation, target="C1"):
        self.queue.append((relation, target))

    def structured(self, messages, response_model):
        self.calls.append(messages)
        rel, target = self.queue.pop(0)
        return RelationOut(reason="scripted", relation=rel, target=target)


@pytest.fixture
def env(tmp_path):
    db = Database(tmp_path / "r.sqlite")
    db.add_conversation(CONV, "synthetic")
    db.add_entity(SUBJ, CONV, "Priya", entity_type="PERSON", is_speaker=True)
    for s, ts in SESSION_TS.items():
        db.add_session(f"{CONV}:s{s:02d}", CONV, s, ts)
        for t in range(1, 10):
            db.append_episode(
                {"turn_uid": make_turn_uid("syn", CONV, s, t), "conversation_id": CONV,
                 "session_id": f"{CONV}:s{s:02d}", "session_ordinal": s, "turn_index": t,
                 "speaker": "Priya", "speaker_entity_id": SUBJ, "session_ts": ts, "text": f"turn {s}.{t}"}
            )  # fmt: skip
    llm = ScriptedClassifier()
    yield db, StateResolver(db.conn, llm), llm
    db.close()


_n = iter(range(10_000))


def fact(text, s, t, key="residence", time=None, polarity="POS", corr=False, event=False):
    expr = time or TimeExpr()
    return NewFact(
        memory_id=f"m{next(_n)}", conversation_id=CONV, subject_entity_id=SUBJ, subject_name="Priya",
        attribute_key=key, fact_text=text, polarity=polarity, mode="EXPLICIT", is_event=event,
        time_expr=expr, session_ts=SESSION_TS[s], rt=resolve(expr, SESSION_TS[s], is_event=event),
        session_ordinal=s, turn_index=t, turn_uid=make_turn_uid("syn", CONV, s, t), quote=text,
        quote_start=0, quote_end=len(text), is_correction=corr, embedding=EMB.encode_passages([text])[0],
    )  # fmt: skip


def row(db, mid):
    return dict(db.conn.execute("SELECT * FROM memories WHERE memory_id = ?", (mid,)).fetchone())


def roles(db, mid):
    return sorted(
        r[0] for r in db.conn.execute("SELECT role FROM fact_sources WHERE memory_id = ?", (mid,))
    )


def test_new_without_candidates_needs_no_llm(env):
    db, res, llm = env
    f = fact("Priya lives in Bengaluru.", 1, 1)
    d = res.apply(f)
    assert d.relation == "NEW" and d.llm_relation is None and llm.calls == []
    r = row(db, f.memory_id)
    assert (r["status"], r["valid_from"], r["valid_from_precision"], r["valid_to"]) == (
        "ACTIVE",
        "2023-01-10",
        "SESSION",
        None,
    )
    assert roles(db, f.memory_id) == ["ASSERTS"]
    assert (
        db.conn.execute(
            "SELECT role FROM fact_entities WHERE memory_id = ?", (f.memory_id,)
        ).fetchone()[0]
        == "SUBJECT"
    )


def test_duplicate_reaffirms_and_counts_distinct_turns(env):
    db, res, llm = env
    a = fact("Priya lives in Bengaluru.", 1, 1)
    res.apply(a)
    llm.say("DUPLICATE")
    res.apply(fact("Priya lives in Bengaluru.", 2, 3))
    llm.say("DUPLICATE")
    res.apply(fact("Priya is based in Bengaluru.", 2, 3))  # same turn again: no extra count
    assert row(db, a.memory_id)["corroboration"] == 2
    assert roles(db, a.memory_id) == ["ASSERTS", "REAFFIRMS"]
    assert db.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1


def test_extends_keeps_both_active(env):
    db, res, llm = env
    a = fact("Priya lives in Bengaluru.", 1, 1)
    res.apply(a)
    llm.say("EXTENDS")
    b = fact("Priya lives in Indiranagar, Bengaluru.", 2, 1)
    res.apply(b)
    assert row(db, a.memory_id)["status"] == row(db, b.memory_id)["status"] == "ACTIVE"


def test_updates_later_supersedes_old(env):
    db, res, llm = env
    old = fact("Priya lives in Austin.", 1, 1, time=TimeExpr(kind="ABSOLUTE", value="2021"))
    res.apply(old)
    llm.say("UPDATES")
    new = fact(
        "Priya moved to Seattle.", 2, 4, time=TimeExpr(kind="RELATIVE", offset=-1, unit="MONTH")
    )
    res.apply(new)
    o, n = row(db, old.memory_id), row(db, new.memory_id)
    assert o["status"] == "SUPERSEDED"
    assert (o["valid_to"], o["valid_to_precision"]) == ("2023-04-01", "MONTH")
    assert (o["retired_session_ordinal"], o["retired_turn_index"]) == (2, 4)
    assert o["superseded_by_id"] == new.memory_id and n["supersedes_id"] == old.memory_id
    assert n["status"] == "ACTIVE" and n["valid_to"] is None


def test_updates_earlier_is_inserted_behind(env):
    db, res, llm = env
    cur = fact("Priya moved to Austin.", 1, 1, time=TimeExpr(kind="ABSOLUTE", value="2022-06"))
    res.apply(cur)
    llm.say("UPDATES")
    older = fact("Priya moved to Denver.", 2, 2, time=TimeExpr(kind="ABSOLUTE", value="2021-03"))
    res.apply(older)
    c, o = row(db, cur.memory_id), row(db, older.memory_id)
    assert (
        c["status"] == "ACTIVE" and c["valid_to"] is None and c["supersedes_id"] == older.memory_id
    )
    assert o["status"] == "SUPERSEDED" and o["superseded_by_id"] == cur.memory_id
    assert (o["valid_from"], o["valid_to"], o["valid_to_precision"]) == (
        "2021-03-01",
        "2022-06-01",
        "MONTH",
    )


def test_insert_behind_splices_into_existing_chain(env):
    db, res, llm = env
    x = fact("Priya moved to Boston.", 1, 1, time=TimeExpr(kind="ABSOLUTE", value="2019"))
    res.apply(x)
    llm.say("UPDATES")
    t = fact("Priya moved to Austin.", 2, 1, time=TimeExpr(kind="ABSOLUTE", value="2022"))
    res.apply(t)  # Boston -> Austin
    llm.say("UPDATES")
    mid = fact("Priya moved to Denver.", 3, 1, time=TimeExpr(kind="ABSOLUTE", value="2021"))
    res.apply(mid)  # earlier than Austin: Boston -> Denver -> Austin
    assert row(db, x.memory_id)["superseded_by_id"] == mid.memory_id
    m = row(db, mid.memory_id)
    assert (m["supersedes_id"], m["superseded_by_id"], m["status"]) == (
        x.memory_id,
        t.memory_id,
        "SUPERSEDED",
    )
    assert row(db, t.memory_id)["supersedes_id"] == mid.memory_id


def test_updates_incomparable_later_order_key_wins_in_session(env):
    db, res, llm = env
    a = fact("Priya lives in Pune.", 2, 1)
    res.apply(a)
    llm.say("UPDATES")
    b = fact("Priya lives in Mumbai.", 2, 5)  # both SESSION precision, same session
    res.apply(b)
    assert row(db, a.memory_id)["status"] == "SUPERSEDED"
    assert (row(db, a.memory_id)["valid_to"], row(db, a.memory_id)["valid_to_precision"]) == (
        "2023-05-17",
        "SESSION",
    )
    assert row(db, b.memory_id)["status"] == "ACTIVE"


def test_updates_incomparable_earlier_order_key_goes_behind(env):
    db, res, llm = env
    late = fact("Priya lives in Mumbai.", 3, 1)
    res.apply(late)  # session 3 ingested first
    llm.say("UPDATES")
    early = fact("Priya lives in Pune.", 2, 1)
    res.apply(early)
    assert row(db, late.memory_id)["status"] == "ACTIVE"
    assert row(db, early.memory_id)["status"] == "SUPERSEDED"


def test_corrects_retracts_old(env):
    db, res, llm = env
    a = fact("Priya lives in Pune.", 2, 2)
    res.apply(a)
    llm.say("CORRECTS")
    b = fact("Priya lives in Mumbai.", 2, 3, corr=True)
    res.apply(b)
    o, n = row(db, a.memory_id), row(db, b.memory_id)
    assert o["status"] == "RETRACTED" and o["valid_to"] is None  # never true, interval untouched
    assert (o["retired_session_ordinal"], o["retired_turn_index"]) == (2, 3)
    assert n["status"] == "ACTIVE" and n["supersedes_id"] == a.memory_id
    assert roles(db, b.memory_id) == ["CORRECTS"]
    assert roles(db, a.memory_id) == ["ASSERTS", "RETRACTS"]


def test_correction_cue_promotes_updates_to_corrects(env):
    db, res, llm = env
    a = fact("Priya lives in Pune.", 2, 2)
    res.apply(a)
    llm.say("UPDATES")
    d = res.apply(fact("Priya lives in Mumbai.", 2, 3, corr=True))
    assert d.relation == "CORRECTS" and row(db, a.memory_id)["status"] == "RETRACTED"


def test_historical_keeps_current_fact(env):
    db, res, llm = env
    cur = fact("Priya lives in Bengaluru.", 1, 1)
    res.apply(cur)
    llm.say("HISTORICAL")
    h = fact(
        "Priya lived in Delhi in 2019.",
        2,
        1,
        time=TimeExpr(kind="ABSOLUTE", value="2019", tense="PAST_ENDED"),
    )
    res.apply(h)
    c, r = row(db, cur.memory_id), row(db, h.memory_id)
    assert c["status"] == "ACTIVE" and c["valid_to"] is None and c["superseded_by_id"] is None
    assert (r["status"], r["valid_from"], r["valid_to"], r["valid_to_precision"]) == (
        "ACTIVE",
        "2019-01-01",
        "2019-12-31",
        "YEAR",
    )


def test_updates_by_closed_fact_becomes_historical(env):
    db, res, llm = env
    cur = fact("Priya lives in Bengaluru.", 1, 1)  # SESSION precision: compare() is INCOMPARABLE
    res.apply(cur)
    llm.say("UPDATES")
    d = res.apply(
        fact(
            "Priya lived in Delhi in 2019.",
            2,
            1,
            time=TimeExpr(kind="ABSOLUTE", value="2019", tense="PAST_ENDED"),
        )
    )
    assert d.relation == "HISTORICAL"
    assert row(db, cur.memory_id)["status"] == "ACTIVE"


def test_historical_with_open_interval_is_closed(env):
    db, res, llm = env
    res.apply(fact("Priya lives in Bengaluru.", 1, 1))
    llm.say("HISTORICAL")
    h = fact(
        "Priya lived in Delhi in 2019.", 2, 1, time=TimeExpr(kind="ABSOLUTE", value="2019")
    )  # tense wrong
    res.apply(h)
    r = row(db, h.memory_id)
    assert (r["valid_from"], r["valid_to"]) == ("2019-01-01", "2019-12-31")


def test_conflict_same_turn_marks_both(env):
    db, res, llm = env
    a = fact("Priya lives in Pune.", 2, 4)
    res.apply(a)
    llm.say("CONFLICT")
    b = fact("Priya lives in Nagpur.", 2, 4)
    res.apply(b)
    o, n = row(db, a.memory_id), row(db, b.memory_id)
    assert o["status"] == n["status"] == "CONFLICTED"
    assert o["conflict_set_id"] == n["conflict_set_id"] is not None


def test_conflict_across_turns_becomes_update(env):
    db, res, llm = env
    a = fact("Priya lives in Pune.", 2, 1)
    res.apply(a)
    llm.say("CONFLICT")
    d = res.apply(fact("Priya lives in Nagpur.", 2, 6))
    assert d.relation == "UPDATES" and row(db, a.memory_id)["status"] == "SUPERSEDED"


@pytest.mark.parametrize("llm_rel", ["UPDATES", "DUPLICATE"])
def test_neg_polarity_closes_matching_fact(env, llm_rel):
    db, res, llm = env
    pos = fact("Priya plays guitar.", 1, 2, key=None)
    res.apply(pos)
    llm.say(llm_rel)
    neg = fact("Priya no longer plays guitar.", 2, 3, key=None, polarity="NEG")
    d = res.apply(neg)
    assert d.relation == "NEGATES"
    p, n = row(db, pos.memory_id), row(db, neg.memory_id)
    assert (p["status"], p["valid_to"], p["valid_to_precision"]) == (
        "SUPERSEDED",
        "2023-05-17",
        "SESSION",
    )
    assert p["superseded_by_id"] == neg.memory_id
    assert (n["status"], n["polarity"]) == ("ACTIVE", "NEG")


def test_target_not_open_falls_back_to_new(env):
    db, res, llm = env
    h = fact(
        "Priya lived in Delhi in 2019.",
        1,
        1,
        time=TimeExpr(kind="ABSOLUTE", value="2019", tense="PAST_ENDED"),
    )
    res.apply(h)  # ACTIVE but closed
    llm.say("UPDATES")
    d = res.apply(fact("Priya lives in Bengaluru.", 2, 1))
    assert d.relation == "NEW" and row(db, h.memory_id)["valid_to"] == "2019-12-31"


def test_missing_target_with_several_candidates_is_new(env):
    db, res, llm = env
    res.apply(fact("Priya plays guitar.", 1, 1, key=None))
    llm.say("NEW", None)
    res.apply(fact("Priya plays the violin.", 1, 2, key=None))
    llm.say("UPDATES", "C9")  # invalid label, two candidates
    d = res.apply(fact("Priya plays music.", 2, 1, key=None))
    assert d.relation == "NEW" and "without a valid target" in d.notes[0]


def test_candidates_attribute_key_vs_embedding(env):
    db, res, llm = env
    res.apply(fact("Priya lives in Bengaluru.", 1, 1))
    llm.say("NEW", None)
    res.apply(fact("Priya plays guitar on weekends.", 1, 2, key=None))
    llm.say("NEW", None)
    res.apply(fact("Priya bakes sourdough bread.", 1, 3, key=None))
    cands = res.candidates(fact("Priya plays the guitar in a band.", 2, 1, key=None))
    assert cands[0].fact_text == "Priya plays guitar on weekends."  # most similar first
    key_cands = res.candidates(fact("Priya lives in Mumbai.", 2, 2))
    assert [c.fact_text for c in key_cands] == ["Priya lives in Bengaluru."]  # attribute key only


def test_reapplying_existing_memory_is_skipped(env):
    db, res, llm = env
    f = fact("Priya lives in Bengaluru.", 1, 1)
    res.apply(f)
    assert res.apply(f).relation == "SKIPPED_EXISTS"
    assert db.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1


def test_classifier_prompt_shows_candidates_with_intervals(env):
    db, res, llm = env
    res.apply(fact("Priya lives in Bengaluru.", 1, 1))
    llm.say("DUPLICATE")
    res.apply(fact("Priya lives in Bengaluru.", 2, 1))
    user = llm.calls[0][1]["content"]
    assert (
        "C1: Priya lives in Bengaluru. [ACTIVE, attribute residence, POS, from 2023-01-10 (SESSION) to now"
        in user
    )
    assert 'Quote: "Priya lives in Bengaluru."' in user


def correction(text, s, t, wrong, right, key="residence", wrong_quote=None):
    from dataclasses import replace

    return replace(fact(text, s, t, key=key, corr=True), correction_wrong=wrong, correction_right=right,
                   wrong_quote=wrong_quote)  # fmt: skip


def test_structured_correction_retracts_existing_original_without_llm(env):
    db, res, llm = env
    pune = fact("Priya lives in Pune.", 2, 2)
    res.apply(pune)
    d = res.apply(correction("Priya lives in Mumbai.", 2, 4, wrong="Pune", right="Mumbai"))
    assert d.relation == "CORRECTS" and d.llm_relation is None and llm.calls == []
    assert d.target_id == pune.memory_id and "retracted existing memory" in d.notes[0]
    assert row(db, pune.memory_id)["status"] == "RETRACTED"


def test_structured_correction_creates_missing_original_then_retracts_it(env):
    db, res, llm = env
    new = correction("Priya's manager is Priyanka.", 3, 3, wrong="Priya", right="Priyanka", key=None,
                     wrong_quote=("Priya", 14, 19))  # fmt: skip
    d = res.apply(new)
    assert d.relation == "CORRECTS" and llm.calls == []
    orig = row(db, new.memory_id + "-orig")
    assert orig["fact_text"] == "Priya's manager is Priya." and orig["status"] == "RETRACTED"
    assert (
        orig["superseded_by_id"] == new.memory_id and row(db, new.memory_id)["status"] == "ACTIVE"
    )
    quotes = db.conn.execute("SELECT role, quote, quote_start FROM fact_sources WHERE memory_id = ? ORDER BY role",
                             (orig["memory_id"],)).fetchall()  # fmt: skip
    assert [tuple(q) for q in quotes] == [
        ("ASSERTS", "Priya", 14),
        ("RETRACTS", "Priya's manager is Priyanka.", 0),
    ]


def test_structured_correction_matches_whole_words_only(env):
    db, res, llm = env
    keep = fact("Priya's manager is Priyanka.", 2, 1, key=None)
    res.apply(keep)  # mentions "Priyanka", must not be taken as the "Priya" being corrected
    new = correction(
        "Priya's manager is Priyanka.", 3, 3, wrong="Priya", right="Priyanka", key=None
    )
    new = new.__class__(**{**new.__dict__, "fact_text": "Priya's boss is Priyanka.", "quote": "x"})
    res.apply(new)
    assert row(db, keep.memory_id)["status"] == "ACTIVE"


def test_correction_without_usable_values_falls_back_to_classifier(env):
    db, res, llm = env
    res.apply(fact("Priya lives in Pune.", 2, 2))
    llm.say("CORRECTS")
    d = res.apply(correction("Priya lives in Mumbai.", 2, 4, wrong="Nagpur", right="Chennai"))
    assert d.llm_relation == "CORRECTS" and len(llm.calls) == 1


def test_value_settled_by_correction_is_not_retracted_by_the_same_turn(env):
    db, res, llm = env
    pune = fact("Priya lives in Pune.", 2, 2)
    res.apply(pune)
    mumbai = correction("Priya lives in Mumbai.", 2, 4, wrong="Pune", right="Mumbai")
    res.apply(mumbai)
    # same turn: a structured "correction" back to Pune, and a classifier CORRECTS, are both refused
    back = correction("Priya used to live in Pune.", 2, 4, wrong="Mumbai", right="Pune")
    llm.say("NEW", None)
    res.apply(back)
    llm.say("CORRECTS", "C2")  # candidates: C1 = "used to live in Pune" (newest), C2 = Mumbai
    d = res.apply(
        fact("Priya eats cheese.", 2, 4, key="residence", corr=True)
    )  # cue set: only the guard stops it
    assert "Mumbai" in llm.calls[-1][1]["content"].split("C2: ")[1].split("\n")[0]
    assert row(db, mumbai.memory_id)["status"] == "ACTIVE"
    assert "established by a correction in this turn" in d.notes[-1]
    # a later turn may still correct it
    res.apply(correction("Priya lives in Nagpur.", 3, 1, wrong="Mumbai", right="Nagpur"))
    assert row(db, mumbai.memory_id)["status"] == "RETRACTED"


def test_corrects_without_correction_cue_is_downgraded(env):
    db, res, llm = env
    veg = fact("Priya is vegetarian.", 2, 1, key="diet")
    res.apply(veg)
    llm.say("CORRECTS")
    d = res.apply(fact("Priya eats cheese.", 2, 3, key=None))  # no cue, different slot
    assert d.relation == "NEW" and row(db, veg.memory_id)["status"] == "ACTIVE"
    llm.say("CORRECTS")
    d = res.apply(
        fact("Priya is vegan.", 3, 1, key="diet")
    )  # no cue, same slot: a change, not a retraction
    assert d.relation == "UPDATES" and row(db, veg.memory_id)["status"] == "SUPERSEDED"


def test_correction_established_value_needs_explicit_wrong_to_be_retracted(env):
    db, res, llm = env
    res.apply(fact("Priya is vegan.", 2, 1, key="diet"))
    veg = correction("Priya is vegetarian.", 2, 1, wrong="vegan", right="vegetarian", key="diet")
    res.apply(veg)  # the model put the correction on turn 1 as well
    llm.say("CORRECTS")
    d = res.apply(
        fact("Priya still eats cheese.", 2, 3, key=None, corr=True)
    )  # flagged, but names nothing
    assert d.relation == "NEW" and row(db, veg.memory_id)["status"] == "ACTIVE"


def test_historical_for_recent_onset_becomes_update(env):
    from dataclasses import replace

    db, res, llm = env
    pune = fact("Priya lives in Pune.", 1, 1)
    res.apply(pune)
    llm.say("HISTORICAL")
    moved = replace(fact("Priya moved to Bengaluru.", 2, 1, time=TimeExpr(kind="RELATIVE", offset=-1, unit="WEEK")),
                    recent_onset=True)  # fmt: skip
    d = res.apply(moved)
    assert d.relation == "UPDATES" and "recently begun state" in " ".join(d.notes)
    assert row(db, pune.memory_id)["status"] == "SUPERSEDED"
    assert (row(db, moved.memory_id)["status"], row(db, moved.memory_id)["valid_to"]) == (
        "ACTIVE",
        None,
    )
