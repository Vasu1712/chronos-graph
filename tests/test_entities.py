import pytest
from fakes import FakeEmbedder

from chronos.config import IngestionConfig
from chronos.ingestion.entities import (
    Descriptor,
    EntityResolver,
    canonical_relation,
    known_entities,
    normalize_name,
    relations_conflict,
)
from chronos.storage.db import Database


@pytest.fixture
def er(tmp_path):
    db = Database(tmp_path / "e.sqlite")
    db.add_conversation("c1", "synthetic")
    r = EntityResolver(
        db.conn, "c1", FakeEmbedder(), IngestionConfig(entity_similarity_threshold=0.8)
    )
    r.register_speakers(["Priya", "Rahul"])
    yield r
    db.close()


def count(er):
    return er.conn.execute("SELECT COUNT(*) FROM entities WHERE is_speaker = 0").fetchone()[0]


def test_speakers_registered_first_and_pronouns_rewrite(er):
    sp = er.speakers()
    assert list(sp) == ["Priya", "Rahul"]
    assert er.resolve("I", speaker="Priya") == sp["Priya"]
    assert er.resolve("my", speaker="Rahul") == sp["Rahul"]
    assert er.resolve("you", speaker="Priya") == sp["Rahul"]
    assert er.resolve("Your", speaker="Rahul") == sp["Priya"]
    assert er.resolve("priya", speaker="Rahul") == sp["Priya"]  # speaker name, any case
    assert er.resolve("she", speaker="Rahul") is None  # unresolvable pronoun
    assert count(er) == 0


def test_brother_and_coworker_alex_stay_separate(er):
    a = er.resolve("Alex", speaker="Priya", relation="brother")
    b = er.resolve("Alex", speaker="Priya", relation="coworker")
    assert a != b and count(er) == 2
    assert er.resolve("Alex", speaker="Priya", relation="older brother") == a  # adjectives ignored
    assert er.resolve("Alex", speaker="Priya", relation="colleague") == b  # synonym


def test_same_name_same_descriptor_merges_across_sessions(er):
    a = er.resolve("Bob", speaker="Priya", relation="coworker", turn_uid=None)
    b = er.resolve("Bob", speaker="Priya", relation="coworker", turn_uid=None)
    assert a == b and count(er) == 1


def test_name_only_mentions_merge_and_upgrade_descriptor(er):
    a = er.resolve("Bob", speaker="Priya")
    b = er.resolve("Bob", speaker="Priya", relation="friend")
    assert a == b
    desc = er.conn.execute("SELECT descriptor FROM entities WHERE entity_id = ?", (a,)).fetchone()[
        0
    ]
    assert Descriptor.loads(desc).relation == "friend"


def test_same_kin_relation_to_different_speakers_does_not_merge(er):
    a = er.resolve("Sam", speaker="Priya", relation="brother")
    b = er.resolve("Sam", speaker="Rahul", relation="brother")
    assert a != b


def test_types_are_not_merged(er):
    a = er.resolve("Paris", speaker="Priya", entity_type="PLACE")
    b = er.resolve("Paris", speaker="Priya", entity_type="PERSON")
    assert a != b


def test_ambiguous_name_uses_tiebreak_or_under_merges(er):
    brother = er.resolve("Alex", speaker="Priya", relation="brother")
    er.resolve("Alex", speaker="Priya", relation="coworker")
    # no relation given: two strong candidates, tie-breaker similarity too low -> new entity
    third = er.resolve("Alex", speaker="Priya")
    assert third not in (brother,) and count(er) == 3
    # relation given that matches only one -> conflict filter leaves one candidate
    assert er.resolve("Alex", speaker="Priya", relation="brother") == brother


def test_weak_partial_name_needs_tiebreak(tmp_path):
    db = Database(tmp_path / "w.sqlite")
    db.add_conversation("c1", "synthetic")
    no_emb = EntityResolver(db.conn, "c1", embedder=None)
    no_emb.register_speakers(["Priya", "Rahul"])
    full = no_emb.resolve("Alex Kim", speaker="Priya")
    assert no_emb.resolve("Alex", speaker="Priya") != full  # no tie-breaker -> under-merge
    with_emb = EntityResolver(
        db.conn, "c1", FakeEmbedder(), IngestionConfig(entity_similarity_threshold=0.5)
    )
    db.conn.execute("UPDATE entities SET embedding = NULL")
    kim = with_emb.resolve("Kim Lee", speaker="Priya")
    assert (
        with_emb.resolve("Kim", speaker="Priya") == kim
    )  # "kim" vs "kim lee": cosine ~0.71 >= 0.5
    db.close()


def test_aliases_recorded_and_known_entities_rendered(er):
    a = er.resolve("Alex", speaker="Priya", relation="brother")
    er.resolve("alex", speaker="Priya", relation="brother")
    aliases = {
        r[0] for r in er.conn.execute("SELECT alias FROM entity_aliases WHERE entity_id = ?", (a,))
    }
    assert aliases == {"Alex", "alex"}
    ke = known_entities(er.conn, "c1")
    assert ke[:2] == ["Priya (PERSON, speaker)", "Rahul (PERSON, speaker)"]
    assert "Alex (PERSON, brother of Priya)" in ke


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("My older brother", "brother"),
        ("co-worker", "coworker"),
        ("Mom", "mother"),
        (None, None),
        ("best friend", "friend"),
    ],
)
def test_canonical_relation(raw, expected):
    assert canonical_relation(raw) == expected


def test_normalize_and_conflict_helpers():
    assert normalize_name("  The Alex's ") == "alex"
    assert relations_conflict(Descriptor("brother", "P"), Descriptor("coworker", "P"))
    assert not relations_conflict(Descriptor("brother", "P"), Descriptor(None, None))
    assert not relations_conflict(Descriptor("friend", "P"), Descriptor("friend", "R"))
