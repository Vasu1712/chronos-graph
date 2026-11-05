"""Entity resolution (SPEC §3 step 1 and the entities / entity_aliases tables).

Order of decisions for a mention:
  1. first/second-person words -> the turn's speaker / the other participant (two-party chats)
  2. a speaker's name or alias -> that speaker
  3. other entities of the same conversation and type whose normalized name or alias equals the
     mention ("strong"), or for PERSON whose name tokens contain the other's ("weak": Alex ~ Alex Kim)
  4. PERSON candidates whose relation to the speaker conflicts (brother vs coworker) are removed
  5. one strong candidate -> merge; several -> embedding similarity of "name + descriptor" as a
     tie-breaker (>= threshold); weak candidates need the tie-breaker too
  6. otherwise create a new entity. When unsure we under-merge.

`entities.descriptor` stores a JSON object {"relation", "anchor", "text"}; relation is the
canonical relation to `anchor` (the speaker who introduced the person).
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from chronos.config import IngestionConfig
from chronos.llm.embed import from_blob, to_blob

FIRST_PERSON = {"i", "me", "my", "mine", "myself"}
SECOND_PERSON = {"you", "your", "yours", "yourself"}
OTHER_PRONOUNS = {
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "they",
    "them",
    "their",
    "we",
    "us",
    "our",
    "it",
}

_RELATION_SYNONYMS = {
    "colleague": "coworker", "co-worker": "coworker", "workmate": "coworker", "co worker": "coworker",
    "bro": "brother", "sis": "sister", "mom": "mother", "mum": "mother", "mommy": "mother",
    "dad": "father", "daddy": "father", "bf": "boyfriend", "gf": "girlfriend", "hubby": "husband",
    "spouse": "spouse", "kid": "child", "boss": "manager",
}  # fmt: skip
_RELATION_ADJECTIVES = {
    "older",
    "younger",
    "little",
    "big",
    "elder",
    "baby",
    "best",
    "old",
    "new",
    "close",
}
KIN = {
    "brother", "sister", "mother", "father", "son", "daughter", "husband", "wife", "spouse",
    "grandmother", "grandfather", "aunt", "uncle", "cousin", "niece", "nephew", "child",
}  # fmt: skip


class Embedder(Protocol):
    def encode_passages(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class Descriptor:
    relation: str | None = None
    anchor: str | None = None
    text: str | None = None

    def dumps(self) -> str:
        return json.dumps(
            {"relation": self.relation, "anchor": self.anchor, "text": self.text}, sort_keys=True
        )

    @staticmethod
    def loads(raw: str | None) -> Descriptor:
        if not raw:
            return Descriptor()
        try:
            d = json.loads(raw)
            return Descriptor(d.get("relation"), d.get("anchor"), d.get("text"))
        except (json.JSONDecodeError, AttributeError):
            return Descriptor(text=raw)

    def phrase(self) -> str:
        parts = []
        if self.relation:
            parts.append(f"{self.relation} of {self.anchor}" if self.anchor else self.relation)
        if self.text:
            parts.append(self.text)
        return ", ".join(parts)


def normalize_name(name: str) -> str:
    n = name.strip().casefold()
    n = re.sub(r"[’`]", "'", n)
    n = re.sub(r"'s\b", "", n)
    n = re.sub(r"[^\w\s-]", " ", n)
    n = re.sub(r"^(the|a|an)\s+", "", n.strip())
    return re.sub(r"\s+", " ", n).strip()


def canonical_relation(rel: str | None) -> str | None:
    if not rel:
        return None
    r = normalize_name(rel)
    r = re.sub(r"^(my|your|his|her|their)\s+", "", r)
    words = [w for w in r.split() if w not in _RELATION_ADJECTIVES]
    r = " ".join(words)
    r = _RELATION_SYNONYMS.get(r, r)
    return r or None


def relations_conflict(a: Descriptor, b: Descriptor) -> bool:
    ra, rb = canonical_relation(a.relation), canonical_relation(b.relation)
    if not ra or not rb:
        return False
    if ra != rb:
        return True
    # the same family relation to *different* speakers is almost always a different person
    return ra in KIN and a.anchor is not None and b.anchor is not None and a.anchor != b.anchor


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", normalize_name(name)).strip("-")
    return s or "entity"


@dataclass(frozen=True)
class _Row:
    entity_id: str
    name: str
    entity_type: str | None
    is_speaker: bool
    descriptor: Descriptor
    embedding: bytes | None
    aliases: frozenset[str]


class EntityResolver:
    """Resolves mentions within one conversation using a connection inside the session txn."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        conversation_id: str,
        embedder: Embedder | None = None,
        config: IngestionConfig | None = None,
    ) -> None:
        self.conn = conn
        self.conversation_id = conversation_id
        self.embedder = embedder
        self.config = config or IngestionConfig()
        self.created = 0
        self.merged = 0

    # ------------------------------------------------------------------ speakers
    def register_speakers(self, speakers: Sequence[str]) -> dict[str, str]:
        ids = {}
        for name in speakers:
            eid = f"{self.conversation_id}:e:{slug(name)}"
            self.conn.execute(
                "INSERT INTO entities(entity_id, conversation_id, canonical_name, entity_type, is_speaker, descriptor) "
                "VALUES (?, ?, ?, 'PERSON', 1, ?) ON CONFLICT(entity_id) DO NOTHING",
                (
                    eid,
                    self.conversation_id,
                    name,
                    Descriptor(text="conversation participant").dumps(),
                ),
            )
            self._add_alias(eid, name, None)
            ids[name] = eid
        return ids

    def speakers(self) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT canonical_name, entity_id FROM entities WHERE conversation_id = ? AND is_speaker = 1 "
            "ORDER BY rowid",
            (self.conversation_id,),
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    # ------------------------------------------------------------------ resolution
    def resolve_person_word(self, word: str, speaker: str) -> str | None:
        """First/second person -> speaker entity ids; None if not a person word or unresolvable."""
        n = normalize_name(word)
        sp = self.speakers()
        if n in FIRST_PERSON:
            return sp.get(speaker)
        if n in SECOND_PERSON:
            others = [eid for name, eid in sp.items() if name != speaker]
            return others[0] if len(others) == 1 else None
        return None

    def resolve(
        self,
        name: str,
        *,
        speaker: str,
        entity_type: str = "PERSON",
        relation: str | None = None,
        turn_uid: str | None = None,
    ) -> str | None:
        """Return an entity_id for the mention, creating one if needed.

        Returns None for unresolvable pronouns (he/she/they or "you" in multi-party chats).
        """
        n = normalize_name(name)
        if not n:
            return None
        if n in FIRST_PERSON or n in SECOND_PERSON:
            return self.resolve_person_word(n, speaker)
        if n in OTHER_PRONOUNS:
            return None

        rows = self._rows()
        for r in rows:  # speakers win over everything else
            if r.is_speaker and (n == normalize_name(r.name) or n in r.aliases):
                self._add_alias(r.entity_id, name, turn_uid)
                return r.entity_id

        etype = entity_type or "OTHER"
        desc = Descriptor(
            relation=canonical_relation(relation) if etype == "PERSON" else None,
            anchor=speaker if (relation and etype == "PERSON") else None,
        )
        same_type = [r for r in rows if not r.is_speaker and (r.entity_type or "OTHER") == etype]
        strong = [r for r in same_type if n == normalize_name(r.name) or n in r.aliases]
        weak = []
        if not strong and etype == "PERSON":
            toks = set(n.split())
            weak = [
                r for r in same_type
                if (rt := set(normalize_name(r.name).split())) and (toks < rt or rt < toks)
            ]  # fmt: skip
        if etype == "PERSON":
            strong = [r for r in strong if not relations_conflict(r.descriptor, desc)]
            weak = [r for r in weak if not relations_conflict(r.descriptor, desc)]

        chosen: _Row | None = None
        if len(strong) == 1:
            chosen = strong[0]
        elif len(strong) > 1:
            chosen = self._tiebreak(name, desc, strong)
        elif weak:
            chosen = self._tiebreak(name, desc, weak)

        if chosen is not None:
            self.merged += 1
            self._add_alias(chosen.entity_id, name, turn_uid)
            if chosen.descriptor.relation is None and desc.relation is not None:
                self._update_descriptor(chosen, desc)
            return chosen.entity_id
        return self._create(name, etype, desc, turn_uid)

    # ------------------------------------------------------------------ internals
    def _rows(self) -> list[_Row]:
        ents = self.conn.execute(
            "SELECT entity_id, canonical_name, entity_type, is_speaker, descriptor, embedding "
            "FROM entities WHERE conversation_id = ? ORDER BY rowid",
            (self.conversation_id,),
        ).fetchall()
        aliases: dict[str, set[str]] = {}
        for eid, alias in self.conn.execute(
            "SELECT a.entity_id, a.alias FROM entity_aliases a JOIN entities e USING(entity_id) "
            "WHERE e.conversation_id = ?",
            (self.conversation_id,),
        ):
            aliases.setdefault(eid, set()).add(normalize_name(alias))
        return [
            _Row(
                e[0],
                e[1],
                e[2],
                bool(e[3]),
                Descriptor.loads(e[4]),
                e[5],
                frozenset(aliases.get(e[0], ())),
            )
            for e in ents
        ]

    def _embed_text(self, name: str, desc: Descriptor) -> str:
        phrase = desc.phrase()
        return f"{name}, {phrase}" if phrase else name

    def _embed(self, name: str, desc: Descriptor) -> np.ndarray | None:
        if self.embedder is None:
            return None
        return self.embedder.encode_passages([self._embed_text(name, desc)])[0]

    def _tiebreak(self, name: str, desc: Descriptor, cands: list[_Row]) -> _Row | None:
        q = self._embed(name, desc)
        if q is None:
            return None
        best, best_sim = None, -1.0
        for r in cands:
            if r.embedding is None:
                continue
            sim = float(np.dot(q, from_blob(r.embedding)))
            if sim > best_sim:
                best, best_sim = r, sim
        return (
            best
            if best is not None and best_sim >= self.config.entity_similarity_threshold
            else None
        )

    def _create(self, name: str, etype: str, desc: Descriptor, turn_uid: str | None) -> str:
        base = f"{self.conversation_id}:e:{slug(name)}"
        eid, k = base, 1
        while self.conn.execute("SELECT 1 FROM entities WHERE entity_id = ?", (eid,)).fetchone():
            k += 1
            eid = f"{base}-{k}"
        vec = self._embed(name, desc)
        self.conn.execute(
            "INSERT INTO entities(entity_id, conversation_id, canonical_name, entity_type, is_speaker, descriptor, embedding) "
            "VALUES (?, ?, ?, ?, 0, ?, ?)",
            (
                eid,
                self.conversation_id,
                name.strip(),
                etype,
                desc.dumps(),
                to_blob(vec) if vec is not None else None,
            ),
        )
        self._add_alias(eid, name, turn_uid)
        self.created += 1
        return eid

    def _update_descriptor(self, row: _Row, desc: Descriptor) -> None:
        merged = Descriptor(desc.relation, desc.anchor, row.descriptor.text)
        vec = self._embed(row.name, merged)
        self.conn.execute(
            "UPDATE entities SET descriptor = ?, embedding = COALESCE(?, embedding) WHERE entity_id = ?",
            (merged.dumps(), to_blob(vec) if vec is not None else None, row.entity_id),
        )

    def _add_alias(self, entity_id: str, alias: str, turn_uid: str | None) -> None:
        exists = self.conn.execute(
            "SELECT 1 FROM entity_aliases WHERE entity_id = ? AND alias = ? AND turn_uid IS ?",
            (entity_id, alias.strip(), turn_uid),
        ).fetchone()
        if not exists:
            self.conn.execute(
                "INSERT INTO entity_aliases(entity_id, alias, turn_uid) VALUES (?, ?, ?)",
                (entity_id, alias.strip(), turn_uid),
            )


def known_entities(conn: sqlite3.Connection, conversation_id: str, limit: int = 40) -> list[str]:
    """Short "Name (TYPE, descriptor)" strings for the extractor prompt, speakers first."""
    rows = conn.execute(
        "SELECT canonical_name, entity_type, is_speaker, descriptor FROM entities WHERE conversation_id = ? "
        "ORDER BY is_speaker DESC, rowid LIMIT ?",
        (conversation_id, limit),
    ).fetchall()
    out = []
    for name, etype, is_speaker, desc in rows:
        extra = "speaker" if is_speaker else Descriptor.loads(desc).phrase()
        out.append(f"{name} ({etype or 'OTHER'}{', ' + extra if extra else ''})")
    return out
