"""Ingestion pipeline: one finished session at a time (SPEC §3).

  1. Tier 1, own transaction: session row + episodes (immutable). Commits even if extraction
     fails later, so the log never loses turns.
  2. If the session is already ingested -> no-op.
  3. Extraction (LLM, outside any transaction) and embeddings of fact_text.
  4. One transaction (Database.session_transaction): entity resolution, state resolution and
     every memory / source / fact_entities write; `ingested_at` is stamped on commit. A crash
     anywhere in this phase leaves no partial session.

Nothing is rebuilt at startup: candidate embeddings are read from the SQLite BLOBs on lookup.
All ids are deterministic so a re-run after a crash produces the same database.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any

from chronos.config import IngestionConfig
from chronos.ingestion.entities import EntityResolver, known_entities, normalize_name
from chronos.ingestion.extractor import ExtractionStats, Extractor, ValidFact
from chronos.ingestion.resolver import Decision, NewFact, StateResolver
from chronos.ingestion.temporal import resolve
from chronos.ingestion.types import Conversation, Session, Turn
from chronos.llm.embed import to_blob
from chronos.ontology import Ontology
from chronos.storage.db import Database

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    session_id: str
    skipped: bool = False
    episodes_added: int = 0
    extraction: ExtractionStats | None = None
    decisions: list[Decision] = field(default_factory=list)
    unresolved_subjects: int = 0
    llm_classifications: int = 0

    @property
    def relation_counts(self) -> Counter:
        return Counter(d.relation for d in self.decisions)

    def summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "skipped": self.skipped,
            "episodes_added": self.episodes_added,
            "extraction": self.extraction.as_dict() if self.extraction else None,
            "relations": dict(self.relation_counts),
            "unresolved_subjects": self.unresolved_subjects,
            "llm_classifications": self.llm_classifications,
        }


def embedder_name(embedder: Any) -> str:
    return getattr(embedder, "name", None) or type(embedder).__name__


def store_episode_embeddings(conn: Any, embedder: Any, items: list[tuple[str, str]]) -> int:
    """Embed and store (turn_uid, text) pairs that have no vector yet. Returns rows written."""
    if embedder is None or not items:
        return 0
    marks = ",".join("?" * len(items))
    have = {r[0] for r in conn.execute(f"SELECT turn_uid FROM episode_embeddings WHERE turn_uid IN ({marks})",
                                       [uid for uid, _ in items])}  # fmt: skip
    todo = [(uid, text) for uid, text in items if uid not in have]
    if not todo:
        return 0
    vecs = embedder.encode_passages([text for _, text in todo])
    name = embedder_name(embedder)
    conn.executemany(
        "INSERT INTO episode_embeddings(turn_uid, model, embedding) VALUES (?, ?, ?)",
        [(uid, name, to_blob(v)) for (uid, _), v in zip(todo, vecs, strict=True)],
    )
    return len(todo)


def memory_id_for(conversation_id: str, vf: ValidFact) -> str:
    h = hashlib.sha1(
        f"{vf.turn.turn_uid}|{vf.ordinal_in_turn}|{vf.fact_text}".encode()
    ).hexdigest()[:12]
    return f"{conversation_id}:m:{h}"


def episode_turn(t: Turn) -> Turn:
    """The text stored in Tier 1 and shown to the extractor: utterance + shared-image caption."""
    return replace(t, text=t.display_text, image_caption=None) if t.image_caption else t


class IngestionPipeline:
    def __init__(
        self,
        db: Database,
        llm: Any,
        embedder: Any,
        *,
        ontology: Ontology | None = None,
        config: IngestionConfig | None = None,
        policy: str = "bitemporal",
        recorded: dict[str, tuple[str | None, str | None]] | None = None,
    ) -> None:
        self.db = db
        self.llm = llm
        self.embedder = embedder
        self.config = config or IngestionConfig()
        self.policy = policy
        self.recorded = recorded
        self.extractor = Extractor(llm, ontology, self.config)

    # ------------------------------------------------------------------ registration
    def register_conversation(
        self, conversation_id: str, dataset: str, speakers: list[str] | tuple[str, ...]
    ) -> None:
        with self.db.transaction() as conn:
            self.db.add_conversation(conversation_id, dataset, participants=list(speakers))
            EntityResolver(conn, conversation_id).register_speakers(list(speakers))

    def speakers(self, conversation_id: str) -> list[str]:
        row = self.db.conn.execute(
            "SELECT participants FROM conversations WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"conversation {conversation_id!r} is not registered")
        return json.loads(row[0])

    # ------------------------------------------------------------------ ingestion
    def ingest_conversation(self, conv: Conversation) -> list[IngestResult]:
        self.register_conversation(conv.conversation_id, conv.dataset, conv.speakers)
        return [self.ingest_session(conv.conversation_id, s) for s in conv.sessions]

    def ingest_session(self, conversation_id: str, session: Session) -> IngestResult:
        if not session.timestamp:
            raise ValueError(f"{session.session_id}: session timestamp is required")
        speakers = self.speakers(conversation_id)
        result = IngestResult(session.session_id)
        turns = [episode_turn(t) for t in session.turns]
        session = replace(session, turns=tuple(turns))

        # 1. Tier 1: episodes, committed on their own
        with self.db.transaction() as conn:
            er = EntityResolver(conn, conversation_id)
            sp_ids = er.speakers()
            missing = [t.speaker for t in turns if t.speaker not in sp_ids]
            if missing:
                sp_ids |= er.register_speakers(sorted(set(missing)))
            self.db.add_session(session.session_id, conversation_id, session.ordinal, session.timestamp,
                                source_ref=session.source_ref)  # fmt: skip
            for t in turns:
                result.episodes_added += self.db.append_episode(
                    {
                        "turn_uid": t.turn_uid,
                        "conversation_id": conversation_id,
                        "session_id": session.session_id,
                        "session_ordinal": session.ordinal,
                        "turn_index": t.turn_index,
                        "speaker": t.speaker,
                        "speaker_entity_id": sp_ids[t.speaker],
                        "session_ts": session.timestamp,
                        "text": t.text,
                        "source_ref": t.source_ref,
                    }  # fmt: skip
                )
            store_episode_embeddings(conn, self.embedder, [(t.turn_uid, t.text) for t in turns])

        # 2. no-op if already ingested
        if self._ingested(session.session_id):
            result.skipped = True
            return result

        # 3. extraction + embeddings (no transaction held)
        facts, stats = self.extractor.extract_session(
            session, speakers, known_entities(self.db.conn, conversation_id)
        )
        result.extraction = stats
        vecs = self.embedder.encode_passages([f.fact_text for f in facts]) if facts else []

        # 4. Tier 2 writes, one transaction
        with self.db.session_transaction(session.session_id) as tx:
            if tx.already_ingested:
                result.skipped = True
                return result
            ents = EntityResolver(tx.conn, conversation_id, self.embedder, self.config)
            state = StateResolver(
                tx.conn, self.llm, self.config, policy=self.policy, recorded=self.recorded
            )
            # a correction is applied after the other facts of its turn, so an original statement
            # extracted from the same turn already exists when the correction looks for it
            order = sorted(
                range(len(facts)),
                key=lambda i: (facts[i].turn.turn_index, facts[i].is_correction, i),
            )
            for vf, vec in ((facts[i], vecs[i]) for i in order):
                nf = self._to_new_fact(conversation_id, session, vf, vec, ents)
                if nf is None:
                    result.unresolved_subjects += 1
                    continue
                result.decisions.append(state.apply(nf))
            result.llm_classifications = state.llm_calls
        log.info("%s: %s", session.session_id, result.summary())
        return result

    # ------------------------------------------------------------------ helpers
    def _ingested(self, session_id: str) -> bool:
        row = self.db.conn.execute(
            "SELECT ingested_at FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row is not None and row[0] is not None

    def _to_new_fact(
        self, conversation_id: str, session: Session, vf: ValidFact, vec: Any, ents: EntityResolver
    ) -> NewFact | None:
        speaker = vf.turn.speaker
        subj_norm = normalize_name(vf.subject)
        subj_mention = next((m for m in vf.entities if normalize_name(m.name) == subj_norm), None)
        subject_id = ents.resolve(
            vf.subject,
            speaker=speaker,
            entity_type=subj_mention.entity_type if subj_mention else "PERSON",
            relation=subj_mention.relation if subj_mention else None,
            turn_uid=vf.turn.turn_uid,
        )
        if subject_id is None:
            return None
        mention_ids = []
        for m in vf.entities:
            eid = ents.resolve(m.name, speaker=speaker, entity_type=m.entity_type, relation=m.relation,
                               turn_uid=vf.turn.turn_uid)  # fmt: skip
            if eid is not None and eid not in mention_ids:
                mention_ids.append(eid)
        return NewFact(
            memory_id=memory_id_for(conversation_id, vf),
            conversation_id=conversation_id,
            subject_entity_id=subject_id,
            subject_name=vf.subject,
            attribute_key=vf.attribute_key,
            fact_text=vf.fact_text,
            polarity=vf.polarity,
            mode=vf.mode,
            is_event=vf.is_event,
            time_expr=vf.time,
            session_ts=session.timestamp,
            rt=resolve(vf.time, session.timestamp, is_event=vf.is_event),
            session_ordinal=session.ordinal,
            turn_index=vf.turn.turn_index,
            turn_uid=vf.turn.turn_uid,
            quote=vf.quote,
            quote_start=vf.quote_start,
            quote_end=vf.quote_end,
            is_correction=vf.is_correction,
            mention_entity_ids=tuple(mention_ids),
            embedding=vec,
            recent_onset=vf.recent_onset,
            correction_wrong=vf.correction_wrong,
            correction_right=vf.correction_right,
            wrong_quote=(
                (vf.turn.text[vf.wrong_start : vf.wrong_end], vf.wrong_start, vf.wrong_end)
                if vf.wrong_start is not None
                else None
            ),
        )


def recorded_decisions(
    conn: Any, conversation_id: str | None = None
) -> dict[str, tuple[str | None, str | None]]:
    """memory_id -> (classifier relation, classifier target memory_id) from a finished run's
    resolver_log; feed it to IngestionPipeline(recorded=...) to replay under another policy."""
    sql = "SELECT memory_id, llm_relation, llm_target_id FROM resolver_log"
    params: tuple = ()
    if conversation_id is not None:
        sql += " WHERE conversation_id = ?"
        params = (conversation_id,)
    return {r[0]: (r[1], r[2]) for r in conn.execute(sql, params)}
