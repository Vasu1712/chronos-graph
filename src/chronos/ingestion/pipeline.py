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
    ) -> None:
        self.db = db
        self.llm = llm
        self.embedder = embedder
        self.config = config or IngestionConfig()
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
            state = StateResolver(tx.conn, self.llm, self.config)
            for vf, vec in zip(facts, vecs, strict=True):
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
        )
