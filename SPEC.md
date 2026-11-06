# ChronosGraph — Specification

ChronosGraph is a local, persistent, **bi-temporal memory** for multi-session conversational
agents, plus a reproducible evaluation on **LoCoMo** and **LongMemEval**.

Every part of the build reads this file first. It is the source of truth for architecture,
schema and write rules.

---

## 0. Environment (hard constraints)

| Concern | Choice |
|---|---|
| Hardware | MacBook M1 Pro, 16 GB unified memory, macOS |
| Python | 3.11, managed with `uv` |
| System LLM | `Qwen2.5-7B-Instruct` via Ollama, tag `qwen2.5:7b-instruct-q4_K_M`, OpenAI-compatible endpoint `http://localhost:11434/v1`. Always `num_ctx=32768`, `temperature=0`. Every structured call uses Ollama's JSON-schema `format`. |
| Embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers on MPS |
| Runtime entailment | `cross-encoder/nli-deberta-v3-base` on MPS |
| Judge (evaluation only) | `openai/gpt-oss-120b`, pooled across Groq and Cerebras free tiers. Keys in `.env` as `GROQ_API_KEY`, `CEREBRAS_API_KEY`. |
| Forbidden | No OpenAI APIs anywhere. |

Rule: never write benchmark numbers that were not produced by a run.

---

## 1. Architecture overview

Three tiers:

1. **Tier 1 — Immutable Episodic Log** (source of truth).
2. **Tier 2 — Bi-Temporal Fact and Entity Store**, populated by the **Ingestion Engine**.
3. **Tier 3 — Hybrid Retriever and Provenance-Enforced Generator**.

```
 sessions ──► Tier 1 episodes (append-only) ──► Ingestion Engine ──► Tier 2 memories/entities
                                                                         │
 query ──► Tier 3 query analysis ─► hybrid retrieval ─► temporal filter ─► episodic expansion
            ─► pack ─► answerability gate ─► structured generation ─► verifier ─► answer
```

---

## 2. Tier 1 — Immutable Episodic Log

SQLite tables `conversations`, `sessions`, `episodes`, plus an **FTS5** index over `episodes.text`.
Episodes are **never updated or deleted**.

```
episodes(
  turn_uid           TEXT PRIMARY KEY,
  conversation_id    TEXT,
  session_id         TEXT,
  session_ordinal    INT,
  turn_index         INT,
  speaker            TEXT,
  speaker_entity_id  TEXT,
  session_ts         TEXT,   -- ISO-8601
  text               TEXT,
  source_ref         TEXT
)
```

- `turn_uid` format: `"{dataset}:{conversation}:s{session:02d}:t{turn:03d}"`,
  e.g. `locomo:conv-26:s03:t007`.
- `source_ref` keeps the dataset's own id: LoCoMo `dia_id` like `"D3:7"`; LongMemEval
  session id + turn index.

---

## 3. Ingestion Engine

Runs **in batch at session end**, **one SQLite transaction per session**.

1. **Speaker-aware coreference.** `I/me/my` bind to the turn's speaker; `you/your` bind to the
   other participant in two-party chats. Speakers are registered as entities **before** ingest.
2. **Extraction.** Extract self-contained atomic facts (resolved text, no pronouns), entities
   mentioned, the verbatim quote that supports each fact, and any time expression.
   - The LLM only **normalizes** the time expression (absolute, or relative offset + unit).
   - Python code computes dates from `session_ts` and assigns a precision:
     `DAY | WEEK | MONTH | YEAR | SESSION | BEFORE | UNKNOWN`.
3. **Candidate lookup.**
   - If the fact maps to a closed single-valued attribute key (§3.1), candidates are ACTIVE
     memories with the same subject and `attribute_key`.
   - Otherwise candidates are the **top-5 ACTIVE memories of the same subject** by embedding
     similarity.
4. **Classify against candidates:**
   `NEW | DUPLICATE | EXTENDS | UPDATES | CORRECTS | HISTORICAL | CONFLICT`.
5. **Write** with the bi-temporal rules (§5).

### 3.1 Closed single-valued attribute keys

Defined in `src/chronos/ontology/attributes.yaml`:

- `residence`
- `hometown`
- `employer`
- `job_title`
- `school`
- `relationship_status`
- `partner`
- `diet`
- `favorite:<category>`
- `primary_tool:<purpose>`

Each key has a description and paraphrase examples shown to the extractor. Anything else has
`attribute_key = NULL` and is **multi-valued**.

---

## 4. Tier 2 — Bi-Temporal Fact and Entity Store

```
entities(
  entity_id        TEXT PRIMARY KEY,
  conversation_id  TEXT,
  canonical_name   TEXT,
  entity_type      TEXT,
  is_speaker       INT,
  descriptor       TEXT,
  embedding        BLOB
)

entity_aliases(entity_id, alias, turn_uid)

memories(
  memory_id                 TEXT PRIMARY KEY,
  conversation_id           TEXT,
  subject_entity_id         TEXT,
  attribute_key             TEXT NULL,
  fact_text                 TEXT,
  polarity                  TEXT,   -- POS | NEG
  mode                      TEXT,   -- EXPLICIT | IMPLIED
  corroboration             INT,
  status                    TEXT,   -- ACTIVE | SUPERSEDED | RETRACTED | CONFLICTED

  -- event time: when it was true in the world
  valid_from                TEXT,
  valid_from_precision      TEXT,
  valid_to                  TEXT,
  valid_to_precision        TEXT,

  -- only for point events ("we went hiking on May 3")
  event_date                TEXT,
  event_date_precision      TEXT,

  -- transaction time: when the system learned it
  recorded_session_ordinal  INT,
  recorded_turn_index       INT,
  retired_session_ordinal   INT,
  retired_turn_index        INT,

  supersedes_id             TEXT,
  superseded_by_id          TEXT,
  conflict_set_id           TEXT,
  embedding                 BLOB
)

fact_sources(
  memory_id    TEXT,
  turn_uid     TEXT REFERENCES episodes(turn_uid),   -- real FK, not a JSON list
  quote        TEXT,
  quote_start  INT,
  quote_end    INT,
  role         TEXT    -- ASSERTS | REAFFIRMS | CORRECTS | RETRACTS
)

fact_entities(
  memory_id  TEXT,
  entity_id  TEXT,
  role       TEXT      -- SUBJECT | OBJECT | MENTION   (used for 1-hop expansion)
)

answer_log(
  query_id              TEXT,
  conversation_id       TEXT,
  query                 TEXT,
  retrieved_memory_ids  JSON,
  retrieved_turn_uids   JSON,
  response_json         JSON,
  verifier_json         JSON
)
```

**Vector index:** embeddings are stored as BLOBs and searched with NumPy in memory
(corpora are small).

---

## 5. Bi-temporal write rules

Order key = `(session_ordinal, turn_index)`.

| Class | Rule |
|---|---|
| **NEW** | Insert ACTIVE. |
| **DUPLICATE** | Add a `fact_sources` row (`REAFFIRMS`); `corroboration += 1` for a new distinct turn. |
| **EXTENDS** | Insert ACTIVE, keep the old one ACTIVE (adds detail, no conflict). |
| **UPDATES** | Compare event time. If `new.valid_from` is later than `old.valid_from` at the coarser of the two precisions, old becomes SUPERSEDED with `valid_to = new.valid_from`, retired at the new fact's order key, chained via `supersedes_id` / `superseded_by_id`. If the new fact is **earlier**, insert it as SUPERSEDED and link it behind the old one. If **incomparable** (equal at shared precision, or SESSION precision), the later order key wins, so in-session updates work. |
| **CORRECTS** | ("Pune, sorry, I meant Mumbai") Old becomes RETRACTED (never true), new is ACTIVE, source role `CORRECTS`. |
| **HISTORICAL** | ("back in 2019 I lived in Delhi") Insert with its closed interval; never touches the current fact. |
| **CONFLICT** | (same turn, contradictory, no correction cue) Both CONFLICTED with a shared `conflict_set_id`; retrieved and shown together. |
| **NEG polarity** | A NEG fact matching an active fact closes it (SUPERSEDED, `valid_to` from the new evidence). |
| **Idempotence** | Re-ingesting a session is a no-op. |

---

## 6. Tier 3 — Hybrid Retriever and Provenance-Enforced Generator

1. **Query analysis** by rules plus entity string matching (**no LLM call**):
   - intent: `CURRENT | AS_OF | HISTORY | TEMPORAL_ORDER | OPEN`
   - subject speaker
   - time constraint resolved to a date
2. **Hybrid retrieval.** Dense + BM25 (FTS5) over `memories.fact_text` and over episodes,
   fused with **Reciprocal Rank Fusion** (`k = 60`).
3. **1-hop expansion** through `fact_entities` from entities named in the query.
4. **Temporal filter by intent:**
   - `CURRENT` = ACTIVE with open `valid_to`.
   - `AS_OF t` = non-retracted facts whose interval contains `t`.
   - `HISTORY` / `TEMPORAL_ORDER` = include version chains with dates.
5. **Episodic expansion.** Fetch each selected memory's source turns plus a ±1 turn window.
6. **Packing** into a token budget (default **1500**); every turn prefixed
   `"[turn_uid | date | speaker]"`.
7. **Answerability gate.** Deterministic speaker-mismatch check, then one LLM call
   "does this evidence answer this exact question?" →
   `ANSWERABLE | PARTIAL | NOT_ANSWERABLE`. Abstain on `NOT_ANSWERABLE`.
8. **Structured generation:**
   `final_answer`, `claims[{text, cited_turn_uids, cited_memory_ids}]`, `abstained`,
   `abstention_reason`.
9. **Verifier, per claim:**
   - cited turns exist;
   - cited turns were in the packed context (checked against `answer_log`);
   - NLI entailment of the claim by the concatenated cited turns.

   Drop failing citations, regenerate once with feedback, drop unsupported claims, abstain if
   none remain.

---

## 7. Evaluation

- Datasets: LoCoMo (`locomo10.json`), LongMemEval (`longmemeval_s.json`), sha256 recorded in
  `data/raw/CHECKSUMS`.
- LoCoMo QA category map: 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop,
  5 adversarial (reference answer in `adversarial_answer`).
- LongMemEval abstention questions are identified by `"_abs"` in `question_id`.
- Manifests (`eval/manifests/`, seed 42, include dataset checksums):
  - `dev_v1`: LoCoMo conversation index 0 + 20 LongMemEval questions stratified over
    knowledge-update, temporal-reasoning, abstention.
  - `test_v1`: 50 questions per LoCoMo category from the remaining 9 conversations + all
    remaining knowledge-update and abstention questions.
  - `quick_v1`: 1 LoCoMo test conversation's sample + 20 knowledge-update questions.
- Judge: `gpt-oss-120b`, routed by `sha256(question_id) mod 2` to Groq or Cerebras so every
  system's answer to a question is judged by the same provider; wait on quota exhaustion
  rather than switch; provider recorded on every verdict.

---

## 8. Implementation notes (Part 2, ingestion)

Decisions the sections above leave open. They refine the architecture; none changes it.

**Extraction schema** (`chronos.ingestion.extractor`). Per fact: `turn_index, subject, fact_text,
attribute_key, polarity, mode, is_event, entities_mentioned[{name, entity_type, relation}], time
{kind, value, offset, unit, tense}, is_correction, quote`.
- `is_event` separates point events (stored in `event_date`) from states (stored in `valid_*`).
- `entities_mentioned` are objects so a PERSON's relation to the speaker is available for the
  entity-merge guard.
- `tense = FUTURE_PLAN` facts never carry an `attribute_key`: a plan does not change a current value.

**Dates.** Memory dates are stored as `YYYY-MM-DD`; `SESSION` and `BEFORE` use the session date.
Spans: `RELATIVE n WEEK` = `session − 7n days` .. `+6 days`; `RELATIVE n MONTH` = the calendar
month shifted by n; `ABSOLUTE` accepts `YYYY`, `YYYY-MM`, `YYYY-MM-DD`; anything else is `UNKNOWN`.

**Entities.** `entities.descriptor` holds a JSON object `{"relation", "anchor", "text"}`, where
`relation` is the canonical relation to `anchor` (the speaker who introduced the person).

**Statuses.** A fact with a closed interval (HISTORICAL, or NEW with `PAST_ENDED`) is stored
`ACTIVE` with a non-null `valid_to`, so CURRENT = `ACTIVE AND valid_to IS NULL`. Candidate lookup
returns all ACTIVE same-key facts (open first); UPDATES / CORRECTS / NEG closing need an open target.

**Deterministic normalization of the classifier's choice** (`chronos.ingestion.resolver`):
1. no valid target label → the only candidate if there is exactly one, else NEW;
2. `is_correction` + UPDATES → CORRECTS;
3. CONFLICT whose target is from a different turn → UPDATES;
4. DUPLICATE with opposite polarity → UPDATES;
5. NEG fact + POS target via UPDATES → NEGATES: the target becomes SUPERSEDED with `valid_to` from
   the NEG fact, and the NEG fact is inserted ACTIVE, linked by `supersedes_id`;
6. target not open → the open same-key candidate, else NEW;
7. UPDATES by a fact whose own interval is closed → HISTORICAL (an ended state never replaces the
   current one; needed because SESSION precision makes `compare` INCOMPARABLE);
8. HISTORICAL with an open interval → the time is re-resolved as `PAST_ENDED`.

**Transactions.** Tier 1 commits first. The Tier 2 transaction (`BEGIN IMMEDIATE`) stays open
during the classifier calls of that session so in-session facts see each other; a crash rolls the
whole session back. All ids (memories, entities, conflict sets) are deterministic.

**Episode text.** For LoCoMo turns that share an image, `episodes.text` is the utterance followed
by `[shares a photo: <blip_caption>]`, so the caption is searchable and quotable.

**Known limitation.** Short-lived states ("in Pune this week") stay open-ended; nothing closes them
when the week is over.

---

## 9. Implementation notes (Part 3, retrieval and answering)

Decisions the Tier 3 description leaves open. They refine the architecture; none changes it.

**Schema v2.** Dense retrieval over episodes needs vectors, and `episodes` is immutable, so
migration 2 adds `episode_embeddings(turn_uid PK FK episodes, model, embedding BLOB)`. The
pipeline writes it in the Tier 1 transaction; the retriever backfills rows missing for databases
ingested before v2. `model` records the embedder so a mismatched query embedder is rejected.

**Query analysis** (`chronos.retrieval.query`). Intent precedence: TEMPORAL_ORDER > HISTORY >
AS_OF (explicit time, "back when", or `as_of`) > CURRENT (cue words or present-tense auxiliary)
> OPEN. A month without a year means its most recent occurrence before the question date. The
asker defaults to the conversation's first speaker; first-person questions are about the asker.

**Fusion.** One RRF ranking over items keyed `(kind, id)`, so memories and episodes compete
directly. 1-hop expansion uses the named entities plus the speaker that "I"/"you" resolves to.

**Temporal filter.** AS_OF uses interval *overlap* with the question's span. `valid_to` is
exclusive (a superseding fact starts on it). A `valid_from` with SESSION / UNKNOWN / BEFORE
precision is treated as an unknown start (no lower bound). CURRENT keeps ACTIVE and CONFLICTED
facts with an open `valid_to`; every other intent drops only RETRACTED. Episodes are never filtered.

**Packing.** Memory lines are written `- (memory <id>) fact | status, attribute | interval |
sources: …` so they cannot be confused with turn lines `[turn_uid | date | speaker] text`.
Version chains are shown for HISTORY, TEMPORAL_ORDER and AS_OF; conflict partners always.

**Generation.** The generator takes `(question, context, allowed_turn_uids)` plus optional hints
`asker` and `question_type` (`CURRENT`, `AS_OF lo..hi`, …) that baselines may omit.

**Verification.** Citations are normalized before checking: a string starting with a turn id → that
turn; a memory id → moved to the memory citations; a pasted `[…] text` line whose text equals
exactly one allowed turn → that turn. Each repair is logged. The NLI premise is the cited turns as
`On <date>, <speaker> said: <text>`. If any claim was removed, `final_answer` becomes the
concatenation of the surviving claims.

**Question date.** Defaults to the latest session timestamp of the conversation.
