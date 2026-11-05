# chronos-graph

ChronosGraph is a local, persistent, **bi-temporal memory** for multi-session conversational
agents, with a reproducible evaluation on LoCoMo and LongMemEval.

The architecture, schema and write rules are in [SPEC.md](SPEC.md).

## Status

- Part 1: spec, scaffold, storage layer, model clients, dataset loaders, eval manifests.
- Part 2: ingestion engine (ontology, temporal resolver, extractor, entity resolution, state
  resolver, pipeline), 30 synthetic probes, restart/idempotence tests, fact audit tool.

No benchmark results have been produced yet.

## Ingestion

```bash
uv run python -m chronos.cli ingest data/synthetic_probes/update_01.json --db /tmp/x.sqlite
uv run python -m chronos.cli ingest data/demo_sessions/demo.json --scripted --embedder hash  # offline
uv run python -m eval.probes --mode live      # 30 probes against the real model, per-case report
uv run python scripts/audit_facts.py --ingest # dev conversation -> unlabelled CSV for hand audit
```

## Quick start

```bash
make setup        # uv sync, create .env from .env.example, download datasets
make test         # offline unit tests (live tests excluded)
make test-live    # optional: tests that need Ollama / Groq / Cerebras / HF models
make manifests    # write eval/manifests/{dev_v1,test_v1,quick_v1}.json
```

Requirements: macOS on Apple Silicon, Python 3.11, [uv](https://docs.astral.sh/uv/),
[Ollama](https://ollama.com). Run `bash scripts/setup_ollama.sh` once: it pulls
`qwen2.5:7b-instruct-q4_K_M` and creates the alias `chronos-qwen2.5-7b-32k` with
`num_ctx 32768` baked in, because Ollama's OpenAI-compatible endpoint may ignore a
per-request `num_ctx`.

## Data notes

- Sources are pinned by revision in `scripts/download_data.sh`; sha256 in `data/raw/CHECKSUMS`
  and in every manifest. LongMemEval is the original `longmemeval_s` (not the later
  `longmemeval-cleaned` release).
- LoCoMo evidence ids: multi-id strings and typos are repaired; 3 ids cannot be resolved
  (`D10:19` and `D4:36` point past the end of their session, one is a bare `"D"`). They are
  reported by the loader, not dropped silently.
- LoCoMo `conv-26` lists dates for sessions 20-35 that have no content; they are skipped.
- LongMemEval haystacks are in order by day but not by time of day, so `session_ordinal` is
  the haystack position, not a timestamp sort.
- `reduce_haystack` counts tokens with the Qwen2.5 tokenizer.

## Layout

```
src/chronos/      config, ontology, storage, llm, ingestion, retrieval, agent, cli
eval/             dataset loaders, manifests, (later) runners and judging
tests/            pytest suite
data/raw/         downloaded datasets (git-ignored, checksums tracked)
data/demo_sessions/, data/synthetic_probes/
artifacts/        databases, LLM response cache, reports
```
