# chronos-graph

ChronosGraph is a local, persistent, **bi-temporal memory** for multi-session conversational
agents, with a reproducible evaluation on LoCoMo and LongMemEval.

The architecture, schema and write rules are in [SPEC.md](SPEC.md).

## Status

Part 1 of 4: spec, scaffold, storage layer, model clients, dataset loaders and eval manifests.
No benchmark results have been produced yet.

## Quick start

```bash
make setup        # uv sync, create .env from .env.example, download datasets
make test         # offline unit tests (live tests excluded)
make test-live    # optional: tests that need Ollama / Groq / Cerebras / HF models
make manifests    # write eval/manifests/{dev_v1,test_v1,quick_v1}.json
```

Requirements: macOS on Apple Silicon, Python 3.11, [uv](https://docs.astral.sh/uv/),
[Ollama](https://ollama.com) with `qwen2.5:7b-instruct-q4_K_M` pulled.

## Layout

```
src/chronos/      config, ontology, storage, llm, ingestion, retrieval, agent, cli
eval/             dataset loaders, manifests, (later) runners and judging
tests/            pytest suite
data/raw/         downloaded datasets (git-ignored, checksums tracked)
data/demo_sessions/, data/synthetic_probes/
artifacts/        databases, LLM response cache, reports
```
