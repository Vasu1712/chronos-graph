# ChronosGraph

A local, persistent, **bi-temporal memory** for multi-session conversational agents, with a reproducible
evaluation on LoCoMo and LongMemEval. It runs entirely on a laptop: a seven-billion-parameter model through
Ollama, small embedding and entailment models on the Apple GPU, SQLite for storage.

- [DESIGN.md](DESIGN.md): architecture, bi-temporal rules with examples, provenance, related work, failure modes.
- [EVAL.md](EVAL.md): setup, every table, ablations, judge agreement, error analysis, limitations.
- [SPEC.md](SPEC.md): the specification every part of the code follows.

Results below come from: **quick50**: 568 answers from 5 systems, 568 scored, 50 questions in the manifest; **quick_v1**: 96 answers from 2 systems, 96 scored, 48 questions in the manifest; synthetic state probes: scripted.

## Setup (macOS, Apple silicon)

Requirements: Python 3.11 with [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com), and about
sixteen gigabytes of unified memory.

```bash
make setup                         # uv sync, create .env, download and checksum the datasets
bash scripts/setup_ollama.sh       # pull the model and create the 32k-context alias
make test                          # offline tests
```

The judge needs `GROQ_API_KEY` and/or `CEREBRAS_API_KEY` in `.env` (evaluation only; the memory itself uses
no external service). With one key the judge runs on that provider alone and the calibration step reports
that it cannot be measured.

## Use

```bash
uv run python -m chronos.cli ingest data/demo_sessions/user_story.json --db artifacts/demo.sqlite
uv run python -m chronos.cli ask "Where do I live now?" --db artifacts/demo.sqlite
uv run python -m chronos.cli inspect <memory_id or query_id> --db artifacts/demo.sqlite
uv run python -m chronos.cli history Meera --attribute residence --db artifacts/demo.sqlite
uv run python -m chronos.cli stats --db artifacts/demo.sqlite
```

## Evaluate

```bash
make eval-quick    # ingest + answer (ChronosGraph and B3) + judge + report on the quick manifest
make eval-50       # 50 questions (same ingestion as quick), every system and ablation + judge + report
make eval-full     # all systems, all ablations, calibration and claim re-check; schedule overnight
make report        # rebuild every table, figure and document from artifacts/; no API call
uv run python -m eval.diagnose --manifest quick50   # seconds: where answerable questions are lost
```

Every step is resumable: ingestion skips sessions already ingested (one transaction per session, so a kill
leaves no partial session), answering skips answered questions, and every model call is cached. Long runs
are wrapped in `caffeinate -i` by the Makefile.

### Rebuilding every table from the committed outputs

`make report` needs only `artifacts/`: the raw outputs, judgments, ingest logs, step timings, calibration and
claim re-check files, and the synthetic probe results. It makes no model call and no API call, and rewrites
`artifacts/summary_metrics.json`, `docs/figures/`, `README.md`, `EVAL.md` and `DESIGN.md`. To re-answer
questions from the cached model calls instead, rerun the `eval-` targets: with a warm cache they do no
generation. The cache itself (`artifacts/llm_cache.sqlite`) is too large for git and stays on the machine
that ran the evaluation; `make report` does not need it.

## Layout

```
src/chronos/      storage, ingestion, retrieval, agent, llm clients, cli
eval/             loaders, manifests, systems, judge, metrics, ablations, report
docs/templates/   sources of README.md, EVAL.md and DESIGN.md
data/             datasets (checksummed), demo sessions, synthetic probes
artifacts/        databases, caches, raw outputs, judgments, summary_metrics.json
tests/            offline tests (live tests are marked and opt-in)
```
