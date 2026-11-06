.PHONY: setup data test test-live lint fmt ingest-demo eval-quick eval-full report manifests

UV ?= uv

setup:
	$(UV) sync
	@test -f .env || cp .env.example .env
	$(MAKE) data

data:
	bash scripts/download_data.sh

manifests:
	$(UV) run python -m eval.make_manifests

test:
	$(UV) run pytest

test-live:
	$(UV) run pytest -m live

lint:
	$(UV) run ruff check src eval tests
	$(UV) run ruff format --check src eval tests

fmt:
	$(UV) run ruff check --fix src eval tests
	$(UV) run ruff format src eval tests

CAFFEINATE := $(shell command -v caffeinate >/dev/null 2>&1 && echo "caffeinate -i")
QUICK := eval/manifests/quick_v1.json
FULL := eval/manifests/test_v1.json

ingest-demo:
	$(UV) run chronos ingest-demo

# Quick tier: one LoCoMo conversation + 20 LongMemEval knowledge-update histories, ChronosGraph and B3.
eval-quick:
	$(CAFFEINATE) $(UV) run python -m eval.ingest_all --manifest $(QUICK)
	$(CAFFEINATE) $(UV) run python -m eval.run_benchmark --manifest $(QUICK) --systems chronos,B3_chunk_rag
	$(UV) run python -m eval.judge --manifest $(QUICK)
	$(UV) run python -m eval.build_report

# Full tier: every system, every ablation, live synthetic probes, calibration and claim re-check.
eval-full:
	$(CAFFEINATE) $(UV) run python -m eval.ingest_all --manifest $(FULL)
	$(CAFFEINATE) $(UV) run python -m eval.run_benchmark --manifest $(FULL) --systems all --ablations all --state-probes live
	$(UV) run python -m eval.judge --manifest $(FULL)
	$(UV) run python -m eval.build_report

# Rebuild every table, figure and document from artifacts/ (no model call, no API call).
report:
	$(UV) run python -m eval.build_report
