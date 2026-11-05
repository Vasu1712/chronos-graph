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

# The targets below are wired to the CLI; their implementations land in Parts 2-4.
ingest-demo:
	$(UV) run chronos ingest-demo

eval-quick:
	$(UV) run chronos eval --manifest eval/manifests/quick_v1.json

eval-full:
	$(UV) run chronos eval --manifest eval/manifests/test_v1.json

report:
	$(UV) run chronos report
