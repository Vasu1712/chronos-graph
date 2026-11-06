"""Shared plumbing for the evaluation scripts: artifact paths, manifest loading, step timing."""

from __future__ import annotations

import hashlib
import json
import pickle
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronos.config import REPO_ROOT, IngestionConfig, Paths
from chronos.ingestion.types import Conversation
from eval.loaders.locomo import load_locomo
from eval.loaders.longmemeval import load_longmemeval, reduce_haystack
from eval.loaders.types import QAItem

ARTIFACTS = REPO_ROOT / "artifacts"
MANIFEST_DIR = REPO_ROOT / "eval" / "manifests"
LME_ASKER = "user"  # in LongMemEval "I" is the user; every system is told so


def artifacts() -> Path:
    """The artifacts directory, looked up at call time so tests can redirect it."""
    return ARTIFACTS


def stop_on_sigterm() -> None:
    """Make SIGTERM (what `pkill` sends) unwind like Ctrl-C, so `finally` blocks run and the step's
    wall time is still recorded as an interrupted segment. SIGKILL cannot be caught."""

    def _raise(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, _raise)


def manifest_name(path: str | Path) -> str:
    return Path(path).stem


def memory_db_path(manifest: str, variant: str | None = None) -> Path:
    """artifacts/memory_<manifest>.sqlite, or ...__<variant>.sqlite for a replayed state policy."""
    return artifacts() / f"memory_{manifest}{'__' + variant if variant else ''}.sqlite"


def ingest_log_path(manifest: str, variant: str | None = None) -> Path:
    return artifacts() / "ingest_logs" / f"{manifest}{'__' + variant if variant else ''}.jsonl"


def raw_output_path(manifest: str, system: str) -> Path:
    return artifacts() / "raw_outputs" / manifest / f"{system.replace('/', '__')}.jsonl"


def judgments_path(manifest: str) -> Path:
    return artifacts() / "judgments" / f"{manifest}.jsonl"


def step_timings_path(manifest: str) -> Path:
    return artifacts() / "step_timings" / f"{manifest}.jsonl"


def llm_cache_path() -> Path:
    return Paths().llm_cache


def judge_cache_path() -> Path:
    return artifacts() / "judge_cache.sqlite"


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        f.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:  # a line cut short by a kill -9; the run will redo it
                continue
    return out


@contextmanager
def timed_step(manifest: str, step: str, **extra: Any):
    """Record the wall time of a pipeline step in artifacts/step_timings/<manifest>.jsonl.

    Yields a dict the caller may fill with counts (items, tokens, ...) to store with the step."""
    started = time.perf_counter()
    started_at = now_iso()
    status = "ok"
    info: dict[str, Any] = dict(extra)
    try:
        yield info
    except BaseException:
        status = "interrupted"
        raise
    finally:
        append_jsonl(step_timings_path(manifest), {
            "step": step, "seconds": time.perf_counter() - started, "started_at": started_at,
            "status": status, **info,
        })  # fmt: skip


# ------------------------------------------------------------------------------ manifest data
@dataclass
class ManifestData:
    name: str
    path: Path
    manifest: dict[str, Any]
    conversations: dict[str, Conversation] = field(default_factory=dict)  # LoCoMo first, then LME
    qas: list[QAItem] = field(default_factory=list)

    def qa_by_id(self) -> dict[str, QAItem]:
        return {q.question_id: q for q in self.qas}

    def conversation_of(self, qa: QAItem) -> Conversation:
        return self.conversations[qa.conversation_id]


def ingestion_config_for(conv: Conversation) -> IngestionConfig:
    """LongMemEval: extract from the user's turns only (assistant answers are 10x longer and carry
    no facts about the user) and clip prompt turns. LoCoMo and synthetic: SPEC defaults."""
    if conv.dataset == "longmemeval":
        return IngestionConfig(target_speakers=("user",), max_prompt_turn_chars=1500)
    return IngestionConfig()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest_data(
    path: str | Path, raw_dir: Path | None = None, *, use_cache: bool = True
) -> ManifestData:
    """Conversations and QA items named by a manifest. LongMemEval histories are reduced with the
    manifest's haystack parameters. The (slow) result is pickled under artifacts/manifest_cache."""
    path = Path(path)
    raw_dir = raw_dir or Paths().data_raw
    manifest = json.loads(path.read_text())
    name = manifest.get("name") or path.stem
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    cache = artifacts() / "manifest_cache" / f"{name}-{key}.pkl"
    if use_cache and cache.exists():
        return pickle.loads(cache.read_bytes())

    for fname, meta in manifest["datasets"].items():
        got = _sha256(raw_dir / fname)
        if got != meta["sha256"]:
            raise SystemExit(
                f"{fname}: sha256 {got} does not match the manifest ({meta['sha256']})"
            )

    data = ManifestData(name=name, path=path, manifest=manifest)
    lo_ids = set(manifest["locomo"]["question_ids"])
    if lo_ids:
        convs, qas, _ = load_locomo(raw_dir / "locomo10.json")
        wanted = set(manifest["locomo"]["conversations"])
        data.conversations |= {c.conversation_id: c for c in convs if c.conversation_id in wanted}
        data.qas += [q for q in qas if q.question_id in lo_ids]
    lme_ids = manifest["longmemeval"]["question_ids"]
    if lme_ids:
        hay = manifest["longmemeval_haystack"]
        by_id = {q.question_id: q for q in load_longmemeval(raw_dir / "longmemeval_s.json")}
        for qid in lme_ids:
            q = reduce_haystack(by_id[qid], max_tokens=hay["max_tokens"], seed=hay["seed"])
            data.conversations[qid] = q.history
            data.qas.append(q.qa)
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps(data))
    return data
