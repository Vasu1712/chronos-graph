"""Synthetic probes: offline with scripted outputs (resolver logic) and live with the real model."""

import copy
import urllib.request

import pytest

from chronos.ingestion.files import conversation_from_dict
from chronos.ingestion.scripted import ScriptedLLM
from chronos.llm.embed import HashingEmbedder
from chronos.storage.db import Database
from eval.probes import check, load_probes, run_probe

PROBES = load_probes()
IDS = [p["probe_id"] for p in PROBES]


def test_thirty_probes_five_per_case():
    from collections import Counter

    assert len(PROBES) == 30
    assert set(Counter(p["case"] for p in PROBES).values()) == {5}


@pytest.mark.parametrize("probe", PROBES, ids=IDS)
def test_probe_scripted(probe, tmp_path):
    r = run_probe(probe, llm=ScriptedLLM(conversation_from_dict(probe), probe["scripted"]),
                  embedder=HashingEmbedder(), db_path=tmp_path / "p.sqlite")  # fmt: skip
    assert r["passed"], "\n".join(r["failures"] + r["memories"])


def test_checker_detects_wrong_expectations(tmp_path):
    """Guard against a vacuous checker: corrupt each kind of expectation and expect failures."""
    probe = next(p for p in PROBES if p["probe_id"] == "late_mention_01")
    run_probe(probe, llm=ScriptedLLM(conversation_from_dict(probe), probe["scripted"]),
              embedder=HashingEmbedder(), db_path=tmp_path / "c.sqlite")  # fmt: skip
    with Database(tmp_path / "c.sqlite") as db:
        bad = copy.deepcopy(probe)
        bad["expected"]["memories"][0]["status"] = "SUPERSEDED"
        assert any("status" in f for f in check(db.conn, bad))
        bad = copy.deepcopy(probe)
        bad["expected"]["memories"][1]["valid_to"] = None
        assert any("valid_to" in f for f in check(db.conn, bad))
        bad = copy.deepcopy(probe)
        bad["expected"]["current"][0]["contains"] = ["Delhi"]
        assert any(f.startswith("current") for f in check(db.conn, bad))
        bad = copy.deepcopy(probe)
        bad["expected"]["absent"] = [
            {"subject": "Priya", "contains": "Bengaluru", "attribute_key": "residence"}
        ]
        assert any("must not be current" in f for f in check(db.conn, bad))
        bad = copy.deepcopy(probe)
        bad["expected"]["memories"].append({"subject": "Priya", "contains": "Kolkata"})
        assert any("not found" in f for f in check(db.conn, bad))


def _ollama_up() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def live_models():
    if not _ollama_up():
        pytest.skip("Ollama is not running")
    from chronos.config import Paths
    from chronos.llm.cache import ResponseCache
    from chronos.llm.embed import Embedder
    from chronos.llm.local_client import LocalLLM

    return LocalLLM(cache=ResponseCache(Paths().llm_cache)), Embedder()


@pytest.mark.live
@pytest.mark.parametrize("probe", PROBES, ids=IDS)
def test_probe_live(probe, tmp_path, live_models, stats):
    llm, emb = live_models
    r = run_probe(probe, llm=llm, embedder=emb, db_path=tmp_path / "live.sqlite")
    stats.append(
        f"live {'PASS' if r['passed'] else 'FAIL'} {probe['probe_id']}: {'; '.join(r['failures'])}"
    )
    assert r["passed"], "\n".join(r["failures"] + r["memories"])
