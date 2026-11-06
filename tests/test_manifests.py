import json
from pathlib import Path

import pytest

from eval.make_manifests import build_manifests, recorded_checksums

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFESTS = ROOT / "eval" / "manifests"


@pytest.fixture(scope="module")
def built():
    if not (RAW / "longmemeval_s.json").exists() or not (RAW / "locomo10.json").exists():
        pytest.skip("datasets missing (run `make data`)")
    return build_manifests(RAW, 42)


def ids(m):
    return set(m["locomo"]["question_ids"]), set(m["longmemeval"]["question_ids"])


def test_committed_manifests_match_regeneration(built):
    for name, m in built.items():
        on_disk = json.loads((MANIFESTS / f"{name}.json").read_text())
        assert on_disk == m, f"{name}.json is stale: run `make manifests`"


def test_manifest_contents(built, stats):
    dev, test, quick = built["dev_v1"], built["test_v1"], built["quick_v1"]
    sums = recorded_checksums(RAW)
    for m in built.values():
        assert {k: v["sha256"] for k, v in m["datasets"].items()} == sums
        assert m["seed"] == 42

    assert dev["locomo"]["conversations"] == ["conv-26"]
    assert len(dev["longmemeval"]["question_ids"]) == 20

    assert test["counts"]["locomo_by_category"] == {
        c: 50 for c in ("adversarial", "multi-hop", "open-domain", "single-hop", "temporal")
    }
    assert "conv-26" not in test["locomo"]["conversations"]
    assert len(test["locomo"]["conversations"]) == 9

    d_lo, d_lme = ids(dev)
    t_lo, t_lme = ids(test)
    q_lo, q_lme = ids(quick)
    assert not (d_lo & t_lo) and not (d_lme & t_lme)
    assert q_lo <= t_lo and q_lme <= t_lme
    assert len(quick["locomo"]["conversations"]) == 1
    assert quick["counts"]["longmemeval_by_type"] == {"knowledge-update": 20}

    # quick50 needs no ingestion beyond quick_v1: same conversation, same histories
    q50 = built["quick50"]
    assert q50["locomo"]["conversations"] == quick["locomo"]["conversations"]
    assert q50["longmemeval"]["question_ids"] == quick["longmemeval"]["question_ids"]
    assert q50["counts"]["locomo_by_category"] == {
        c: 6 for c in ("adversarial", "multi-hop", "open-domain", "single-hop", "temporal")
    }
    assert not (set(q50["locomo"]["question_ids"]) & d_lo)

    for m in built.values():
        c = m["counts"]
        stats.append(
            f"manifest {m['name']}: locomo {c['locomo_total']} {m['locomo']['conversations'] if len(m['locomo']['conversations']) == 1 else ''}"
            f"{c['locomo_by_category']}; longmemeval {c['longmemeval_total']} "
            f"{c['longmemeval_by_type']} (abstention {c['longmemeval_abstention']})"
        )
