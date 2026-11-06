"""Write eval/manifests/{dev_v1,test_v1,quick_v1}.json (seed 42).

dev_v1   LoCoMo conversation index 0 (all its QA)
         + 20 LongMemEval questions stratified over knowledge-update (7),
           temporal-reasoning (7) and abstention (6). KU/TR strata exclude abstention items.
test_v1  50 QA per LoCoMo category sampled from the remaining 9 conversations
         + every LongMemEval knowledge-update or abstention question not in dev.
quick_v1 One LoCoMo test conversation's share of the test sample
         + 20 knowledge-update questions from the test set. quick is a strict subset of test.
quick50  The same LoCoMo conversation and the same 20 knowledge-update histories as quick_v1 (so
         nothing new is ingested), with LoCoMo rebalanced to 6 questions per category: quick_v1's
         questions are kept (trimmed by seed where a category has more than 6) and topped up from
         the rest of that conversation. 30 LoCoMo + 20 LongMemEval = 50. Not a subset of test_v1.

Each stratum uses its own RNG, random.Random(f"{seed}:{stratum}"), so changing one stratum
never reshuffles another. Output contains no timestamps: re-running yields identical files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from chronos.config import SEED, Paths
from eval.loaders.locomo import CATEGORY_MAP, load_locomo
from eval.loaders.longmemeval import TOKENIZER_REPO, load_longmemeval

LOCOMO_FILE = "locomo10.json"
LME_FILE = "longmemeval_s.json"
LOCOMO_PER_CATEGORY = 50
DEV_LME_STRATA = {"knowledge-update": 7, "temporal-reasoning": 7, "abstention": 6}
QUICK_KU = 20
QUICK50_PER_CATEGORY = 6
HAYSTACK = {"max_tokens": 24000, "seed": SEED, "tokenizer": TOKENIZER_REPO}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def recorded_checksums(raw: Path) -> dict[str, str]:
    sums = {}
    for line in (raw / "CHECKSUMS").read_text().splitlines():
        if line.strip():
            digest, name = line.split()
            sums[name] = digest
    return sums


def verify_checksums(raw: Path) -> dict[str, str]:
    recorded = recorded_checksums(raw)
    out = {}
    for name in (LOCOMO_FILE, LME_FILE):
        got = sha256_file(raw / name)
        if recorded.get(name) != got:
            raise SystemExit(
                f"checksum mismatch for {name}: recorded {recorded.get(name)}, got {got}"
            )
        out[name] = got
    return out


def _rng(seed: int, stratum: str) -> random.Random:
    return random.Random(f"{seed}:{stratum}")


def build_manifests(raw: Path, seed: int = SEED) -> dict[str, dict]:
    checksums = verify_checksums(raw)
    convs, qas, _ = load_locomo(raw / LOCOMO_FILE)
    lme = load_longmemeval(raw / LME_FILE)

    conv_ids = [c.conversation_id for c in convs]  # file order
    dev_conv, test_convs = conv_ids[0], conv_ids[1:]
    lme_by_id = {q.question_id: q for q in lme}
    qa_by_id = {q.question_id: q for q in qas}

    # ---------------- dev
    dev_locomo = [q.question_id for q in qas if q.conversation_id == dev_conv]
    pools = {
        "knowledge-update": [
            q.question_id
            for q in lme
            if q.question_type == "knowledge-update" and not q.qa.is_abstention
        ],
        "temporal-reasoning": [
            q.question_id
            for q in lme
            if q.question_type == "temporal-reasoning" and not q.qa.is_abstention
        ],
        "abstention": [q.question_id for q in lme if q.qa.is_abstention],
    }
    dev_lme: list[str] = []
    for stratum, n in DEV_LME_STRATA.items():
        dev_lme += sorted(_rng(seed, f"dev:lme:{stratum}").sample(sorted(pools[stratum]), n))

    # ---------------- test
    test_locomo: list[str] = []
    for cat_id in sorted(CATEGORY_MAP):
        cat = CATEGORY_MAP[cat_id]
        pool = sorted(
            q.question_id for q in qas if q.category == cat and q.conversation_id in test_convs
        )
        if len(pool) < LOCOMO_PER_CATEGORY:
            raise SystemExit(f"only {len(pool)} {cat} questions available")
        test_locomo += _rng(seed, f"test:locomo:{cat}").sample(pool, LOCOMO_PER_CATEGORY)
    test_locomo = _locomo_sorted(test_locomo)
    dev_lme_set = set(dev_lme)
    test_lme = [
        q.question_id
        for q in lme
        if (q.question_type == "knowledge-update" or q.qa.is_abstention)
        and q.question_id not in dev_lme_set
    ]

    # ---------------- quick (subset of test)
    quick_conv = _rng(seed, "quick:locomo:conversation").choice(test_convs)
    quick_locomo = [qid for qid in test_locomo if qa_by_id[qid].conversation_id == quick_conv]
    test_ku = sorted(qid for qid in test_lme if lme_by_id[qid].question_type == "knowledge-update")
    quick_lme = sorted(_rng(seed, "quick:lme:knowledge-update").sample(test_ku, QUICK_KU))
    quick_lme = [qid for qid in test_lme if qid in set(quick_lme)]  # dataset order

    quick50_locomo = quick50_select(qas, quick_conv, quick_locomo, seed)

    def manifest(name: str, description: str, locomo_q: list[str], lme_q: list[str]) -> dict:
        return {
            "name": name,
            "description": description,
            "seed": seed,
            "generator": "eval/make_manifests.py",
            "datasets": {
                LOCOMO_FILE: {"sha256": checksums[LOCOMO_FILE]},
                LME_FILE: {"sha256": checksums[LME_FILE]},
            },
            "longmemeval_haystack": HAYSTACK,
            "locomo": {
                "conversations": sorted(
                    {qa_by_id[q].conversation_id for q in locomo_q}, key=conv_ids.index
                ),
                "question_ids": locomo_q,
            },
            "longmemeval": {"question_ids": lme_q},
            "counts": {
                "locomo_total": len(locomo_q),
                "locomo_by_category": _ordered_counts(qa_by_id[q].category for q in locomo_q),
                "longmemeval_total": len(lme_q),
                "longmemeval_by_type": _ordered_counts(lme_by_id[q].question_type for q in lme_q),
                "longmemeval_abstention": sum(lme_by_id[q].qa.is_abstention for q in lme_q),
            },
        }

    return {
        "dev_v1": manifest(
            "dev_v1",
            f"LoCoMo {dev_conv} (all QA) + 20 LongMemEval questions stratified over "
            "knowledge-update / temporal-reasoning / abstention",
            dev_locomo,
            [q for q in (x.question_id for x in lme) if q in dev_lme_set],
        ),
        "test_v1": manifest(
            "test_v1",
            f"{LOCOMO_PER_CATEGORY} QA per LoCoMo category from {len(test_convs)} held-out conversations "
            "+ all LongMemEval knowledge-update and abstention questions not in dev_v1",
            test_locomo,
            test_lme,
        ),
        "quick_v1": manifest(
            "quick_v1",
            f"test_v1 LoCoMo sample for {quick_conv} + {QUICK_KU} test_v1 knowledge-update questions",
            quick_locomo,
            quick_lme,
        ),
        "quick50": manifest(
            "quick50",
            f"quick_v1 rebalanced: {QUICK50_PER_CATEGORY} LoCoMo questions per category from "
            f"{quick_conv} + the same {QUICK_KU} knowledge-update questions (no new ingestion)",
            quick50_locomo,
            quick_lme,
        ),
    }


def quick50_select(
    qas, conv: str, base: list[str], seed: int, per: int = QUICK50_PER_CATEGORY
) -> list[str]:
    """`per` questions per LoCoMo category from `conv`: keep `base` (trimmed by seed), top up by seed."""
    cat_of = {q.question_id: q.category for q in qas}
    out: list[str] = []
    for cat_id in sorted(CATEGORY_MAP):
        cat = CATEGORY_MAP[cat_id]
        have = sorted(q for q in base if cat_of.get(q) == cat)
        if len(have) > per:
            have = sorted(_rng(seed, f"quick50:trim:{cat}").sample(have, per))
        pool = sorted(
            q.question_id
            for q in qas
            if q.conversation_id == conv and q.category == cat and q.question_id not in have
        )
        need = max(0, per - len(have))
        out += have + _rng(seed, f"quick50:add:{cat}").sample(pool, min(need, len(pool)))
    return _locomo_sorted(out)


def _locomo_sorted(qids: list[str]) -> list[str]:
    def key(q: str) -> tuple[int, int]:
        _, conv, idx = q.split(":")
        return int(conv.split("-")[1]), int(idx[1:])

    return sorted(qids, key=key)


def _ordered_counts(values) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def write_manifests(manifests: dict[str, dict], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, m in manifests.items():
        p = out_dir / f"{name}.json"
        p.write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
        paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw", type=Path, default=Paths().data_raw)
    ap.add_argument("--out", type=Path, default=Paths().manifests)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    manifests = build_manifests(args.raw, args.seed)
    for p in write_manifests(manifests, args.out):
        c = manifests[p.stem]["counts"]
        print(f"{p}: locomo {c['locomo_total']} {c['locomo_by_category']}; "
              f"longmemeval {c['longmemeval_total']} {c['longmemeval_by_type']} "
              f"(abstention {c['longmemeval_abstention']})")  # fmt: skip


if __name__ == "__main__":
    main()
