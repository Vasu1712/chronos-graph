"""Where do answerable questions get lost? A seconds-long look at the raw outputs of one manifest.

    uv run python -m eval.diagnose --manifest quick50

Per system, over answerable questions only (gold not marked unanswerable):
  declined        the final answer is an abstention
  at gate / model / verifier   which stage declined (the gate is ChronosGraph's only)
  gold text shown  a gold evidence turn's text is in the context (its "[turn_uid |" line), not just
                   its id in a "sources:" list
  declined|shown   declined although a gold turn's text was shown: the generator's fault, not retrieval's
  forced retries   forced-answer retries that rescued a decline (needs runs after that change)
Then a few of ChronosGraph's declines with the gold turn shown, with the model's stated reason.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from eval.common import artifacts


def load(manifest: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path in sorted((artifacts() / "raw_outputs" / manifest).glob("*.jsonl")):
        rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
        out[path.stem] = [r for r in rows if "skipped" not in r]
    return out


def gold_shown(r: dict) -> bool:
    text = (r.get("context") or {}).get("text") or ""
    return any(f"[{uid} |" in text for uid in r.get("gold_evidence_turn_uids") or [])


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:5.1f}% ({a}/{b})" if b else "    - "


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", default="quick50")
    ap.add_argument("--examples", type=int, default=6)
    args = ap.parse_args(argv)
    runs = load(args.manifest)
    if not runs:
        raise SystemExit(f"no raw outputs under {artifacts() / 'raw_outputs' / args.manifest}")
    print(f"{'system':<24}{'answerable':>10}  {'declined':<16}{'gate/model/verifier':<22}"
          f"{'gold text shown':<18}{'declined|shown':<16}forced")  # fmt: skip
    for system, rows in runs.items():
        ans = [r for r in rows if not r.get("gold_unanswerable")]
        dec = [r for r in ans if r["abstained"]]
        stages = Counter(r.get("abstained_at") for r in dec)
        shown = [r for r in ans if gold_shown(r)]
        dec_shown = [r for r in shown if r["abstained"]]
        forced = sum((r.get("verifier") or {}).get("forced_answer_retries", 0) for r in ans)
        print(f"{system:<24}{len(ans):>10}  {pct(len(dec), len(ans)):<16}"
              f"{stages.get('gate', 0)}/{stages.get('model', 0)}/{stages.get('verifier', 0):<18}"
              f"{pct(len(shown), len(ans)):<18}{pct(len(dec_shown), len(shown)):<16}{forced}")  # fmt: skip
    main_run = next((v for k, v in runs.items() if k == "chronos"), None)
    if main_run:
        bad = [
            r
            for r in main_run
            if not r.get("gold_unanswerable") and r["abstained"] and gold_shown(r)
        ]
        print(f"\nChronosGraph declines with a gold turn shown: {len(bad)}")
        for r in bad[: args.examples]:
            print(f"- [{r.get('abstained_at')}] {r['question'][:90]}\n    gold: {str(r['gold_answer'])[:70]}"
                  f"\n    reason: {(r.get('abstention_reason') or '')[:140]}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
