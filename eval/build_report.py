"""Build the report from artifacts/ alone: no model, no network, no API call.

  uv run python -m eval.build_report            # = make report

Reads artifacts/ (raw outputs, judgments, ingest logs, step timings, calibration, claim re-check,
state probes) and writes
  artifacts/summary_metrics.json    every number, once
  docs/figures/*.png                accuracy by category (CI whiskers), accuracy vs token budget
  README.md, EVAL.md, DESIGN.md     docs/templates/*.tmpl.md with every {{placeholder}} filled

Nothing in the three documents is typed by hand: prose lives in the templates, every figure in them is
a placeholder filled from the summary. tests/test_docs_numbers.py rejects a number in a template.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from chronos.config import (
    REPO_ROOT,
    AgentConfig,
    LocalLLMConfig,
    ModelConfig,
    RetrievalConfig,
    judge_providers,
)
from eval import report_tables as T
from eval.ablations import Ablation, load_ablations
from eval.common import (
    MANIFEST_DIR,
    artifacts,
    judgments_path,
    read_jsonl,
)
from eval.failure_modes import MODES, analyze_failures
from eval.make_manifests import HAYSTACK
from eval.metrics import (
    accuracy_cell,
    ci_cell,
    compare_systems,
    ingest_cost,
    load_rows,
    step_seconds,
    summarize_system,
)
from eval.stats import N_BOOT, bootstrap_ci, cohens_kappa
from eval.systems import MAIN_SYSTEMS, OPTIONAL_SYSTEMS

TEMPLATES = REPO_ROOT / "docs" / "templates"
FIGURES = REPO_ROOT / "docs" / "figures"
PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_.\-]+)\s*\}\}")
REFERENCE_SYSTEM = "chronos"
ORDER = list(MAIN_SYSTEMS) + list(OPTIONAL_SYSTEMS)


# ------------------------------------------------------------------------------ helpers
def sanitize(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [sanitize(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return o


def available_manifests() -> list[str]:
    base = artifacts() / "raw_outputs"
    names = (
        [p.name for p in base.iterdir() if p.is_dir() and any(p.glob("*.jsonl"))]
        if base.exists()
        else []
    )
    return sorted(names, key=lambda n: (n != "test_v1", n))  # the headline manifest first


def _sort_systems(ids: list[str]) -> list[str]:
    return sorted(ids, key=lambda s: (ORDER.index(s) if s in ORDER else len(ORDER), s))


# ------------------------------------------------------------------------------ collection
def collect_ablation(ab: Ablation, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    sel = lambda r: ab.selects(SimpleNamespace(dataset=r["dataset"], category=r["category"]))  # noqa: E731
    ref = next((v for v in ab.variants if v.name == ab.reference), None)
    if ref is None:
        return None
    ref_sys = ref.alias_of or ref.system_id
    ref_rows = [r for r in rows if r["system"] == ref_sys and sel(r)]
    variants = []
    for v in ab.variants:
        sid = v.alias_of or v.system_id
        rs = [r for r in rows if r["system"] == sid and sel(r)]
        if not rs:
            continue
        s = summarize_system(rs)
        budget = v.spec.get("budget") or RetrievalConfig().token_budget
        entry = {
            "name": v.name, "system": sid, "is_reference": v.name == ab.reference, "n": len(rs), "budget": budget,
            "accuracy": s["accuracy"], "gold_in_context": s["citation"]["gold_in_context"], "context_tokens": s["context_tokens"],
            "abstention": s["abstention"], "token_f1": s["token_f1"],
            "vs_reference": None if v.name == ab.reference else compare_systems(rows, sid, ref_sys, subset=sel),
            "gate": {
                "locomo_adversarial_abstained": ci_cell([float(r["abstained"]) for r in rs if r["dataset"] == "locomo" and r["unanswerable"]]),
                "longmemeval_abs_abstained": ci_cell([float(r["abstained"]) for r in rs if r["dataset"] == "longmemeval" and r["unanswerable"]]),
                "false_abstention": ci_cell([float(r["abstained"]) for r in rs if not r["unanswerable"]]),
            },
        }  # fmt: skip
        variants.append(entry)
    if not variants:
        return None
    pop = sorted({(r["dataset"], r["category"]) for r in ref_rows})
    return {
        "id": ab.id, "title": ab.title, "question": ab.question, "reference": ab.reference, "reference_system": ref_sys,
        "population": f"{len(ref_rows)} questions (" + ", ".join(f"{d}/{c}" for d, c in pop) + ")" if ref_rows else "reference variant not run",
        "variants": variants,
    }  # fmt: skip


def collect_recheck(manifest: str) -> dict[str, Any] | None:
    rows = read_jsonl(artifacts() / "claim_recheck" / f"{manifest}.jsonl")
    if not rows:
        return None
    nli = [bool(r["nli_supported"]) for r in rows]
    judge = [bool(r["judge_supported"]) for r in rows]
    agree = [float(a == b) for a, b in zip(nli, judge, strict=True)]
    mean, lo, hi = bootstrap_ci(agree)
    return {
        "n": len(rows), "agreement": mean, "lo": lo, "hi": hi,
        "kappa": None if len(set(nli)) < 2 or len(set(judge)) < 2 else cohens_kappa(nli, judge),
        "both_support": sum(a and b for a, b in zip(nli, judge, strict=True)),
        "nli_only": sum(a and not b for a, b in zip(nli, judge, strict=True)),
        "judge_only": sum(b and not a for a, b in zip(nli, judge, strict=True)),
        "both_reject": sum(not a and not b for a, b in zip(nli, judge, strict=True)),
    }  # fmt: skip


def collect_judge(manifest: str) -> dict[str, Any]:
    j = [r for r in read_jsonl(judgments_path(manifest)) if r.get("type") == "judgment"]
    methods: dict[str, int] = {}
    providers: dict[str, int] = {}
    for r in j:
        methods[r["method"]] = methods.get(r["method"], 0) + 1
        if r.get("provider"):
            providers[r["provider"]] = providers.get(r["provider"], 0) + 1
    cal_path = artifacts() / "judge_calibration.json"
    audit_path = artifacts() / "judge_audit.csv"
    audit = {"exists": audit_path.exists(), "rows": 0, "labelled": 0}
    if audit_path.exists():
        recs = list(csv.DictReader(audit_path.open(encoding="utf-8")))
        audit |= {
            "rows": len(recs),
            "labelled": sum(bool((r.get("human_correct?") or "").strip()) for r in recs),
        }
    return {"methods": methods, "providers": providers, "audit": audit,
            "calibration": json.loads(cal_path.read_text()) if cal_path.exists() else None}  # fmt: skip


def steps_with_ingest(name: str) -> dict[str, float]:
    """Recorded step times, with ingestion never below the sum of its per-session wall times: a run
    killed with SIGKILL cannot log its own segment, but every finished session logged its time."""
    steps = step_seconds(name)
    per_session = sum(c["seconds"] for c in ingest_cost(name)["conversations"].values())
    if per_session:
        steps["ingest"] = max(steps.get("ingest", 0.0), per_session)
    return steps


def collect_manifest(name: str, ablations: dict[str, Ablation]) -> dict[str, Any]:
    rows, skip_recs = load_rows(name)
    mpath = MANIFEST_DIR / f"{name}.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {"counts": {}}
    counts = manifest.get("counts", {})
    by_dc = {f"locomo/{c}": n for c, n in counts.get("locomo_by_category", {}).items()}
    by_dc |= {f"longmemeval/{c}": n for c, n in counts.get("longmemeval_by_type", {}).items()}
    main_ids = _sort_systems(sorted({r["system"] for r in rows if "/" not in r["system"]}))
    by_sys = {s: [r for r in rows if r["system"] == s] for s in main_ids}
    systems = {s: summarize_system(rs) for s, rs in by_sys.items()}
    by_category = {}
    for s, rs in by_sys.items():
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in rs:
            groups.setdefault(f"{r['dataset']}/{r['category']}", []).append(r)
        by_category[s] = {c: accuracy_cell(g) for c, g in sorted(groups.items())}
    pairwise = []
    if REFERENCE_SYSTEM in by_sys:
        for s in main_ids:
            if s != REFERENCE_SYSTEM and (c := compare_systems(rows, REFERENCE_SYSTEM, s)):
                pairwise.append(c)
    skips: dict[str, dict[str, int]] = {}
    for r in skip_recs:
        head = r["skipped"].split(":")[0]
        skips.setdefault(r["system"], {}).setdefault(head, 0)
        skips[r["system"]][head] += 1
    judgments = read_jsonl(judgments_path(name))
    abl = {}
    for key, ab in ablations.items():
        if (c := collect_ablation(ab, rows)) is not None:
            abl[key] = c
    return {
        "name": name, "manifest": {k: manifest.get(k) for k in ("name", "description", "seed")},
        "questions": {"total": counts.get("locomo_total", 0) + counts.get("longmemeval_total", 0), "by_dataset_category": by_dc,
                      "answers_scored": sum(1 for r in rows if r["correct"] is not None), "answers": len(rows)},
        "systems": systems, "by_category": by_category, "pairwise": pairwise, "skips": skips, "ablations": abl,
        "judge": collect_judge(name), "recheck": collect_recheck(name), "ingest": ingest_cost(name),
        "steps": steps_with_ingest(name), "failure_modes": analyze_failures(name, judgments) if rows else {},
    }  # fmt: skip


def collect(manifests: list[str], *, with_estimate: bool = True) -> dict[str, Any]:
    ablations = load_ablations()
    summary: dict[str, Any] = {"generated_by": "eval/build_report.py", "bootstrap_resamples": N_BOOT,
                               "manifests": {n: collect_manifest(n, ablations) for n in manifests}}  # fmt: skip
    summary["state_probes"] = {mode: json.loads(p.read_text()) for mode in ("live", "scripted")
                               if (p := artifacts() / f"state_probes_{mode}.json").exists()}  # fmt: skip
    summary["estimate"] = None
    if with_estimate and "quick_v1" in manifests and (MANIFEST_DIR / "test_v1.json").exists():
        try:
            from eval.estimate import estimate
            from eval.judge import build_pool

            try:
                providers = [p.config.name for p in build_pool().providers]
            except RuntimeError:
                providers = [p.name for p in judge_providers() if p.api_key]
            summary["estimate"] = estimate("quick_v1", "test_v1", only_providers=providers or None)
        except Exception as exc:  # the projection is a convenience: never block the report on it
            summary["estimate"] = {"error": f"{type(exc).__name__}: {exc}"}
    return summary


# ------------------------------------------------------------------------------ plots
def plot_category_accuracy(m: dict[str, Any], path: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cats = sorted({c for v in m["by_category"].values() for c in v})
    systems = [s for s in m["by_category"] if any(c in m["by_category"][s] for c in cats)]
    if not cats or not systems:
        return False
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(cats) + 3), 4.2))
    width = 0.8 / len(systems)
    colors = plt.get_cmap("tab10")
    for i, s in enumerate(systems):
        xs, ys, lo, hi = [], [], [], []
        for j, c in enumerate(cats):
            cl = m["by_category"][s].get(c)
            if cl and cl["mean"] is not None:
                xs.append(j + (i - (len(systems) - 1) / 2) * width)
                ys.append(100 * cl["mean"])
                lo.append(100 * (cl["mean"] - cl["lo"]))
                hi.append(100 * (cl["hi"] - cl["mean"]))
        ax.bar(
            xs,
            ys,
            width * 0.92,
            yerr=[lo, hi],
            capsize=2,
            label=T.name(s),
            color=colors(i),
            error_kw={"linewidth": 0.9},
        )
    ax.set_xticks(range(len(cats)), [c.replace("/", "\n") for c in cats], fontsize=8)
    ax.set_ylabel("accuracy (%), 95% bootstrap CI")
    ax.set_ylim(0, 105)
    ax.set_title(f"Accuracy by category, manifest {m['name']}")
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def plot_budget(m: dict[str, Any], path: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ab = m["ablations"].get("budget")
    if not ab or len(ab["variants"]) < 2:
        return False
    pts = sorted(
        (v["budget"], v["accuracy"]) for v in ab["variants"] if v["accuracy"]["mean"] is not None
    )
    if len(pts) < 2:
        return False
    xs = [p[0] for p in pts]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(
        xs, [100 * p[1]["mean"] for p in pts], marker="o", color="tab:blue", label="ChronosGraph"
    )
    ax.fill_between(
        xs,
        [100 * p[1]["lo"] for p in pts],
        [100 * p[1]["hi"] for p in pts],
        alpha=0.2,
        color="tab:blue",
    )
    ax.set_xlabel("context budget (tokens)")
    ax.set_ylabel("accuracy (%), 95% bootstrap CI")
    ax.set_title(f"Accuracy vs token budget, manifest {m['name']}")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


# ------------------------------------------------------------------------------ values for the templates
def spec_section(prefix: str) -> str:
    text = (REPO_ROOT / "SPEC.md").read_text(encoding="utf-8")
    m = re.search(rf"^## {re.escape(prefix)}.*?(?=^## |\Z)", text, re.S | re.M)
    if not m:
        raise KeyError(f"SPEC.md has no section starting with '## {prefix}'")
    body = re.sub(r"\n---\s*$", "", m.group(0).strip(), flags=re.M).strip()
    # embedded under a DESIGN.md subsection: demote SPEC's headings two levels
    return re.sub(r"^(#+) ", lambda h: "#" * (len(h.group(1)) + 2) + " ", body, flags=re.M)


def bitemporal_examples() -> str:
    """Real ingestions of the repository's scripted probes: what the write rules do, as the database sees it."""
    from chronos.cli import _memory_table
    from chronos.ingestion.files import conversation_from_dict
    from chronos.ingestion.pipeline import IngestionPipeline
    from chronos.ingestion.scripted import ScriptedLLM
    from chronos.llm.embed import HashingEmbedder
    from chronos.storage.db import Database

    cases = [
        (
            "update_01",
            "UPDATES: a later session moves the user; the old value is closed, not deleted",
        ),
        (
            "late_mention_01",
            "HISTORICAL: a late mention of an old fact never replaces the current one",
        ),
        (
            "correction_01",
            "CORRECTS: 'Pune, sorry, I meant Mumbai' retracts the old value (it was never true)",
        ),
        ("neg_01", "NEG polarity: 'I don't play guitar anymore' closes the matching fact"),
        (
            "out_of_order_01",
            "Event time vs arrival: the earlier move is told later and goes behind the current fact",
        ),
    ]
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for pid, title in cases:
            probe = json.loads(
                (REPO_ROOT / "data" / "synthetic_probes" / f"{pid}.json").read_text()
            )
            conv = conversation_from_dict(probe)
            db = Database(Path(tmp) / f"{pid}.sqlite")
            IngestionPipeline(
                db, ScriptedLLM(conv, probe["scripted"]), HashingEmbedder()
            ).ingest_conversation(conv)
            convo = "\n".join(
                f"  [{s['ts'][:10]}] {t['speaker']}: {t['text']}"
                for s in probe["sessions"]
                for t in s["turns"]
            )
            out.append(
                f"**{title}** (probe `{pid}`)\n\n```\n{convo}\n\n"
                + "\n".join(_memory_table(db, conv.conversation_id))
                + "\n```"
            )
            db.close()
    return "\n\n".join(out)


def python_version() -> str:
    """The Python the project pins, read from pyproject.toml."""
    m = re.search(
        r'requires-python\s*=\s*"[^"]*?(\d+\.\d+)', (REPO_ROOT / "pyproject.toml").read_text()
    )
    return m.group(1) if m else "?"


def models_table() -> str:
    llm, mc = LocalLLMConfig(), ModelConfig()
    judge = judge_providers()[0]
    rows = [
        [
            "system LLM",
            f"{llm.model} via Ollama (served as `{llm.served_model}`), temperature {llm.temperature:g}, context {llm.num_ctx} tokens",
        ],
        ["embeddings", f"{mc.embed_model} (sentence-transformers, {mc.device})"],
        [
            "entailment (runtime verifier)",
            f"{mc.nli_model}, threshold {AgentConfig().nli_threshold:g}",
        ],
        [
            "judge (evaluation only)",
            f"{judge.model}, pooled over "
            + " and ".join(p.name for p in judge_providers())
            + ", reasoning effort low",
        ],
        [
            "context budget",
            f"{RetrievalConfig().token_budget} tokens (Qwen2.5 tokenizer) unless a table says otherwise",
        ],
        ["bootstrap", f"{N_BOOT:,} resamples over questions, 95% percentile intervals, seed fixed"],
    ]
    return T.table(["component", "setting"], rows)


def per_manifest(summary: dict[str, Any], fn, *, level: str = "####") -> str:
    parts = []
    for name, m in summary["manifests"].items():
        parts.append(f"{level} {name}\n\n{fn(m)}")
    return (
        "\n\n".join(parts)
        if parts
        else T.not_measured("no manifest has been run yet; run `make eval-quick`")
    )


def headline(summary: dict[str, Any]) -> str:
    lines = []
    for name, m in summary["manifests"].items():
        c = m["systems"].get(REFERENCE_SYSTEM)
        if not c or c["accuracy"]["n"] == 0:
            lines.append(f"- **{name}**: no scored answers yet.")
            continue
        s = f"- **{name}**: ChronosGraph answers {T.cell(c['accuracy'])}."
        for p in m["pairwise"]:
            if p["b"] in ("B3_chunk_rag", "B4_turns_plus_notes"):
                s += f" Against {T.name(p['b'])} ({T.pct(p['acc_b'])}, n={p['n']} paired): {T.verdict(p, p['a'], p['b'])}"
                s += f" (difference {100 * p['mean_diff']:+.1f} points, 95% CI [{100 * p['lo']:+.1f}, {100 * p['hi']:+.1f}])."
        lines.append(s)
    return "\n".join(lines) if lines else T.not_measured("no scored answers yet")


def status_banner(summary: dict[str, Any]) -> str:
    bits = []
    for name, m in summary["manifests"].items():
        q = m["questions"]
        bits.append(
            f"**{name}**: {q['answers']} answers from {len(m['systems'])} systems, {q['answers_scored']} scored, {q['total']} questions in the manifest"
        )
        cal = m["judge"]["calibration"]
        if not cal or cal.get("status") != "measured":
            bits[-1] += "; judge calibration (Cohen's kappa) **not measured**"
    probes = summary["state_probes"]
    bits.append("synthetic state probes: " + (", ".join(sorted(probes)) if probes else "not run"))
    return "Results below come from: " + "; ".join(bits) + "."


def systems_table(summary: dict[str, Any]) -> str:
    present = {s for m in summary["manifests"].values() for s in m["systems"]}
    rows = [
        [T.name(s), T.SYSTEM_DESCRIPTIONS.get(s, ""), "run" if s in present else "not run yet"]
        for s in ORDER
    ]
    return T.table(["system", "what it is", "status"], rows)


def limitations_sizes(summary: dict[str, Any]) -> str:
    lines = []
    for name, m in summary["manifests"].items():
        ds = {}
        for k, n in m["questions"]["by_dataset_category"].items():
            ds[k.split("/")[0]] = ds.get(k.split("/")[0], 0) + n
        lines.append(
            f"- `{name}`: "
            + ", ".join(f"{n} {d} questions" for d, n in sorted(ds.items()))
            + f" ({m['questions']['total']} in all)"
        )
    return "\n".join(lines) if lines else T.not_measured("no manifest has been run yet")


def figures(summary: dict[str, Any], *, plots: bool) -> tuple[str, str]:
    cat, bud = [], []
    for name, m in summary["manifests"].items():
        if plots:
            if plot_category_accuracy(m, FIGURES / f"accuracy_by_category_{name}.png"):
                cat.append(
                    f"![Accuracy by category, {name}](docs/figures/accuracy_by_category_{name}.png)"
                )
            if plot_budget(m, FIGURES / f"accuracy_vs_budget_{name}.png"):
                bud.append(
                    f"![Accuracy vs token budget, {name}](docs/figures/accuracy_vs_budget_{name}.png)"
                )
    return "\n\n".join(cat) or T.not_measured("no scored answers to plot"), "\n\n".join(
        bud
    ) or T.not_measured("the budget ablation has not run")


def build_values(summary: dict[str, Any], *, plots: bool = True) -> dict[str, str]:
    M = lambda fn: per_manifest(summary, fn)  # noqa: E731
    fig_cat, fig_bud = figures(summary, plots=plots)

    def ab(key: str, render):
        def f(m):
            a = m["ablations"].get(key)
            return (
                render(a)
                if a
                else T.not_measured(f"the {key} ablation has not run on this manifest")
            )

        return M(f)

    probes = summary["state_probes"]
    probe_text = (
        "\n\n".join(f"{T.probe_table(p)}" for _, p in sorted(probes.items(), reverse=True))
        if probes
        else T.probe_table(None)
    )
    first = next(iter(summary["manifests"].values()), None)
    values = {
        "status_banner": status_banner(summary),
        "headline": headline(summary),
        "models_table": models_table(),
        "budget": str(RetrievalConfig().token_budget),
        "nli_threshold": f"{AgentConfig().nli_threshold:g}",
        "bootstrap_resamples": f"{N_BOOT:,}",
        "systems_table": systems_table(summary),
        "manifest_tables": M(T.manifest_table),
        "main_results": M(T.main_results),
        "category_results": M(T.category_results),
        "pairwise_results": M(T.pairwise_results),
        "figure_category": fig_cat,
        "figure_budget": fig_bud,
        "citation_results": M(T.citation_results),
        "cost_results": M(T.cost_results),
        "ingest_results": M(T.ingest_results),
        "skips_results": M(T.skips_results),
        "ablation_a": ab("retrieval", T.ablation_table),
        "ablation_b": ab("state", T.ablation_table),
        "ablation_b_probes": probe_text,
        "ablation_c": ab("gate", T.gate_table),
        "ablation_d": ab("budget", T.ablation_table),
        "judge_results": M(T.judge_results),
        "recheck_results": M(T.recheck_results),
        "error_analysis": M(
            lambda m: (
                T.failure_results(m["failure_modes"], MODES)
                if m["failure_modes"]
                else T.not_measured("no answers yet")
            )
        ),
        "limitations_sizes": limitations_sizes(summary),
        "runtime_table": M(T.runtime_table),
        "projection_table": T.projection_table(
            summary["estimate"]
            if summary["estimate"] and "error" not in summary["estimate"]
            else None
        )
        if summary["estimate"] and "error" not in summary["estimate"]
        else T.not_measured(
            (summary["estimate"] or {}).get(
                "error", "a measured quick run and the full manifest are both needed to project"
            )
        ),
        "bitemporal_examples": bitemporal_examples(),
        "spec_overview": spec_section("1."),
        "spec_ingestion": spec_section("3."),
        "spec_schema": spec_section("4."),
        "spec_write_rules": spec_section("5."),
        "spec_tier3": spec_section("6."),
        "ci_level": "95%",
        "python_version": python_version(),
        "haystack_tokens": f"{HAYSTACK['max_tokens']:,}",
        "seed": str(HAYSTACK["seed"]),
        "n_manifests": str(len(summary["manifests"])),
        "first_manifest": first["name"] if first else "none",
    }
    return values


def fill(template: str, values: dict[str, str]) -> str:
    missing = sorted({k for k in PLACEHOLDER.findall(template) if k not in values})
    if missing:
        raise KeyError(f"template uses unknown placeholders: {missing}")
    return PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


def build(out_dir: Path = REPO_ROOT, manifests: list[str] | None = None, *, plots: bool = True,
          with_estimate: bool = True, templates: Path = TEMPLATES) -> dict[str, Any]:  # fmt: skip
    manifests = manifests if manifests is not None else available_manifests()
    summary = sanitize(collect(manifests, with_estimate=with_estimate))
    artifacts().mkdir(parents=True, exist_ok=True)
    (artifacts() / "summary_metrics.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8"
    )
    values = build_values(summary, plots=plots)
    for tmpl in sorted(templates.glob("*.tmpl.md")):
        target = out_dir / tmpl.name.replace(".tmpl.md", ".md")
        target.write_text(fill(tmpl.read_text(encoding="utf-8"), values), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifests", help="comma list; default: every manifest with raw outputs")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-estimate", action="store_true")
    args = ap.parse_args(argv)
    manifests = [m for m in args.manifests.split(",") if m] if args.manifests else None
    summary = build(
        manifests=manifests, plots=not args.no_plots, with_estimate=not args.no_estimate
    )
    print(
        f"manifests: {list(summary['manifests'])}; wrote artifacts/summary_metrics.json, README.md, EVAL.md, DESIGN.md"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
