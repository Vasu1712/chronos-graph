"""Ablation configs (eval/configs/*.yaml) -> runnable system specs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from eval.common import MANIFEST_DIR  # noqa: F401  (re-exported for callers)
from eval.loaders.types import QAItem

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
LETTERS = {"A": "retrieval", "B": "state", "C": "gate", "D": "budget"}
POLICY_DBS = ("append_only", "arrival_order")


@dataclass
class Variant:
    name: str
    system_id: str  # the id under which results are stored, e.g. "A_retrieval/dense_only"
    system: str | None = None  # base system to build (None for an alias)
    alias_of: str | None = None  # results of another system id are reused as this variant
    spec: dict[str, Any] = field(default_factory=dict)  # db / retrieval / agent / budget


@dataclass
class Ablation:
    id: str
    title: str
    question: str
    filter: dict[str, Any]
    variants: list[Variant]
    probes: bool = False
    reference: str = ""  # variant every other variant is compared against (the full system)

    def selects(self, qa: QAItem) -> bool:
        f = self.filter or {}
        if f.get("dataset") and qa.dataset != f["dataset"]:
            return False
        return not (f.get("categories") and qa.category not in f["categories"])

    @property
    def aliases(self) -> set[str]:
        return {v.alias_of for v in self.variants if v.alias_of}

    @property
    def db_variants(self) -> set[str]:
        return {v.spec["db"] for v in self.variants if v.spec.get("db")}


def load_ablations(config_dir: Path = CONFIG_DIR) -> dict[str, Ablation]:
    out: dict[str, Ablation] = {}
    for path in sorted(config_dir.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text())
        variants = []
        for name, body in raw["variants"].items():
            body = dict(body or {})
            variants.append(Variant(
                name=name, system_id=f"{raw['id']}/{name}", system=body.pop("system", None),
                alias_of=body.pop("alias_of", None), spec=body,
            ))  # fmt: skip
        out[path.stem] = Ablation(raw["id"], raw["title"], raw.get("question", ""), raw.get("filter") or {},
                                  variants, bool(raw.get("probes", False)), str(raw.get("reference", "")))  # fmt: skip
    return out


def resolve_selection(spec: str, available: dict[str, Ablation]) -> list[Ablation]:
    """'A,C' / 'retrieval,gate' / 'all' / '' -> ablations in letter order."""
    if not spec:
        return []
    if spec == "all":
        return [available[k] for k in LETTERS.values() if k in available]
    picked = []
    for token in spec.split(","):
        key = LETTERS.get(token.strip().upper(), token.strip())
        if key not in available:
            raise SystemExit(
                f"unknown ablation {token!r}; have {sorted(available)} / letters {sorted(LETTERS)}"
            )
        picked.append(available[key])
    return picked
