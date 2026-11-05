"""Closed single-valued attribute ontology (SPEC §3.1)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

ONTOLOGY_PATH = Path(__file__).with_name("attributes.yaml")

_SLOT_RE = re.compile(r"^<([a-z_]+)>$")
_SLOT_VALUE_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")


class OntologyError(ValueError):
    pass


@dataclass(frozen=True)
class AttributeSpec:
    key: str  # "residence" or the template "favorite:<category>"
    description: str
    paraphrases: tuple[str, ...]
    not_examples: tuple[str, ...]

    @property
    def is_template(self) -> bool:
        return ":" in self.key

    @property
    def base(self) -> str:
        return self.key.split(":", 1)[0]

    @property
    def slot_name(self) -> str | None:
        return _SLOT_RE.match(self.key.split(":", 1)[1]).group(1) if self.is_template else None


class Ontology:
    def __init__(self, specs: list[AttributeSpec]) -> None:
        self.specs = {s.key: s for s in specs}
        self._by_base = {s.base: s for s in specs}
        if len(self._by_base) != len(self.specs):
            raise OntologyError("two keys share the same base name")

    def __iter__(self):
        return iter(self.specs.values())

    def __len__(self) -> int:
        return len(self.specs)

    def canonicalize(self, key: str | None) -> str | None:
        """Return the canonical attribute key, or None if `key` is not in the ontology.

        "residence" -> "residence"; "Favorite:Book" -> "favorite:book"; a bare template
        ("favorite"), an unfilled template ("favorite:<category>") or an unknown key -> None.
        """
        if not key or not isinstance(key, str):
            return None
        raw = key.strip()
        base, sep, slot = raw.partition(":")
        base = base.strip().lower()
        spec = self._by_base.get(base)
        if spec is None:
            return None
        if not spec.is_template:
            return None if sep else spec.key
        if not sep:
            return None
        slot = re.sub(r"[^a-z0-9]+", "_", slot.strip().lower()).strip("_")
        if not _SLOT_VALUE_RE.match(slot) or slot == spec.slot_name:
            return None
        return f"{base}:{slot}"

    def spec_for(self, canonical_key: str) -> AttributeSpec:
        return self._by_base[canonical_key.split(":", 1)[0]]

    def render_for_prompt(self) -> str:
        """Compact text block listing every key for the extractor prompt (deterministic)."""
        lines = []
        for s in self:
            lines.append(f"- {s.key}: {s.description}")
            lines.append(f"    maps from: {'; '.join(s.paraphrases)}")
            lines.append(f"    NOT this key: {'; '.join(s.not_examples)}")
        return "\n".join(lines)


def parse_ontology(data: dict) -> Ontology:
    if not isinstance(data, dict) or not data:
        raise OntologyError("ontology must be a non-empty mapping")
    specs = []
    for key, body in data.items():
        base, sep, slot = str(key).partition(":")
        if not re.fullmatch(r"[a-z_]+", base) or (sep and not _SLOT_RE.match(slot)):
            raise OntologyError(f"bad key {key!r}")
        for field in ("description", "paraphrases", "not"):
            if field not in body:
                raise OntologyError(f"{key}: missing {field!r}")
        desc = str(body["description"]).strip()
        if not desc or "\n" in desc:
            raise OntologyError(f"{key}: description must be one non-empty line")
        para, nots = body["paraphrases"], body["not"]
        if not isinstance(para, list) or not 3 <= len(para) <= 5:
            raise OntologyError(f"{key}: needs 3-5 paraphrases, got {para!r}")
        if not isinstance(nots, list) or not nots:
            raise OntologyError(f"{key}: needs at least one 'not' example")
        specs.append(AttributeSpec(str(key), desc, tuple(map(str, para)), tuple(map(str, nots))))
    return Ontology(specs)


@lru_cache(maxsize=1)
def load_ontology(path: str | Path = ONTOLOGY_PATH) -> Ontology:
    return parse_ontology(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
