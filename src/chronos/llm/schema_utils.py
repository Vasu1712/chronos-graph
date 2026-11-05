"""Helpers for turning Pydantic models into JSON schemas that constrained decoders accept."""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve local `$ref`s into `$defs` so grammar-based decoders get a flat schema."""
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {}) | schema.pop("definitions", {})

    def resolve(node: Any, stack: tuple[str, ...]) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith(("#/$defs/", "#/definitions/")):
                name = ref.rsplit("/", 1)[-1]
                if name in stack:
                    raise ValueError(f"recursive schema via {name} cannot be inlined")
                merged = {**defs[name], **{k: v for k, v in node.items() if k != "$ref"}}
                return resolve(merged, (*stack, name))
            return {k: resolve(v, stack) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(v, stack) for v in node]
        return node

    return resolve(schema, ())


def json_schema_for(model: type[BaseModel]) -> dict[str, Any]:
    return inline_refs(model.model_json_schema())
