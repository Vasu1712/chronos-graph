"""ScriptedLLM: replays hand-written extractor and classifier outputs (offline tests, demos).

`scripted` block of a conversation file:
  {"extraction": [{"session": 1, "turn_index": 2, "subject": ..., <ExtractedFact fields>}],
   "relations": {"<fact_text>": {"relation": "UPDATES", "target": "<substring of candidate>"}}}

Extraction calls return the scripted facts whose (turn_index, turn text) appear among the
prompt's TARGET turns. Classification calls look up the new fact's text; unknown facts are NEW.
These are *scripted* outputs of an ideal model, not recordings of a real one.
"""

from __future__ import annotations

import re
from typing import Any

from chronos.ingestion.extractor import ExtractionOutput
from chronos.ingestion.resolver import RelationOut
from chronos.ingestion.types import Conversation

_TARGET_LINE = re.compile(r"^\[t(\d+)\] [^:]+: (.*)$")
_NEW_FACT = re.compile(r"^NEW fact: (.*?) \[", re.MULTILINE)
_CAND = re.compile(r"^(C\d+): (.*?) \[", re.MULTILINE)


class ScriptedLLM:
    def __init__(self, conversation: Conversation, scripted: dict[str, Any]) -> None:
        self.facts = []
        texts = {
            (s.ordinal, t.turn_index): t.display_text
            for s in conversation.sessions
            for t in s.turns
        }
        for f in scripted.get("extraction", []):
            f = dict(f)
            f.setdefault("correction", None)  # older scripted files predate this field
            session = f.pop("session")
            self.facts.append((f["turn_index"], texts[(session, f["turn_index"])], f))
        self.relations: dict[str, dict[str, Any]] = scripted.get("relations", {})
        self.extraction_calls = 0
        self.classification_calls = 0

    def structured(self, messages: list[dict[str, str]], response_model: type[Any]) -> Any:
        user = messages[-1]["content"]
        if response_model is ExtractionOutput:
            self.extraction_calls += 1
            targets = {}
            for line in user.split("TARGET turns:", 1)[1].splitlines():
                if m := _TARGET_LINE.match(line):
                    targets[int(m.group(1))] = m.group(2)
            facts = [f for idx, text, f in self.facts if targets.get(idx) == text]
            return ExtractionOutput.model_validate({"facts": facts})
        if response_model is RelationOut:
            self.classification_calls += 1
            new_text = _NEW_FACT.search(user).group(1)
            spec = self.relations.get(new_text, {"relation": "NEW"})
            label = None
            if spec["relation"] != "NEW":
                cands = _CAND.findall(user)
                want = spec.get("target")
                label = next(
                    (lab for lab, text in cands if want and want.lower() in text.lower()),
                    cands[0][0],
                )
            return RelationOut(reason="scripted", relation=spec["relation"], target=label)
        raise TypeError(f"ScriptedLLM cannot answer {response_model.__name__}")
