"""Runtime entailment with cross-encoder/nli-deberta-v3-base (MPS).

Used by the verifier (SPEC §6.9): does the concatenation of cited turns entail the claim?
Note: the cross-encoder truncates at 512 tokens; callers should keep premises short.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from chronos.config import ModelConfig
from chronos.llm.embed import resolve_device

# Label order documented on the model card; overridden by the model's own id2label if present.
DEFAULT_LABELS = ("contradiction", "entailment", "neutral")


@dataclass(frozen=True)
class NLIScore:
    entailment: float
    neutral: float
    contradiction: float

    @property
    def label(self) -> str:
        return max(
            ("entailment", "neutral", "contradiction"), key=lambda k: getattr(self, k)
        ).upper()


class NLI:
    def __init__(
        self,
        config: ModelConfig | None = None,
        model: Any | None = None,
        labels: Sequence[str] | None = None,
    ) -> None:
        self.config = config or ModelConfig()
        self._model = model
        self._labels = tuple(labels) if labels else None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.config.nli_model, device=resolve_device(self.config.device), max_length=512
            )
        return self._model

    @property
    def labels(self) -> tuple[str, ...]:
        if self._labels is None:
            id2label = getattr(
                getattr(getattr(self.model, "model", None), "config", None), "id2label", None
            )
            if id2label:
                self._labels = tuple(id2label[i].lower() for i in sorted(id2label))
            else:
                self._labels = DEFAULT_LABELS
        return self._labels

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScore]:
        """Score (premise, hypothesis) pairs in batches; returns softmax probabilities."""
        if not pairs:
            return []
        logits = np.asarray(
            self.model.predict(
                [list(p) for p in pairs],
                batch_size=self.config.nli_batch_size,
                show_progress_bar=False,
            ),
            dtype=np.float64,
        )
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        idx = {name: self.labels.index(name) for name in ("entailment", "neutral", "contradiction")}
        return [
            NLIScore(
                entailment=float(row[idx["entailment"]]),
                neutral=float(row[idx["neutral"]]),
                contradiction=float(row[idx["contradiction"]]),
            )
            for row in probs
        ]

    def entails(self, premise: str, hypothesis: str, threshold: float = 0.5) -> bool:
        return self.score([(premise, hypothesis)])[0].entailment >= threshold
