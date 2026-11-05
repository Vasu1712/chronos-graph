"""Dense embeddings with BAAI/bge-small-en-v1.5 (sentence-transformers, MPS)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from chronos.config import ModelConfig

# bge v1.5 recommends this instruction on the *query* side only for retrieval.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
EMBED_DIM = 384


def resolve_device(requested: str) -> str:
    if requested == "mps":
        import torch

        if not torch.backends.mps.is_available():
            return "cpu"
    return requested


class Embedder:
    def __init__(self, config: ModelConfig | None = None, model: Any | None = None) -> None:
        self.config = config or ModelConfig()
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.config.embed_model, device=resolve_device(self.config.device)
            )
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """L2-normalized float32 matrix of shape (len(texts), dim)."""
        if len(texts) == 0:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        vecs = self.model.encode(
            list(texts),
            batch_size=self.config.embed_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode([BGE_QUERY_INSTRUCTION + t for t in texts])


def to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def top_k(query: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Cosine top-k over normalized rows (in-memory NumPy vector index)."""
    if matrix.shape[0] == 0:
        return []
    scores = matrix @ query
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return [(int(i), float(scores[i])) for i in idx]


class HashingEmbedder:
    """Deterministic bag-of-words embedder for offline tests and scripted demos (not for eval)."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        import hashlib
        import re

        v = np.zeros(self.dim, dtype=np.float32)
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vec(t) for t in texts])

    encode_passages = encode
    encode_queries = encode
