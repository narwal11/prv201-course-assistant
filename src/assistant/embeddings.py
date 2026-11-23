"""Embedding back ends. Production uses all-MiniLM-L6-v2 (Reimers & Gurevych 2019).
HashingEmbedder is a dependency-light stand-in used only by unit tests and the
offline smoke test; it is lexical, not semantic, so never use it for evaluation."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class MiniLMEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, batch_size=64, normalize_embeddings=True,
                                  show_progress_bar=len(texts) > 200)
        return np.asarray(vecs, dtype=np.float32)


class HashingEmbedder:
    """Deterministic bag-of-words hashing vectors (TEST ONLY)."""

    def __init__(self, dim: int = 2048):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.name = f"hashing-{dim}"
        self._vec = HashingVectorizer(n_features=dim, alternate_sign=False, norm="l2",
                                      stop_words="english", ngram_range=(1, 2))

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._vec.transform(texts).toarray().astype(np.float32)


def get_embedder(name: str) -> Embedder:
    if name.startswith("hashing"):
        return HashingEmbedder()
    return MiniLMEmbedder(name)
