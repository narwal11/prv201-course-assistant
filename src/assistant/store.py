"""Vector stores. ChromaStore is the real one (local, persistent, metadata filters).
InMemoryStore has the same interface and is used in tests."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from .embeddings import Embedder
from .types import Chunk, Retrieved


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk]) -> None: ...
    def query(self, text: str, k: int, lecture: int | None = None) -> list[Retrieved]: ...
    def count(self) -> int: ...


def _chunk_from(id_: str, text: str, meta: dict) -> Chunk:
    lec = meta.get("lecture", -1)
    return Chunk(id=id_, text=text, source=meta["source"], page=int(meta["page"]),
                 lecture=None if lec in (-1, None) else int(lec), kind=meta.get("kind", "other"))


class ChromaStore:
    def __init__(self, path: Path, collection: str, embedder: Embedder):
        import chromadb

        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._name = collection
        self._embedder = embedder
        self._col = self._get()

    def _get(self):
        # Cosine space so that similarity = 1 - distance is in [-1, 1].
        return self._client.get_or_create_collection(
            self._name, metadata={"hnsw:space": "cosine", "embedder": self._embedder.name})

    def reset(self) -> None:
        try:
            self._client.delete_collection(self._name)
        except Exception:
            pass
        self._col = self._get()

    def add(self, chunks: list[Chunk], batch: int = 256) -> None:
        stored = (self._col.metadata or {}).get("embedder")
        if stored and stored != self._embedder.name:
            raise RuntimeError(f"Collection was built with {stored}, not {self._embedder.name}; re-run with --reset")
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            vecs = self._embedder.embed([c.text for c in part])
            self._col.upsert(ids=[c.id for c in part], documents=[c.text for c in part],
                             embeddings=vecs.tolist(), metadatas=[c.meta() for c in part])

    def query(self, text: str, k: int, lecture: int | None = None) -> list[Retrieved]:
        if self._col.count() == 0:
            return []
        vec = self._embedder.embed([text])[0].tolist()
        res = self._col.query(query_embeddings=[vec], n_results=k,
                              where={"lecture": lecture} if lecture is not None else None,
                              include=["documents", "metadatas", "distances"])
        out = []
        for id_, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append(Retrieved(_chunk_from(id_, doc, meta), similarity=1.0 - float(dist)))
        return out

    def count(self) -> int:
        return self._col.count()


class InMemoryStore:
    def __init__(self, embedder: Embedder):
        self._embedder = embedder
        self._chunks: list[Chunk] = []
        self._vecs: np.ndarray | None = None

    def add(self, chunks: list[Chunk]) -> None:
        vecs = self._embedder.embed([c.text for c in chunks])
        self._chunks.extend(chunks)
        self._vecs = vecs if self._vecs is None else np.vstack([self._vecs, vecs])

    def query(self, text: str, k: int, lecture: int | None = None) -> list[Retrieved]:
        if self._vecs is None:
            return []
        q = self._embedder.embed([text])[0]
        sims = self._vecs @ q
        order = np.argsort(-sims)
        out = []
        for i in order:
            c = self._chunks[i]
            if lecture is not None and c.lecture != lecture:
                continue
            out.append(Retrieved(c, float(sims[i])))
            if len(out) == k:
                break
        return out

    def count(self) -> int:
        return len(self._chunks)
