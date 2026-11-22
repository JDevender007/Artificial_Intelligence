"""Sparse, dense, hybrid (reciprocal rank fusion) and reranked retrieval over chunks.

The embedder and reranker are protocols so tests can use the tiny deterministic fakes defined here;
the Hugging Face implementations live in :mod:`ragqa.hf`.
"""

from __future__ import annotations

import hashlib
from typing import Literal, Protocol

import numpy as np

from ragqa.bm25 import BM25
from ragqa.chunking import Chunk
from ragqa.text import tokenize


class Embedder(Protocol):
    def encode(self, texts: list[str], kind: Literal["query", "doc"]) -> np.ndarray:
        """L2-normalised float32 embeddings, one row per text."""
        ...


class Reranker(Protocol):
    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Relevance score per (query, passage) pair; higher is more relevant."""
        ...


class HashEmbedder:
    """Bag-of-words hashing embedder: a deterministic stand-in for tests (no model download)."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts: list[str], kind: Literal["query", "doc"]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n)


class OverlapReranker:
    """Query-token overlap reranker: a deterministic stand-in for tests."""

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        return np.array([len(set(tokenize(q)) & set(tokenize(p))) for q, p in pairs], float)


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal rank fusion: score(d) = sum_r 1 / (k + rank_r(d)), ranks starting at 1. Ties: first seen."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for r, d in enumerate(ranking, start=1):
            score[d] = score.get(d, 0.0) + 1.0 / (k + r)
    return sorted(score, key=lambda d: -score[d])  # sorted() is stable, so ties keep insertion order


def top_k_rows(S: np.ndarray, k: int) -> list[list[int]]:
    """Top-k column indices per row of a score matrix, best first."""
    k = min(k, S.shape[1])
    part = np.argpartition(-S, k - 1, axis=1)[:, :k]
    order = np.take_along_axis(S, part, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(part, order, axis=1).tolist()


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None, reranker: Reranker | None = None,
                 n_candidates: int = 20, rrf_k: int = 60):
        self.chunks = chunks
        self.texts = [c.index_text for c in chunks]
        self.bm25 = BM25().fit(self.texts)
        self.embedder, self.reranker = embedder, reranker
        self.n_candidates, self.rrf_k = n_candidates, rrf_k
        self._doc_emb: np.ndarray | None = None

    def doc_embeddings(self) -> np.ndarray:
        if self._doc_emb is None:
            if self.embedder is None:
                raise ValueError("dense retrieval needs an embedder")
            self._doc_emb = self.embedder.encode(self.texts, "doc")
        return self._doc_emb

    def _bm25(self, queries: list[str], k: int) -> list[list[int]]:
        return [self.bm25.search(q, k) for q in queries]

    def _dense(self, queries: list[str], k: int) -> list[list[int]]:
        D = self.doc_embeddings()
        assert self.embedder is not None
        Q = self.embedder.encode(queries, "query")
        return top_k_rows(Q @ D.T, k)

    def search(self, queries: list[str], method: str, k: int) -> list[list[int]]:
        """Top-k chunk ids per query for ``method`` in bm25 | dense | hybrid | hybrid_rerank."""
        if method == "bm25":
            return self._bm25(queries, k)
        if method == "dense":
            return self._dense(queries, k)
        if method not in ("hybrid", "hybrid_rerank"):
            raise ValueError(method)
        n = max(self.n_candidates, k)
        fused = [rrf([a, b], self.rrf_k)[:n] for a, b in zip(self._bm25(queries, n), self._dense(queries, n),
                                                            strict=True)]
        if method == "hybrid":
            return [f[:k] for f in fused]
        if self.reranker is None:
            raise ValueError("hybrid_rerank needs a reranker")
        pairs = [(q, self.texts[c]) for q, f in zip(queries, fused, strict=True) for c in f]
        scores = self.reranker.score(pairs)
        out, pos = [], 0
        for f in fused:
            s = scores[pos:pos + len(f)]
            pos += len(f)
            order = sorted(range(len(f)), key=lambda i: (-s[i], i))
            out.append([f[i] for i in order[:k]])
        return out
