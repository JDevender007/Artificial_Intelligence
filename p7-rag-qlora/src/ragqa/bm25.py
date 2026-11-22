"""Okapi BM25 over a sparse document-term matrix (no extra dependency)."""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import sparse

from ragqa.text import tokenize


class BM25:
    """score(q, d) = sum over query terms t of idf(t) * tf(t,d) (k1+1) / (tf(t,d) + k1 (1 - b + b |d| / avgdl)).

    idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5)) (the Lucene form, always positive). A term repeated in the
    query counts once per occurrence.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vocab: dict[str, int] = {}
        self.W: sparse.csc_matrix | None = None

    def fit(self, docs: list[str]) -> BM25:
        rows, cols, vals, lengths = [], [], [], []
        for i, d in enumerate(docs):
            toks = tokenize(d)
            lengths.append(len(toks))
            for t, c in Counter(toks).items():
                rows.append(i)
                cols.append(self.vocab.setdefault(t, len(self.vocab)))
                vals.append(c)
        n = len(docs)
        tf = sparse.csr_matrix((np.asarray(vals, float), (rows, cols)), shape=(n, len(self.vocab)))
        dl = np.asarray(lengths, float)
        avgdl = dl.mean() if n and dl.mean() > 0 else 1.0
        df = np.bincount(np.asarray(cols, int), minlength=len(self.vocab)).astype(float)
        self.idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
        tf = tf.tocoo()
        denom = tf.data + self.k1 * (1 - self.b + self.b * dl[tf.row] / avgdl)
        w = self.idf[tf.col] * tf.data * (self.k1 + 1) / denom
        self.W = sparse.csc_matrix((w, (tf.row, tf.col)), shape=(n, len(self.vocab)))
        self.n_docs = n
        return self

    def scores(self, query: str) -> np.ndarray:
        assert self.W is not None, "call fit() first"
        counts = Counter(t for t in tokenize(query) if t in self.vocab)
        if not counts:
            return np.zeros(self.n_docs)
        idx = np.fromiter((self.vocab[t] for t in counts), int)
        qv = np.fromiter(counts.values(), float)
        return np.asarray(self.W[:, idx] @ qv).ravel()

    def search(self, query: str, k: int) -> list[int]:
        """Top-k document indices, best first (ties broken by lower index)."""
        s = self.scores(query)
        k = min(k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        return sorted(top.tolist(), key=lambda i: (-s[i], i))
