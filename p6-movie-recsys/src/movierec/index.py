"""Nearest-neighbour retrieval over item embeddings: FAISS (exact IndexFlatIP, approximate HNSW) or NumPy.

FAISS is imported lazily so the package and its tests work without it (``faiss-cpu`` is optional).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
    except ImportError:
        return False
    return True


class ItemIndex:
    """Inner-product search. ``kind``: "flat" (exact), "hnsw" (approximate) or "numpy" (exact, no FAISS)."""

    def __init__(self, item_vecs: np.ndarray, kind: str = "flat", hnsw_m: int = 32, ef_search: int = 64) -> None:
        self.vecs = np.ascontiguousarray(item_vecs, dtype=np.float32)
        self.kind = kind
        self.index: Any = None
        if kind == "numpy":
            return
        import faiss

        d = self.vecs.shape[1]
        if kind == "flat":
            self.index = faiss.IndexFlatIP(d)
        elif kind == "hnsw":
            self.index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = ef_search
        else:
            raise ValueError(kind)
        self.index.add(self.vecs)

    def set_ef(self, ef: int) -> None:
        if self.kind == "hnsw":
            self.index.hnsw.efSearch = ef

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.float32)
        k = min(k, len(self.vecs))
        if self.index is not None:
            scores, ids = self.index.search(q, k)
            return scores, ids.astype(np.int64)
        s = q @ self.vecs.T
        part = np.argpartition(-s, k - 1, axis=1)[:, :k]
        order = np.take_along_axis(s, part, axis=1).argsort(axis=1)[:, ::-1]
        ids = np.take_along_axis(part, order, axis=1)
        return np.take_along_axis(s, ids, axis=1), ids

    def recommend(self, query: np.ndarray, k: int, exclude: set[int] | None = None) -> list[tuple[int, float]]:
        """Top ``k`` items for one query vector, skipping ``exclude`` (asks the index for k + |exclude|)."""
        exclude = exclude or set()
        scores, ids = self.search(query, k + len(exclude))
        out = [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i >= 0 and int(i) not in exclude]
        return out[:k]


def benchmark(item_vecs: np.ndarray, queries: np.ndarray, k: int, hnsw_m: int, ef_values: list[int],
              n_latency: int = 1000) -> list[dict[str, float | str]]:
    """Recall of each index's raw top-k against exact search, and single-query latency (p50/p95 ms)."""
    kinds: list[tuple[str, int | None]] = [("numpy", None)]
    if faiss_available():
        kinds += [("flat", None)] + [("hnsw", ef) for ef in ef_values]
    exact = ItemIndex(item_vecs, "numpy").search(queries, k)[1]
    rows = []
    lat_q = queries[: min(n_latency, len(queries))]
    for kind, ef in kinds:
        t0 = time.perf_counter()
        idx = ItemIndex(item_vecs, kind, hnsw_m=hnsw_m, ef_search=ef or 64)
        build = time.perf_counter() - t0
        ids = idx.search(queries, k)[1]
        recall = np.mean([len(set(a) & set(b)) / k for a, b in zip(ids, exact, strict=True)])
        times = []
        for q in lat_q:
            t = time.perf_counter()
            idx.search(q[None, :], k)
            times.append((time.perf_counter() - t) * 1000)
        rows.append({"index": kind if ef is None else f"hnsw(ef={ef})", "recall_vs_exact": float(recall),
                     "p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)),
                     "build_s": build})
    return rows
