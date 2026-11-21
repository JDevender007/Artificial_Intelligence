"""Top-k ranking with seen-item exclusion, Recall/NDCG@k, coverage, popularity bias and a paired bootstrap."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

ScoreFn = Callable[[np.ndarray], np.ndarray]  # user indices -> (len, n_items) scores


@dataclass
class EvalResult:
    users: np.ndarray  # evaluated user indices
    topk: np.ndarray  # (n_users_eval, max_k) recommended items, best first
    recall: dict[int, np.ndarray]  # k -> per-user recall@k
    ndcg: dict[int, np.ndarray]
    hits_cold_items: dict[int, tuple[int, int]]  # k -> (hits, total) over (user, cold target item) pairs


def topk_excluding(scores: np.ndarray, seen: sp.csr_matrix, k: int) -> np.ndarray:
    """Indices of the ``k`` best unseen items per row (``seen`` has one row per score row)."""
    scores = np.array(scores, dtype=np.float32, copy=True)
    rows, cols = seen.nonzero()
    scores[rows, cols] = -np.inf
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.take_along_axis(scores, part, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(part, order, axis=1)


def recall_ndcg(topk: np.ndarray, targets: list[np.ndarray], k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-user recall@k = hits / |targets| and NDCG@k with the ideal DCG of min(|targets|, k) hits."""
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    rec = np.zeros(len(targets))
    ndcg = np.zeros(len(targets))
    for n, t in enumerate(targets):
        hit = np.isin(topk[n, :k], t)
        rec[n] = hit.sum() / len(t)
        ndcg[n] = (hit * discounts).sum() / discounts[: min(len(t), k)].sum()
    return rec, ndcg


def evaluate(score_fn: ScoreFn, targets: dict[int, np.ndarray], seen: sp.csr_matrix, ks: list[int],
             cold_items: np.ndarray | None = None, batch: int = 1024) -> EvalResult:
    """Rank all unseen items for every user in ``targets`` and score the top ``max(ks)``."""
    users = np.array(sorted(targets), dtype=np.int64)
    kmax = max(ks)
    tops = []
    for s in range(0, len(users), batch):
        u = users[s: s + batch]
        tops.append(topk_excluding(score_fn(u), seen[u], kmax))
    topk = np.concatenate(tops) if tops else np.zeros((0, kmax), dtype=np.int64)
    tl = [targets[int(u)] for u in users]
    recall, ndcg, cold = {}, {}, {}
    cold_set = set() if cold_items is None else set(cold_items.tolist())
    for k in ks:
        recall[k], ndcg[k] = recall_ndcg(topk, tl, k)
        hits = total = 0
        if cold_set:
            for n, t in enumerate(tl):
                ct = [i for i in t if i in cold_set]
                if ct:
                    total += len(ct)
                    hits += int(np.isin(ct, topk[n, :k]).sum())
        cold[k] = (hits, total)
    return EvalResult(users=users, topk=topk, recall=recall, ndcg=ndcg, hits_cold_items=cold)


def coverage(topk: np.ndarray, n_items: int, k: int) -> float:
    """Share of the catalogue that appears in at least one user's top-k."""
    return len(np.unique(topk[:, :k])) / n_items


def head_items(counts: np.ndarray, frac: float = 0.1) -> np.ndarray:
    """The ``frac`` most popular items by training positives."""
    n = max(1, int(round(frac * len(counts))))
    return np.argsort(-counts, kind="stable")[:n]


def tail_share(topk: np.ndarray, head: np.ndarray, k: int) -> float:
    """Share of recommended slots that are not head items (higher = less popularity bias)."""
    if topk.size == 0:
        return float("nan")
    return float(1 - np.isin(topk[:, :k], head).mean())


def summarize(res: EvalResult, ks: list[int], n_items: int, head: np.ndarray,
              mask: np.ndarray | None = None) -> dict[str, float]:
    """Mean metrics over users (optionally a boolean ``mask`` over ``res.users``)."""
    m = np.ones(len(res.users), dtype=bool) if mask is None else mask
    out: dict[str, float] = {"n_users": int(m.sum())}
    for k in ks:
        out[f"recall@{k}"] = float(res.recall[k][m].mean()) if m.any() else float("nan")
        out[f"ndcg@{k}"] = float(res.ndcg[k][m].mean()) if m.any() else float("nan")
        out[f"coverage@{k}"] = coverage(res.topk[m], n_items, k) if m.any() else float("nan")
        out[f"tail_share@{k}"] = tail_share(res.topk[m], head, k)
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 1000, seed: int = 0) -> dict[str, float]:
    """Mean of ``a - b`` over users with a 95% percentile bootstrap CI (users resampled with replacement)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "p_le_0": float((means <= 0).mean())}
