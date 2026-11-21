from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from movierec.index import ItemIndex, benchmark
from movierec.metrics import coverage, evaluate, paired_bootstrap, recall_ndcg, tail_share, topk_excluding


def test_topk_excludes_seen():
    scores = np.array([[5.0, 4, 3, 2, 1]])
    seen = sp.csr_matrix(np.array([[1, 0, 1, 0, 0]], dtype=np.float32))
    assert topk_excluding(scores, seen, 2).tolist() == [[1, 3]]


def test_recall_ndcg_by_hand():
    topk = np.array([[3, 1, 7]])
    rec, ndcg = recall_ndcg(topk, [np.array([1, 9])], 3)
    assert rec[0] == pytest.approx(0.5)
    assert ndcg[0] == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)))
    rec, ndcg = recall_ndcg(np.array([[1, 9, 0]]), [np.array([1, 9])], 3)
    assert rec[0] == 1 and ndcg[0] == pytest.approx(1)


def test_evaluate_cold_items_coverage_tail():
    scores = np.tile(np.arange(6, 0, -1, dtype=float), (2, 1))  # item 0 best
    seen = sp.csr_matrix((2, 6), dtype=np.float32)
    res = evaluate(lambda u: scores[u], {0: np.array([0, 5]), 1: np.array([1])}, seen, [2],
                   cold_items=np.array([5]))
    assert res.recall[2].tolist() == [0.5, 1.0]
    assert res.hits_cold_items[2] == (0, 1)
    assert coverage(res.topk, 6, 2) == pytest.approx(2 / 6)
    assert tail_share(res.topk, np.array([0]), 2) == pytest.approx(0.5)


def test_bootstrap_sign():
    rng = np.random.default_rng(0)
    a = 0.3 + 0.05 * rng.normal(size=200)
    r = paired_bootstrap(a, a - 0.2, 200)
    assert r["diff"] == pytest.approx(0.2) and r["ci_low"] > 0 and r["p_le_0"] == 0


def test_numpy_index_matches_bruteforce_and_excludes():
    rng = np.random.default_rng(0)
    items, q = rng.normal(size=(50, 8)), rng.normal(size=(3, 8))
    idx = ItemIndex(items, "numpy")
    _, ids = idx.search(q, 5)
    assert ids.tolist() == np.argsort(-(q @ items.T), axis=1)[:, :5].tolist()
    top = [i for i, _ in idx.recommend(q[0], 5, exclude={int(ids[0, 0])})]
    assert int(ids[0, 0]) not in top and len(top) == 5


def test_faiss_flat_equals_exact():
    pytest.importorskip("faiss")
    rng = np.random.default_rng(1)
    items, q = rng.normal(size=(200, 16)).astype(np.float32), rng.normal(size=(20, 16)).astype(np.float32)
    rows = benchmark(items, q, 10, 16, [64], n_latency=5)
    flat = next(r for r in rows if r["index"] == "flat")
    assert flat["recall_vs_exact"] == pytest.approx(1.0)
