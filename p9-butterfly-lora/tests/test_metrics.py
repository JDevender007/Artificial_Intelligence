import numpy as np

from sdlora.metrics import clip_scores, frechet_distance, kid, max_similarity, paired_bootstrap, retrieval_hits


def test_fid_zero_for_identical_and_grows_with_shift():
    x = np.random.default_rng(0).normal(size=(300, 8))
    assert abs(frechet_distance(x, x)) < 1e-6
    assert frechet_distance(x, x + 1.0) > frechet_distance(x, x + 0.1) > 0


def test_kid_near_zero_same_distribution():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(200, 8)), rng.normal(size=(200, 8))
    m_same, _ = kid(a, b, subset_size=50, n_subsets=50)
    m_diff, _ = kid(a, b + 2.0, subset_size=50, n_subsets=50)
    assert abs(m_same) < 0.05 < m_diff


def test_clip_retrieval_memorisation():
    e = np.eye(3)
    assert np.allclose(clip_scores(e, e), 100) and np.allclose(clip_scores(e, -e), 0)
    hits = retrieval_hits(e, e, np.array([0, 1, 0]), ks=(1, 2))
    assert hits[1].tolist() == [1.0, 1.0, 0.0] and hits[2].tolist() == [1.0, 1.0, 1.0]
    assert np.allclose(max_similarity(e, np.vstack([e, e]), chunk=2), 1)


def test_paired_bootstrap():
    r = paired_bootstrap(np.ones(50) + 0.01 * np.arange(50), np.zeros(50), n=500)
    assert r["diff"] > 1 and r["ci_low"] > 0 and r["p_le_0"] == 0.0 and r["n_items"] == 50
