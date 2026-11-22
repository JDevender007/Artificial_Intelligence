"""Image-generation metrics on pre-extracted features (numpy only, so they are unit-tested without a GPU).

* FID: Fréchet distance between Gaussian fits of Inception-v3 pool features (2048-d).
* KID: unbiased MMD² with the cubic polynomial kernel, averaged over random subsets (unbiased at small n, unlike FID).
* CLIP score: 100 · cos(image, caption), clipped at 0 (the torchmetrics definition), per image.
* Species retrieval: does the image's CLIP embedding rank its own species name in the top-k among all test species?
* Memorization: max cosine between each generated image and every training image (CLIP image space).
"""

from __future__ import annotations

import numpy as np


def _stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    return x.mean(axis=0), np.cov(x, rowvar=False)


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    """||mu_a - mu_b||² + Tr(S_a + S_b - 2 (S_a S_b)^½); the trace of the root uses the eigenvalues of S_a S_b."""
    mu1, s1 = _stats(a)
    mu2, s2 = _stats(b)
    eig = np.linalg.eigvals(s1 @ s2)
    tr_sqrt = np.sqrt(np.clip(eig.real, 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * tr_sqrt)


def kid(a: np.ndarray, b: np.ndarray, subset_size: int = 100, n_subsets: int = 100, seed: int = 0) -> tuple[float, float]:
    """Kernel Inception Distance (mean, std over subsets), kernel k(x, y) = (x·y / d + 1)³."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    m = min(subset_size, len(a), len(b))
    d = a.shape[1]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_subsets):
        x = a[rng.choice(len(a), m, replace=False)]
        y = b[rng.choice(len(b), m, replace=False)]
        kxx, kyy, kxy = ((x @ x.T / d + 1) ** 3, (y @ y.T / d + 1) ** 3, (x @ y.T / d + 1) ** 3)
        mmd = ((kxx.sum() - np.trace(kxx)) + (kyy.sum() - np.trace(kyy))) / (m * (m - 1)) - 2 * kxy.mean()
        vals.append(mmd)
    return float(np.mean(vals)), float(np.std(vals))


def l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)


def clip_scores(img_emb: np.ndarray, txt_emb: np.ndarray) -> np.ndarray:
    """Row-wise 100 · max(cos, 0) for matched (image, caption) pairs."""
    return 100 * np.clip((l2n(img_emb) * l2n(txt_emb)).sum(axis=1), 0, None)


def retrieval_hits(img_emb: np.ndarray, class_emb: np.ndarray, labels: np.ndarray, ks: tuple[int, ...] = (1, 5)
                   ) -> dict[int, np.ndarray]:
    """Per-image top-k hit (0/1) of the true class among all classes, ranked by cosine similarity."""
    sims = l2n(img_emb) @ l2n(class_emb).T
    true = sims[np.arange(len(labels)), labels]
    rank = (sims > true[:, None]).sum(axis=1)  # 0 = best; ties count in the image's favour
    return {k: (rank < k).astype(float) for k in ks}


def max_similarity(query: np.ndarray, reference: np.ndarray, chunk: int = 512) -> np.ndarray:
    """For each query row, the highest cosine similarity to any reference row."""
    r = l2n(reference)
    q = l2n(query)
    return np.concatenate([(q[i:i + chunk] @ r.T).max(axis=1) for i in range(0, len(q), chunk)])


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> dict[str, float]:
    """Mean of (a - b) over paired items with a 95% percentile CI from resampling items; p = P(diff <= 0)."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "p_le_0": float((boots <= 0).mean()),
            "n_items": len(d)}
