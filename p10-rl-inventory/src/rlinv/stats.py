"""Seed-level statistics: t-intervals for learning-curve bands, and a hierarchical paired bootstrap."""

from __future__ import annotations

import numpy as np
from scipy import stats


def t_interval(x: np.ndarray, axis: int = 0, level: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and the (low, high) Student-t interval over ``axis``. With n=5 seeds the half-width is 2.78 * s / sqrt(5).
    A single seed gives a zero-width band."""
    x = np.asarray(x, dtype=float)
    n = x.shape[axis]
    m = x.mean(axis=axis)
    if n < 2:
        return m, m, m
    half = stats.t.ppf(0.5 + level / 2, n - 1) * x.std(axis=axis, ddof=1) / np.sqrt(n)
    return m, m - half, m + half


def hierarchical_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 5000, seed: int = 0) -> dict[str, float]:
    """Difference mean(a) - mean(b) for score matrices of shape (n_seeds, n_episodes). Both are evaluated on the
    same episode seeds, so the episodes are resampled jointly (paired). Training seeds are resampled
    independently for each side. A deterministic heuristic is a (1, n_episodes) matrix."""
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    if a.shape[1] != b.shape[1]:
        raise ValueError("a and b must share the episode axis")
    rng = np.random.default_rng(seed)
    n_eps = a.shape[1]
    diffs = np.empty(n)
    for i in range(n):
        e = rng.integers(0, n_eps, n_eps)
        sa = rng.integers(0, a.shape[0], a.shape[0])
        sb = rng.integers(0, b.shape[0], b.shape[0])
        diffs[i] = a[np.ix_(sa, e)].mean() - b[np.ix_(sb, e)].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(a.mean() - b.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_le_0": float((diffs <= 0).mean())}


def iqm(x: np.ndarray) -> float:
    """Interquartile mean (the robust aggregate recommended by Agarwal et al., 2021)."""
    x = np.sort(np.asarray(x, dtype=float).ravel())
    k = int(np.floor(0.25 * len(x)))
    return float(x[k:len(x) - k].mean()) if len(x) - 2 * k > 0 else float(x.mean())
