"""Classification metrics, calibration, error analysis by class and length, and a paired bootstrap."""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def classification_metrics(y: np.ndarray, pred: np.ndarray, n_labels: int) -> dict[str, float]:
    labels = list(range(n_labels))
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0))}


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error of the top-1 confidence (equal-width bins)."""
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == y).astype(float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def per_class(y: np.ndarray, pred: np.ndarray, labels: list[str]) -> list[dict]:
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=list(range(len(labels))), zero_division=0)
    return [{"intent": labels[i], "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
             "support": int(s[i])} for i in range(len(labels))]


def length_bucket_names(edges: list[int]) -> list[str]:
    names, lo = [], 1
    for e in edges:
        names.append(f"{lo}-{e}")
        lo = e + 1
    return names + [f">{edges[-1]}"]


def by_length(word_counts: np.ndarray, y: np.ndarray, pred: np.ndarray, edges: list[int],
              n_labels: int) -> list[dict]:
    """Accuracy and macro-F1 per word-count bucket (``edges`` are inclusive upper bounds)."""
    idx = np.searchsorted(np.asarray(edges), word_counts, side="left")
    out = []
    for b, name in enumerate(length_bucket_names(edges)):
        m = idx == b
        row = {"words": name, "n": int(m.sum()), "accuracy": float("nan"), "macro_f1": float("nan")}
        if m.any():
            row.update(classification_metrics(y[m], pred[m], n_labels))
            # macro-F1 inside a bucket only averages the classes that occur there
            present = np.unique(np.concatenate([y[m], pred[m]]))
            row["macro_f1"] = float(f1_score(y[m], pred[m], labels=present, average="macro", zero_division=0))
        out.append(row)
    return out


def top_confusions(y: np.ndarray, pred: np.ndarray, labels: list[str], k: int = 10) -> list[dict]:
    c = Counter((int(t), int(p)) for t, p in zip(y, pred, strict=True) if t != p)
    return [{"true": labels[t], "predicted": labels[p], "count": n} for (t, p), n in c.most_common(k)]


def paired_bootstrap(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_labels: int,
                     n: int = 1000, seed: int = 0) -> dict[str, float]:
    """Bootstrap the test set: macro-F1(b) - macro-F1(a), with a 95% percentile interval."""
    rng = np.random.default_rng(seed)
    labels = list(range(n_labels))
    obs = (f1_score(y, pred_b, labels=labels, average="macro", zero_division=0)
           - f1_score(y, pred_a, labels=labels, average="macro", zero_division=0))
    diffs = np.empty(n)
    for i in range(n):
        s = rng.integers(0, len(y), len(y))
        diffs[i] = (f1_score(y[s], pred_b[s], labels=labels, average="macro", zero_division=0)
                    - f1_score(y[s], pred_a[s], labels=labels, average="macro", zero_division=0))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(obs), "ci_low": float(lo), "ci_high": float(hi), "p_b_not_better": float((diffs <= 0).mean())}
