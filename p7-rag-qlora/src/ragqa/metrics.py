"""SQuAD 2.0 EM/F1, retrieval metrics, RAGAS-style context/faithfulness metrics and bootstrap CIs.

The RAGAS definitions (context recall, context precision, faithfulness) are computed against SQuAD's gold
labels instead of an LLM judge: the gold supporting passage and the extractive answer are known exactly, so the
judge is unnecessary and the numbers are deterministic.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from ragqa.text import normalize_answer


def _f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def squad_em(pred: str, golds: Sequence[str]) -> float:
    """Official SQuAD 2.0 rule: an unanswerable question (no golds) scores 1 only for an empty prediction."""
    if not golds:
        return float(normalize_answer(pred) == "")
    return max(float(normalize_answer(pred) == normalize_answer(g)) for g in golds)


def squad_f1(pred: str, golds: Sequence[str]) -> float:
    if not golds:
        return float(normalize_answer(pred) == "")
    if normalize_answer(pred) == "":
        return 0.0
    return max(_f1(pred, g) for g in golds)


def recall_at_k(ranking: Sequence[int], relevant: set[int], k: int) -> float:
    """Hit rate: 1 if any relevant chunk is in the top k."""
    return float(any(c in relevant for c in ranking[:k]))


def mrr_at_k(ranking: Sequence[int], relevant: set[int], k: int = 10) -> float:
    for r, c in enumerate(ranking[:k], start=1):
        if c in relevant:
            return 1.0 / r
    return 0.0


def context_precision(ranking: Sequence[int], relevant: set[int]) -> float:
    """RAGAS context precision: mean of precision@i over the positions i that hold a relevant chunk (0 if none)."""
    hits, total = 0, 0.0
    for i, c in enumerate(ranking, start=1):
        if c in relevant:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0


def contains(haystack: str, needle: str) -> bool:
    n = normalize_answer(needle)
    return bool(n) and f" {n} " in f" {normalize_answer(haystack)} "


def bootstrap_ci(x: np.ndarray, n: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 1000, seed: int = 0) -> dict[str, float]:
    """Mean(b) - mean(a) over the same questions, with a 95% percentile CI from resampling questions."""
    if len(a) != len(b):
        raise ValueError("paired bootstrap needs equal-length arrays")
    d = np.asarray(b, float) - np.asarray(a, float)
    lo, hi = bootstrap_ci(d, n, seed)
    return {"diff": float(d.mean()), "lo": lo, "hi": hi, "n": len(d)}


def _mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else float("nan")


def summarize(records: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate per-question records (see ``experiment.score_record``) into the results-table metrics."""
    ans = [r for r in records if not r["is_impossible"]]
    imp = [r for r in records if r["is_impossible"]]
    answered = [r for r in records if not r["abstained"]]
    out = {
        "n": len(records),
        "em": _mean([r["em"] for r in records]),
        "f1": _mean([r["f1"] for r in records]),
        "has_ans_f1": _mean([r["f1"] for r in ans]),
        "no_ans_acc": _mean([float(r["abstained"]) for r in imp]),
        "abstain_rate": _mean([float(r["abstained"]) for r in records]),
        "false_abstain": _mean([float(r["abstained"]) for r in ans]),
    }
    if records and records[0].get("context") is not None:
        cited = [r for r in answered if r["citations"]]
        out.update({
            "context_recall": _mean([float(r["retrieval_hit"]) for r in ans]),
            "context_precision": _mean([r["context_precision"] for r in ans]),
            "faithfulness": _mean([float(r["grounded"]) for r in answered]),
            "cite_rate": _mean([float(bool(r["citations"])) for r in answered]),
            "cite_supports_answer": _mean([float(r["cite_supports_answer"]) for r in cited]),
            "cite_gold": _mean([float(r["cite_gold"]) for r in cited if not r["is_impossible"]]),
        })
    return out
