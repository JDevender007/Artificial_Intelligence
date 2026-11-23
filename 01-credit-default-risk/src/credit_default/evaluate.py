"""Metrics: discrimination (ROC/PR-AUC), thresholded F1, and calibration (ECE, Brier)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_ORDER = [
    "roc_auc", "pr_auc", "f1", "precision", "recall", "brier", "ece", "log_loss",
]


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Equal-width-bin ECE: sum_b (n_b / n) * |mean(y_b) - mean(p_b)|."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape or y_true.size == 0:
        raise ValueError("y_true and y_prob must be non-empty and the same shape")
    bins = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold that maximises F1. Call this on OUT-OF-FOLD train predictions only."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        return 0.5
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(thresholds[int(np.argmax(f1))])


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, n_bins: int = 15
) -> dict[str, float]:
    """All reported metrics for one set of predictions."""
    y_true = np.asarray(y_true)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1 - 1e-7)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob, n_bins),
        "log_loss": float(log_loss(y_true, y_prob)),
        "threshold": float(threshold),
    }


def aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over seeds per model, as numeric columns ``<metric>_mean/_std``."""
    cols = ["cv_pr_auc", *METRIC_ORDER]
    g = per_seed.groupby("model")[cols]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std"))
    out["n_seeds"] = per_seed.groupby("model").size()
    return out.sort_values("pr_auc_mean", ascending=False)


def to_markdown(agg: pd.DataFrame, baseline: str = "logreg_tuned") -> str:
    """Render ``mean ± std`` table; the baseline row is marked for quick comparison."""
    cols = ["cv_pr_auc", *METRIC_ORDER]
    lines = [
        "| model | " + " | ".join(f"{c} (test)" if c != "cv_pr_auc" else c for c in cols) + " |",
        "|---|" + "---|" * len(cols) + "---|",
    ]
    lines[0] += " n_seeds |"
    for model, row in agg.iterrows():
        name = f"{model} (baseline)" if model == baseline else str(model)
        cells = []
        for c in cols:
            std = row[f"{c}_std"]
            std_s = "n/a" if pd.isna(std) else f"{std:.4f}"
            cells.append(f"{row[f'{c}_mean']:.4f} ± {std_s}")
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {int(row['n_seeds'])} |")
    return "\n".join(lines)
