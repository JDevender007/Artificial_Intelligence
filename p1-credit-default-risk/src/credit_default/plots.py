"""Matplotlib figures saved to disk (headless-safe)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve  # noqa: E402


def roc_pr_curves(y: np.ndarray, preds: dict[str, np.ndarray], path: Path) -> None:
    """Side-by-side ROC and precision-recall curves for several models."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        a1.plot(fpr, tpr, label=name)
        a2.plot(rec, prec, label=name)
    a1.plot([0, 1], [0, 1], "k--", lw=0.8)
    a2.axhline(float(np.mean(y)), color="k", ls="--", lw=0.8, label="prevalence")
    a1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (test)")
    a2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (test)")
    a1.legend(fontsize=8)
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def reliability_diagram(
    y: np.ndarray, preds: dict[str, np.ndarray], path: Path, n_bins: int = 15
) -> None:
    """Observed default rate vs mean predicted probability per bin."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    for name, p in preds.items():
        bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
        xs, ys = [], []
        for b in range(n_bins):
            m = bins == b
            if m.sum() >= 10:
                xs.append(p[m].mean())
                ys.append(y[m].mean())
        ax.plot(xs, ys, "o-", label=name)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed default rate",
           title="Reliability diagram (test)", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def score_histogram(y: np.ndarray, p: np.ndarray, threshold: float, path: Path) -> None:
    """Predicted-probability distribution per true class with the decision threshold."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(p[y == 0], bins=40, alpha=0.6, density=True, label="no default")
    ax.hist(p[y == 1], bins=40, alpha=0.6, density=True, label="default")
    ax.axvline(threshold, color="k", ls="--", label=f"threshold={threshold:.3f}")
    ax.set(xlabel="Predicted P(default)", ylabel="Density", title="Score distribution (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def confusion(y: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    """Confusion matrix at the chosen threshold."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y, y_pred, display_labels=["no default", "default"], ax=ax, colorbar=False
    )
    ax.set_title("Confusion matrix (test)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
