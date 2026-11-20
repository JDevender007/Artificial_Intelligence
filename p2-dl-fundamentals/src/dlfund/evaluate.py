"""Classification metrics and mean ± std aggregation over seeds."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

METRICS = ["test_acc", "test_macro_f1", "test_loss", "val_acc", "best_epoch", "epochs_run",
           "train_minutes", "images_per_s", "params_m"]


def classification_metrics(logits: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Accuracy, macro-F1 and cross-entropy from raw logits."""
    pred = logits.argmax(1)
    z = logits - logits.max(1, keepdims=True)  # stable log-softmax
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return {
        "acc": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "loss": float(-logp[np.arange(len(y)), y].mean()),
    }


def per_class_accuracy(y: np.ndarray, pred: np.ndarray, classes: list[str]) -> pd.Series:
    """Recall of each class (diagonal of the row-normalised confusion matrix)."""
    cm = confusion_matrix(y, pred, labels=range(len(classes)))
    return pd.Series(cm.diagonal() / cm.sum(1).clip(min=1), index=classes, name="accuracy")


def top_confusions(y: np.ndarray, pred: np.ndarray, classes: list[str], k: int = 10) -> pd.DataFrame:
    """The ``k`` most frequent (true -> predicted) mistakes."""
    cm = confusion_matrix(y, pred, labels=range(len(classes)))
    np.fill_diagonal(cm, 0)
    flat = np.argsort(cm.ravel())[::-1][:k]
    rows = [(classes[i // len(classes)], classes[i % len(classes)], int(cm.ravel()[i])) for i in flat]
    rows = [r for r in rows if r[2] > 0]
    out = pd.DataFrame(rows, columns=["true", "predicted", "count"])
    out["share_of_true_class"] = out["count"] / [int((y == classes.index(t)).sum()) for t in out["true"]]
    return out


def aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean and std (ddof=1) per model; ``n_seeds`` counts rows."""
    cols = [c for c in METRICS if c in per_seed]
    g = per_seed.groupby("model")[cols]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std"))
    out["n_seeds"] = per_seed.groupby("model").size()
    return out.sort_values("test_acc_mean", ascending=False)


def to_markdown(agg: pd.DataFrame, baseline: str | None = None) -> str:
    """``mean ± std`` table; the baseline row is labelled."""
    cols = [c for c in METRICS if f"{c}_mean" in agg]
    lines = ["| model | " + " | ".join(cols) + " | n_seeds |", "|---|" + "---|" * (len(cols) + 1)]
    for model, row in agg.iterrows():
        name = f"{model} (baseline)" if model == baseline else str(model)
        cells = []
        for c in cols:
            fmt = ".4f" if c.startswith(("test_", "val_")) else ".1f"
            std = row[f"{c}_std"]
            cells.append(f"{row[f'{c}_mean']:{fmt}} ± " + ("n/a" if pd.isna(std) else f"{std:{fmt}}"))
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {int(row['n_seeds'])} |")
    return "\n".join(lines)
