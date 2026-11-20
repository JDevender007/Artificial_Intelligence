"""Loss/accuracy curves and confusion-matrix figures (headless Agg backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402


def learning_curves(histories: dict[str, pd.DataFrame], path: Path) -> None:
    """Train (dashed) vs val (solid) loss and accuracy per model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, (name, h) in enumerate(histories.items()):
        color = f"C{i}"
        axes[0].plot(h["epoch"], h["train_loss"], "--", color=color, alpha=0.7)
        axes[0].plot(h["epoch"], h["val_loss"], "-", color=color, label=name)
        axes[1].plot(h["epoch"], h["train_acc"], "--", color=color, alpha=0.7)
        axes[1].plot(h["epoch"], h["val_acc"], "-", color=color, label=name)
    axes[0].set(xlabel="epoch", ylabel="cross-entropy", title="Loss (dashed = train, solid = val)")
    axes[1].set(xlabel="epoch", ylabel="accuracy", title="Accuracy (dashed = train, solid = val)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def confusion(y: np.ndarray, pred: np.ndarray, classes: list[str], path: Path, title: str) -> None:
    """Row-normalised confusion matrix (each row sums to 1 = recall per class)."""
    cm = confusion_matrix(y, pred, labels=range(len(classes))).astype(float)
    cm /= cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > 0.5 else "black")
    ax.set(xlabel="predicted", ylabel="true", title=title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
