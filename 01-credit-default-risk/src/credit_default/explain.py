"""SHAP explanations for the tree model inside a fitted pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402


def shap_summary(
    pipe: Pipeline, X: pd.DataFrame, out_dir: Path, max_samples: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """Save beeswarm + bar plots and return mean |SHAP| per feature (log-odds units).

    Explains the *uncalibrated* booster: isotonic calibration is a monotone map on top,
    so feature rankings are unchanged.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = X.sample(n=min(max_samples, len(X)), random_state=seed)
    Xt = pipe[:-1].transform(sample)
    explanation = shap.TreeExplainer(pipe[-1])(Xt)
    if explanation.values.ndim == 3:  # some versions return one slice per class
        explanation = explanation[..., 1]

    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_beeswarm.png", dpi=130)
    plt.close("all")

    shap.plots.bar(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_bar.png", dpi=130)
    plt.close("all")

    importance = pd.DataFrame(
        {"feature": Xt.columns, "mean_abs_shap": np.abs(explanation.values).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(out_dir / "shap_importance.csv", index=False)
    return importance
