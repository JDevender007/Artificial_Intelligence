"""Slice-level error analysis and group-fairness gaps."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def add_slices(test: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable slice columns computed from raw test features."""
    s = test.copy()
    s["recent_status"] = np.where(s["PAY_1"] > 0, "delayed_last_month", "on_time_last_month")
    delayed = (s[[f"PAY_{i}" for i in range(1, 7)]] > 0).sum(axis=1)
    s["months_delayed"] = pd.cut(delayed, [-1, 0, 2, 6], labels=["0", "1-2", "3-6"]).astype(str)
    s["limit_quartile"] = pd.qcut(s["LIMIT_BAL"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"]).astype(str)
    s["age_band"] = pd.cut(s["AGE"], [0, 29, 39, 49, 200], labels=["<30", "30-39", "40-49", "50+"]).astype(str)
    s["sex"] = s["SEX"].map({1: "male", 2: "female"})
    s["education"] = s["EDUCATION"].map({1: "graduate", 2: "university", 3: "high_school", 4: "other"})
    s["marriage"] = s["MARRIAGE"].map({1: "married", 2: "single", 3: "other"})
    return s


def slice_metrics(y: np.ndarray, p: np.ndarray, pred: np.ndarray, groups: pd.Series) -> pd.DataFrame:
    """Per-group size, prevalence, PR-AUC, TPR (recall), FPR, precision and selection rate."""
    rows = []
    for g in sorted(groups.unique()):
        m = (groups == g).to_numpy()
        yg, pg, dg = y[m], p[m], pred[m]
        pos, neg = yg == 1, yg == 0
        rows.append({
            "group": g,
            "n": int(m.sum()),
            "prevalence": yg.mean(),
            "mean_pred_prob": pg.mean(),
            "pr_auc": average_precision_score(yg, pg) if 0 < pos.sum() < len(yg) else np.nan,
            "tpr_recall": dg[pos].mean() if pos.any() else np.nan,
            "fpr": dg[neg].mean() if neg.any() else np.nan,
            "precision": yg[dg == 1].mean() if (dg == 1).any() else np.nan,
            "selection_rate": dg.mean(),
        })
    return pd.DataFrame(rows)


def fairness_gaps(table: pd.DataFrame) -> dict[str, float]:
    """Max-min gaps across groups: equal-opportunity (TPR), FPR and demographic parity."""
    return {
        "tpr_gap": float(table["tpr_recall"].max() - table["tpr_recall"].min()),
        "fpr_gap": float(table["fpr"].max() - table["fpr"].min()),
        "selection_rate_gap": float(table["selection_rate"].max() - table["selection_rate"].min()),
    }


def outcome_profile(features: pd.DataFrame, y: np.ndarray, pred: np.ndarray, cols: list[str]) -> pd.DataFrame:
    """Mean of selected features for TP / FN / FP / TN — shows what missed defaulters look like."""
    outcome = np.select(
        [(y == 1) & (pred == 1), (y == 1) & (pred == 0), (y == 0) & (pred == 1)],
        ["TP", "FN", "FP"], default="TN",
    )
    prof = features[cols].groupby(outcome).mean().T
    prof.loc["count"] = pd.Series(outcome).value_counts()
    return prof[[c for c in ["TP", "FN", "FP", "TN"] if c in prof.columns]]
