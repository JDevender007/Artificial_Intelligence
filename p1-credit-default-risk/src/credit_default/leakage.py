"""Automated leakage sanity checks run before any model is trained."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def duplicate_overlap(train_X: pd.DataFrame, test_X: pd.DataFrame) -> int:
    """Number of test rows whose full feature vector also appears in train."""
    cols = list(train_X.columns)
    train_keys = set(map(tuple, train_X[cols].to_numpy()))
    return int(sum(tuple(r) in train_keys for r in test_X[cols].to_numpy()))


def single_feature_auc(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Direction-free ROC-AUC of each raw feature alone. Values near 1.0 suggest a
    feature that encodes the target (e.g. something recorded after the outcome)."""
    out = {}
    for col in X.columns:
        values = X[col].to_numpy(dtype=float)
        if np.unique(values).size < 2:
            out[col] = 0.5
            continue
        auc = roc_auc_score(y, values)
        out[col] = max(auc, 1 - auc)
    return pd.Series(out).sort_values(ascending=False)


def run_leakage_checks(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    test_X: pd.DataFrame,
    test_y: pd.Series,
    auc_flag: float = 0.9,
) -> dict[str, Any]:
    """Run all checks and return a JSON-serialisable report."""
    aucs = single_feature_auc(train_X, train_y)
    report: dict[str, Any] = {
        "n_train": len(train_X),
        "n_test": len(test_X),
        "train_positive_rate": float(train_y.mean()),
        "test_positive_rate": float(test_y.mean()),
        "test_rows_duplicated_in_train": duplicate_overlap(train_X, test_X),
        "top_single_feature_auc": aucs.head(5).round(4).to_dict(),
        "suspicious_features": aucs[aucs >= auc_flag].index.tolist(),
        "target_in_features": bool(train_y.name in train_X.columns),
    }
    report["passed"] = not report["suspicious_features"] and not report["target_in_features"]
    return report
