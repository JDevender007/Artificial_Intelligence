"""Model pipelines, metrics and the F1 threshold. Training data never reaches the test set here."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credops.features import CreditFeatures


def build(kind: str, seed: int, params: dict[str, Any] | None = None, calibrate: bool = True) -> Pipeline:
    """``logreg`` (baseline) or ``lgbm``; the GBDT is wrapped in 5-fold isotonic calibration."""
    params = dict(params or {})
    if kind == "logreg":
        return Pipeline([("features", CreditFeatures()), ("scale", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000, C=params.get("C", 1.0)))])
    if kind == "lgbm":
        clf = LGBMClassifier(random_state=seed, n_jobs=params.pop("n_jobs", 1), verbose=-1, **params)
        if calibrate:
            clf = CalibratedClassifierCV(clf, method="isotonic", cv=StratifiedKFold(5, shuffle=True, random_state=seed))
        return Pipeline([("features", CreditFeatures()), ("clf", clf)])
    raise ValueError(f"unknown model kind {kind!r}")


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    y, p = np.asarray(y), np.asarray(p)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(n_bins) if (idx == b).any()))


def f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold that maximises F1 on (out-of-fold) predictions."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.nanargmax(f1))])


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    return {"pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "ece": ece(y, p),
            "f1": float(f1_score(y, p >= threshold)), "positive_rate_pred": float(np.mean(p >= threshold))}


def oof_threshold(model: Pipeline, X: pd.DataFrame, y: pd.Series, seed: int) -> float:
    """F1 threshold chosen on 5-fold out-of-fold predictions of the training set."""
    oof = cross_val_predict(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                            method="predict_proba")[:, 1]
    return f1_threshold(y.to_numpy(), oof)
