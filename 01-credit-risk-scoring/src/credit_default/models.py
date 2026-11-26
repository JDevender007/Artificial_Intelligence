"""Model + preprocessing pipelines. All fitting happens inside a single sklearn Pipeline,
so cross-validation refits preprocessing on each training fold (no leakage)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from credit_default.data import CATEGORICAL
from credit_default.features import CreditFeatureEngineer

MODEL_NAMES = ("dummy", "logreg", "xgboost", "lightgbm")
TREE_MODELS = ("xgboost", "lightgbm")
MONEY_PREFIXES = ("LIMIT_BAL", "BILL_AMT", "PAY_AMT")


def signed_log1p(x: Any) -> Any:
    """sign(x) * log(1 + |x|): tames heavy-tailed money columns, keeps negative bills."""
    return np.sign(x) * np.log1p(np.abs(x))


def _categorical(X: pd.DataFrame) -> list[str]:
    return [c for c in CATEGORICAL if c in X.columns]


def _money(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c.startswith(MONEY_PREFIXES)]


def _non_categorical(X: pd.DataFrame) -> list[str]:
    cats = set(_categorical(X))
    return [c for c in X.columns if c not in cats]


def _other_numeric(X: pd.DataFrame) -> list[str]:
    skip = set(_categorical(X)) | set(_money(X))
    return [c for c in X.columns if c not in skip]


def build_preprocessor(linear: bool) -> ColumnTransformer:
    """Scale + log-transform for linear models; trees get raw numerics."""
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if linear:
        money = Pipeline(
            [
                ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
            ]
        )
        transformers = [
            ("cat", onehot, _categorical),
            ("money", money, _money),
            ("num", StandardScaler(), _other_numeric),
        ]
    else:
        transformers = [
            ("cat", onehot, _categorical),
            ("num", "passthrough", _non_categorical),
        ]
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def build_estimator(
    name: str, params: dict[str, Any], seed: int, pos_weight: float | None
) -> Any:
    """Create an unfitted classifier.

    Args:
        name: one of :data:`MODEL_NAMES`.
        params: hyper-parameters overriding the defaults.
        seed: random seed for the estimator.
        pos_weight: n_negative / n_positive if re-weighting the minority class, else None.
    """
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "logreg":
        base = {"C": 1.0, "max_iter": 5000}
        return LogisticRegression(
            **{**base, **params},
            class_weight="balanced" if pos_weight else None,
            random_state=seed,
        )
    if name == "xgboost":
        base = {
            "n_estimators": 300, "learning_rate": 0.05, "max_depth": 4,
            "subsample": 0.8, "colsample_bytree": 0.8,
        }
        return XGBClassifier(
            **{**base, **params},
            tree_method="hist",
            eval_metric="logloss",
            scale_pos_weight=pos_weight or 1.0,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "lightgbm":
        base = {
            "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
            "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
        }
        return LGBMClassifier(
            **{**base, **params},
            scale_pos_weight=pos_weight or 1.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")


def build_pipeline(
    name: str,
    params: dict[str, Any] | None = None,
    seed: int = 0,
    pos_weight: float | None = None,
    drop_protected: bool = True,
    engineered: bool = True,
) -> Pipeline:
    """Feature engineering -> preprocessing -> classifier, with pandas output for SHAP."""
    pipe = Pipeline(
        [
            ("features", CreditFeatureEngineer(engineered, drop_protected)),
            ("prep", build_preprocessor(linear=name not in TREE_MODELS)),
            ("model", build_estimator(name, params or {}, seed, pos_weight)),
        ]
    )
    pipe[:-1].set_output(transform="pandas")
    return pipe
