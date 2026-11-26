"""Stateless feature engineering (row-wise only, so it cannot leak across splits)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from credit_default.data import BILL_AMT, PAY_AMT, PAY_STATUS, PROTECTED

RATIO_CAP = 5.0


def safe_ratio(num: pd.Series, den: pd.Series, when_den_nonpos: float = 1.0) -> pd.Series:
    """``num / den`` clipped to [0, RATIO_CAP]; returns ``when_den_nonpos`` where den <= 0.

    A non-positive bill means nothing was owed, so a payment ratio of 1 ("fully paid")
    is the natural fill value.
    """
    num_f = num.astype(float)
    den_f = den.astype(float)
    out = np.where(den_f > 0, num_f / den_f.where(den_f > 0, 1.0), when_den_nonpos)
    return pd.Series(np.clip(out, 0.0, RATIO_CAP), index=num.index)


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add utilisation, repayment-ratio and delinquency-history features.

    Every feature is a deterministic function of a single row, so ``fit`` learns
    nothing from the data and the transformer is leakage-free by construction.

    Args:
        add_engineered: add the derived features below.
        drop_protected: drop demographic columns (SEX, EDUCATION, MARRIAGE, AGE).
    """

    def __init__(self, add_engineered: bool = True, drop_protected: bool = True) -> None:
        self.add_engineered = add_engineered
        self.drop_protected = drop_protected

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> CreditFeatureEngineer:
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        # Use the unwrapped implementation: sklearn's set_output wrapper around
        # ``transform`` would ask for feature names that do not exist yet.
        self.feature_names_out_ = np.asarray(self._engineer(X.head(1)).columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._engineer(X)

    def _engineer(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.add_engineered:
            limit = X["LIMIT_BAL"].astype(float)
            util = pd.DataFrame(
                {f"util_{i}": X[f"BILL_AMT{i}"] / limit for i in range(1, 7)}, index=X.index
            )
            # PAY_AMT{k} (month k) settles the statement of the previous month, BILL_AMT{k+1}.
            payr = pd.DataFrame(
                {f"payratio_{k}": safe_ratio(X[f"PAY_AMT{k}"], X[f"BILL_AMT{k + 1}"])
                 for k in range(1, 6)},
                index=X.index,
            )
            status = X[PAY_STATUS]
            X = pd.concat([X, util, payr], axis=1)
            X["util_mean"] = util.mean(axis=1)
            X["util_max"] = util.max(axis=1)
            X["util_trend"] = util["util_1"] - util["util_6"]
            X["payratio_mean"] = payr.mean(axis=1)
            X["payratio_min"] = payr.min(axis=1)
            X["n_months_delayed"] = (status > 0).sum(axis=1)
            X["max_delay"] = status.max(axis=1)
            X["n_months_no_use"] = (status == -2).sum(axis=1)
            X["delay_trend"] = X["PAY_1"] - X["PAY_6"]
            X["pay_total_to_limit"] = X[PAY_AMT].sum(axis=1) / limit
            X["bill_total_to_limit"] = X[BILL_AMT].sum(axis=1) / limit
        if self.drop_protected:
            X = X.drop(columns=[c for c in PROTECTED if c in X.columns])
        return X

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        return self.feature_names_out_
