"""Stateless, row-wise feature engineering, ported from P1 (``credit_default.features``).

Every feature is a function of a single row, so ``fit`` learns nothing and serving cannot skew from training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from credops.schema import BILL_AMT, MODEL_INPUTS, PAY_AMT, PAY_STATUS

RATIO_CAP = 5.0


def safe_ratio(num: pd.Series, den: pd.Series, when_den_nonpos: float = 1.0) -> pd.Series:
    """``num / den`` clipped to [0, RATIO_CAP]; ``when_den_nonpos`` where nothing was owed."""
    num_f, den_f = num.astype(float), den.astype(float)
    out = np.where(den_f > 0, num_f / den_f.where(den_f > 0, 1.0), when_den_nonpos)
    return pd.Series(np.clip(out, 0.0, RATIO_CAP), index=num.index)


def engineer(X: pd.DataFrame) -> pd.DataFrame:
    """Raw model inputs plus utilisation, repayment-ratio and delinquency-history features."""
    X = X[MODEL_INPUTS].astype(float)
    limit = X["LIMIT_BAL"]
    util = pd.DataFrame({f"util_{i}": X[f"BILL_AMT{i}"] / limit for i in range(1, 7)}, index=X.index)
    # PAY_AMT{k} settles the previous month's statement, BILL_AMT{k+1}.
    payr = pd.DataFrame({f"payratio_{k}": safe_ratio(X[f"PAY_AMT{k}"], X[f"BILL_AMT{k + 1}"]) for k in range(1, 6)},
                        index=X.index)
    status = X[PAY_STATUS]
    extra = pd.DataFrame({
        "util_mean": util.mean(axis=1), "util_max": util.max(axis=1), "util_trend": util["util_1"] - util["util_6"],
        "payratio_mean": payr.mean(axis=1), "payratio_min": payr.min(axis=1),
        "n_months_delayed": (status > 0).sum(axis=1), "max_delay": status.max(axis=1),
        "n_months_no_use": (status == -2).sum(axis=1), "delay_trend": X["PAY_1"] - X["PAY_6"],
        "pay_total_to_limit": X[PAY_AMT].sum(axis=1) / limit, "bill_total_to_limit": X[BILL_AMT].sum(axis=1) / limit,
    }, index=X.index)
    return pd.concat([X, util, payr, extra], axis=1)


class CreditFeatures(BaseEstimator, TransformerMixin):
    """sklearn wrapper around :func:`engineer` so it is pickled inside the model pipeline."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> CreditFeatures:
        self.feature_names_out_ = np.asarray(engineer(X.head(1)).columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return engineer(X)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return self.feature_names_out_
