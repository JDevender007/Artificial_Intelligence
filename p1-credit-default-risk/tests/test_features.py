import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from credit_default.data import PROTECTED, clean, get_xy
from credit_default.features import CreditFeatureEngineer, safe_ratio


def test_safe_ratio_handles_zero_and_negative_bills():
    out = safe_ratio(pd.Series([50, 10, 10, 1000]), pd.Series([100, 0, -5, 10]))
    assert out.tolist() == pytest.approx([0.5, 1.0, 1.0, 5.0])  # last one capped


def test_engineered_features_are_finite_and_correct(synthetic):
    X, _ = get_xy(synthetic)
    Xt = CreditFeatureEngineer().fit(X).transform(X)
    assert np.isfinite(Xt.to_numpy(dtype=float)).all()
    expected = X["BILL_AMT1"] / X["LIMIT_BAL"]
    assert np.allclose(Xt["util_1"], expected)
    assert (Xt["n_months_delayed"] == (X[[f"PAY_{i}" for i in range(1, 7)]] > 0).sum(axis=1)).all()


def test_protected_columns_dropped_by_default(synthetic):
    X, _ = get_xy(synthetic)
    Xt = CreditFeatureEngineer().fit(X).transform(X)
    assert not set(PROTECTED) & set(Xt.columns)
    kept = CreditFeatureEngineer(drop_protected=False).fit(X).transform(X)
    assert set(PROTECTED) <= set(kept.columns)


def test_transformer_is_stateless_and_cloneable(synthetic):
    X, _ = get_xy(synthetic)
    fe = CreditFeatureEngineer()
    a = fe.fit(X.iloc[:100]).transform(X.iloc[200:])
    b = clone(fe).fit(X.iloc[100:200]).transform(X.iloc[200:])
    pd.testing.assert_frame_equal(a, b)  # fit data must not influence output
    assert list(fe.get_feature_names_out()) == list(a.columns)


def test_clean_maps_undocumented_codes(synthetic):
    raw = synthetic.copy()
    raw.loc[0, "EDUCATION"] = 0
    raw.loc[1, "EDUCATION"] = 6
    raw.loc[2, "MARRIAGE"] = 0
    out = clean(raw)
    assert out.loc[[0, 1], "EDUCATION"].tolist() == [4, 4]
    assert out.loc[2, "MARRIAGE"] == 3
