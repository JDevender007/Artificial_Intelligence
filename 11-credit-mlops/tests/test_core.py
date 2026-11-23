import numpy as np
import pandas as pd
import pytest

from credops.drift import drift_table, drift_verdict, psi
from credops.features import engineer, safe_ratio
from credops.model import ece, f1_threshold
from credops.schema import MODEL_INPUTS, PROTECTED
from credops.simulate import scenario
from credops.synthetic import synthetic_frame


def test_features_rowwise_and_no_protected():
    df = synthetic_frame(200, seed=3)
    full, one = engineer(df), engineer(df.iloc[[7]])
    assert not set(PROTECTED) & set(full.columns) and full.shape == (200, len(MODEL_INPUTS) + 22)
    pd.testing.assert_frame_equal(full.iloc[[7]], one)  # row-wise: no batch statistics leak in
    r = safe_ratio(pd.Series([10, 5, 100]), pd.Series([0, 10, 1]))
    assert r.tolist() == [1.0, 0.5, 5.0]


def test_ece_and_threshold():
    rng = np.random.default_rng(0)
    p = rng.random(200_000)
    assert ece(rng.random(200_000) < p, p) < 0.01  # calibrated by construction
    assert ece(np.zeros(10), np.full(10, 0.9)) == pytest.approx(0.9)
    y = np.array([0, 0, 1, 1])
    assert f1_threshold(y, np.array([0.1, 0.2, 0.8, 0.9])) == 0.8


def test_psi_and_drift_rules():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert psi(a, rng.normal(size=5000)) < 0.02 and psi(a, rng.normal(1.0, size=5000)) > 0.25
    assert psi(np.ones(10), np.ones(5)) == 0.0
    df = synthetic_frame(4000, seed=1)
    ref, test = df.iloc[:2000], df.iloc[2000:].reset_index(drop=True)
    quiet = drift_verdict(drift_table(ref, scenario(test, "none"), MODEL_INPUTS))
    assert not quiet["alert"] and quiet["n_drifted"] == 0
    cut = drift_table(ref, scenario(test, "limit_cut"), MODEL_INPUTS)
    assert cut.loc[cut.drifted, "column"].tolist() == ["LIMIT_BAL"]


def test_scenarios():
    df = synthetic_frame(3000, seed=2)
    rec = scenario(df, "recession", seed=0)
    assert rec["default"].mean() > df["default"].mean() + 0.1 and rec["PAY_1"].max() <= 9
    with pytest.raises(ValueError):
        scenario(df, "nope")
