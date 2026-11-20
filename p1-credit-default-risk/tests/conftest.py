"""Synthetic data with the real schema, so tests never need the download."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_synthetic(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "ID": np.arange(1, n + 1),
        "LIMIT_BAL": rng.integers(1, 50, n) * 10_000,
        "SEX": rng.integers(1, 3, n),
        "EDUCATION": rng.integers(1, 5, n),
        "MARRIAGE": rng.integers(1, 4, n),
        "AGE": rng.integers(21, 70, n),
    })
    for i in range(1, 7):
        df[f"PAY_{i}"] = rng.integers(-2, 4, n)
    for i in range(1, 7):
        df[f"BILL_AMT{i}"] = rng.integers(-1_000, 100_000, n)
    for i in range(1, 7):
        df[f"PAY_AMT{i}"] = rng.integers(0, 20_000, n)
    logit = 0.9 * df["PAY_1"] - 1.5
    df["default"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


@pytest.fixture
def synthetic() -> pd.DataFrame:
    return make_synthetic()
