"""Schema-valid synthetic clients for tests, CI smoke runs and load tests (no real data needed).

The generator only mimics the column ranges and a plausible default/delay relationship; it is not a model of
the real population and no result in the README comes from it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credops.schema import BILL_AMT, ID_COL, PAY_AMT, PAY_STATUS, TARGET


def synthetic_frame(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """A frame with the cleaned UCI schema (ID, LIMIT_BAL, demographics, 18 history columns, target)."""
    rng = np.random.default_rng(seed)
    risk = rng.normal(size=n)
    limit = np.clip(np.round(rng.lognormal(np.log(120_000), 0.7, n), -4), 10_000, 1_000_000)
    df = pd.DataFrame({ID_COL: np.arange(1, n + 1), "LIMIT_BAL": limit,
                       "SEX": rng.integers(1, 3, n), "EDUCATION": rng.integers(1, 5, n),
                       "MARRIAGE": rng.integers(1, 4, n), "AGE": rng.integers(21, 70, n)})
    for c in PAY_STATUS:
        df[c] = np.clip(np.round(risk + rng.normal(scale=0.8, size=n)), -2, 8).astype(int)
    for i, c in enumerate(BILL_AMT):
        df[c] = (limit * np.clip(0.4 + 0.2 * risk + rng.normal(scale=0.2, size=n), -0.05, 1.2) * (1 - 0.02 * i)).round()
    for c in PAY_AMT:
        df[c] = np.abs(limit * np.clip(0.05 - 0.03 * risk + rng.normal(scale=0.03, size=n), 0, None)).round()
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * risk - 1.3)))).astype(int)
    return df.astype("int64")
