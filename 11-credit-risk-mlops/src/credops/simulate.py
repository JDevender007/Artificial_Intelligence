"""Synthetic 'production' scenarios built from the held-out test set, for drift and degradation drills.

These are illustrative stress scenarios, not forecasts: the source data is Taiwan 2005 and no real
post-deployment data exists. Labels are kept so the performance impact of each shift can be measured.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credops.schema import BILL_AMT, PAY_STATUS


def scenario(df: pd.DataFrame, name: str, seed: int = 0) -> pd.DataFrame:
    """Return a shifted copy of ``df``.

    * ``none``: a bootstrap resample (no shift: the drift check must stay quiet).
    * ``limit_cut``: the lender cuts every credit limit by 30% (covariate shift; bills unchanged).
    * ``recession``: 30% of clients slip one more month behind, bills rise 25%
      (covariate shift), and a random share of non-defaulters equal to the base default rate become defaulters
      (concept shift: about +17 points of default rate on this data).
    """
    rng = np.random.default_rng(seed)
    out = df.sample(len(df), replace=True, random_state=seed).reset_index(drop=True)
    if name == "none":
        return out
    if name == "limit_cut":
        out["LIMIT_BAL"] = (out["LIMIT_BAL"] * 0.7).round().astype("int64")
        return out
    if name == "recession":
        hit = rng.random(len(out)) < 0.3
        out.loc[hit, PAY_STATUS] = np.clip(out.loc[hit, PAY_STATUS] + 1, -2, 9)
        out[BILL_AMT] = (out[BILL_AMT] * 1.25).round().astype("int64")
        if "default" in out:
            flip = (out["default"] == 0) & (rng.random(len(out)) < out["default"].mean())
            out.loc[flip, "default"] = 1
        return out
    raise ValueError(f"unknown scenario {name!r}")


SCENARIOS = ("none", "limit_cut", "recession")
