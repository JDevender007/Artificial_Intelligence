"""Point-forecast metrics (MAE, RMSE, MASE), breakdowns and a paired bootstrap over forecast days."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mase_scale(train: np.ndarray, season: int) -> float:
    """In-sample MAE of the seasonal naive forecast y[t-season] on the training data (Hyndman & Koehler, 2006)."""
    train = np.asarray(train, dtype=float)
    if len(train) <= season:
        raise ValueError("training series must be longer than the season")
    scale = float(np.mean(np.abs(train[season:] - train[:-season])))
    if scale == 0:
        raise ValueError("MASE scale is zero (constant training series)")
    return scale


def point_metrics(fc: pd.DataFrame, scale: float) -> dict[str, float]:
    """``fc`` has columns ``y`` and ``yhat`` (one row per forecast hour)."""
    m = mae(fc.y, fc.yhat)
    return {"mae": m, "rmse": rmse(fc.y, fc.yhat), "mase": m / scale}


def by_step(fc: pd.DataFrame) -> pd.Series:
    """MAE per horizon step (1 = first hour after the origin)."""
    return (fc.y - fc.yhat).abs().groupby(fc.step).mean()


def by_hour(fc: pd.DataFrame) -> pd.Series:
    return (fc.y - fc.yhat).abs().groupby(pd.DatetimeIndex(fc.ts).hour).mean()


def by_month(fc: pd.DataFrame) -> pd.Series:
    return (fc.y - fc.yhat).abs().groupby(pd.DatetimeIndex(fc.ts).to_period("M").astype(str)).mean()


def by_daytype(fc: pd.DataFrame) -> pd.Series:
    wk = np.where(pd.DatetimeIndex(fc.ts).dayofweek >= 5, "weekend", "weekday")
    return (fc.y - fc.yhat).abs().groupby(wk).mean()


def fold_mae(fc: pd.DataFrame) -> pd.Series:
    return (fc.y - fc.yhat).abs().groupby(fc.origin).mean()


def paired_bootstrap(fc_a: pd.DataFrame, fc_b: pd.DataFrame, n: int = 1000, seed: int = 0) -> dict[str, float]:
    """MAE(b) - MAE(a), resampling whole forecast origins (days) so the 24 correlated hours stay together.

    Negative means ``b`` is better than ``a``.
    """
    a, b = fold_mae(fc_a), fold_mae(fc_b)
    common = a.index.intersection(b.index)
    d = (b.loc[common] - a.loc[common]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return {
        "diff": float(d.mean()),
        "lo": float(np.percentile(boots, 2.5)),
        "hi": float(np.percentile(boots, 97.5)),
        "n_folds": len(d),
        "b_wins": float((d < 0).mean()),
    }
