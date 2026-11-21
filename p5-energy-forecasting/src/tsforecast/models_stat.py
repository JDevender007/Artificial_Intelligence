"""Classical forecasters: seasonal naive (daily and weekly), ETS (Holt-Winters) and SARIMA via statsmodels.

All of them are deterministic, so one seed is enough; the experiment runner records ``n_seeds=1`` for them.
statsmodels is imported lazily so the naive baselines and the tests run without it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import StatConfig


class SeasonalNaive:
    """yhat[t0 + h] = y[t0 + h - lag * ceil((h + 1) / lag)]: repeat the last full season."""

    deterministic = True

    def __init__(self, lag: int = 24, name: str = "seasonal_naive"):
        self.lag, self.name = lag, name

    def fit(self, train: pd.Series, val: pd.Series) -> dict:
        return {"lag": self.lag}

    def predict(self, hist: pd.Series, horizon: int) -> np.ndarray:
        last = hist.to_numpy()[-self.lag :]
        if len(last) < self.lag:
            raise ValueError("history shorter than one season")
        return np.resize(last, horizon)


class ETS:
    """Additive Holt-Winters with a daily season, re-fitted on the last ``window_days`` before every origin."""

    deterministic = True
    name = "ets"

    def __init__(self, cfg: StatConfig, season: int = 24):
        self.cfg, self.season = cfg, season
        self.n_fail = 0

    def fit(self, train: pd.Series, val: pd.Series) -> dict:
        return {"window_days": self.cfg.window_days, "season": self.season}

    def predict(self, hist: pd.Series, horizon: int) -> np.ndarray:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        w = hist.to_numpy()[-self.cfg.window_days * 24 :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = ExponentialSmoothing(
                    w, trend=None, seasonal="add", seasonal_periods=self.season, initialization_method="estimated"
                ).fit()
                out = res.forecast(horizon)
            except (ValueError, np.linalg.LinAlgError):
                out = np.full(horizon, np.nan)
        if not np.isfinite(out).all():  # fall back to seasonal naive rather than dropping the fold
            self.n_fail += 1
            out = np.resize(w[-self.season :], horizon)
        return np.asarray(out)


class SARIMA:
    """SARIMA on the last ``window_days``. Parameters are re-estimated every ``refit_every`` origins; in between,
    the fitted parameters are re-applied to the newest window (Kalman filtering only), which is much cheaper."""

    deterministic = True
    name = "sarima"

    def __init__(self, cfg: StatConfig):
        self.cfg = cfg
        self._res = None
        self._calls = 0
        self.n_refits = 0
        self.n_fail = 0

    def fit(self, train: pd.Series, val: pd.Series) -> dict:
        return {"order": self.cfg.sarima_order, "seasonal": self.cfg.sarima_seasonal, "refit_every": self.cfg.refit_every}

    def predict(self, hist: pd.Series, horizon: int) -> np.ndarray:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        w = hist.to_numpy()[-self.cfg.window_days * 24 :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ConvergenceWarning on some windows; the fallback below covers failures
            try:
                if self._res is None or self._calls % self.cfg.refit_every == 0:
                    mod = SARIMAX(w, order=tuple(self.cfg.sarima_order), seasonal_order=tuple(self.cfg.sarima_seasonal))
                    self._res = mod.fit(disp=False, maxiter=50)
                    self.n_refits += 1
                    res = self._res
                else:
                    res = self._res.apply(w)
                out = res.forecast(horizon)
            except (ValueError, np.linalg.LinAlgError):
                out = np.full(horizon, np.nan)
        self._calls += 1
        if not np.isfinite(out).all():
            self.n_fail += 1
            out = np.resize(w[-24:], horizon)
        return np.asarray(out)
