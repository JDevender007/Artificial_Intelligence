"""Walk-forward (rolling-origin) backtest.

At each origin ``t0`` the forecaster receives only ``y[:t0]`` (everything strictly before the origin) and returns
``horizon`` values. With the default stride = horizon = 24, origins are every midnight of the test period and the
forecasts tile the test period without overlap, so each test hour is scored exactly once.
"""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .utils import get_logger

log = get_logger(__name__)


class Forecaster(Protocol):
    name: str

    def fit(self, train: pd.Series, val: pd.Series) -> dict: ...

    def predict(self, hist: pd.Series, horizon: int) -> np.ndarray: ...


def origins(y: pd.Series, start: pd.Timestamp, end: pd.Timestamp, cfg: BacktestConfig) -> list[int]:
    """Integer positions of forecast origins in ``[start, end - horizon]``, every ``stride`` hours."""
    t_start = int(y.index.searchsorted(start))
    t_end = int(y.index.searchsorted(end))  # exclusive
    out = list(range(t_start, t_end - cfg.horizon + 1, cfg.stride))
    if cfg.max_folds is not None:
        out = out[: cfg.max_folds]
    if not out:
        raise ValueError("no forecast origins: the period is shorter than the horizon")
    return out


def run_backtest(
    model: Forecaster, y: pd.Series, start: pd.Timestamp, end: pd.Timestamp, cfg: BacktestConfig
) -> tuple[pd.DataFrame, float]:
    """Forecasts as a long frame (origin, step, ts, y, yhat) and the total predict time in seconds."""
    rows = []
    t_pred = 0.0
    pos = origins(y, start, end, cfg)
    vals = y.to_numpy()
    for i, t0 in enumerate(pos):
        hist = y.iloc[:t0]  # nothing at or after the origin
        t = time.perf_counter()
        yhat = np.asarray(model.predict(hist, cfg.horizon), dtype=float)
        t_pred += time.perf_counter() - t
        if yhat.shape != (cfg.horizon,) or not np.isfinite(yhat).all():
            raise ValueError(f"{model.name} returned shape {yhat.shape} or non-finite values at origin {y.index[t0]}")
        rows.append(
            pd.DataFrame(
                {
                    "origin": y.index[t0],
                    "step": np.arange(1, cfg.horizon + 1),
                    "ts": y.index[t0 : t0 + cfg.horizon],
                    "y": vals[t0 : t0 + cfg.horizon],
                    "yhat": yhat,
                }
            )
        )
        if (i + 1) % 50 == 0:
            log.info("%s: %d/%d folds", model.name, i + 1, len(pos))
    return pd.concat(rows, ignore_index=True), t_pred
