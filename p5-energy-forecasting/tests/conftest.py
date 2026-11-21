from __future__ import annotations

import pandas as pd
import pytest

from tsforecast.config import load_config
from tsforecast.data import load_splits, synthetic_series


@pytest.fixture
def series() -> pd.Series:
    return synthetic_series("2009-01-01", "2009-03-01", seed=0)


@pytest.fixture
def tiny_cfg(tmp_path):
    return load_config(
        None,
        [
            f"output_dir={tmp_path / 'out'}",
            "seeds=[0, 1]",
            "data.synthetic=true",
            "data.val_start=2009-11-01",
            "data.test_start=2010-01-01",
            "data.test_end=2010-01-15",
            "backtest.max_folds=5",
            "stat.window_days=7",
            "nn.lookback=48",
            "nn.hidden=8",
            "nn.layers=1",
            "nn.heads=2",
            "nn.epochs=1",
            "nn.device=cpu",
            "anomaly.n_events=3",
            "anomaly.window=48",
        ],
    )


@pytest.fixture
def splits(tiny_cfg):
    return load_splits(tiny_cfg.data)
