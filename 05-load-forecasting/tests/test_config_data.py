import numpy as np
import pandas as pd
import pytest

from tsforecast.config import load_config
from tsforecast.data import clean_hourly, split_stats, stats_markdown


def test_defaults_and_overrides():
    cfg = load_config(None, ["nn.lr=1e-4", "backtest.horizon=12", "models=[seasonal_naive]"])
    assert cfg.nn.lr == pytest.approx(1e-4) and isinstance(cfg.nn.lr, float)
    assert cfg.backtest.horizon == 12 and cfg.models == ["seasonal_naive"]


@pytest.mark.parametrize(
    "bad", ["models=[prophet]", "seeds=[0, 0]", "data.val_start=2011-01-01", "stat.sarima_order=[1, 0]"]
)
def test_validation_rejects(bad):
    with pytest.raises((ValueError, KeyError)):
        load_config(None, [bad])


def test_unknown_key():
    with pytest.raises(KeyError):
        load_config(None, ["nn.hiden=3"])


def test_clean_hourly_fills_short_and_long_gaps():
    idx = pd.date_range("2009-01-01 00:00", periods=60 * 24 * 21, freq="min")  # 3 weeks of minutes
    v = np.sin(np.arange(len(idx)) / 600.0) + 2.0
    s = pd.Series(v, index=idx)
    s.iloc[60 * 30 : 60 * 33] = np.nan  # a 3-hour gap -> interpolated
    s.iloc[60 * 24 * 10 : 60 * 24 * 11] = np.nan  # a 24-hour gap -> filled from one week earlier
    y, rep = clean_hourly(s, max_gap_hours=6)
    assert not y.isna().any() and y.index.freq is not None
    assert rep.hours_all_missing == 3 + 24
    assert rep.hours_interpolated == 3 and rep.hours_week_filled == 24
    day10 = y["2009-01-11"].to_numpy()
    assert np.allclose(day10, y["2009-01-04"].to_numpy())
    assert y.index[0].hour == 0 and len(y) % 24 == 0


def test_splits_are_ordered_and_disjoint(splits):
    assert splits.train.index.max() < splits.val.index.min()
    assert splits.val.index.max() < splits.test.index.min()
    assert splits.test.index.max() < splits.test_end
    st = split_stats(splits)
    assert st["train"]["hours"] == len(splits.train)
    assert "Autocorrelation" in stats_markdown(st)
