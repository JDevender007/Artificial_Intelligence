import numpy as np
import pandas as pd
import pytest

from tsforecast import metrics as M
from tsforecast.backtest import origins, run_backtest
from tsforecast.config import BacktestConfig, NNConfig, StatConfig
from tsforecast.models_stat import SeasonalNaive


def test_metrics_known_values():
    y, yhat = np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 5.0])
    assert M.mae(y, yhat) == pytest.approx(1.0)
    assert M.rmse(y, yhat) == pytest.approx(np.sqrt(5 / 3))
    assert M.mase_scale(np.array([1.0, 2.0, 1.0, 2.0, 4.0]), 2) == pytest.approx(1.0)


def test_origins_tile_the_period(series):
    cfg = BacktestConfig(horizon=24, stride=24)
    start, end = pd.Timestamp("2009-02-01"), pd.Timestamp("2009-02-08")
    pos = origins(series, start, end, cfg)
    assert len(pos) == 7 and series.index[pos[0]] == start
    assert all(b - a == 24 for a, b in zip(pos, pos[1:], strict=False))


class _Spy:
    """Records the last timestamp it was shown, to prove no future data leaks into a forecast."""

    name = "spy"

    def __init__(self):
        self.seen = []

    def fit(self, train, val):
        return {}

    def predict(self, hist, horizon):
        self.seen.append(hist.index[-1])
        return np.zeros(horizon)


def test_no_lookahead(series):
    spy = _Spy()
    fc, _ = run_backtest(spy, series, pd.Timestamp("2009-02-01"), pd.Timestamp("2009-02-05"), BacktestConfig())
    for last_seen, origin in zip(spy.seen, sorted(fc.origin.unique()), strict=True):
        assert last_seen < origin
    assert len(fc) == 4 * 24 and set(fc.step) == set(range(1, 25))


def test_future_corruption_does_not_change_forecasts(series):
    cfg = BacktestConfig()
    a, _ = run_backtest(SeasonalNaive(24), series, pd.Timestamp("2009-02-01"), pd.Timestamp("2009-02-03"), cfg)
    s2 = series.copy()
    s2[s2.index >= pd.Timestamp("2009-02-02")] = 1e6
    b, _ = run_backtest(SeasonalNaive(24), s2, pd.Timestamp("2009-02-01"), pd.Timestamp("2009-02-03"), cfg)
    first = a.origin == pd.Timestamp("2009-02-01")
    assert np.allclose(a.yhat[first], b.yhat[first])


def test_seasonal_naive_repeats_last_season():
    hist = pd.Series(np.arange(48.0), index=pd.date_range("2009-01-01", periods=48, freq="h"))
    assert np.array_equal(SeasonalNaive(24).predict(hist, 24), np.arange(24.0, 48.0))
    assert np.array_equal(SeasonalNaive(24).predict(hist, 30)[24:], np.arange(24.0, 30.0))


def test_seasonal_naive_is_perfect_on_a_pure_daily_cycle():
    idx = pd.date_range("2009-01-01", periods=24 * 20, freq="h")
    y = pd.Series(np.tile(np.arange(24.0), 20), index=idx)
    fc, _ = run_backtest(SeasonalNaive(24), y, idx[24 * 10], idx[-1] + pd.Timedelta(hours=1), BacktestConfig())
    assert M.mae(fc.y, fc.yhat) == 0


def test_paired_bootstrap_sign(series):
    cfg = BacktestConfig()
    start, end = pd.Timestamp("2009-02-01"), pd.Timestamp("2009-02-20")
    good, _ = run_backtest(SeasonalNaive(24), series, start, end, cfg)
    bad = good.assign(yhat=good.yhat + 1.0)
    r = M.paired_bootstrap(bad, good, n=200)
    assert r["diff"] < 0 and r["hi"] < 0 and r["b_wins"] == 1.0


def test_ets_and_sarima_beat_a_constant(series):
    pytest.importorskip("statsmodels")
    from tsforecast.models_stat import ETS, SARIMA

    cfg = BacktestConfig()
    start, end = pd.Timestamp("2009-02-10"), pd.Timestamp("2009-02-13")
    st = StatConfig(window_days=10, sarima_order=[1, 0, 0], sarima_seasonal=[0, 1, 1, 24], refit_every=2)
    for model in (ETS(st), SARIMA(st)):
        fc, _ = run_backtest(model, series, start, end, cfg)
        assert M.mae(fc.y, fc.yhat) < M.mae(fc.y, np.full(len(fc), fc.y.mean()))
    assert model.n_refits == 2  # 3 origins, refit_every=2 -> refit at origins 0 and 2


@pytest.mark.parametrize("kind", ["lstm", "transformer"])
def test_neural_fit_predict_save_load(kind, series, tmp_path):
    pytest.importorskip("torch")
    from tsforecast.models_nn import NeuralForecaster

    cfg = NNConfig(lookback=48, hidden=8, layers=1, heads=2, epochs=2, batch_size=64, device="cpu")
    train, val = series[: 24 * 40], series[24 * 40 : 24 * 50]
    m = NeuralForecaster(kind, cfg, horizon=24, seed=0)
    info = m.fit(train, val)
    assert info["params"] > 0 and 1 <= info["best_epoch"] <= 2
    p = m.predict(series[: 24 * 52], 24)
    assert p.shape == (24,) and np.isfinite(p).all()
    m.save(tmp_path / "m.pt")
    m2 = NeuralForecaster(kind, cfg, horizon=24, seed=0).load(tmp_path / "m.pt")
    assert np.allclose(m2.predict(series[: 24 * 52], 24), p)
    m3 = NeuralForecaster(kind, cfg, horizon=24, seed=0)
    m3.fit(train, val)
    assert np.allclose(m3.predict(series[: 24 * 52], 24), p, atol=1e-5)  # same seed -> same model
