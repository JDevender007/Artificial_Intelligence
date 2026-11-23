import json

import numpy as np
import pandas as pd
import pytest

from tsforecast import anomaly as an
from tsforecast.experiment import run_experiment


def test_inject_labels_match_events(series):
    yc, lab, ev = an.inject(series, 9, ["spike", "dropout", "level_shift"], sigma=0.5, seed=1)
    assert len(ev) == 9 and lab.sum() == sum(e.length for e in ev)
    assert {e.kind for e in ev} == {"spike", "dropout", "level_shift"}
    changed = ~np.isclose(yc.to_numpy(), series.to_numpy())
    assert not (changed & ~lab).any()  # nothing outside the labels was touched
    for a, b in zip(ev, ev[1:], strict=False):
        assert b.start - (a.start + a.length) >= 48


def test_robust_z_is_causal():
    r = pd.Series(np.r_[np.random.default_rng(0).normal(0, 1, 300), [50.0], np.zeros(10)])
    z = an.robust_z(r, 100)
    r2 = r.copy()
    r2.iloc[301:] = 1e6
    assert np.allclose(an.robust_z(r2, 100)[:301], z[:301])
    assert z[300] > 20


def test_evaluate_and_threshold():
    lab = np.zeros(100, dtype=bool)
    lab[10:12] = lab[50:55] = True
    ev = [an.Event("spike", 10, 2, 1.0), an.Event("dropout", 50, 5, -1.0)]
    z = np.zeros(100)
    z[10:12] = 8
    z[50:55] = 4
    z[80] = 3.2
    res = an.evaluate(z > 3.5, lab, ev)
    assert res["precision"] == 1.0 and res["recall"] == 1.0 and res["event_recall"] == 1.0
    best, table = an.pick_threshold(z, lab, ev, [3.0, 3.5, 5.0])
    assert best == 3.5 and len(table) == 3


def test_seasonal_residual_detector_finds_spikes(series):
    from tsforecast.backtest import run_backtest
    from tsforecast.config import BacktestConfig
    from tsforecast.models_stat import SeasonalNaive

    period = series["2009-02-01":]
    yc, lab, ev = an.inject(period, 4, ["spike"], sigma=float(series.std()), seed=2)
    full = pd.concat([series[: "2009-01-31"], yc])
    fc, _ = run_backtest(SeasonalNaive(168, "weekly_naive"), full, period.index[0], period.index[-1] + pd.Timedelta(hours=1), BacktestConfig())
    z = an.scores(yc, pd.Series(fc.yhat.to_numpy(), index=pd.DatetimeIndex(fc.ts)), 168)
    assert an.evaluate(z > 4, lab, ev)["event_recall"] == 1.0


def test_end_to_end_smoke_and_resume(tiny_cfg):
    pytest.importorskip("torch")
    models = ["seasonal_naive", "weekly_naive", "lstm"]
    tiny_cfg.models = models
    tiny_cfg.anomaly.detectors = ["raw_zscore", "seasonal_naive", "lstm"]
    out = run_experiment(tiny_cfg)
    rows = [json.loads(x) for x in (out / "results.jsonl").read_text().splitlines()]
    assert [(r["model"], r["seed"]) for r in rows] == [("seasonal_naive", 0), ("weekly_naive", 0), ("lstm", 0), ("lstm", 1)]
    assert all(r["n_folds"] == 5 for r in rows)
    for f in ("results.md", "results.csv", "error_analysis.md", "anomaly.md", "data_summary.md", "versions.json"):
        assert (out / f).exists(), f
    md = (out / "results.md").read_text(encoding="utf-8")
    assert "| lstm |" in md and "±" in md and "vs seasonal naive" in md
    assert (out / "figures" / "forecast_week.png").exists()
    n_an = len((out / "anomaly.jsonl").read_text().splitlines())
    run_experiment(tiny_cfg)  # resume: nothing re-run
    assert len((out / "results.jsonl").read_text().splitlines()) == 4
    assert len((out / "anomaly.jsonl").read_text().splitlines()) == n_an == 3
