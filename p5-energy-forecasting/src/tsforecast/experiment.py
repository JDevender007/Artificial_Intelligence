"""End-to-end runner: data -> walk-forward backtest per (model, seed) -> anomaly detection -> reports.

Each (model, seed) appends one row to ``results.jsonl`` and saves its forecasts, so an interrupted run resumes,
and every table is rebuilt from the saved forecasts by ``summarize``. Deterministic models run once.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from . import anomaly as an
from . import metrics as M
from .backtest import run_backtest
from .config import ExperimentConfig
from .data import Splits, load_splits, split_stats, stats_markdown
from .models_stat import ETS, SARIMA, SeasonalNaive
from .utils import aggregate, append_jsonl, get_logger, mean_std_markdown, read_jsonl, write_json

log = get_logger(__name__)
METRICS = ["mae", "rmse", "mase"]
DETERMINISTIC = {"seasonal_naive", "weekly_naive", "ets", "sarima"}


def build_model(name: str, cfg: ExperimentConfig, seed: int):
    if name == "seasonal_naive":
        return SeasonalNaive(24, name)
    if name == "weekly_naive":
        return SeasonalNaive(168, name)
    if name == "ets":
        return ETS(cfg.stat, cfg.backtest.season)
    if name == "sarima":
        return SARIMA(cfg.stat)
    from .models_nn import NeuralForecaster  # lazy: torch

    return NeuralForecaster(name, cfg.nn, cfg.backtest.horizon, seed)


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("numpy", "pandas", "torch", "statsmodels", "sklearn"):
        try:
            out[mod] = __import__(mod).__version__
        except ImportError:
            out[mod] = None
    try:
        import torch

        out["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return out


def _paths(out: Path, model: str, seed: int) -> tuple[Path, Path]:
    return out / "forecasts" / f"{model}_s{seed}.parquet", out / "models" / f"{model}_s{seed}.pt"


def run_models(cfg: ExperimentConfig, s: Splits, out: Path, scale: float) -> None:
    done = {(r["model"], r["seed"]) for r in read_jsonl(out / "results.jsonl")}
    for name in cfg.models:
        seeds = cfg.seeds[:1] if name in DETERMINISTIC else cfg.seeds
        for seed in seeds:
            if (name, seed) in done:
                log.info("skip %s seed %d (done)", name, seed)
                continue
            log.info("=== %s seed %d ===", name, seed)
            model = build_model(name, cfg, seed)
            info = model.fit(s.train, s.val)
            fc, t_pred = run_backtest(model, s.y, s.test_start, s.test_end, cfg.backtest)
            fc_path, model_path = _paths(out, name, seed)
            fc_path.parent.mkdir(parents=True, exist_ok=True)
            fc.to_parquet(fc_path, index=False)
            if hasattr(model, "save"):
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model.save(model_path)
            history = info.pop("history", None)
            if history:
                write_json(history, out / "train_logs" / f"{name}_s{seed}.json")
            for attr in ("n_fail", "n_refits"):
                if hasattr(model, attr):
                    info[attr] = getattr(model, attr)
            n_folds = int(fc.origin.nunique())
            row = {
                "model": name,
                "seed": seed,
                **M.point_metrics(fc, scale),
                "n_folds": n_folds,
                "predict_ms_per_origin": 1000 * t_pred / n_folds,
                "info": info,
            }
            append_jsonl(row, out / "results.jsonl")
            log.info("%s seed %d: MAE %.4f RMSE %.4f MASE %.3f", name, seed, row["mae"], row["rmse"], row["mase"])


def _detector_forecasts(det: str, cfg: ExperimentConfig, out: Path, y: pd.Series, start, end) -> pd.Series | None:
    if det == "raw_zscore":
        return None
    seed = cfg.seeds[0]
    model = build_model(det, cfg, seed)
    if hasattr(model, "load"):
        model.load(_paths(out, det, seed)[1])
    fc, _ = run_backtest(model, y, start, end, cfg.backtest)
    return pd.Series(fc.yhat.to_numpy(), index=pd.DatetimeIndex(fc.ts))


def run_anomaly(cfg: ExperimentConfig, s: Splits, out: Path) -> None:
    path = out / "anomaly.jsonl"
    done = {r["detector"] for r in read_jsonl(path)}
    a = cfg.anomaly
    sigma = float(s.train.std())
    yv, lab_v, ev_v = an.inject(s.val, a.n_events, a.kinds, sigma, a.inject_seed + 1)
    yt, lab_t, ev_t = an.inject(s.test, a.n_events, a.kinds, sigma, a.inject_seed)
    y_val_c = pd.concat([s.train, yv])
    y_test_c = pd.concat([s.train, s.val, yt])
    write_json({"val": [asdict(e) for e in ev_v], "test": [asdict(e) for e in ev_t]}, out / "anomaly_events.json")
    for det in a.detectors:
        if det in done:
            continue
        if det != "raw_zscore" and det not in cfg.models:
            log.warning("anomaly detector %s skipped: model not in this run", det)
            continue
        log.info("anomaly detector %s", det)
        fv = _detector_forecasts(det, cfg, out, y_val_c, s.val_start, s.test_start)
        zv = an.scores(yv, fv, a.window) if fv is not None else an.scores(y_val_c, None, a.window)[-len(yv) :]
        z_star, table = an.pick_threshold(zv, lab_v, ev_v, a.z_grid)
        if z_star in (min(a.z_grid), max(a.z_grid)):
            log.warning("%s: chosen z=%.1f is at the edge of z_grid %s; consider widening it", det, z_star, a.z_grid)
        ft = _detector_forecasts(det, cfg, out, y_test_c, s.test_start, s.test_end)
        zt = an.scores(yt, ft, a.window) if ft is not None else an.scores(y_test_c, None, a.window)[-len(yt) :]
        res = an.evaluate(zt > z_star, lab_t, ev_t)
        append_jsonl({"detector": det, "z": z_star, **res, "val_grid": table}, path)
        log.info("%s: z*=%.1f P %.3f R %.3f F1 %.3f", det, z_star, res["precision"], res["recall"], res["f1"])


def _fmt_table(df: pd.DataFrame, digits: int = 3) -> str:
    return df.to_markdown(index=False, floatfmt=f".{digits}f")


def summarize(cfg: ExperimentConfig, s: Splits, out: Path) -> None:
    rows = read_jsonl(out / "results.jsonl")
    if not rows:
        return
    order = [m for m in cfg.models if any(r["model"] == m for r in rows)]
    rows = sorted(rows, key=lambda r: order.index(r["model"]))
    agg = aggregate(rows, METRICS + ["predict_ms_per_origin"])
    lines = [f"# {cfg.name}: walk-forward test, {rows[0]['n_folds']} origins x {cfg.backtest.horizon} h", ""]
    lines.append(mean_std_markdown(agg, METRICS, digits=4))
    lines += ["", "Deterministic models (naive, ETS, SARIMA) run once, so they show no ±.", ""]
    extra = []
    for r in rows:
        info = r["info"]
        extra.append(
            {
                "model": r["model"],
                "seed": r["seed"],
                "params": info.get("params", ""),
                "best_epoch": info.get("best_epoch", ""),
                "train_s": info.get("train_s", float("nan")),
                "predict_ms_per_origin": r["predict_ms_per_origin"],
                "fallbacks": info.get("n_fail", ""),
            }
        )
    lines += ["## Cost", "", _fmt_table(pd.DataFrame(extra), 1), ""]

    # forecasts of the first seed per model
    fcs = {m: pd.read_parquet(_paths(out, m, next(r["seed"] for r in rows if r["model"] == m))[0]) for m in order}
    if "seasonal_naive" in fcs:
        lines += ["## Difference in MAE vs seasonal naive (paired bootstrap over forecast days, first seed)", ""]
        bt = []
        for m in order:
            if m == "seasonal_naive":
                continue
            b = M.paired_bootstrap(fcs["seasonal_naive"], fcs[m], n=1000, seed=0)
            bt.append({"model": m, "mae_diff_kw": b["diff"], "ci95_lo": b["lo"], "ci95_hi": b["hi"], "days_better": b["b_wins"]})
        if bt:
            lines += [_fmt_table(pd.DataFrame(bt), 4), "", "Negative = lower MAE than seasonal naive.", ""]
    steps = sorted({1, 6, 12, 18, cfg.backtest.horizon})
    st = pd.DataFrame({m: M.by_step(fc).reindex(steps) for m, fc in fcs.items()}).T
    st.columns = [f"h={c}" for c in st.columns]
    lines += ["## MAE (kW) by horizon step", "", st.reset_index(names="model").to_markdown(index=False, floatfmt=".4f")]
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(rows).drop(columns=["info"]).to_csv(out / "results.csv", index=False)

    best = agg.sort_values("mae_mean").iloc[0]["model"]
    write_error_analysis(fcs, best, out)
    write_figures(fcs, out)
    an_rows = read_jsonl(out / "anomaly.jsonl")
    if an_rows:
        keep = ["detector", "z", "precision", "recall", "f1", "event_recall"]
        keep += [c for c in an_rows[0] if c.startswith("recall_")] + ["false_alarms_per_week"]
        tab = pd.DataFrame(an_rows)[keep]
        text = [
            "# Anomaly detection on injected events (test period)",
            "",
            f"{cfg.anomaly.n_events} events per period ({', '.join(cfg.anomaly.kinds)}); the z threshold is picked on a "
            "separate validation injection and applied unchanged here.",
            "",
            _fmt_table(tab),
        ]
        (out / "anomaly.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def write_error_analysis(fcs: dict[str, pd.DataFrame], best: str, out: Path) -> None:
    fc = fcs[best]
    lines = [f"# Error analysis: {best} (lowest mean MAE)", ""]
    hr = pd.DataFrame({m: M.by_hour(f) for m, f in fcs.items()})
    lines += ["## MAE (kW) by hour of day", "", hr.reset_index(names="hour").to_markdown(index=False, floatfmt=".3f"), ""]
    dt = pd.DataFrame({m: M.by_daytype(f) for m, f in fcs.items()})
    lines += ["## Weekday vs weekend", "", dt.reset_index(names="day").to_markdown(index=False, floatfmt=".4f"), ""]
    mo = pd.DataFrame({m: M.by_month(f) for m, f in fcs.items()})
    lines += ["## By month", "", mo.reset_index(names="month").to_markdown(index=False, floatfmt=".4f"), ""]
    fm = M.fold_mae(fc).sort_values(ascending=False)
    worst = pd.DataFrame({"origin": fm.index[:8].astype(str), f"{best}_mae": fm.to_numpy()[:8]})
    if "seasonal_naive" in fcs and best != "seasonal_naive":
        sn = M.fold_mae(fcs["seasonal_naive"])
        worst["seasonal_naive_mae"] = sn.reindex(fm.index[:8]).to_numpy()
    day_tot = fc.groupby("origin").y.mean().reindex(fm.index[:8]).to_numpy()
    worst["mean_load_kw"] = day_tot
    lines += ["## Worst 8 forecast days", "", worst.to_markdown(index=False, floatfmt=".3f"), ""]
    q = fm.quantile([0.5, 0.9, 0.99])
    lines += [f"Daily MAE of {best}: median {q.iloc[0]:.3f}, p90 {q.iloc[1]:.3f}, p99 {q.iloc[2]:.3f} kW."]
    (out / "error_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(fcs: dict[str, pd.DataFrame], out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    any_fc = next(iter(fcs.values()))
    first = sorted(any_fc.origin.unique())
    week = set(first[7:14]) if len(first) >= 14 else set(first)
    fig, ax = plt.subplots(figsize=(12, 4))
    sub = any_fc[any_fc.origin.isin(week)]
    ax.plot(sub.ts, sub.y, color="black", lw=1.5, label="actual")
    for m, fc in fcs.items():
        f = fc[fc.origin.isin(week)]
        ax.plot(f.ts, f.yhat, lw=1, label=m)
    ax.set_ylabel("kW (hourly mean)")
    ax.legend(ncol=4, fontsize=8)
    ax.set_title("One test week of 24-h-ahead forecasts (one origin per midnight)")
    fig.tight_layout()
    fig.savefig(fig_dir / "forecast_week.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, fc in fcs.items():
        ax.plot(M.by_step(fc), label=m)
    ax.set_xlabel("hours ahead")
    ax.set_ylabel("MAE (kW)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "mae_by_step.png", dpi=120)
    plt.close(fig)


def run_experiment(cfg: ExperimentConfig) -> Path:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    write_json(_versions(), out / "versions.json")
    s = load_splits(cfg.data)
    st = split_stats(s)
    write_json(st, out / "data_summary.json")
    (out / "data_summary.md").write_text(stats_markdown(st) + "\n", encoding="utf-8")
    scale = M.mase_scale(s.train.to_numpy(), cfg.backtest.season)
    write_json({"mase_scale_kw": scale, "season": cfg.backtest.season}, out / "mase_scale.json")
    log.info("data: %s", json.dumps({k: st[k]["hours"] for k in ("train", "val", "test")}))
    run_models(cfg, s, out, scale)
    run_anomaly(cfg, s, out)
    summarize(cfg, s, out)
    return out

