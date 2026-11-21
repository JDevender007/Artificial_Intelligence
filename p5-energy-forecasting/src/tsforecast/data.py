"""Download, parse and clean the UCI household power data into one hourly series, with computed statistics.

The raw file has one row per minute (Dec 2006 to Nov 2010) and marks missing readings with ``?``.
Cleaning, in order:

1. resample minutes to hourly means (an hour with no reading at all becomes NaN);
2. interpolate gaps of at most ``max_gap_hours`` linearly;
3. fill longer gaps with the value one week (168 h) earlier, repeated until filled;
4. trim to whole days (first midnight to the last complete day).

Every step is counted in ``CleanReport`` so the README can quote computed numbers, not typed ones.
"""

from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig
from .utils import get_logger

log = get_logger(__name__)
RAW_NAME = "household_power.zip"
TXT_NAME = "household_power_consumption.txt"


@dataclass
class CleanReport:
    minutes_total: int
    minutes_missing: int
    hours_total: int
    hours_all_missing: int
    hours_interpolated: int
    hours_week_filled: int
    max_gap_hours: int
    start: str
    end: str


@dataclass
class Splits:
    """One hourly series plus the boundaries. Train < val_start <= val < test_start <= test."""

    y: pd.Series
    val_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # exclusive
    report: CleanReport | None = None

    @property
    def train(self) -> pd.Series:
        return self.y[self.y.index < self.val_start]

    @property
    def val(self) -> pd.Series:
        return self.y[(self.y.index >= self.val_start) & (self.y.index < self.test_start)]

    @property
    def test(self) -> pd.Series:
        return self.y[(self.y.index >= self.test_start) & (self.y.index < self.test_end)]


def download(cfg: DataConfig) -> Path:
    path = Path(cfg.raw_dir) / RAW_NAME
    if path.exists() and zipfile.is_zipfile(path):  # a cut-off download has no central directory
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s", cfg.url)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(cfg.url, tmp)
    if not zipfile.is_zipfile(tmp):
        raise OSError(f"{tmp} is not a complete zip; delete it and retry")
    tmp.replace(path)
    return path


def read_minutes(zip_path: str | Path, target: str) -> pd.Series:
    """Minute-level target as float (NaN where the file has ``?``), indexed by timestamp."""
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.endswith(TXT_NAME))
        with z.open(name) as f:
            df = pd.read_csv(f, sep=";", usecols=["Date", "Time", target], na_values=["?"], dtype={target: "float64"})
    idx = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    return pd.Series(df[target].to_numpy(), index=idx, name=target).sort_index()


def clean_hourly(minutes: pd.Series, max_gap_hours: int = 6, freq: str = "h") -> tuple[pd.Series, CleanReport]:
    hourly = minutes.resample(freq).mean()
    all_missing = int(hourly.isna().sum())
    # 2) short gaps: linear interpolation, only inside runs of NaN no longer than max_gap_hours
    isna = hourly.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    short = isna & (run_len <= max_gap_hours)
    interp = hourly.interpolate(method="time", limit_area="inside")
    y = hourly.copy()
    y[short] = interp[short]
    n_interp = int(short.sum() - y[short].isna().sum())
    # 3) long gaps: copy the value one week earlier (repeat for gaps longer than a week)
    n_week = 0
    for _ in range(20):
        miss = y.isna()
        if not miss.any():
            break
        prev = y.shift(168)
        fill = miss & prev.notna()
        y[fill] = prev[fill]
        n_week += int(fill.sum())
    # 4) whole days only
    y = y.dropna()
    first = y.index[0].normalize()
    if y.index[0] != first:
        first += pd.Timedelta(days=1)
    last_full = (y.index[-1] + pd.Timedelta(hours=1)).normalize()
    y = y[(y.index >= first) & (y.index < last_full)].asfreq(freq)
    if y.isna().any():
        raise ValueError(f"{int(y.isna().sum())} hours still missing after cleaning")
    rep = CleanReport(
        minutes_total=int(len(minutes)),
        minutes_missing=int(minutes.isna().sum()),
        hours_total=int(len(y)),
        hours_all_missing=all_missing,
        hours_interpolated=n_interp,
        hours_week_filled=n_week,
        max_gap_hours=max_gap_hours,
        start=str(y.index[0]),
        end=str(y.index[-1]),
    )
    return y, rep


def synthetic_series(start: str, end: str, seed: int = 0, freq: str = "h") -> pd.Series:
    """Daily + weekly seasonality, a slow annual swing and noise; positive like a load in kW."""
    idx = pd.date_range(start, end, freq=freq, inclusive="left")
    rng = np.random.default_rng(seed)
    t = np.arange(len(idx))
    h = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    daily = 0.6 * np.exp(-((h - 8) ** 2) / 6) + 1.0 * np.exp(-((h - 20) ** 2) / 8)
    weekly = 0.25 * (dow >= 5)
    annual = 0.3 * np.cos(2 * np.pi * t / (24 * 365.25))
    y = 0.5 + daily + weekly + annual + rng.normal(0, 0.08, len(idx))
    return pd.Series(np.clip(y, 0.05, None), index=idx, name="load")


def load_splits(cfg: DataConfig) -> Splits:
    val_start, test_start = pd.Timestamp(cfg.val_start), pd.Timestamp(cfg.test_start)
    if cfg.synthetic:
        end = cfg.test_end or str((test_start + pd.Timedelta(days=60)).date())
        y = synthetic_series(str((val_start - pd.Timedelta(days=120)).date()), end, freq=cfg.freq)
        rep = None
    else:
        y, rep = clean_hourly(read_minutes(download(cfg), cfg.target), cfg.max_gap_hours, cfg.freq)
    test_end = pd.Timestamp(cfg.test_end) if cfg.test_end else y.index[-1] + pd.Timedelta(hours=1)
    s = Splits(y=y, val_start=val_start, test_start=test_start, test_end=test_end, report=rep)
    if len(s.train) == 0 or len(s.val) == 0 or len(s.test) == 0:
        raise ValueError("empty train, val or test split; check the dates in the config")
    return s


def split_stats(s: Splits) -> dict:
    out = {}
    for name, part in (("train", s.train), ("val", s.val), ("test", s.test)):
        out[name] = {
            "hours": int(len(part)),
            "start": str(part.index[0]),
            "end": str(part.index[-1]),
            "mean_kw": float(part.mean()),
            "std_kw": float(part.std()),
            "p95_kw": float(part.quantile(0.95)),
        }
    by_hour = s.train.groupby(s.train.index.hour).mean()
    out["train_peak_hour"] = int(by_hour.idxmax())
    out["train_trough_hour"] = int(by_hour.idxmin())
    # autocorrelation at the seasonal lags, the reason for seasonal naive baselines
    out["train_acf"] = {str(k): float(s.train.autocorr(k)) for k in (1, 24, 168)}
    if s.report is not None:
        out["cleaning"] = asdict(s.report)
    return out


def stats_markdown(st: dict) -> str:
    lines = ["| split | hours | start | end | mean kW | std kW | p95 kW |", "|---|---|---|---|---|---|---|"]
    for k in ("train", "val", "test"):
        r = st[k]
        lines.append(
            f"| {k} | {r['hours']:,} | {r['start']} | {r['end']} | {r['mean_kw']:.3f} | {r['std_kw']:.3f} | {r['p95_kw']:.3f} |"
        )
    acf = st["train_acf"]
    lines += [
        "",
        f"Train peak hour {st['train_peak_hour']}:00, trough {st['train_trough_hour']}:00. "
        f"Autocorrelation at lag 1 / 24 / 168 h: {acf['1']:.3f} / {acf['24']:.3f} / {acf['168']:.3f}.",
    ]
    if "cleaning" in st:
        c = st["cleaning"]
        lines += [
            "",
            f"Cleaning: {c['minutes_missing']:,} of {c['minutes_total']:,} minute readings missing; "
            f"{c['hours_all_missing']:,} hours had no reading, {c['hours_interpolated']:,} were interpolated "
            f"(gaps of {c['max_gap_hours']} h or less) and {c['hours_week_filled']:,} were filled from one week earlier. "
            f"Series: {c['start']} to {c['end']} ({c['hours_total']:,} hours).",
        ]
    return "\n".join(lines)
