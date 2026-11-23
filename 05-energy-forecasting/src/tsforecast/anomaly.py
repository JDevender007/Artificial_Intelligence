"""Anomaly detection on the load series, evaluated on *injected* anomalies with known labels.

The raw data has no anomaly labels, so we corrupt a copy of a period with three kinds of synthetic events and
measure how well each detector finds them:

- ``spike``: 1-2 hours raised by 3-6 training standard deviations (a short heavy appliance or a meter glitch);
- ``dropout``: 3-12 hours set to ~0 kW (meter or supply failure);
- ``level_shift``: 6-24 hours shifted by +/- 1.5-3 standard deviations (a device left on, or a mis-scaled meter).

Detectors score each hour with a *causal* robust z-score of a residual: (r_t - median) / (1.4826 * MAD), where the
median and MAD come from the previous ``window`` hours only. Residuals are either the raw value
(``raw_zscore``, no model) or the observed value minus a 24-h-ahead forecast. The threshold is picked on a
validation-period injection (different seed) and then applied unchanged to the test period.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

KIND_LENGTHS = {"spike": (1, 2), "dropout": (3, 12), "level_shift": (6, 24)}


@dataclass
class Event:
    kind: str
    start: int  # position in the series
    length: int
    magnitude: float


def inject(y: pd.Series, n_events: int, kinds: list[str], sigma: float, seed: int, margin: int = 48):
    """Corrupted copy of ``y``, a boolean label per hour, and the event list. Events never overlap and keep
    ``margin`` hours apart so each one can be scored on its own."""
    rng = np.random.default_rng(seed)
    v = y.to_numpy(dtype=float).copy()
    lab = np.zeros(len(v), dtype=bool)
    events: list[Event] = []
    tries = 0
    while len(events) < n_events and tries < 100 * n_events:
        tries += 1
        kind = kinds[len(events) % len(kinds)]
        lo, hi = KIND_LENGTHS[kind]
        length = int(rng.integers(lo, hi + 1))
        s = int(rng.integers(margin, len(v) - length - margin))
        if lab[max(0, s - margin) : s + length + margin].any():
            continue
        if kind == "spike":
            mag = float(rng.uniform(3, 6)) * sigma
            v[s : s + length] += mag
        elif kind == "dropout":
            mag = -float(np.mean(v[s : s + length]))
            v[s : s + length] = 0.02
        else:
            mag = float(rng.uniform(1.5, 3) * sigma * rng.choice([-1, 1]))
            v[s : s + length] = np.clip(v[s : s + length] + mag, 0.02, None)
        lab[s : s + length] = True
        events.append(Event(kind, s, length, mag))
    if len(events) < n_events:
        raise ValueError(f"could only place {len(events)} of {n_events} events; the period is too short")
    return pd.Series(v, index=y.index, name=y.name), lab, sorted(events, key=lambda e: e.start)


def robust_z(r: pd.Series, window: int) -> np.ndarray:
    """|r_t - median(r_{t-window..t-1})| / (1.4826 * MAD). Uses past values only; the first hours get 0."""
    warm = max(24, window // 4)
    med = r.shift(1).rolling(window, min_periods=warm).median()
    # MAD around the current past-median; min_periods=1 here so the warm-up is ``warm`` hours, not twice that
    mad = (r.shift(1) - med).abs().rolling(window, min_periods=1).median()
    z = (r - med).abs() / (1.4826 * mad.replace(0, np.nan))
    return z.fillna(0).to_numpy()


def scores(observed: pd.Series, forecast: pd.Series | None, window: int) -> np.ndarray:
    resid = observed if forecast is None else observed - forecast.reindex(observed.index)
    return robust_z(resid.fillna(0), window)


def evaluate(flags: np.ndarray, labels: np.ndarray, events: list[Event], hours_per_week: int = 168) -> dict:
    tp = int((flags & labels).sum())
    fp = int((flags & ~labels).sum())
    fn = int((~flags & labels).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    hit = {k: [] for k in KIND_LENGTHS}
    for e in events:
        hit[e.kind].append(bool(flags[e.start : e.start + e.length].any()))
    out = {
        "precision": p,
        "recall": r,
        "f1": f1,
        "event_recall": float(np.mean([h for v in hit.values() for h in v])),
        "false_alarms_per_week": fp / (len(flags) / hours_per_week),
    }
    for k, v in hit.items():
        if v:
            out[f"recall_{k}"] = float(np.mean(v))
    return out


def pick_threshold(z: np.ndarray, labels: np.ndarray, events: list[Event], grid: list[float]) -> tuple[float, list[dict]]:
    """Best point-wise F1 on the validation injection; ties go to the higher (more conservative) threshold."""
    table = [{"z": g, **evaluate(z > g, labels, events)} for g in grid]
    best = max(table, key=lambda r: (round(r["f1"], 6), r["z"]))
    return best["z"], table
