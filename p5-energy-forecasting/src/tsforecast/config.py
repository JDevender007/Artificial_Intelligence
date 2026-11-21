"""Experiment configuration: dataclasses loaded from YAML, with ``--set key=value`` overrides."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

UCI_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
MODELS = ("seasonal_naive", "weekly_naive", "ets", "sarima", "lstm", "transformer")


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    url: str = UCI_URL
    target: str = "Global_active_power"  # kW, minute averages; resampled to hourly means
    freq: str = "h"
    max_gap_hours: int = 6  # longer gaps are filled with the value one week earlier
    val_start: str = "2009-07-01"  # neural early stopping only
    test_start: str = "2010-01-01"  # walk-forward origins start here
    test_end: str | None = None  # None = last full day in the data
    synthetic: bool = False  # tests and smoke runs: generate a series instead of downloading


@dataclass
class BacktestConfig:
    horizon: int = 24  # hours ahead
    stride: int = 24  # hours between forecast origins (24 + horizon 24 = contiguous, non-overlapping)
    season: int = 24  # MASE scale: in-sample seasonal-naive MAE with this lag
    max_folds: int | None = None  # None = every origin in the test period


@dataclass
class StatConfig:
    window_days: int = 28  # ETS / SARIMA fit on this many days before each origin
    sarima_order: list[int] = field(default_factory=lambda: [2, 0, 1])
    sarima_seasonal: list[int] = field(default_factory=lambda: [1, 1, 1, 24])
    refit_every: int = 7  # SARIMA: re-estimate parameters every k folds; between refits only re-filter new data


@dataclass
class NNConfig:
    lookback: int = 168  # one week of hourly inputs
    hidden: int = 64  # LSTM hidden size / transformer d_model
    layers: int = 2
    heads: int = 4
    dropout: float = 0.1
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    device: str = "auto"


@dataclass
class AnomalyConfig:
    n_events: int = 40  # injected events in the test period
    kinds: list[str] = field(default_factory=lambda: ["spike", "dropout", "level_shift"])
    z_grid: list[float] = field(default_factory=lambda: [2.5, 3.0, 3.5, 4.0, 5.0, 6.0])
    window: int = 168  # rolling window (hours) for the robust z-score
    detectors: list[str] = field(default_factory=lambda: ["raw_zscore", "seasonal_naive", "lstm"])
    inject_seed: int = 123


@dataclass
class ExperimentConfig:
    name: str = "household_power"
    output_dir: str = "outputs/household_power"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    models: list[str] = field(default_factory=lambda: list(MODELS))
    data: DataConfig = field(default_factory=DataConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    stat: StatConfig = field(default_factory=StatConfig)
    nn: NNConfig = field(default_factory=NNConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls: type, values: dict[str, Any]) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise KeyError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for k, v in values.items():
        default = known[k].default_factory() if callable(known[k].default_factory) else known[k].default
        if is_dataclass(default) and isinstance(v, dict):
            v = _build(type(default), v)
        elif isinstance(default, float) and isinstance(v, str):
            v = float(v)  # PyYAML (YAML 1.1) reads "1e-4" without a dot as a string
        elif isinstance(v, date):
            v = str(v)  # and reads unquoted 2010-01-01 as a datetime.date; the configs keep dates as strings
        kwargs[k] = v
    return cls(**kwargs)


def _apply_override(d: dict[str, Any], dotted: str) -> None:
    if "=" not in dotted:
        raise ValueError(f"override must be key=value, got {dotted!r}")
    key, raw = dotted.split("=", 1)
    *parents, leaf = key.strip().split(".")
    node = d
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = yaml.safe_load(raw)


def validate(cfg: ExperimentConfig) -> ExperimentConfig:
    bad = [m for m in cfg.models if m not in MODELS]
    if bad:
        raise ValueError(f"unknown models {bad}; choose from {MODELS}")
    if not cfg.seeds or len(set(cfg.seeds)) != len(cfg.seeds):
        raise ValueError("seeds must be a non-empty list of distinct ints")
    b = cfg.backtest
    if b.horizon < 1 or b.stride < 1 or b.season < 1:
        raise ValueError("backtest.horizon, stride and season must be >= 1")
    if cfg.data.val_start >= cfg.data.test_start:
        raise ValueError("data.val_start must be before data.test_start")
    if len(cfg.stat.sarima_order) != 3 or len(cfg.stat.sarima_seasonal) != 4:
        raise ValueError("sarima_order is [p,d,q] and sarima_seasonal is [P,D,Q,s]")
    bad_det = [d for d in cfg.anomaly.detectors if d not in ("raw_zscore", *MODELS)]
    if bad_det:
        raise ValueError(f"unknown anomaly detectors {bad_det}")
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load YAML (missing keys keep their defaults), apply ``key.sub=value`` overrides, validate."""
    raw: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for o in overrides or []:
        _apply_override(raw, o)
    return validate(_build(ExperimentConfig, raw))
