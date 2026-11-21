"""Experiment configuration: dataclasses loaded from YAML, with ``--set key=value`` overrides."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

BASE_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
MODELS = ("majority", "tfidf_lr", "distilbert")


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    train_url: str = f"{BASE_URL}/train.csv"
    test_url: str = f"{BASE_URL}/test.csv"
    val_fraction: float = 0.1
    split_seed: int = 42
    max_train: int | None = None  # subsample training rows (smoke runs only)
    length_edges: list[int] = field(default_factory=lambda: [5, 10, 15, 25])  # word-count bucket edges


@dataclass
class TfidfConfig:
    word_ngrams: list[int] = field(default_factory=lambda: [1, 2])
    char_ngrams: list[int] = field(default_factory=lambda: [2, 5])  # [] disables char features
    min_df: int = 1
    c_grid: list[float] = field(default_factory=lambda: [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0])
    max_iter: int = 3000


@dataclass
class BertConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 64
    epochs: int = 5
    batch_size: int = 32
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    patience: int = 2
    fp16: bool = True
    eval_batch_size: int = 128


@dataclass
class LatencyConfig:
    batch_sizes: list[int] = field(default_factory=lambda: [1, 32])
    n_texts: int = 256
    warmup: int = 5
    devices: list[str] = field(default_factory=lambda: ["cpu", "cuda"])


@dataclass
class ExperimentConfig:
    name: str = "banking77"
    output_dir: str = "outputs/banking77"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    models: list[str] = field(default_factory=lambda: list(MODELS))
    n_bootstrap: int = 1000
    data: DataConfig = field(default_factory=DataConfig)
    tfidf: TfidfConfig = field(default_factory=TfidfConfig)
    bert: BertConfig = field(default_factory=BertConfig)
    latency: LatencyConfig = field(default_factory=LatencyConfig)

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
    if not 0 < cfg.data.val_fraction < 0.5:
        raise ValueError("data.val_fraction must be in (0, 0.5)")
    if sorted(cfg.data.length_edges) != cfg.data.length_edges:
        raise ValueError("data.length_edges must be increasing")
    if len(cfg.tfidf.word_ngrams) != 2 or cfg.tfidf.char_ngrams and len(cfg.tfidf.char_ngrams) != 2:
        raise ValueError("ngram ranges must be [min, max]")
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load YAML (missing keys keep their defaults), apply ``key.sub=value`` overrides, validate."""
    raw: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for o in overrides or []:
        _apply_override(raw, o)
    return validate(_build(ExperimentConfig, raw))
