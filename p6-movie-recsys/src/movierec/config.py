"""Experiment configuration: dataclasses loaded from YAML, with ``--set key=value`` overrides."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
# two_tower_* are ablations of the main model: no side features, no logQ correction, no extra random negatives
MODELS = ("popularity", "bpr_mf", "two_tower", "two_tower_nofeat", "two_tower_nologq", "two_tower_inbatch")
DETERMINISTIC = {"popularity"}


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    url: str = ML1M_URL
    min_rating: int = 4  # a rating >= 4 is a positive (the user liked it); every rating counts as "seen"
    val_frac: float = 0.1  # global temporal split on positives: last 10% by time = test, the 10% before = val
    test_frac: float = 0.1
    synthetic: bool = False  # tests and smoke runs: write a small fake ml-1m.zip instead of downloading
    synthetic_users: int = 300
    synthetic_items: int = 120


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 1024
    patience: int = 3  # early stopping on val Recall@k_main
    device: str = "auto"


@dataclass
class BPRConfig:
    dim: int = 64
    lr: float = 5e-3
    reg: float = 1e-5  # L2 on the embeddings in each batch


@dataclass
class TowerConfig:
    dim: int = 64  # output embedding size
    emb: int = 32  # size of every input embedding (ids, features, history)
    hidden: int = 128
    history_len: int = 30  # last positives before each interaction (causal)
    temperature: float = 0.05
    n_random_neg: int = 256  # uniform negatives shared by the batch, on top of in-batch negatives
    logq: bool = True  # subtract log(sampling probability) from in-batch logits (popularity correction)
    use_features: bool = True  # user gender/age/occupation, item genres/year
    id_dropout: float = 0.1  # map ids to "unknown" at this rate so cold users/items get a trained embedding
    lr: float = 1e-3
    weight_decay: float = 1e-6


@dataclass
class EvalConfig:
    ks: list[int] = field(default_factory=lambda: [10, 20, 50])
    k_main: int = 20
    n_bootstrap: int = 1000
    hnsw_m: int = 32
    hnsw_ef: list[int] = field(default_factory=lambda: [16, 32, 64, 128])
    n_latency_queries: int = 1000
    n_examples: int = 5  # users shown with their history and top-10 in error_analysis.md


@dataclass
class ExperimentConfig:
    name: str = "ml1m"
    output_dir: str = "outputs/ml1m"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    models: list[str] = field(default_factory=lambda: list(MODELS))
    serve_model: str = "two_tower"  # seed 0 of this model is exported for FAISS + the API
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    bpr: BPRConfig = field(default_factory=BPRConfig)
    tower: TowerConfig = field(default_factory=TowerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tower_variant(base: TowerConfig, name: str) -> TowerConfig:
    """The ablation ``name`` applied to the main two-tower config."""
    d = asdict(base)
    if name == "two_tower_nofeat":
        d["use_features"] = False
    elif name == "two_tower_nologq":
        d["logq"] = False
    elif name == "two_tower_inbatch":
        d["n_random_neg"] = 0
    elif name != "two_tower":
        raise ValueError(name)
    return TowerConfig(**d)


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
        elif isinstance(default, float) and isinstance(v, str | int) and not isinstance(v, bool):
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
    d = cfg.data
    if not (d.val_frac > 0 and d.test_frac > 0 and d.val_frac + d.test_frac < 1):
        raise ValueError("val_frac and test_frac must be > 0 and sum to < 1")
    if cfg.eval.k_main not in cfg.eval.ks:
        raise ValueError("eval.k_main must be one of eval.ks")
    if cfg.serve_model not in MODELS or cfg.serve_model in ("popularity",):
        raise ValueError("serve_model must be an embedding model")
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load YAML (missing keys keep their defaults), apply ``key.sub=value`` overrides, validate."""
    raw: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for o in overrides or []:
        _apply_override(raw, o)
    return validate(_build(ExperimentConfig, raw))
