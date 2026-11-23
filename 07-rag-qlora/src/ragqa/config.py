"""Experiment configuration: dataclasses loaded from YAML, with ``--set key=value`` overrides."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

SQUAD_TRAIN_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
SQUAD_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
CHUNKERS = ("paragraph", "window")
METHODS = ("bm25", "dense", "hybrid", "hybrid_rerank")
# Generation variants without training. Fine-tuned variants (ft_closed, rag_ft) come from ``ft.modes`` x seeds.
BASE_VARIANTS = ("base", "prompted", "rag", "rag_nf4")
FT_MODES = ("closed", "rag")
FT_VARIANT = {"closed": "ft_closed", "rag": "rag_ft"}


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    train_url: str = SQUAD_TRAIN_URL
    dev_url: str = SQUAD_DEV_URL
    synthetic: bool = False  # tests: write small fake SQuAD-format files instead of downloading
    synthetic_articles: int = 12
    n_val_articles: int = 8  # dev articles held out for choosing the retrieval setup; the rest are the test set
    n_val_questions: int = 300
    n_test_questions: int = 1000
    n_train_examples: int = 3000  # sampled from SQuAD train (disjoint articles) for the QLoRA runs
    split_seed: int = 0


@dataclass
class RetrievalConfig:
    chunkers: list[str] = field(default_factory=lambda: list(CHUNKERS))
    window_words: int = 100
    window_overlap: int = 25
    methods: list[str] = field(default_factory=lambda: list(METHODS))
    ks: list[int] = field(default_factory=lambda: [1, 5, 20])
    k_context: int = 5  # passages given to the generator; the setup is chosen by val recall@k_context
    n_candidates: int = 20  # per retriever before fusion; the reranker reorders this many
    rrf_k: int = 60
    embedder: str = "BAAI/bge-small-en-v1.5"
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    reranker: str = "BAAI/bge-reranker-base"
    batch_size: int = 64
    max_len: int = 512


@dataclass
class GenConfig:
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    variants: list[str] = field(default_factory=lambda: list(BASE_VARIANTS))
    dtype: str = "float16"  # T4 has no bf16; set float32 if fp16 outputs look broken
    max_new_tokens: int = 32
    batch_size: int = 16


@dataclass
class FTConfig:
    modes: list[str] = field(default_factory=lambda: list(FT_MODES))
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    lr: float = 2e-4
    epochs: int = 1
    batch_size: int = 4
    grad_accum: int = 4
    max_seq_len: int = 1536  # longer training examples are skipped (and counted)
    warmup_ratio: float = 0.03
    p_drop_gold: float = 0.2  # rag mode: share of answerable examples whose supporting passages are removed
    max_steps: int = 0  # 0 = full epochs; smoke runs cap it


@dataclass
class ExperimentConfig:
    name: str = "squad"
    output_dir: str = "outputs/squad"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    device: str = "auto"
    n_bootstrap: int = 1000
    n_examples: int = 8  # failure examples per category in error_analysis.md
    data: DataConfig = field(default_factory=DataConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    gen: GenConfig = field(default_factory=GenConfig)
    ft: FTConfig = field(default_factory=FTConfig)

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
        elif isinstance(default, float) and isinstance(v, str | int) and not isinstance(v, bool):
            v = float(v)  # PyYAML (YAML 1.1) reads "2e-4" without a dot as a string
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
    r, g = cfg.retrieval, cfg.gen
    for name, got, allowed in [("retrieval.chunkers", r.chunkers, CHUNKERS), ("retrieval.methods", r.methods, METHODS),
                               ("gen.variants", g.variants, BASE_VARIANTS), ("ft.modes", cfg.ft.modes, FT_MODES)]:
        bad = [x for x in got if x not in allowed]
        if bad:
            raise ValueError(f"unknown {name} {bad}; choose from {allowed}")
    if not r.chunkers or not r.methods:
        raise ValueError("need at least one chunker and one retrieval method")
    if r.k_context not in r.ks or max(r.ks) > r.n_candidates:
        raise ValueError("retrieval.k_context must be in retrieval.ks, and max(ks) <= n_candidates")
    if not 0 <= r.window_overlap < r.window_words:
        raise ValueError("need 0 <= window_overlap < window_words")
    if not cfg.seeds or len(set(cfg.seeds)) != len(cfg.seeds):
        raise ValueError("seeds must be a non-empty list of distinct ints")
    if g.dtype not in ("float16", "float32", "bfloat16"):
        raise ValueError("gen.dtype must be float16, float32 or bfloat16")
    if not 0 <= cfg.ft.p_drop_gold < 1:
        raise ValueError("ft.p_drop_gold must be in [0, 1)")
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load YAML (missing keys keep their defaults), apply ``key.sub=value`` overrides, validate."""
    raw: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for o in overrides or []:
        _apply_override(raw, o)
    return validate(_build(ExperimentConfig, raw))
