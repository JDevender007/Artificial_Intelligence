"""Experiment configuration: dataclasses loaded from YAML, with ``--set key=value`` overrides."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from toolagent.agent import AGENTS

BACKENDS = ("hf", "oracle")


@dataclass
class ModelConfig:
    name: str = "qwen7b_nf4"  # label in the tables
    backend: str = "hf"  # hf | oracle (replays the reference plans; a solvability check, not a baseline)
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    nf4: bool = True
    dtype: str = "float16"  # T4 has no bf16
    max_new_tokens: int = 256
    temperature: float = 0.7  # sampling, so the seeds differ; 0 = greedy
    top_p: float = 0.9
    batch_size: int = 16


@dataclass
class ExperimentConfig:
    name: str = "eval"
    output_dir: str = "outputs/eval"
    world_dir: str = ""  # default: <output_dir>/world
    world_seed: int = 0
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    agents: list[str] = field(default_factory=lambda: list(AGENTS))
    tasks: list[str] = field(default_factory=list)  # task ids; empty = all 30
    flaky_p: float = 0.5  # chance of each extra consecutive exchange-rate API failure (max 2 per episode)
    n_bootstrap: int = 2000
    models: list[ModelConfig] = field(default_factory=lambda: [ModelConfig()])

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
        if k == "models":
            v = [m if isinstance(m, ModelConfig) else _build(ModelConfig, m) for m in v]
        elif is_dataclass(default) and isinstance(v, dict):
            v = _build(type(default), v)
        elif isinstance(default, float) and isinstance(v, str | int) and not isinstance(v, bool):
            v = float(v)
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
    bad = [a for a in cfg.agents if a not in AGENTS]
    if bad or not cfg.agents:
        raise ValueError(f"unknown agents {bad}; choose from {list(AGENTS)}")
    if not cfg.seeds or len(set(cfg.seeds)) != len(cfg.seeds):
        raise ValueError("seeds must be a non-empty list of distinct ints")
    if not cfg.models or len({m.name for m in cfg.models}) != len(cfg.models):
        raise ValueError("models must be a non-empty list with distinct names")
    for m in cfg.models:
        if m.backend not in BACKENDS:
            raise ValueError(f"model {m.name}: backend must be one of {BACKENDS}")
        if m.dtype not in ("float16", "float32", "bfloat16"):
            raise ValueError(f"model {m.name}: dtype must be float16, float32 or bfloat16")
    if not 0 <= cfg.flaky_p < 1:
        raise ValueError("flaky_p must be in [0, 1)")
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load YAML (missing keys keep their defaults), apply ``key.sub=value`` overrides, validate."""
    raw: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for o in overrides or []:
        _apply_override(raw, o)
    return validate(_build(ExperimentConfig, raw))
