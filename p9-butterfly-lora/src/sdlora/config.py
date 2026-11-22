"""Experiment configuration: dataclasses loaded from YAML, with ``--set key.sub=value`` overrides."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    n_test: int = 200  # held-out species: FID/KID reference set and generation prompts
    split_seed: int = 0
    resolution: int = 512
    limit: int = 0  # > 0 keeps only the first rows (smoke runs)


@dataclass
class ModelConfig:
    base: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"  # CreativeML OpenRAIL-M
    revision: str = ""  # empty = main
    clip: str = "openai/clip-vit-base-patch16"  # CLIP score, retrieval, memorization, concept filter
    dtype: str = "float16"  # T4 has no bf16


@dataclass
class TrainConfig:
    rank: int = 8
    alpha: int = 8
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-2
    batch_size: int = 4
    steps: int = 800
    checkpoints: list[int] = field(default_factory=lambda: [400, 800])  # each is evaluated (steps ablation)
    warmup: int = 50
    max_grad_norm: float = 1.0
    grad_checkpointing: bool = True


@dataclass
class GenConfig:
    steps: int = 25  # DPM-Solver++ steps
    guidance: float = 7.5
    batch_size: int = 8
    n_prompts: int = 0  # 0 = one image per test caption


@dataclass
class SafetyConfig:
    sd_checker: bool = True  # the pipeline's own safety checker (blacks out flagged images)
    clip_threshold: float = 0.5  # unsafe probability mass for the CLIP concept filter


@dataclass
class ExperimentConfig:
    name: str = "main"
    output_dir: str = "outputs/main"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])  # training seed = generation seed
    n_bootstrap: int = 2000
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    gen: GenConfig = field(default_factory=GenConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def variants(self) -> list[str]:
        """``base`` (no LoRA, the baseline) plus one LoRA variant per checkpoint."""
        return ["base"] + [f"lora_r{self.train.rank}@{s}" for s in self.train.checkpoints]


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
    if not cfg.seeds or len(set(cfg.seeds)) != len(cfg.seeds):
        raise ValueError("seeds must be a non-empty list of distinct ints")
    t = cfg.train
    if not t.checkpoints or sorted(set(t.checkpoints)) != t.checkpoints or t.checkpoints[-1] != t.steps:
        raise ValueError("train.checkpoints must be strictly increasing and end at train.steps")
    if t.checkpoints[0] <= 0 or t.rank <= 0 or t.batch_size <= 0:
        raise ValueError("checkpoints, rank and batch_size must be positive")
    if cfg.data.resolution % 64:
        raise ValueError("data.resolution must be a multiple of 64")
    if cfg.model.dtype not in ("float16", "float32"):
        raise ValueError("model.dtype must be float16 or float32")
    if not 0 < cfg.safety.clip_threshold < 1:
        raise ValueError("safety.clip_threshold must be in (0, 1)")
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load YAML (missing keys keep their defaults), apply ``key.sub=value`` overrides, validate."""
    raw: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for o in overrides or []:
        _apply_override(raw, o)
    return validate(_build(ExperimentConfig, raw))
