"""Typed experiment configuration loaded from YAML with ``key.sub=value`` CLI overrides."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import yaml

DATASETS = {"mnist", "cifar10", "fake"}
MODELS = {"linear", "mlp", "cnn", "resnet18"}
SCHEDULERS = {"onecycle", "cosine", "none"}


@dataclass
class DataConfig:
    dataset: str = "cifar10"  # mnist | cifar10 | fake (synthetic, for tests/CI)
    data_dir: str = "data"
    val_size: int = 5000  # carved from the official train split
    split_seed: int = 42  # fixed: every seed sees the SAME train/val/test split
    augment: bool = True
    num_workers: int = 2
    limit_train: int | None = None  # subsample train for smoke runs
    resnet_img_size: int = 128  # pretrained ResNet sees upsampled images


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    lr: float = 0.05
    weight_decay: float = 5e-4
    optimizer: str = "sgd"  # sgd | adamw
    momentum: float = 0.9
    scheduler: str = "onecycle"  # onecycle | cosine | none
    amp: bool = True
    early_stop_patience: int = 5  # epochs without val-loss improvement; 0 disables
    label_smoothing: float = 0.0
    grad_clip: float | None = None


@dataclass
class ExperimentConfig:
    name: str = "default"
    output_dir: str = "outputs/default"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    models: list[str] = field(default_factory=lambda: ["linear", "mlp", "cnn", "resnet18"])
    deterministic: bool = False  # True = bitwise-reproducible but slower cuDNN kernels
    tensorboard: bool = True
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    # Per-model TrainConfig overrides, e.g. {"resnet18": {"lr": 0.01, "epochs": 10}}
    model_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def train_for(self, model: str) -> TrainConfig:
        """Return the TrainConfig for ``model`` with its overrides applied."""
        over = self.model_overrides.get(model, {})
        unknown = set(over) - {f.name for f in fields(TrainConfig)}
        if unknown:
            raise KeyError(f"Unknown train keys in model_overrides[{model}]: {sorted(unknown)}")
        return replace(copy.deepcopy(self.train), **over)


def _build(cls: type, raw: dict[str, Any]) -> Any:
    """Recursively build dataclass ``cls`` from ``raw``, rejecting unknown keys."""
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise KeyError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    defaults = cls()
    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        current = getattr(defaults, name)
        kwargs[name] = _build(type(current), value) if is_dataclass(current) else value
    return cls(**kwargs)


def _apply_override(raw: dict[str, Any], override: str) -> None:
    """Apply one ``a.b=value`` override in place; the value is parsed as YAML."""
    if "=" not in override:
        raise ValueError(f"Override must look like key=value, got {override!r}")
    key, value = override.split("=", 1)
    parts = key.strip().split(".")
    node = raw
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = yaml.safe_load(value)


def validate(cfg: ExperimentConfig) -> ExperimentConfig:
    """Raise ``ValueError`` on invalid choices; return ``cfg`` unchanged otherwise."""
    if cfg.data.dataset not in DATASETS:
        raise ValueError(f"data.dataset must be one of {sorted(DATASETS)}")
    bad = set(cfg.models) - MODELS
    if bad:
        raise ValueError(f"Unknown models {sorted(bad)}; choose from {sorted(MODELS)}")
    for m in cfg.models:
        t = cfg.train_for(m)
        if t.scheduler not in SCHEDULERS:
            raise ValueError(f"scheduler must be one of {sorted(SCHEDULERS)}")
        if t.optimizer not in {"sgd", "adamw"}:
            raise ValueError("optimizer must be sgd or adamw")
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from YAML and apply CLI overrides."""
    raw: dict[str, Any] = {}
    if path is not None:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    for ov in overrides or []:
        _apply_override(raw, ov)
    return validate(_build(ExperimentConfig, raw))
