"""Typed experiment configuration loaded from YAML with ``key.sub=value`` CLI overrides."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    yaml: str = "construction-ppe.yaml"  # Ultralytics built-in dataset (auto-download)
    val_fraction: float = 0.15  # carved from the official train split
    split_seed: int = 42  # fixed: every seed/variant sees the SAME split
    # The official "val" split (200 images) is our held-out TEST set, used once per run.


@dataclass
class TrainConfig:
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    patience: int = 15  # early stopping on val fitness (0.9*mAP50-95 + 0.1*mAP50)
    workers: int = 4
    fraction: float = 1.0  # <1 trains on a subset (smoke runs)
    optimizer: str = "auto"


@dataclass
class Variant:
    """One row of the comparison: a model + optional Ultralytics train-arg overrides."""

    name: str = "pretrained_n"
    model: str = "yolo26n.pt"  # *.pt = COCO-pretrained, *.yaml = random init
    pretrained: bool = True
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalConfig:
    conf: float = 0.25  # operating threshold for error analysis / compliance / tracking
    iou: float = 0.5  # TP if IoU >= this (error analysis)
    main_variant: str = "pretrained_n"  # analysed in depth (seed 0)


@dataclass
class TrackConfig:
    enabled: bool = True
    trackers: list[str] = field(default_factory=lambda: ["bytetrack.yaml", "botsort.yaml"])
    n_sequences: int = 20
    n_frames: int = 30
    min_objects: int = 3  # only images with at least this many GT boxes become sequences
    seed: int = 0


@dataclass
class ExperimentConfig:
    name: str = "ppe"
    output_dir: str = "outputs/ppe"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    variants: list[Variant] = field(default_factory=lambda: [Variant()])
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    track: TrackConfig = field(default_factory=TrackConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls: type, raw: dict[str, Any]) -> Any:
    """Recursively build dataclass ``cls`` from ``raw``, rejecting unknown keys."""
    unknown = set(raw) - {f.name for f in fields(cls)}
    if unknown:
        raise KeyError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    defaults = cls()
    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        current = getattr(defaults, name)
        if name == "variants":
            kwargs[name] = [_build(Variant, v) for v in value]
        elif is_dataclass(current):
            kwargs[name] = _build(type(current), value)
        else:
            kwargs[name] = value
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
    """Raise ``ValueError`` on inconsistent settings; return ``cfg`` otherwise."""
    names = [v.name for v in cfg.variants]
    if len(names) != len(set(names)):
        raise ValueError(f"Variant names must be unique: {names}")
    if not 0 < cfg.data.val_fraction < 1:
        raise ValueError("data.val_fraction must be in (0, 1)")
    for v in cfg.variants:
        if v.model.endswith(".yaml") and v.pretrained:
            raise ValueError(f"{v.name}: a .yaml model is randomly initialised; set pretrained: false")
    if cfg.eval.main_variant not in names:
        raise ValueError(f"eval.main_variant {cfg.eval.main_variant!r} is not one of {names}")
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
