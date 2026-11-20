"""Typed experiment configuration loaded from YAML with ``key.sub=value`` overrides."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    test_size: float = 0.2
    split_seed: int = 42  # fixed: every seed is evaluated on the SAME test set


@dataclass
class FeatureConfig:
    drop_protected: bool = True  # SEX, EDUCATION, MARRIAGE, AGE used only for auditing
    engineered: bool = True


@dataclass
class TuningConfig:
    n_trials: int = 50
    timeout_s: int | None = None


@dataclass
class ExperimentConfig:
    name: str = "default"
    output_dir: str = "outputs/default"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    models: list[str] = field(default_factory=lambda: ["dummy", "logreg", "xgboost", "lightgbm"])
    tune: list[str] = field(default_factory=lambda: ["logreg", "xgboost", "lightgbm"])
    imbalance: str = "class_weight"  # "none" | "class_weight"
    calibration: str = "isotonic"  # "isotonic" | "sigmoid" | "none"
    cv_folds: int = 5
    ece_bins: int = 15
    shap_samples: int = 2000
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls: type, raw: dict[str, Any]) -> Any:
    """Recursively build dataclass ``cls`` from ``raw``, rejecting unknown keys."""
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise KeyError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    defaults = cls()
    for name, value in raw.items():
        current = getattr(defaults, name)
        kwargs[name] = _build(type(current), value) if is_dataclass(current) else value
    return cls(**kwargs)


def _apply_override(raw: dict[str, Any], override: str) -> None:
    """Apply a single ``a.b=value`` override in place; value is parsed as YAML."""
    if "=" not in override:
        raise ValueError(f"Override must look like key=value, got {override!r}")
    key, value = override.split("=", 1)
    parts = key.strip().split(".")
    node = raw
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = yaml.safe_load(value)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from YAML and apply CLI overrides."""
    raw: dict[str, Any] = {}
    if path is not None:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    for ov in overrides or []:
        _apply_override(raw, ov)
    cfg = _build(ExperimentConfig, raw)
    if cfg.imbalance not in {"none", "class_weight"}:
        raise ValueError(f"imbalance must be 'none' or 'class_weight', got {cfg.imbalance!r}")
    if cfg.calibration not in {"isotonic", "sigmoid", "none"}:
        raise ValueError(f"calibration must be isotonic|sigmoid|none, got {cfg.calibration!r}")
    return cfg
