"""Experiment configuration: YAML into dataclasses, with ``key=value`` overrides (top-level keys or ``a.b`` keys)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from rlinv.agents import ALGOS
from rlinv.heuristics import FACTORIES


@dataclass
class EnvSpec:
    id: str = "CartPole-v1"
    total_steps: int = 50_000
    eval_every: int = 5_000
    eval_episodes: int = 10
    threshold: float | None = None  # "solved" score for the steps-to-threshold metric
    algos: list[str] = field(default_factory=lambda: list(ALGOS))
    heuristics: list[str] = field(default_factory=list)  # inventory baselines, tuned on training seeds
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    dqn: dict[str, Any] = field(default_factory=dict)  # DQNConfig fields, shared by the three scratch variants
    sb3_dqn: dict[str, Any] = field(default_factory=dict)
    sb3_ppo: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    name: str = "main"
    output_dir: str = "outputs/main"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    final_episodes: int = 50  # held-out episodes per trained policy (seeds final_seed_base + i)
    final_seed_base: int = 100_000
    eval_seed_base: int = 50_000  # learning-curve evaluation episodes
    tune_episodes: int = 30  # heuristic grid search episodes (seeds tune_seed_base + i), disjoint from the above
    tune_seed_base: int = 200_000
    steps_scale: float = 1.0  # multiply every env's total_steps / eval_every (the smoke config uses < 1)
    n_workers: int = 0  # 0 = os.cpu_count()
    n_bootstrap: int = 5000
    envs: list[EnvSpec] = field(default_factory=lambda: [EnvSpec()])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> ExperimentConfig:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        if not 0 < self.steps_scale <= 10:
            raise ValueError("steps_scale must be in (0, 10]")
        if self.final_episodes < 2 or self.tune_episodes < 1:
            raise ValueError("need final_episodes >= 2 and tune_episodes >= 1")
        bases = sorted([self.final_seed_base, self.eval_seed_base, self.tune_seed_base])
        if any(b2 - b1 < 10_000 for b1, b2 in zip(bases, bases[1:], strict=False)):
            raise ValueError("seed bases must be >= 10000 apart so evaluation / tuning episodes never overlap")
        for e in self.envs:
            bad = set(e.algos) - set(ALGOS)
            if bad:
                raise ValueError(f"unknown algos {sorted(bad)}")
            if set(e.heuristics) - set(FACTORIES) - {"random"}:
                raise ValueError(f"unknown heuristics {e.heuristics}")
            if e.total_steps <= 0 or e.eval_every <= 0 or e.eval_episodes < 1:
                raise ValueError("total_steps, eval_every and eval_episodes must be positive")
        return self


def _build(values: dict[str, Any]) -> ExperimentConfig:
    known = {f.name for f in fields(ExperimentConfig)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    values = dict(values)
    if "envs" in values:
        env_known = {f.name for f in fields(EnvSpec)}
        envs = []
        for e in values["envs"]:
            if isinstance(e, EnvSpec):
                envs.append(e)
                continue
            if set(e) - env_known:
                raise ValueError(f"unknown env keys: {sorted(set(e) - env_known)}")
            envs.append(EnvSpec(**e))
        values["envs"] = envs
    return ExperimentConfig(**values)


def load_config(path: str | Path | None, overrides: list[str] | None = None) -> ExperimentConfig:
    raw: dict[str, Any] = {}
    if path:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, val = item.split("=", 1)
        parsed = yaml.safe_load(val)
        node = raw
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot override inside non-mapping key {key!r}")
        node[parts[-1]] = parsed
    return _build(raw).validate()
