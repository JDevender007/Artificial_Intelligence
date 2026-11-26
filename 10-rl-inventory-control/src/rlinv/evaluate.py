"""Policy evaluation shared by every method: fixed seed lists, so all methods face the same episodes (paired)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym
import numpy as np

from rlinv.envs import make_env

Policy = Callable[[np.ndarray, gym.Env], int]  # (obs, unwrapped env) -> action


def episode_score(reward: float, info: dict[str, Any]) -> float:
    """CartPole: the reward. Inventory: unscaled profit (``info['profit']``)."""
    return float(info["profit"]) if "profit" in info else float(reward)


def run_episode(env: gym.Env, policy: Policy, seed: int) -> dict[str, float]:
    obs, _ = env.reset(seed=seed)
    total, length, lost, demand = 0.0, 0, 0, 0
    done = False
    while not done:
        a = policy(obs, env.unwrapped)
        obs, r, term, trunc, info = env.step(a)
        total += episode_score(r, info)
        lost += info.get("lost", 0)
        demand += info.get("demand", 0)
        length += 1
        done = term or trunc
    out = {"seed": seed, "score": total, "length": length}
    if demand:
        out["fill_rate"] = 1 - lost / demand
    return out


def evaluate_policy(env_id: str, policy: Policy, seeds: Sequence[int], env_kwargs: dict | None = None) -> list[dict]:
    env = make_env(env_id, **(env_kwargs or {}))
    try:
        return [run_episode(env, policy, int(s)) for s in seeds]
    finally:
        env.close()


def mean_score(rows: Sequence[dict]) -> float:
    return float(np.mean([r["score"] for r in rows]))
