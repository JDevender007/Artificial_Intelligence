"""Train one (env, algorithm, seed) and return its learning curve plus its final held-out evaluation.

The algorithms:
- ``dqn``: my DQN with Double-DQN targets and a target network
- ``dqn_nodouble``: my DQN with the plain max target
- ``dqn_notarget``: my DQN with no target network and no double
- ``sb3_dqn`` and ``sb3_ppo``: the Stable-Baselines3 reference implementations

Every algorithm is evaluated with the same greedy or deterministic protocol on the same seed lists.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch

from rlinv.dqn import DQNAgent, DQNConfig
from rlinv.envs import make_env
from rlinv.evaluate import Policy, evaluate_policy, mean_score

log = logging.getLogger(__name__)

SCRATCH = {"dqn": {}, "dqn_nodouble": {"double": False}, "dqn_notarget": {"double": False, "target_update": 1}}
SB3 = ("sb3_dqn", "sb3_ppo")
ALGOS = (*SCRATCH, *SB3)


def parse_schedule(v: Any) -> Any:
    """'lin_0.001' -> a linear decay from 0.001 to 0, as an SB3 schedule. Numbers pass through unchanged."""
    if isinstance(v, str) and v.startswith("lin_"):
        start = float(v[4:])
        return lambda progress_remaining: progress_remaining * start
    return v


def _sb3_kwargs(raw: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    kw = dict(raw)
    n_envs = int(kw.pop("n_envs", 1))
    for k in ("learning_rate", "clip_range"):
        if k in kw:
            kw[k] = parse_schedule(kw[k])
    if "net_arch" in kw:
        kw["policy_kwargs"] = {"net_arch": list(kw.pop("net_arch"))}
    return n_envs, kw


def train_one(job: dict[str, Any]) -> dict[str, Any]:
    """``job``: env_id, algo, seed, total_steps, eval_every, eval_seeds, final_seeds, hp (algorithm kwargs),
    env_kwargs. Returns {curve: [(step, score)], final: [episode rows], train_s, ...}."""
    torch.set_num_threads(1)
    env_id, algo, seed = job["env_id"], job["algo"], int(job["seed"])
    env_kwargs = job.get("env_kwargs") or {}
    curve: list[tuple[int, float]] = []
    t0 = time.perf_counter()

    def record(step: int, policy: Policy) -> None:
        curve.append((int(step), mean_score(evaluate_policy(env_id, policy, job["eval_seeds"], env_kwargs))))

    extra: dict[str, Any] = {}
    if algo in SCRATCH:
        cfg = DQNConfig(**{**job.get("hp", {}), **SCRATCH[algo]})
        env = make_env(env_id, **env_kwargs)
        env.action_space.seed(seed)
        agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, cfg, seed=seed)
        extra = agent.learn(env, job["total_steps"], seed=seed * 1000 + 1,
                            callback=lambda s: record(s, agent.policy()), callback_every=job["eval_every"])
        env.close()
        final_policy = agent.policy()
        model_state = agent.q.state_dict()
    elif algo in SB3:
        from stable_baselines3 import DQN, PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.env_util import make_vec_env

        n_envs, kw = _sb3_kwargs(job.get("hp", {}))
        vec = make_vec_env(lambda: make_env(env_id, **env_kwargs), n_envs=n_envs, seed=seed * 1000 + 1)
        cls = DQN if algo == "sb3_dqn" else PPO
        model = cls("MlpPolicy", vec, seed=seed, verbose=0, device="cpu", **kw)

        def sb3_policy(obs: np.ndarray, env: Any) -> int:
            return int(model.predict(obs, deterministic=True)[0])

        class EvalCurve(BaseCallback):
            def __init__(self) -> None:
                super().__init__()
                self.next_eval = 0

            def _on_training_start(self) -> None:
                record(0, sb3_policy)
                self.next_eval = job["eval_every"]

            def _on_step(self) -> bool:
                while self.num_timesteps >= self.next_eval:  # PPO with n_envs advances several steps at a time
                    record(self.next_eval, sb3_policy)
                    self.next_eval += job["eval_every"]
                return True

        t_train = time.perf_counter()
        model.learn(total_timesteps=job["total_steps"], callback=EvalCurve())
        extra = {"train_s": time.perf_counter() - t_train}
        vec.close()
        final_policy = sb3_policy
        model_state = None
    else:
        raise ValueError(f"unknown algo {algo!r}; choose from {ALGOS}")

    final = evaluate_policy(env_id, final_policy, job["final_seeds"], env_kwargs)
    out = {"env_id": env_id, "algo": algo, "seed": seed, "curve": curve, "final": final,
           "wall_s": time.perf_counter() - t0, **extra}
    if model_state is not None and job.get("keep_model"):
        out["model_state"] = {k: v.cpu() for k, v in model_state.items()}
    log.info("%s %s seed %d: final %.1f (%.0fs)", env_id, algo, seed, mean_score(final), out["wall_s"])
    return out
