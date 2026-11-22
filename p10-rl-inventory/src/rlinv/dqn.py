"""DQN from scratch in PyTorch: replay buffer, target network, Double-DQN targets, Huber loss, epsilon-greedy.

The ablation switches:

* ``double=False``: the vanilla max target. The target net both selects and evaluates the next action.
* ``target_update=1``: no separate target network. It is synced every step, so the target is the online network.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import nn

log = logging.getLogger(__name__)


@dataclass
class DQNConfig:
    lr: float = 5e-4
    gamma: float = 0.99
    buffer_size: int = 50_000
    batch_size: int = 64
    learning_starts: int = 1_000
    train_freq: int = 1  # env steps per gradient step
    target_update: int = 500  # env steps between hard target syncs; 1 = no target network
    double: bool = True
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.2  # fraction of total steps over which epsilon decays linearly
    hidden: tuple[int, ...] = (128, 128)
    grad_clip: float = 10.0

    def to_dict(self) -> dict:
        return asdict(self)


class ReplayBuffer:
    """A circular numpy buffer. ``done`` marks real termination only; time-limit truncation still bootstraps."""

    def __init__(self, capacity: int, obs_dim: int, rng: np.random.Generator):
        self.capacity, self.rng = capacity, rng
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros(capacity, np.int64)
        self.rew = np.zeros(capacity, np.float32)
        self.done = np.zeros(capacity, np.float32)
        self.pos, self.size = 0, 0

    def add(self, o: np.ndarray, a: int, r: float, o2: np.ndarray, done: bool) -> None:
        i = self.pos
        self.obs[i], self.act[i], self.rew[i], self.next_obs[i], self.done[i] = o, a, r, o2, float(done)
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int) -> tuple[torch.Tensor, ...]:
        idx = self.rng.integers(0, self.size, size=n)
        return tuple(torch.as_tensor(x[idx]) for x in (self.obs, self.act, self.rew, self.next_obs, self.done))


def mlp(inp: int, out: int, hidden: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = inp
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


def td_targets(q_online_next: torch.Tensor, q_target_next: torch.Tensor, rew: torch.Tensor, done: torch.Tensor,
               gamma: float, double: bool) -> torch.Tensor:
    """r + gamma * (1 - done) * Q_target(s', a*). With Double DQN, a* = argmax of the online net. Without it,
    a* = argmax of the target net (the plain max)."""
    if double:
        a_star = q_online_next.argmax(dim=1, keepdim=True)
        nxt = q_target_next.gather(1, a_star).squeeze(1)
    else:
        nxt = q_target_next.max(dim=1).values
    return rew + gamma * (1.0 - done) * nxt


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig, seed: int = 0):
        self.cfg, self.n_actions = cfg, n_actions
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.q = mlp(obs_dim, n_actions, cfg.hidden)
        self.q_target = mlp(obs_dim, n_actions, cfg.hidden)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size, obs_dim, self.rng)
        self.steps = 0

    def epsilon(self, total_steps: int) -> float:
        c = self.cfg
        frac = min(1.0, self.steps / max(1, c.eps_fraction * total_steps))
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    @torch.no_grad()
    def greedy(self, obs: np.ndarray) -> int:
        return int(self.q(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).argmax(1).item())

    def act(self, obs: np.ndarray, total_steps: int) -> int:
        if self.rng.random() < self.epsilon(total_steps):
            return int(self.rng.integers(self.n_actions))
        return self.greedy(obs)

    def train_step(self) -> float:
        c = self.cfg
        o, a, r, o2, d = self.buffer.sample(c.batch_size)
        with torch.no_grad():
            target = td_targets(self.q(o2) if c.double else self.q_target(o2), self.q_target(o2), r, d, c.gamma,
                                c.double)
        q = self.q(o).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), c.grad_clip)
        self.opt.step()
        return float(loss.item())

    def learn(self, env: gym.Env, total_steps: int, seed: int,
              callback: Callable[[int], None] | None = None, callback_every: int = 0) -> dict[str, float]:
        """Interact for ``total_steps`` env steps. ``callback(step)`` is called every ``callback_every`` steps
        (and at step 0) for periodic evaluation."""
        c = self.cfg
        t0 = time.perf_counter()
        obs, _ = env.reset(seed=seed)
        losses: list[float] = []
        if callback and callback_every:
            callback(0)
        for _ in range(total_steps):
            a = self.act(obs, total_steps)
            o2, r, term, trunc, _ = env.step(a)
            self.buffer.add(obs, a, r, o2, term)
            obs = o2
            if term or trunc:
                obs, _ = env.reset()
            self.steps += 1
            if self.steps >= c.learning_starts and self.steps % c.train_freq == 0:
                losses.append(self.train_step())
            if self.steps % c.target_update == 0:
                self.q_target.load_state_dict(self.q.state_dict())
            if callback and callback_every and self.steps % callback_every == 0:
                callback(self.steps)
        return {"train_s": time.perf_counter() - t0, "final_loss_avg100": float(np.mean(losses[-100:]) if losses else 0)}

    def policy(self) -> Callable[[np.ndarray, gym.Env], int]:
        return lambda obs, env: self.greedy(obs)
