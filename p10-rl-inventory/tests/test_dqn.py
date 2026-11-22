import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from rlinv.dqn import DQNAgent, DQNConfig, ReplayBuffer, td_targets


def test_replay_buffer_wraps():
    b = ReplayBuffer(3, 2, np.random.default_rng(0))
    for i in range(5):
        b.add(np.full(2, i), i % 2, float(i), np.full(2, i + 1), i == 4)
    assert b.size == 3 and b.pos == 2 and sorted(b.rew.tolist()) == [2.0, 3.0, 4.0]
    o, a, r, o2, d = b.sample(8)
    assert o.shape == (8, 2) and a.dtype == torch.int64 and set(d.tolist()) <= {0.0, 1.0}


def test_double_vs_vanilla_targets():
    online = torch.tensor([[1.0, 5.0]])  # the online net picks action 1
    target = torch.tensor([[3.0, 2.0]])  # the target net would pick action 0
    r, d = torch.tensor([1.0]), torch.tensor([0.0])
    assert td_targets(online, target, r, d, 0.5, double=True).item() == 2.0  # 1 + 0.5 * target[1]
    assert td_targets(online, target, r, d, 0.5, double=False).item() == 2.5  # 1 + 0.5 * max(target)
    assert td_targets(online, target, r, torch.tensor([1.0]), 0.5, double=True).item() == 1.0


class Bandit(gym.Env):
    """One-step env: action 1 pays 1 and action 0 pays 0. DQN must learn to pick action 1."""

    observation_space = spaces.Box(-1, 1, (2,), np.float32)
    action_space = spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(2, np.float32), {}

    def step(self, a):
        return np.zeros(2, np.float32), float(a == 1), True, False, {}


def test_dqn_learns_bandit_and_calls_back():
    cfg = DQNConfig(learning_starts=50, buffer_size=500, target_update=50, eps_fraction=0.5, hidden=(16,), lr=1e-2)
    agent = DQNAgent(2, 2, cfg, seed=0)
    calls = []
    out = agent.learn(Bandit(), 600, seed=0, callback=calls.append, callback_every=200)
    assert agent.greedy(np.zeros(2, np.float32)) == 1
    assert calls == [0, 200, 400, 600] and out["train_s"] > 0


def test_epsilon_schedule():
    agent = DQNAgent(2, 2, DQNConfig(eps_start=1.0, eps_end=0.1, eps_fraction=0.5), seed=0)
    agent.steps = 0
    assert agent.epsilon(100) == 1.0
    agent.steps = 25
    assert abs(agent.epsilon(100) - 0.55) < 1e-9
    agent.steps = 90
    assert abs(agent.epsilon(100) - 0.1) < 1e-9
