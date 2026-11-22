import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rlinv.envs import InventoryEnv, InventoryParams, make_env


def test_passes_gymnasium_checker():
    check_env(InventoryEnv(), skip_render_check=True)


def test_same_seed_same_episode():
    def roll(seed):
        env = InventoryEnv()
        env.reset(seed=seed)
        return [env.step(3)[1] for _ in range(20)]

    assert roll(7) == roll(7) and roll(7) != roll(8)


def test_lead_time_and_accounting():
    p = InventoryParams(base_demand=0.0, lead_time=2, holding_cost=1.0)  # no demand: pure pipeline mechanics
    env = InventoryEnv(p)
    env.reset(seed=0, options={"on_hand": 4, "weekday": 0})
    _, r, _, _, info = env.step(2)  # order 10 today
    assert info["order"] == 10 and info["arrived"] == 0 and info["on_hand"] == 4
    assert info["profit"] == pytest.approx(-(p.fixed_cost + 10 * p.unit_cost) - 4 * p.holding_cost)
    assert r == pytest.approx(info["profit"] * p.reward_scale)
    _, _, _, _, info = env.step(0)
    assert info["arrived"] == 0 and info["profit"] == pytest.approx(-4.0)
    _, _, _, _, info = env.step(0)  # arrives lead_time steps after it was ordered
    assert info["arrived"] == 10 and info["on_hand"] == 14


def test_lost_sales_and_capacity():
    env = InventoryEnv(InventoryParams(base_demand=50.0, capacity=20))
    env.reset(seed=1, options={"on_hand": 20, "pipeline": [40, 0]})
    _, _, _, _, info = env.step(0)
    assert info["wasted"] == 40 and info["sales"] == min(20, info["demand"])
    assert info["lost"] == info["demand"] - info["sales"] and info["on_hand"] >= 0


def test_truncates_at_horizon_and_registry():
    env = make_env("Inventory-v0", horizon=5)
    env.reset(seed=0)
    flags = [env.step(0)[3] for _ in range(5)]
    assert flags == [False] * 4 + [True]
    obs, _ = env.reset(seed=0)
    assert obs.shape == (6,) and np.all(np.abs(obs) <= 1)


def test_rejects_bad_params():
    with pytest.raises(ValueError):
        InventoryEnv(lead_time=0)
