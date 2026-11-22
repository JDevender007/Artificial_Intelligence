"""Custom environment: single-item inventory control with lead time, weekly demand seasonality and lost sales.

Each step is one day:

1. The order placed ``lead_time`` days ago arrives. Stock above ``capacity`` is discarded, but it was still paid for.
2. Poisson demand with a day-of-week multiplier is served from stock. Unmet demand is lost and pays a goodwill
   penalty on top of the lost margin.
3. The agent chooses an order quantity from ``order_sizes``. It pays a fixed cost if the quantity is > 0, plus a unit
   cost.
4. Holding cost is charged on the stock left at the end of the day.

``info["profit"]`` is the day's profit in currency units. The reward is that profit times ``reward_scale``, so value
targets stay O(1) for DQN. Every result reports unscaled profit per episode (``episode_profit``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

WEEKDAY_MULT = (0.7, 0.8, 0.9, 1.0, 1.2, 1.6, 1.3)  # Mon..Sun; the weekend peak is what a static policy can't follow


@dataclass(frozen=True)
class InventoryParams:
    base_demand: float = 10.0
    weekday_mult: tuple[float, ...] = WEEKDAY_MULT
    lead_time: int = 2
    capacity: int = 60
    order_sizes: tuple[int, ...] = (0, 5, 10, 15, 20, 25, 30, 35, 40)
    price: float = 5.0
    unit_cost: float = 2.0
    fixed_cost: float = 10.0
    holding_cost: float = 0.1
    stockout_penalty: float = 3.0
    horizon: int = 91  # 13 weeks
    max_initial_stock: int = 30
    reward_scale: float = 0.01

    def demand_rate(self, weekday: int) -> float:
        return self.base_demand * self.weekday_mult[weekday % 7]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InventoryState:
    on_hand: int
    pipeline: list[int] = field(default_factory=list)  # pipeline[0] arrives tomorrow ... pipeline[-1] in lead_time days
    weekday: int = 0
    t: int = 0


class InventoryEnv(gym.Env):
    """Gymnasium env. Observation (float32, 4 + lead_time dims): on-hand / capacity, each pipeline slot / max order,
    sin and cos of the weekday, and the fraction of the horizon elapsed."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, params: InventoryParams | None = None, render_mode: str | None = None, **overrides: Any):
        p = params or InventoryParams()
        if overrides:
            p = InventoryParams(**{**p.to_dict(), **overrides})
        if p.lead_time < 1:
            raise ValueError("lead_time must be >= 1")
        self.p = p
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(p.order_sizes))
        dim = 4 + p.lead_time
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(dim,), dtype=np.float32)
        self.state = InventoryState(on_hand=0, pipeline=[0] * p.lead_time)
        self._last: dict[str, Any] = {}

    # ----------------------------------------------------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        s, p = self.state, self.p
        ang = 2 * np.pi * s.weekday / 7
        max_order = max(p.order_sizes)
        o = [s.on_hand / p.capacity, *[q / max_order for q in s.pipeline], np.sin(ang), np.cos(ang), s.t / p.horizon]
        return np.clip(np.asarray(o, dtype=np.float32), -1.0, 1.0)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        p = self.p
        opts = options or {}
        self.state = InventoryState(
            on_hand=int(opts.get("on_hand", self.np_random.integers(0, p.max_initial_stock + 1))),
            pipeline=list(opts.get("pipeline", [0] * p.lead_time)),
            weekday=int(opts.get("weekday", self.np_random.integers(0, 7))),
            t=0,
        )
        if len(self.state.pipeline) != p.lead_time:
            raise ValueError("pipeline length must equal lead_time")
        self._last = {}
        return self._obs(), {"position": self.position}

    @property
    def position(self) -> int:
        """Inventory position: on hand plus everything ordered but not yet arrived (what base-stock policies use)."""
        return self.state.on_hand + sum(self.state.pipeline)

    def step(self, action: int):
        p, s = self.p, self.state
        if not self.action_space.contains(int(action)):
            raise ValueError(f"invalid action {action}")
        arriving = s.pipeline.pop(0)
        stock = s.on_hand + arriving
        wasted = max(0, stock - p.capacity)
        stock -= wasted
        demand = int(self.np_random.poisson(p.demand_rate(s.weekday)))
        sales = min(stock, demand)
        lost = demand - sales
        stock -= sales
        qty = int(p.order_sizes[int(action)])
        s.pipeline.append(qty)
        revenue = p.price * sales
        order_cost = (p.fixed_cost if qty > 0 else 0.0) + p.unit_cost * qty
        holding = p.holding_cost * stock
        penalty = p.stockout_penalty * lost
        profit = revenue - order_cost - holding - penalty
        s.on_hand = stock
        s.weekday = (s.weekday + 1) % 7
        s.t += 1
        terminated = False
        truncated = s.t >= p.horizon
        info = {"profit": profit, "demand": demand, "sales": sales, "lost": lost, "wasted": wasted, "order": qty,
                "arrived": arriving, "on_hand": stock, "position": self.position}
        self._last = info
        return self._obs(), float(profit * p.reward_scale), terminated, truncated, info

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        i, s = self._last, self.state
        if not i:
            return f"t=0 on_hand={s.on_hand} pipeline={s.pipeline}"
        return (f"t={s.t:3d} demand={i['demand']:3d} sold={i['sales']:3d} lost={i['lost']:3d} order={i['order']:3d} "
                f"on_hand={i['on_hand']:3d} profit={i['profit']:7.1f}")


ENV_ID = "Inventory-v0"


def register() -> None:
    if ENV_ID not in gym.registry:
        gym.register(id=ENV_ID, entry_point="rlinv.envs:InventoryEnv")


def make_env(env_id: str, **kwargs: Any) -> gym.Env:
    register()
    return gym.make(env_id, **kwargs)
