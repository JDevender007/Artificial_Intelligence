"""Classical inventory baselines, tuned by grid search on training seeds only and scored on the held-out eval seeds.

* ``random``: a uniform random order size. This is the floor.
* ``base_stock(S)``: order up to inventory position S every day. The textbook policy without a fixed cost.
* ``sS(s, S)``: if the position is below s, order up to S. Optimal for i.i.d. demand with a fixed cost.
* ``dow_sS(q, batch)``: a weekday-aware (s, S). The reorder point s_d is the q-quantile of Poisson demand over the
  lead time plus one day starting at weekday d, and the order-up-to level is S_d = s_d + batch. It knows the true
  demand model, so it is an informed baseline. The RL agents have to learn the seasonality from reward.

Order quantities are rounded to the nearest allowed size, with a floor of 0.
"""

from __future__ import annotations

import functools
import itertools
import logging
from collections.abc import Sequence

import numpy as np
from scipy.stats import poisson

from rlinv.envs import ENV_ID, InventoryEnv, InventoryParams
from rlinv.evaluate import Policy, evaluate_policy, mean_score

log = logging.getLogger(__name__)


def nearest_action(qty: float, sizes: Sequence[int]) -> int:
    """Index of the allowed order size closest to ``qty`` (ties go to the smaller size)."""
    qty = max(0.0, qty)
    return int(np.argmin([abs(s - qty) + 1e-9 * i for i, s in enumerate(sizes)]))


def random_policy(seed: int = 0) -> Policy:
    rng = np.random.default_rng(seed)
    return lambda obs, env: int(rng.integers(env.action_space.n))


def base_stock(S: float) -> Policy:
    def act(obs: np.ndarray, env: InventoryEnv) -> int:
        return nearest_action(S - env.position, env.p.order_sizes)

    return act


def sS(s: float, S: float) -> Policy:
    def act(obs: np.ndarray, env: InventoryEnv) -> int:
        pos = env.position
        return nearest_action(S - pos, env.p.order_sizes) if pos < s else 0

    return act


def dow_levels(p: InventoryParams, q: float) -> list[float]:
    """Reorder point per weekday: the q-quantile of demand over the lead time + 1 days starting that day."""
    out = []
    for d in range(7):
        lam = sum(p.demand_rate(d + k) for k in range(p.lead_time + 1))
        out.append(float(poisson.ppf(q, lam)))
    return out


@functools.lru_cache(maxsize=256)
def _cached_levels(p: InventoryParams, q: float) -> tuple[float, ...]:
    return tuple(dow_levels(p, q))


def dow_sS(q: float, batch: float) -> Policy:
    def act(obs: np.ndarray, env: InventoryEnv) -> int:
        s = _cached_levels(env.p, q)[env.state.weekday]  # from the env's own demand model
        pos = env.position
        return nearest_action(s + batch - pos, env.p.order_sizes) if pos < s else 0

    return act


GRIDS: dict[str, dict[str, list[float]]] = {
    "base_stock": {"S": list(range(20, 81, 2))},
    "sS": {"s": list(range(10, 71, 4)), "S": list(range(30, 91, 4))},
    "dow_sS": {"q": [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99, 0.995],
               "batch": [0, 5, 10, 15, 20, 25, 30, 35, 40, 50]},
}
FACTORIES = {"base_stock": base_stock, "sS": sS, "dow_sS": dow_sS}


def tune(name: str, train_seeds: Sequence[int], env_kwargs: dict | None = None,
         grid: dict[str, list[float]] | None = None) -> tuple[dict[str, float], float]:
    """Grid search over ``grid`` (default ``GRIDS[name]``) and return (best params, mean training profit)."""
    grid = grid or GRIDS[name]
    keys = list(grid)
    best: tuple[dict[str, float], float] = ({}, -np.inf)
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values, strict=True))
        if name == "sS" and params["s"] > params["S"]:
            continue
        score = mean_score(evaluate_policy(ENV_ID, FACTORIES[name](**params), train_seeds, env_kwargs))
        if score > best[1]:
            best = (params, score)
    log.info("tuned %s: %s (train profit %.1f)", name, best[0], best[1])
    return best
