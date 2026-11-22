"""Demo: run the trained DQN and the tuned weekday (s, S) heuristic on the same inventory episode, day by day.

    python scripts/play.py --run-dir outputs/main --days 14 --seed 100000

The DQN weights come from ``<run-dir>/dqn_Inventory-v0_seed0.pt`` (written by the main experiment). The heuristic
parameters are read from ``<run-dir>/results.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rlinv.dqn import DQNConfig, mlp
from rlinv.envs import ENV_ID, make_env
from rlinv.heuristics import FACTORIES

DAYS = "Mon Tue Wed Thu Fri Sat Sun".split()


def run(env, policy, seed: int, days: int) -> list[str]:
    obs, _ = env.reset(seed=seed)
    lines, total = [], 0.0
    for _ in range(days):
        wd = env.unwrapped.state.weekday
        obs, _, term, trunc, i = env.step(policy(obs, env.unwrapped))
        total += i["profit"]
        lines.append(f"{DAYS[wd]} demand={i['demand']:3d} sold={i['sales']:3d} lost={i['lost']:2d} "
                     f"order={i['order']:3d} stock={i['on_hand']:3d} profit={i['profit']:7.1f}")
        if term or trunc:
            break
    lines.append(f"total profit over {len(lines)} days: {total:,.1f}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="outputs/main")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--seed", type=int, default=100_000)
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    inv = next(e for e in cfg["envs"] if e["id"] == ENV_ID)
    env = make_env(ENV_ID, **inv["env_kwargs"])
    hidden = tuple(DQNConfig(**inv["dqn"]).hidden)
    q = mlp(env.observation_space.shape[0], env.action_space.n, hidden)
    q.load_state_dict(torch.load(run_dir / f"dqn_{ENV_ID}_seed0.pt", weights_only=True))

    def dqn_policy(obs, _env):
        with torch.no_grad():
            return int(q(torch.as_tensor(obs).unsqueeze(0)).argmax(1).item())

    params = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))[ENV_ID]["summary"]["dow_sS"]["params"]
    for name, pol in (("DQN (scratch, seed 0)", dqn_policy), (f"weekday (s,S) {params}", FACTORIES["dow_sS"](**params))):
        print(f"\n=== {name}")
        print("\n".join(run(env, pol, a.seed, a.days)))


if __name__ == "__main__":
    main()
