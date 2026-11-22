"""P10 error analysis: cost breakdown of seed-0 DQN vs tuned dow_sS over the 50 held-out episodes,
plus peak-vs-final learning-curve gap for every trained Inventory run.

    python scripts/error_analysis.py <run-dir> reports/kaggle_cpu   # run-dir holds dqn_Inventory-v0_seed0.pt
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from rlinv.dqn import DQNConfig, mlp
from rlinv.envs import ENV_ID, make_env
from rlinv.heuristics import FACTORIES

run_dir, rep = Path(sys.argv[1]), Path(sys.argv[2])
cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
inv = next(e for e in cfg["envs"] if e["id"] == ENV_ID)
env = make_env(ENV_ID, **inv["env_kwargs"])
p = env.unwrapped.p
q = mlp(env.observation_space.shape[0], env.action_space.n, tuple(DQNConfig(**inv["dqn"]).hidden))
q.load_state_dict(torch.load(run_dir / f"dqn_{ENV_ID}_seed0.pt", weights_only=True))


def dqn(obs, _e):
    with torch.no_grad():
        return int(q(torch.as_tensor(obs).unsqueeze(0)).argmax(1).item())


params = json.loads((rep / "results.json").read_text(encoding="utf-8"))[ENV_ID]["summary"]["dow_sS"]["params"]
pols = {"dqn_seed0": dqn, "dow_sS": FACTORIES["dow_sS"](**params)}
keys = ["profit", "orders", "units_ordered", "fixed_cost", "lost", "wasted", "holding", "end_stock"]
print("| policy | " + " | ".join(keys) + " |\n|---" * 1 + "|---" * len(keys) + "|")
for name, pol in pols.items():
    agg = {k: [] for k in keys}
    for ep in range(cfg["final_episodes"]):
        obs, _ = env.reset(seed=cfg["final_seed_base"] + ep)
        t = dict.fromkeys(keys, 0.0)
        done = False
        while not done:
            obs, _, term, trunc, i = env.step(pol(obs, env.unwrapped))
            t["profit"] += i["profit"]
            t["orders"] += i["order"] > 0
            t["units_ordered"] += i["order"]
            t["fixed_cost"] += p.fixed_cost * (i["order"] > 0)
            t["lost"] += i["lost"]
            t["wasted"] += i["wasted"]
            t["holding"] += p.holding_cost * i["on_hand"]
            done = term or trunc
        t["end_stock"] = i["on_hand"] + sum(env.unwrapped.state.pipeline)
        for k in keys:
            agg[k].append(t[k])
    print(f"| {name} | " + " | ".join(f"{np.mean(agg[k]):,.1f}" for k in keys) + " |")

print("\n| algo | mean final eval | mean peak eval | mean (peak - final) | seeds where final < 95% of peak |")
print("|---|---|---|---|---|")
runs = json.loads((rep / "runs.json").read_text(encoding="utf-8"))
for algo in ("dqn", "dqn_nodouble", "sb3_dqn", "sb3_ppo"):
    rs = [r for r in runs if r["env_id"] == ENV_ID and r["algo"] == algo]
    fin = np.array([r["curve"][-1][1] for r in rs])
    peak = np.array([max(v for _, v in r["curve"]) for r in rs])
    print(f"| {algo} | {fin.mean():,.1f} | {peak.mean():,.1f} | {(peak - fin).mean():,.1f} | "
          f"{(fin < 0.95 * peak).sum()}/{len(rs)} |")
