"""Run every (env, algorithm, seed) job in a process pool, tune and score the heuristics, and write the reports."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import torch

from rlinv.agents import SCRATCH, train_one
from rlinv.config import EnvSpec, ExperimentConfig
from rlinv.envs import ENV_ID
from rlinv.evaluate import evaluate_policy
from rlinv.heuristics import FACTORIES, random_policy, tune
from rlinv.report import write_reports

log = logging.getLogger(__name__)


def build_jobs(cfg: ExperimentConfig) -> list[dict[str, Any]]:
    eval_seeds = [cfg.eval_seed_base + i for i in range(max(e.eval_episodes for e in cfg.envs))]
    final_seeds = [cfg.final_seed_base + i for i in range(cfg.final_episodes)]
    jobs = []
    for e in cfg.envs:
        steps = max(1, int(round(e.total_steps * cfg.steps_scale)))
        every = max(1, int(round(e.eval_every * cfg.steps_scale)))
        for algo in e.algos:
            hp = e.dqn if algo in SCRATCH else getattr(e, algo)
            for s in cfg.seeds:
                jobs.append({"env_id": e.id, "algo": algo, "seed": s, "total_steps": steps, "eval_every": every,
                             "eval_seeds": eval_seeds[:e.eval_episodes], "final_seeds": final_seeds, "hp": dict(hp),
                             "env_kwargs": dict(e.env_kwargs), "keep_model": algo == "dqn" and s == cfg.seeds[0]})
    # Longest jobs first, so the pool doesn't end on one straggler.
    return sorted(jobs, key=lambda j: -j["total_steps"])


def run_heuristics(cfg: ExperimentConfig, e: EnvSpec) -> list[dict[str, Any]]:
    """Tune each heuristic on the tuning seeds, then score it on the same final seeds as the RL policies."""
    final_seeds = [cfg.final_seed_base + i for i in range(cfg.final_episodes)]
    tune_seeds = [cfg.tune_seed_base + i for i in range(cfg.tune_episodes)]
    out = []
    for name in e.heuristics:
        if name == "random":
            params, train_score, pol = {}, None, random_policy(0)
        else:
            params, train_score = tune(name, tune_seeds, e.env_kwargs)
            pol = FACTORIES[name](**params)
        final = evaluate_policy(e.id, pol, final_seeds, e.env_kwargs)
        out.append({"env_id": e.id, "algo": name, "seed": 0, "params": params, "tune_score": train_score,
                    "curve": [], "final": final, "heuristic": True})
    return out


def run_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    jobs = build_jobs(cfg)
    n_workers = cfg.n_workers or os.cpu_count() or 1
    log.info("%d training jobs on %d workers", len(jobs), n_workers)
    if n_workers == 1:
        runs = [train_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            runs = list(pool.map(train_one, jobs))
    for r in runs:
        state = r.pop("model_state", None)
        if state is not None:
            tag = r["env_id"].replace("/", "_")
            torch.save(state, out_dir / f"dqn_{tag}_seed{r['seed']}.pt")
    for e in cfg.envs:
        if e.heuristics:
            if e.id != ENV_ID:
                raise ValueError("heuristics are only defined for the inventory env")
            runs.extend(run_heuristics(cfg, e))
    thresholds = {e.id: e.threshold for e in cfg.envs}
    return write_reports(out_dir, runs, thresholds, cfg.n_bootstrap)
