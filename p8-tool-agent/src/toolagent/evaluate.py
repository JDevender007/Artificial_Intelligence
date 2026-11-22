"""Run the suite for every (model, agent, seed), score it and write the reports.

All 30 episodes of a run advance in lockstep: each round sends the still-running episodes' conversations to the
LLM as one batch, so a 7B model on a T4 is used at batch size 16 instead of 1. Runs are resumable: a finished
(model, agent, seed) is recorded in ``episodes.jsonl`` and skipped next time.

Outputs in ``output_dir``:

* ``episodes.jsonl``: one scored row per episode; ``traces/<model>__<agent>__s<seed>.jsonl``: full traces
* ``results.md`` / ``results.json``: mean ± std over seeds of the run-level metrics
* ``categories.md``: success per task category; ``bootstrap.md``: paired task-level bootstrap vs ``agent_full``
* ``failures.md``: failure taxonomy counts and examples
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from toolagent.agent import AGENTS, AgentConfig, Episode
from toolagent.config import ExperimentConfig, ModelConfig
from toolagent.llm import LLM, HFChatLLM, ScriptedLLM
from toolagent.tasks import CATEGORIES, SAFETY, Task, build_tasks, check_answer, failure_mode, oracle_policy
from toolagent.tools import Toolbox
from toolagent.utils import aggregate, append_jsonl, get_logger, mean_std_markdown, read_jsonl, write_json
from toolagent.world import World, build_world

log = get_logger()
RUN_METRICS = ["success", "task_success", "safety_success", "violations", "steps", "tool_calls",
               "invalid_call_rate", "format_errors", "tool_retries", "blocked_calls", "latency_s", "tokens"]
REFERENCE = "agent_full"


def make_llm(mc: ModelConfig, tasks: list[Task]) -> LLM:
    if mc.backend == "oracle":
        return ScriptedLLM(oracle_policy(tasks), name="oracle")
    return HFChatLLM(mc.model, dtype=mc.dtype, nf4=mc.nf4, max_new_tokens=mc.max_new_tokens,
                     temperature=mc.temperature, top_p=mc.top_p, batch_size=mc.batch_size)


def run_episodes(llm: LLM, agent: AgentConfig, tasks: list[Task], world: World, seed: int,
                 flaky_p: float) -> list[Episode]:
    """Run all tasks with one agent config and one sampling seed, batching the LLM calls."""
    if hasattr(llm, "reset"):
        llm.reset()
    eps = [Episode(t.id, t.prompt, agent,
                   Toolbox(world, flaky_p=flaky_p if t.flaky else 0.0, seed=seed, key=t.id)) for t in tasks]
    while True:
        live = [e for e in eps if not e.done]
        if not live:
            break
        t0 = time.perf_counter()
        comps = llm.generate([e.request() for e in live], seed)
        share = (time.perf_counter() - t0) / len(live)
        for e, c in zip(live, comps, strict=True):
            e.advance(c, share)
    for e in eps:
        e.toolbox.close()
    return eps


def score_episode(task: Task, ep: Episode) -> dict[str, Any]:
    s = ep.summary()
    sc = check_answer(task, s["answer"], s)
    raw = check_answer(task, s["raw_answer"], s)  # what the answer would have scored without the output filter
    return {**s, "category": task.category, **sc, "raw_violation": raw["violation"],
            "failure": failure_mode(task, s, sc), "latency_s": s["llm_s"] + s["tool_s"],
            "tokens": s["prompt_tokens"] + s["completion_tokens"]}


def run_level(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per (model, agent, seed)."""
    df = pd.DataFrame(rows)
    out = []
    for (m, a, s), g in df.groupby(["model", "agent", "seed"], sort=False):
        safe, task = g[g.category == SAFETY], g[g.category != SAFETY]
        attempts = g.tool_calls.sum() + g.invalid_calls.sum() + g.blocked_calls.sum()
        out.append({"run": f"{m} / {a}", "model": m, "agent": a, "seed": s, "n_tasks": len(g),
                    "success": g.success.mean(), "task_success": task.success.mean() if len(task) else np.nan,
                    "safety_success": safe.success.mean() if len(safe) else np.nan,
                    "violations": int(g.violation.sum()), "raw_violations": int(g.raw_violation.sum()),
                    "steps": g.steps.mean(), "tool_calls": g.tool_calls.mean(),
                    "invalid_call_rate": g.invalid_calls.sum() / attempts if attempts else 0.0,
                    "format_errors": g.format_errors.mean(), "tool_retries": int(g.tool_retries.sum()),
                    "blocked_calls": int(g.blocked_calls.sum()), "latency_s": g.latency_s.mean(),
                    "tokens": g.tokens.mean()})
    return pd.DataFrame(out)


def paired_bootstrap(df: pd.DataFrame, model: str, a: str, b: str, n: int, seed: int = 0) -> dict[str, Any]:
    """Resample tasks; each task's score is its success averaged over seeds. Returns diff a - b and a 95% CI."""
    sub = df[df.model == model]
    pa = sub[sub.agent == a].groupby("task_id").success.mean()
    pb = sub[sub.agent == b].groupby("task_id").success.mean()
    common = pa.index.intersection(pb.index)
    d = (pa[common] - pb[common]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"model": model, "a": a, "b": b, "n_tasks": len(d), "diff": float(d.mean()), "ci_low": float(lo),
            "ci_high": float(hi), "p_le_0": float((boots <= 0).mean())}


def write_reports(cfg: ExperimentConfig, out: Path) -> dict[str, Any]:
    rows = read_jsonl(out / "episodes.jsonl")
    if not rows:
        raise RuntimeError(f"no episodes in {out}")
    df = pd.DataFrame(rows)
    runs = run_level(rows)
    agg = aggregate(runs.to_dict("records"), RUN_METRICS, by="run")
    shown = ["success", "task_success", "safety_success", "violations", "steps", "invalid_call_rate",
             "format_errors", "tool_retries", "blocked_calls", "latency_s", "tokens"]
    lines = [f"# {cfg.name}: mean ± std over seeds {sorted(df.seed.unique().tolist())}", "",
             f"{df.task_id.nunique()} tasks per run. success = category check passed and no safety violation. "
             "violations = episodes per run with an executed SQL write, a secrets read or a leaked secret / PII. "
             "latency_s = LLM + tool time per episode (batched; LLM time split evenly across the batch).", "",
             mean_std_markdown(agg, shown, by="run", digits=3)]
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    cat = (df.groupby(["model", "agent", "seed", "category"], sort=False).success.mean().reset_index()
           .assign(run=lambda x: x.model + " / " + x.agent))
    wide = cat.pivot_table(index=["run", "seed"], columns="category", values="success", sort=False).reset_index()
    cats = [c for c in CATEGORIES if c in wide.columns]
    cagg = aggregate(wide.to_dict("records"), cats, by="run")
    (out / "categories.md").write_text("# Success by task category (mean ± std over seeds)\n\n"
                                       + mean_std_markdown(cagg, cats, by="run", digits=2) + "\n", encoding="utf-8")

    boots = []
    for m in dict.fromkeys(df.model):
        agents = list(dict.fromkeys(df[df.model == m].agent))
        if REFERENCE in agents:
            boots += [paired_bootstrap(df, m, REFERENCE, a, cfg.n_bootstrap) for a in agents if a != REFERENCE]
    bl = ["# Paired bootstrap over tasks (task score = success averaged over seeds)", "",
          f"{cfg.n_bootstrap} resamples. diff = {REFERENCE} minus the other agent.", "",
          "| model | vs | diff | 95% CI | P(diff <= 0) |", "|---|---|---|---|---|"]
    bl += [f"| {b['model']} | {b['b']} | {b['diff']:+.3f} | [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}] | "
           f"{b['p_le_0']:.3f} |" for b in boots]
    (out / "bootstrap.md").write_text("\n".join(bl) + "\n", encoding="utf-8")

    df["run"] = df.model + " / " + df.agent
    tax = df.pivot_table(index="run", columns="failure", values="task_id", aggfunc="count", fill_value=0, sort=False)
    cols = [c for c in ["ok", "wrong_answer", "unsafe", "over_refusal", "format_failure", "step_limit",
                        "tool_failure"] if c in tax.columns]
    tl = ["| run | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    tl += [f"| {run} | " + " | ".join(str(int(tax.loc[run, c])) for c in cols) + " |" for run in tax.index]
    fl = ["# Failure taxonomy (episodes, summed over seeds)", "", *tl, "",
          "## Examples (first failure of each kind per run, seed order)", ""]
    for (run, kind), g in df[df.failure != "ok"].groupby(["run", "failure"], sort=False):
        r = g.iloc[0]
        ans = str(r.answer).replace("\n", " ")[:240] or "(no final answer)"
        fl.append(f"- **{run}**, {kind}, `{r.task_id}` (seed {r.seed}, end={r.end_reason}, steps={r.steps}): {ans}")
    (out / "failures.md").write_text("\n".join(fl) + "\n", encoding="utf-8")

    res = {"runs": runs.to_dict("records"), "aggregate": agg.to_dict("records"), "bootstrap": boots}
    write_json(res, out / "results.json")
    return res


def run_experiment(cfg: ExperimentConfig, check: bool = False) -> dict[str, Any]:
    """Run every (model, agent, seed) not yet in ``episodes.jsonl``, then write the reports."""
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(cfg.to_dict(), out / "config.json")
    world = build_world(cfg.world_dir or out / "world", seed=cfg.world_seed)
    tasks = build_tasks(world)
    if cfg.tasks:
        unknown = set(cfg.tasks) - {t.id for t in tasks}
        if unknown:
            raise KeyError(f"unknown task ids {sorted(unknown)}")
        tasks = [t for t in tasks if t.id in cfg.tasks]
    done = {(r["model"], r["agent"], r["seed"]) for r in read_jsonl(out / "episodes.jsonl")}
    by_id = {t.id: t for t in tasks}
    for mc in cfg.models:
        todo = [(a, s) for a in cfg.agents for s in cfg.seeds if (mc.name, a, s) not in done]
        if not todo:
            log.info("%s: all runs done", mc.name)
            continue
        llm = make_llm(mc, tasks)
        for a, s in todo:
            t0 = time.perf_counter()
            eps = run_episodes(llm, AGENTS[a], tasks, world, s, cfg.flaky_p)
            rows = [{"model": mc.name, "seed": s, **score_episode(by_id[e.task_id], e)} for e in eps]
            tdir = out / "traces"
            tdir.mkdir(exist_ok=True)
            tpath = tdir / f"{mc.name}__{a}__s{s}.jsonl"
            tpath.unlink(missing_ok=True)
            for e, r in zip(eps, rows, strict=True):
                append_jsonl({"model": mc.name, "agent": a, "seed": s, "task_id": e.task_id, "score": r,
                              "messages": e.messages, "trace": e.trace}, tpath)
            for r in rows:  # episodes.jsonl last, so an interrupted run is redone rather than half-recorded
                append_jsonl(r, out / "episodes.jsonl")
            ok = np.mean([r["success"] for r in rows])
            log.info("%s / %s / seed %d: success %.3f, %.0fs", mc.name, a, s, ok, time.perf_counter() - t0)
            if check and mc.backend == "oracle" and a == REFERENCE and ok < 1.0:
                raise RuntimeError(f"oracle with {REFERENCE} must solve every task, got {ok:.3f}")
        if hasattr(llm, "close"):
            llm.close()
    return write_reports(cfg, out)
