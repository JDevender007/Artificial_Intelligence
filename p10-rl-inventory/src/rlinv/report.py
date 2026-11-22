"""Reports: per-env results tables, a hierarchical bootstrap between methods, and learning-curve plots with 95% bands."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from rlinv.agents import ALGOS
from rlinv.stats import hierarchical_bootstrap, iqm, t_interval

log = logging.getLogger(__name__)

PAIRS = [("dqn", "dqn_nodouble"), ("dqn", "dqn_notarget"), ("dqn_nodouble", "dqn_notarget"), ("dqn", "sb3_dqn"),
         ("sb3_ppo", "sb3_dqn"), ("sb3_ppo", "dqn")]


def _order(algos: set[str]) -> list[str]:
    rl = [a for a in ALGOS if a in algos]
    return sorted(algos - set(rl), key=lambda a: (a != "random", a)) + rl


def score_matrix(runs: list[dict[str, Any]]) -> np.ndarray:
    """(n_seeds, n_episodes) final scores, rows ordered by seed."""
    runs = sorted(runs, key=lambda r: r["seed"])
    return np.array([[ep["score"] for ep in r["final"]] for r in runs], dtype=float)


def curve_matrix(runs: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Common eval steps and an (n_seeds, n_points) score matrix, truncated to the shortest curve."""
    n = min(len(r["curve"]) for r in runs)
    steps = np.array([s for s, _ in runs[0]["curve"][:n]])
    return steps, np.array([[v for _, v in r["curve"][:n]] for r in sorted(runs, key=lambda r: r["seed"])])


def summarize(runs: list[dict[str, Any]], threshold: float | None) -> dict[str, Any]:
    m = score_matrix(runs)
    per_seed = m.mean(axis=1)
    mean, lo, hi = t_interval(per_seed)
    row: dict[str, Any] = {"n_seeds": len(per_seed), "final_mean": float(mean), "final_std": float(per_seed.std(ddof=1))
                           if len(per_seed) > 1 else None, "ci_low": float(lo), "ci_high": float(hi),
                           "iqm": iqm(m), "episode_std": float(m.std())}
    fills = [ep["fill_rate"] for r in runs for ep in r["final"] if "fill_rate" in ep]
    if fills:
        row["fill_rate"] = float(np.mean(fills))
    if runs[0]["curve"]:
        steps, c = curve_matrix(runs)
        row["auc"] = float(c.mean())  # mean eval score over the run: rewards learning fast and staying good
        if threshold is not None:
            hits = [int(steps[np.argmax(seed_c >= threshold)]) for seed_c in c if (seed_c >= threshold).any()]
            row["solved_seeds"] = len(hits)
            row["median_steps_to_threshold"] = float(np.median(hits)) if hits else None
    if "train_s" in runs[0]:
        row["train_s"] = float(np.mean([r["train_s"] for r in runs]))
    if runs[0].get("params") is not None and runs[0].get("heuristic"):
        row["params"] = runs[0]["params"]
    return row


def _fmt(v: Any, nd: int = 1) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    return str(v)


def plot_curves(path: Path, env_id: str, by_algo: dict[str, list[dict]], summaries: dict[str, dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ci = 0
    for algo in _order(set(by_algo)):
        runs = by_algo[algo]
        if runs[0]["curve"]:
            steps, c = curve_matrix(runs)
            m, lo, hi = t_interval(c)
            col = colors[ci % len(colors)]
            ci += 1
            ax.plot(steps, m, label=f"{algo} (n={len(runs)})", color=col)
            ax.fill_between(steps, lo, hi, color=col, alpha=0.2)
        else:
            ax.axhline(summaries[algo]["final_mean"], ls="--", lw=1, color="gray" if algo == "random" else "black",
                       alpha=0.7)
            ax.annotate(algo, (0, summaries[algo]["final_mean"]), fontsize=8, va="bottom")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode profit" if "Inventory" in env_id else "episode return")
    ax.set_title(f"{env_id}: mean over seeds, shaded = 95% t-interval")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_reports(out_dir: str | Path, runs: list[dict[str, Any]], thresholds: dict[str, float | None],
                  n_bootstrap: int = 5000) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs.json").write_text(json.dumps(runs), encoding="utf-8")
    result: dict[str, Any] = {}
    lines_md, boot_md = ["# Results: final held-out evaluation", ""], ["# Hierarchical paired bootstrap", "",
                                                                      f"{n_bootstrap} resamples of training seeds "
                                                                      "(per side) and shared evaluation episodes.", ""]
    for env_id in dict.fromkeys(r["env_id"] for r in runs):
        by_algo: dict[str, list[dict]] = {}
        for r in runs:
            if r["env_id"] == env_id:
                by_algo.setdefault(r["algo"], []).append(r)
        thr = thresholds.get(env_id)
        summ = {a: summarize(rs, thr) for a, rs in by_algo.items()}
        n_eps = len(next(iter(by_algo.values()))[0]["final"])
        lines_md += [f"## {env_id}", "", f"Final score = mean over {n_eps} held-out episodes per trained policy, "
                     "then mean ± std over training seeds (t 95% CI). Heuristics are deterministic and tuned once "
                     "on separate seeds.", ""]
        has_fill = any("fill_rate" in s for s in summ.values())
        head = ["method", "seeds", "final mean ± std", "95% CI", "IQM", "AUC"]
        head += ["solved seeds", "median steps to solve"] if thr is not None else []
        head += ["fill rate"] if has_fill else []
        head += ["train s"]
        lines_md += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for a in _order(set(by_algo)):
            s = summ[a]
            name = a + (f" {s['params']}" if s.get("params") else "")
            cells = [name, str(s["n_seeds"]),
                     f"{_fmt(s['final_mean'])} ± {_fmt(s['final_std'])}" if s["final_std"] is not None
                     else _fmt(s["final_mean"]),
                     f"[{_fmt(s['ci_low'])}, {_fmt(s['ci_high'])}]" if s["n_seeds"] > 1 else "n/a",
                     _fmt(s["iqm"]), _fmt(s.get("auc"))]
            if thr is not None:
                cells += [f"{s.get('solved_seeds', 'n/a')}/{s['n_seeds']}", _fmt(s.get("median_steps_to_threshold"), 0)]
            if has_fill:
                cells.append(_fmt(s.get("fill_rate"), 3))
            cells.append(_fmt(s.get("train_s"), 0))
            lines_md.append("| " + " | ".join(cells) + " |")
        lines_md.append("")
        pairs = [p for p in PAIRS if p[0] in by_algo and p[1] in by_algo]
        heur = [a for a in by_algo if by_algo[a][0].get("heuristic") and a != "random"]
        pairs += [(a, h) for a in ALGOS if a in by_algo for h in heur]
        boots = []
        for a, b in pairs:
            r = hierarchical_bootstrap(score_matrix(by_algo[a]), score_matrix(by_algo[b]), n=n_bootstrap)
            boots.append({"a": a, "b": b, **r})
        if boots:
            boot_md += [f"## {env_id}", "", "| a - b | diff | 95% CI | P(diff <= 0) |", "|---|---|---|---|"]
            boot_md += [f"| {x['a']} - {x['b']} | {x['diff']:+,.2f} | [{x['ci_low']:+,.2f}, {x['ci_high']:+,.2f}] | "
                        f"{x['p_le_0']:.3f} |" for x in boots]
            boot_md.append("")
        tag = env_id.replace("/", "_")
        try:
            plot_curves(out / f"curves_{tag}.png", env_id, by_algo, summ)
        except Exception as exc:  # plotting must never lose the numbers
            log.warning("plot failed for %s: %s", env_id, exc)
        result[env_id] = {"summary": summ, "bootstrap": boots}
    (out / "results.md").write_text("\n".join(lines_md), encoding="utf-8")
    (out / "bootstrap.md").write_text("\n".join(boot_md), encoding="utf-8")
    (out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
