"""One command per experiment: train every model x seed, evaluate on the test period, write reports.

Outputs (in ``cfg.output_dir``): results.jsonl / results.md (mean ± std over seeds), segments.md (by user
activity, cold start), bootstrap.md, index.md (FAISS exact vs HNSW), error_analysis.md, data_summary.md and
serve/ (seed-0 two-tower checkpoint + item vectors for the API). Rerunning skips (model, seed) rows already done.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig, tower_variant
from .data import Dataset, build_dataset, summary
from .index import benchmark, faiss_available
from .metrics import EvalResult, evaluate, head_items, paired_bootstrap, summarize
from .models import BPRRecommender, Popularity, TwoTowerRecommender, known_items, known_users, pick_device
from .utils import aggregate, append_jsonl, get_logger, mean_std_markdown, read_jsonl, set_seed, write_json

log = get_logger()
ACTIVITY_BINS = [(0, 0, "cold (0)"), (1, 19, "1-19"), (20, 99, "20-99"), (100, 299, "100-299"),
                 (300, 10**9, "300+")]


def activity_segments(ds: Dataset, users: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean masks over ``users`` by number of ratings before the test period."""
    prior = np.bincount(ds.part("train", "val")["u"].to_numpy(), minlength=ds.n_users)[users]
    segs = {name: (prior >= lo) & (prior <= hi) for lo, hi, name in ACTIVITY_BINS}
    segs["no train ratings (MF-cold)"] = ~known_users(ds)[users]
    return segs


def build_model(name: str, ds: Dataset, cfg: ExperimentConfig, device: torch.device) -> Any:
    if name == "popularity":
        return Popularity(ds)
    if name == "bpr_mf":
        return BPRRecommender(ds, cfg.bpr, cfg.train, device)
    return TwoTowerRecommender(ds, tower_variant(cfg.tower, name), cfg.train, device)


def _md_table(df: pd.DataFrame, digits: int = 3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [f"{v:.{digits}f}" if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_one(name: str, seed: int, ds: Dataset, cfg: ExperimentConfig, device: torch.device,
            ctx: dict[str, Any], out: Path) -> dict[str, Any]:
    set_seed(seed)
    k = cfg.eval.k_main
    model = build_model(name, ds, cfg, device)
    info: dict[str, Any] = {"best_epoch": 0, "epochs_run": 0, "train_seconds": 0.0, "best_val": float("nan")}
    if name != "popularity":
        def val_metric(score_fn: Any) -> float:
            return float(evaluate(score_fn, ctx["val_targets"], ctx["val_seen"], [k]).recall[k].mean())
        info = model.fit(val_metric)
        write_json(info["history"], out / "history" / f"{name}_s{seed}.json")
    res: EvalResult = evaluate(model.score_fn("test"), ctx["test_targets"], ctx["test_seen"], cfg.eval.ks,
                               cold_items=ctx["cold_items"])
    row = {"model": name, "seed": seed, **summarize(res, cfg.eval.ks, ds.n_items, ctx["head"])}
    for kk in cfg.eval.ks:
        hits, total = res.hits_cold_items[kk]
        row[f"cold_item_recall@{kk}"] = hits / total if total else float("nan")
    row.update({x: info[x] for x in ("best_epoch", "epochs_run", "train_seconds", "best_val")})
    for seg, mask in activity_segments(ds, res.users).items():
        append_jsonl({"model": name, "seed": seed, "segment": seg,
                      **summarize(res, [k], ds.n_items, ctx["head"], mask)}, out / "segments.jsonl")
    np.savez_compressed(out / "per_user" / f"{name}_s{seed}.npz", users=res.users, topk=res.topk[:, :max(cfg.eval.ks)],
                        **{f"recall@{kk}": res.recall[kk] for kk in cfg.eval.ks},
                        **{f"ndcg@{kk}": res.ndcg[kk] for kk in cfg.eval.ks})
    if name == cfg.serve_model and seed == cfg.seeds[0]:
        export_serving(model, ds, out / "serve")
    return row


def export_serving(model: TwoTowerRecommender, ds: Dataset, d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.checkpoint(), d / "two_tower.pt")
    np.save(d / "item_vecs.npy", model.item_vectors().cpu().numpy())
    users = np.arange(ds.n_users)
    np.save(d / "user_vecs.npy", model.user_vectors(users, "test").cpu().numpy())
    seen = ds.part("train", "val")
    meta = {
        "item_raw": ds.item_raw.tolist(), "titles": ds.titles, "genres": ds.genres,
        "user_raw": ds.user_raw.tolist(),
        "seen": {int(u): g["i"].astype(int).tolist() for u, g in seen.groupby("u")},
        "liked": {int(u): g["i"].astype(int).tolist() for u, g in seen[seen["pos"]].groupby("u")},
    }
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def write_reports(cfg: ExperimentConfig, ds: Dataset, ctx: dict[str, Any], out: Path) -> None:
    k = cfg.eval.k_main
    rows = [r for r in read_jsonl(out / "results.jsonl") if r["model"] in cfg.models]
    order = [m for m in cfg.models if any(r["model"] == m for r in rows)]
    rows.sort(key=lambda r: order.index(r["model"]))
    metrics = [f"recall@{kk}" for kk in cfg.eval.ks] + [f"ndcg@{kk}" for kk in cfg.eval.ks if kk <= k] + [
        f"coverage@{k}", f"tail_share@{k}", f"cold_item_recall@{k}"]
    agg = aggregate(rows, metrics + ["train_seconds"])
    text = [f"# Results: {cfg.name} (test period, {len(ctx['test_targets'])} users, mean ± std over seeds)\n",
            mean_std_markdown(agg, metrics), "",
            "Training time (s): " + ", ".join(f"{r['model']} {r['train_seconds_mean']:.0f}" for _, r in agg.iterrows()),
            "", f"Head = top 10% items by train positives ({len(ctx['head'])} items). "
            f"Cold items = no train ratings ({len(ctx['cold_items'])}). popularity is deterministic: seeds identical."]
    (out / "results.md").write_text("\n".join(text) + "\n", encoding="utf-8")

    seg = pd.DataFrame([r for r in read_jsonl(out / "segments.jsonl") if r["model"] in order])
    if len(seg):
        lines = [f"# Recall@{k} by user segment (mean over seeds)\n"]
        piv = seg.groupby(["segment", "model"], sort=False)[f"recall@{k}"].mean().unstack("model")[order]
        n = seg.groupby("segment", sort=False)["n_users"].first()
        piv.insert(0, "users", n.reindex(piv.index).astype(int))
        lines.append(_md_table(piv.reset_index()))
        (out / "segments.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # paired bootstrap on per-user recall averaged over seeds (same users in every run)
    per = {}
    for m in order:
        arrs = [np.load(out / "per_user" / f"{m}_s{s}.npz")[f"recall@{k}"] for s in cfg.seeds
                if (out / "per_user" / f"{m}_s{s}.npz").exists()]
        if arrs:
            per[m] = np.mean(arrs, axis=0)
    pairs = [("two_tower", b) for b in ("popularity", "bpr_mf", "two_tower_nofeat", "two_tower_nologq",
                                        "two_tower_inbatch")] + [("bpr_mf", "popularity")]
    bl = [f"# Paired bootstrap over users: recall@{k} difference (A - B), {cfg.eval.n_bootstrap} resamples\n",
          "| A | B | diff | 95% CI | share of resamples <= 0 |", "|---|---|---|---|---|"]
    for a, b in pairs:
        if a in per and b in per:
            r = paired_bootstrap(per[a], per[b], cfg.eval.n_bootstrap)
            bl.append(f"| {a} | {b} | {r['diff']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['p_le_0']:.3f} |")
    (out / "bootstrap.md").write_text("\n".join(bl) + "\n", encoding="utf-8")
    write_error_analysis(cfg, ds, ctx, out, order)


def write_error_analysis(cfg: ExperimentConfig, ds: Dataset, ctx: dict[str, Any], out: Path,
                         order: list[str]) -> None:
    k = cfg.eval.k_main
    s0 = cfg.seeds[0]
    targets = ctx["test_targets"]
    head = set(ctx["head"].tolist())
    lines = [f"# Error analysis (seed {s0}, k={k})\n", "## Hit rate of test positives by item popularity\n",
             "Share of (user, liked test item) pairs found in the user's top-k, split by whether the item is a head "
             "item (top 10% by train positives), a tail item, or cold (no train ratings).\n",
             "| model | head | tail (seen in train) | cold item |", "|---|---|---|---|"]
    cold = set(ctx["cold_items"].tolist())
    loaded = {}
    for m in order:
        p = out / "per_user" / f"{m}_s{s0}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        loaded[m] = z
        cnt = {"head": [0, 0], "tail": [0, 0], "cold": [0, 0]}
        for n, u in enumerate(z["users"]):
            top = set(z["topk"][n, :k].tolist())
            for i in targets[int(u)]:
                b = "cold" if i in cold else "head" if i in head else "tail"
                cnt[b][1] += 1
                cnt[b][0] += int(i in top)
        cells = [f"{h / t:.3f} ({t})" if t else "n/a" for h, t in cnt.values()]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines.append("\n(counts in brackets are the number of pairs)\n")

    show = [m for m in (cfg.serve_model, "bpr_mf") if m in loaded]
    if show:
        rng = np.random.default_rng(0)
        users = loaded[show[0]]["users"]
        picks = rng.choice(len(users), size=min(cfg.eval.n_examples, len(users)), replace=False)
        lines.append("## Example users (random, test period)\n")
        liked = ds.part("train", "val", positives=True)
        for n in picks:
            u = int(users[n])
            f = ds.user_feat[u]
            hist = liked[liked["u"] == u]["i"].to_numpy()[-5:]
            tgt = set(targets[u].tolist())
            lines.append(f"### user {ds.user_raw[u]} (gender {'M' if f[0] else 'F'}, age bucket {f[1]}, "
                         f"occupation {f[2]}, {len(liked[liked['u'] == u])} liked before test)\n")
            lines.append("- last liked: " + ("; ".join(ds.titles[i] for i in hist) or "(none: cold start)"))
            lines.append("- liked in test: " + "; ".join(ds.titles[i] for i in sorted(tgt)[:8]))
            for m in show:
                top = loaded[m]["topk"][n, :10]
                lines.append(f"- {m} top-10: " + "; ".join(("**" + ds.titles[i] + "**") if i in tgt else ds.titles[i]
                                                           for i in top))
            lines.append("")
        lines.append("Bold = liked by the user in the test period.")
    (out / "error_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_index_report(cfg: ExperimentConfig, ctx: dict[str, Any], out: Path) -> None:
    d = out / "serve"
    if not (d / "item_vecs.npy").exists():
        return
    items = np.load(d / "item_vecs.npy")
    users = np.load(d / "user_vecs.npy")[np.array(sorted(ctx["test_targets"]))]
    k = cfg.eval.k_main
    rows = benchmark(items, users, k, cfg.eval.hnsw_m, cfg.eval.hnsw_ef, cfg.eval.n_latency_queries)
    note = "" if faiss_available() else "\nfaiss is not installed here: only the NumPy baseline was measured.\n"
    text = (f"# Retrieval index ({cfg.serve_model} seed {cfg.seeds[0]}, {items.shape[0]} items, dim {items.shape[1]}, "
            f"{len(users)} test users, k={k})\n\nrecall_vs_exact = overlap of raw top-{k} with exact search "
            "(before seen-item filtering); latency = one query at a time on this machine's CPU.\n\n"
            + _md_table(pd.DataFrame(rows), digits=4) + "\n" + note)
    (out / "index.md").write_text(text, encoding="utf-8")


def run_experiment(cfg: ExperimentConfig) -> Path:
    out = Path(cfg.output_dir)
    (out / "per_user").mkdir(parents=True, exist_ok=True)
    write_json(cfg.to_dict(), out / "config.json")
    ds = build_dataset(cfg.data)
    s = summary(ds)
    write_json(s, out / "data_summary.json")
    (out / "data_summary.md").write_text("# Data summary\n\n" + "\n".join(f"- {a}: {b}" for a, b in s.items())
                                         + "\n", encoding="utf-8")
    log.info("data: %s", s)
    ctx = {
        "val_targets": ds.targets("val"), "val_seen": ds.seen_matrix("train"),
        "test_targets": ds.targets("test"), "test_seen": ds.seen_matrix("train", "val"),
        "head": head_items(ds.item_counts("train")), "cold_items": np.flatnonzero(~known_items(ds)),
    }
    device = pick_device(cfg.train.device)
    done = {(r["model"], r["seed"]) for r in read_jsonl(out / "results.jsonl")}
    for name in cfg.models:
        for seed in cfg.seeds:
            if (name, seed) in done:
                log.info("skip %s seed %d (done)", name, seed)
                continue
            log.info("== %s seed %d", name, seed)
            row = run_one(name, seed, ds, cfg, device, ctx, out)
            append_jsonl(row, out / "results.jsonl")
            log.info("%s seed %d: recall@%d %.4f ndcg@%d %.4f", name, seed, cfg.eval.k_main,
                     row[f"recall@{cfg.eval.k_main}"], cfg.eval.k_main, row[f"ndcg@{cfg.eval.k_main}"])
    write_reports(cfg, ds, ctx, out)
    write_index_report(cfg, ctx, out)
    return out
