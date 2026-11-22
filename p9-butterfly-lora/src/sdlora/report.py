"""Markdown reports from ``runs.jsonl`` / ``per_prompt.jsonl`` rows (pandas only; no torch)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sdlora.metrics import paired_bootstrap
from sdlora.utils import aggregate, mean_std_markdown, write_json

RUN_METRICS = ["fid", "kid_x1000", "clip_score", "top1", "top5", "mem_max_sim", "mem_frac_ge_095", "sd_nsfw_rate",
               "clip_flag_rate", "gen_s_per_img"]
PROMPT_METRICS = ["clip_score", "top1", "top5", "mem_sim"]


def results_markdown(rows: list[dict[str, Any]], reference: dict[str, Any], variants: list[str]) -> str:
    df = pd.DataFrame(rows)
    df["variant"] = pd.Categorical(df["variant"], [v for v in variants if v in set(df["variant"])], ordered=True)
    df = df.sort_values(["variant", "seed"])
    df["variant"] = df["variant"].astype(str)
    agg = aggregate(df.to_dict("records"), RUN_METRICS, by="variant")
    n = int(df["n_images"].iloc[0])
    lines = [
        f"# Results: mean ± std over seeds ({n} images per run, one per held-out species caption)", "",
        mean_std_markdown(agg, RUN_METRICS, by="variant", digits=3), "",
        "## Reference: real held-out images", "",
        "| reference | fid | kid_x1000 | clip_score | top1 | top5 | mem_max_sim | mem_frac_ge_095 | clip_flag_rate |",
        "|---|---|---|---|---|---|---|---|---|",
        "| real test vs real train subset (floor) | {fid:.3f} | {kid_x1000:.3f} | {clip_score:.3f} | {top1:.3f} | "
        "{top5:.3f} | {mem_max_sim:.3f} | {mem_frac_ge_095:.3f} | {clip_flag_rate:.3f} |".format(**reference),
        "",
        "* fid / kid_x1000: against the held-out real images (lower is better). With only "
        f"{n} images FID is biased upward; compare variants with each other and with the real-vs-real floor, "
        "not with published FIDs. KID is unbiased at small n.",
        "* clip_score: 100 * cos(image, caption) with CLIP ViT-B/16. top1/top5: the image retrieves its own species "
        "name among all held-out species (chance = 1/n and 5/n).",
        "* mem_max_sim: mean over images of the max CLIP cosine to any training image; mem_frac_ge_095: share of "
        "images with a near-copy (>= 0.95) in the training set. The real held-out images give the natural level.",
        "* sd_nsfw_rate: images blacked out by the SD safety checker; clip_flag_rate: images flagged by the CLIP "
        "concept filter. All prompts are benign, so both are false-positive rates.",
    ]
    return "\n".join(lines) + "\n"


def per_prompt_means(prompt_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Average each per-prompt metric over seeds: one row per (variant, idx)."""
    df = pd.DataFrame(prompt_rows)
    return df.groupby(["variant", "idx"], sort=False)[PROMPT_METRICS].mean().reset_index()


def bootstrap_markdown(prompt_rows: list[dict[str, Any]], pairs: list[tuple[str, str]], n: int,
                       seed: int = 0) -> tuple[str, list[dict[str, Any]]]:
    pm = per_prompt_means(prompt_rows)
    out = []
    lines = ["# Paired bootstrap over prompts (per-prompt metric averaged over seeds first)", "",
             "| comparison | metric | diff | 95% CI | P(diff <= 0) | prompts |", "|---|---|---|---|---|---|"]
    for a, b in pairs:
        x = pm[pm.variant == a].set_index("idx")
        y = pm[pm.variant == b].set_index("idx")
        common = x.index.intersection(y.index)
        if len(common) == 0:
            continue
        for m in PROMPT_METRICS:
            r = paired_bootstrap(x.loc[common, m].to_numpy(), y.loc[common, m].to_numpy(), n=n, seed=seed)
            out.append({"a": a, "b": b, "metric": m, **r})
            lines.append(f"| {a} - {b} | {m} | {r['diff']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
                         f"{r['p_le_0']:.3f} | {r['n_items']} |")
    lines += ["", "Prompts are resampled with replacement (2.5/97.5 percentiles). Base and LoRA images for the same "
              "prompt and seed share their initial noise, so the comparison is paired."]
    return "\n".join(lines) + "\n", out


def safety_markdown(s: dict[str, Any]) -> str:
    rt, cap = s["red_team"], s["captions"]
    lines = [
        "# Safety filter", "",
        "## 1. Prompt filter (regex categories on raw + de-obfuscated text)", "",
        "| set | n | metric | value |", "|---|---|---|---|",
        f"| red-team unsafe prompts (hand-written, in-sample) | {rt['n_unsafe']} | recall | {rt['recall']:.3f} |",
        f"| red-team benign hard negatives (in-sample) | {rt['n_benign']} | false-positive rate | "
        f"{rt['false_positive_rate']:.3f} |",
        f"| all dataset captions (real species names, out-of-sample) | {cap['n']} | false-positive rate | "
        f"{cap['false_positive_rate']:.4f} |", "",
        f"Missed unsafe prompts: {rt['missed'] or 'none'}. False blocks on hard negatives: {rt['false_blocks'] or 'none'}."
        f" Blocked dataset captions: {cap['blocked'] or 'none'}.", "",
        "The red-team set was written together with the filter, so its numbers are an upper bound, not a held-out "
        "estimate; the caption false-positive rate is the honest number.", "",
        "## 2. Image filters on benign generations (false-positive rates)", "",
        "| images | n | SD safety checker | CLIP concept filter |", "|---|---|---|---|",
    ]
    for r in s["image_filters"]:
        sd = "n/a" if r["sd_nsfw_rate"] is None else f"{r['sd_nsfw_rate']:.4f}"
        lines.append(f"| {r['images']} | {r['n']} | {sd} | {r['clip_flag_rate']:.4f} |")
    lines += ["", "Recall of the image filters is not measured: we do not generate unsafe images to test them.",
              "", "## 3. Provenance", "", "Every generated PNG carries `ai_generated=true`, the base model, LoRA "
              "variant, prompt and seed as PNG text chunks (removable by re-encoding; not a robust watermark)."]
    return "\n".join(lines) + "\n"


def training_markdown(train_rows: list[dict[str, Any]]) -> str:
    lines = ["# LoRA training runs", "", "| seed | steps | LoRA params | train s | s/step | final loss (avg 50) |",
             "|---|---|---|---|---|---|"]
    for r in train_rows:
        lines.append(f"| {r['seed']} | {r['steps']} | {r['lora_params']:,} | {r['train_s']:.0f} | "
                     f"{r['s_per_step']:.3f} | {r['final_loss_avg50']:.4f} |")
    return "\n".join(lines) + "\n"


def write_reports(out: Path, rows: list[dict[str, Any]], prompt_rows: list[dict[str, Any]], reference: dict[str, Any],
                  safety: dict[str, Any], train_rows: list[dict[str, Any]], variants: list[str],
                  n_bootstrap: int) -> dict[str, Any]:
    out = Path(out)
    (out / "results.md").write_text(results_markdown(rows, reference, variants), encoding="utf-8")
    final, first = variants[-1], variants[1]
    pairs = [(final, "base")] + ([(final, first)] if first != final else [])
    md, boot = bootstrap_markdown(prompt_rows, pairs, n_bootstrap)
    (out / "bootstrap.md").write_text(md, encoding="utf-8")
    (out / "safety.md").write_text(safety_markdown(safety), encoding="utf-8")
    if train_rows:
        (out / "training.md").write_text(training_markdown(train_rows), encoding="utf-8")
    agg = aggregate(rows, RUN_METRICS, by="variant")
    res = {"aggregate": agg.replace({np.nan: None}).to_dict("records"), "bootstrap": boot, "reference": reference,
           "safety": safety, "training": train_rows}
    write_json(res, out / "results.json")
    return res
