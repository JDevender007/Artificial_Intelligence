"""One command per experiment: data -> real-image features -> (base | LoRA x checkpoints) x seeds -> reports.

Resumable: each finished (variant, seed) is a row in ``runs.jsonl`` and is skipped on re-run; trained LoRAs and
feature caches are reused.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sdlora.config import ExperimentConfig
from sdlora.data import image_path, prepare_dataset
from sdlora.metrics import clip_scores, frechet_distance, kid, max_similarity, retrieval_hits
from sdlora.report import write_reports
from sdlora.safety import (
    SAFE_CONCEPTS,
    UNSAFE_CONCEPTS,
    check_prompt,
    concept_flags,
    evaluate_prompt_filter,
    save_with_provenance,
)
from sdlora.utils import append_jsonl, get_logger, read_jsonl, set_seed, write_json

log = get_logger(__name__)
NOISE_SEED_STRIDE = 100_000  # noise seed of prompt i under run seed s = s * stride + i (shared by every variant)


def make_grid(images: list[Image.Image], cols: int = 4, size: int = 256) -> Image.Image:
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * size, rows * size), "white")
    for i, im in enumerate(images):
        grid.paste(im.resize((size, size)), ((i % cols) * size, (i // cols) * size))
    return grid


def _cached(path: Path, fn) -> np.ndarray:
    if path.exists():
        return np.load(path)
    x = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, x)
    return x


def _slug(variant: str, seed: int) -> str:
    return f"{variant.replace('@', '_step')}_s{seed}"


def run_experiment(cfg: ExperimentConfig, check: bool = False) -> dict[str, Any]:
    import torch

    from sdlora import gpu

    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required (Kaggle T4); the unit tests cover the CPU-side logic")
    device = "cuda"
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(cfg.to_dict(), out / "config.json")
    data_dir = out / "data"
    recs = prepare_dataset(data_dir, cfg.data.n_test, cfg.data.split_seed, cfg.data.resolution, cfg.data.limit)
    train = [r for r in recs if r["split"] == "train"]
    test = [r for r in recs if r["split"] == "test"]
    if cfg.gen.n_prompts:
        test = test[: cfg.gen.n_prompts]
    log.info("%d train / %d eval species", len(train), len(test))

    # 1. prompt filter: red-team set + every dataset caption (false positives on real species names)
    blocked = [r for r in recs if not check_prompt(r["caption"]).allowed]
    safety: dict[str, Any] = {
        "red_team": evaluate_prompt_filter(),
        "captions": {"n": len(recs), "false_positive_rate": len(blocked) / len(recs),
                     "blocked": [r["caption"] for r in blocked]},
        "image_filters": [],
    }
    gen_recs = [r for r in test if check_prompt(r["caption"]).allowed]  # blocked prompts are never generated
    prompts = [r["caption"] for r in gen_recs]
    labels = np.array([test.index(r) for r in gen_recs])

    # 2. real-image features (cached)
    feat = gpu.Featurizer(cfg.model.clip, device)
    fdir = out / "features"
    load = lambda rs: [Image.open(image_path(data_dir, r)).convert("RGB") for r in rs]  # noqa: E731
    test_imgs = load(test)
    test_clip = _cached(fdir / "test_clip.npy", lambda: feat.image_clip(test_imgs))
    test_inc = _cached(fdir / "test_inception.npy", lambda: feat.inception(test_imgs))
    train_clip = _cached(fdir / "train_clip.npy", lambda: feat.image_clip(load(train)))
    rng = np.random.default_rng(cfg.data.split_seed)
    floor_recs = [train[i] for i in sorted(rng.choice(len(train), min(len(test), len(train)), replace=False))]
    floor_inc = _cached(fdir / "train_subset_inception.npy", lambda: feat.inception(load(floor_recs)))
    cap_emb = feat.text_clip(prompts)
    class_emb = feat.text_clip([f"a photo of a {r['name']}" for r in test])
    unsafe_emb, safe_emb = feat.text_clip(UNSAFE_CONCEPTS), feat.text_clip(SAFE_CONCEPTS)

    def image_metrics(img_clip: np.ndarray, inc: np.ndarray) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        cs = clip_scores(img_clip, cap_emb)
        hits = retrieval_hits(img_clip, class_emb, labels)
        mem = max_similarity(img_clip, train_clip)
        flags, _ = concept_flags(img_clip, unsafe_emb, safe_emb, cfg.safety.clip_threshold)
        k_mean, k_std = kid(inc, test_inc, subset_size=min(100, len(inc), len(test_inc)))
        summary = {"fid": frechet_distance(inc, test_inc), "kid_x1000": 1000 * k_mean, "kid_std_x1000": 1000 * k_std,
                   "clip_score": float(cs.mean()), "top1": float(hits[1].mean()), "top5": float(hits[5].mean()),
                   "mem_max_sim": float(mem.mean()), "mem_frac_ge_095": float((mem >= 0.95).mean()),
                   "clip_flag_rate": float(flags.mean())}
        return summary, {"clip_score": cs, "top1": hits[1], "top5": hits[5], "mem_sim": mem}

    real_idx = labels  # real test images for the prompts that were generated
    ref, _ = image_metrics(test_clip[real_idx], test_inc[real_idx])
    k_mean, _ = kid(floor_inc, test_inc, subset_size=min(100, len(floor_inc), len(test_inc)))
    ref.update(fid=frechet_distance(floor_inc, test_inc), kid_x1000=1000 * k_mean)
    write_json(ref, out / "reference.json")
    safety["image_filters"].append({"images": "real held-out photos", "n": len(real_idx), "sd_nsfw_rate": None,
                                    "clip_flag_rate": ref["clip_flag_rate"]})

    # 3. generation runs
    runs_path, prompts_path, train_log = out / "runs.jsonl", out / "per_prompt.jsonl", out / "train_log.jsonl"
    done = {(r["variant"], r["seed"]) for r in read_jsonl(runs_path)}
    pipe = gpu.load_pipeline(cfg.model, device, cfg.safety.sd_checker)
    cache: dict[str, Any] = {}

    def run_variant(variant: str, seed: int) -> None:
        set_seed(seed)
        noise = [seed * NOISE_SEED_STRIDE + r["idx"] for r in gen_recs]
        t0 = time.time()
        images, nsfw = gpu.generate(pipe, prompts, noise, cfg.gen)
        gen_s = (time.time() - t0) / len(images)
        gdir = out / "gen" / _slug(variant, seed)
        for im, r, s in zip(images, gen_recs, noise, strict=True):
            save_with_provenance(im, gdir / f"{r['idx']:04d}.png", generator=cfg.model.base, variant=variant,
                                 prompt=r["caption"], seed=s)
        (out / "grids").mkdir(exist_ok=True)
        make_grid(images[:8]).save(out / "grids" / f"{_slug(variant, seed)}.jpg", quality=90)
        summary, per = image_metrics(feat.image_clip(images), feat.inception(images))
        row = {"variant": variant, "seed": seed, "n_images": len(images), **summary,
               "sd_nsfw_rate": float(np.mean(nsfw)), "gen_s_per_img": gen_s}
        for i, r in enumerate(gen_recs):
            append_jsonl({"variant": variant, "seed": seed, "idx": r["idx"],
                          **{k: float(v[i]) for k, v in per.items()}}, prompts_path)
        append_jsonl(row, runs_path)
        log.info("%s seed %d: FID %.2f KID %.3f CLIP %.2f top5 %.3f (%.2f s/img)", variant, seed, row["fid"],
                 row["kid_x1000"], row["clip_score"], row["top5"], gen_s)

    lora_variants = cfg.variants()[1:]
    for seed in cfg.seeds:
        if ("base", seed) not in done:
            run_variant("base", seed)
        todo = [(v, k) for v, k in zip(lora_variants, cfg.train.checkpoints, strict=True) if (v, seed) not in done]
        if not todo:
            continue
        ldir = out / "lora" / f"s{seed}"
        if not all((ldir / f"step{k}").exists() for k in cfg.train.checkpoints):
            if "latents" not in cache:
                cache["latents"], cache["embeds"] = gpu.encode_dataset(pipe, load(train),
                                                                       [r["caption"] for r in train])
            info = gpu.train_lora(cfg.model, cfg.train, cache["latents"], cache["embeds"], seed, ldir, device,
                                  on_log=lambda d: append_jsonl(d, train_log))
            write_json({"seed": seed, "steps": cfg.train.steps, **info}, ldir / "train.json")
        for variant, k in todo:
            pipe.load_lora_weights(str(ldir / f"step{k}"))
            try:
                run_variant(variant, seed)
            finally:
                pipe.unload_lora_weights()

    # 4. reports
    rows = read_jsonl(runs_path)
    for v in cfg.variants():
        vr = [r for r in rows if r["variant"] == v]
        if vr:
            safety["image_filters"].append({"images": f"generated: {v}", "n": sum(r["n_images"] for r in vr),
                                            "sd_nsfw_rate": float(np.mean([r["sd_nsfw_rate"] for r in vr])),
                                            "clip_flag_rate": float(np.mean([r["clip_flag_rate"] for r in vr]))})
    train_rows = [json.loads((out / "lora" / f"s{s}" / "train.json").read_text())
                  for s in cfg.seeds if (out / "lora" / f"s{s}" / "train.json").exists()]
    res = write_reports(out, rows, read_jsonl(prompts_path), ref, safety, train_rows, cfg.variants(),
                        cfg.n_bootstrap)
    if check:
        _check(out, cfg, rows)
    return res


def _check(out: Path, cfg: ExperimentConfig, rows: list[dict[str, Any]]) -> None:
    """Smoke assertions: every run present, metrics finite, and the LoRA really changes the images."""
    want = {(v, s) for v in cfg.variants() for s in cfg.seeds}
    got = {(r["variant"], r["seed"]) for r in rows}
    assert want <= got, f"missing runs: {sorted(want - got)}"
    for r in rows:
        bad = [k for k, v in r.items() if isinstance(v, float) and not np.isfinite(v)]
        assert not bad, f"non-finite metrics {bad} in {r['variant']} seed {r['seed']}"
    s = cfg.seeds[0]
    a, b = sorted((out / "gen" / _slug("base", s)).glob("*.png"))[:4], \
        sorted((out / "gen" / _slug(cfg.variants()[-1], s)).glob("*.png"))[:4]
    diff = np.mean([np.abs(np.asarray(Image.open(x), np.float32) - np.asarray(Image.open(y), np.float32)).mean()
                    for x, y in zip(a, b, strict=True)])
    assert diff > 1.0, f"LoRA images are identical to base (mean abs diff {diff:.3f}); LoRA not applied?"
    log.info("check passed: %d runs, LoRA vs base mean abs pixel diff %.2f", len(rows), diff)
