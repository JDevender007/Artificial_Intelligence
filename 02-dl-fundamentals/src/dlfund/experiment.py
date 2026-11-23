"""Run every (seed, model) in a config, then aggregate mean ± std and draw figures.

Resumable at two levels: finished (seed, model) rows in ``results_per_seed.csv`` are
skipped, and an interrupted run continues from its ``checkpoints/<run>/last.pt``.
"""

from __future__ import annotations

import json
from logging import Logger
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from dlfund import plots
from dlfund.config import ExperimentConfig
from dlfund.data import get_datasets, make_loaders
from dlfund.evaluate import (
    aggregate,
    classification_metrics,
    per_class_accuracy,
    to_markdown,
    top_confusions,
)
from dlfund.models import build_model
from dlfund.train import fit, predict
from dlfund.utils import count_params, get_device, set_seed

BASELINE = "linear"


def _append_row(path: Path, row: dict[str, Any]) -> None:
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def run_one(cfg: ExperimentConfig, model_name: str, seed: int, out: Path, device: torch.device,
            logger: Logger) -> dict[str, Any]:
    """Train and test one model with one seed; save history, test logits and best weights."""
    set_seed(seed, cfg.deterministic)
    t = cfg.train_for(model_name)
    train_ds, val_ds, test_ds, info = get_datasets(cfg.data, model_name)
    tr, va, te = make_loaders(train_ds, val_ds, test_ds, t.batch_size, cfg.data.num_workers, seed,
                              pin_memory=device.type == "cuda")
    model = build_model(model_name, info.in_channels, info.img_size, info.n_classes)
    run = f"seed{seed}_{model_name}"
    writer = None
    if cfg.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(out / "tb" / run))
    logger.info("seed=%d model=%s params=%.2fM train=%d val=%d test=%d epochs<=%d",
                seed, model_name, count_params(model) / 1e6, len(train_ds), len(val_ds), len(test_ds),
                t.epochs)
    res = fit(model, tr, va, t, device, out / "checkpoints" / run, writer, logger)
    if writer is not None:
        writer.close()

    _, logits, y = predict(model, te, device, t.amp)
    m = classification_metrics(logits, y)
    np.savez_compressed(out / "preds" / f"{run}.npz", logits=logits, y=y)
    pd.DataFrame(res.history).to_csv(out / "history" / f"{run}.csv", index=False)
    best_val = max(h["val_acc"] for h in res.history if h["epoch"] == res.best_epoch)
    row = {
        "seed": seed, "model": model_name,
        "test_acc": m["acc"], "test_macro_f1": m["macro_f1"], "test_loss": m["loss"],
        "val_acc": best_val, "best_epoch": res.best_epoch, "epochs_run": res.epochs_run,
        "stopped_early": res.stopped_early, "train_minutes": res.train_seconds / 60,
        "images_per_s": res.images_per_s, "params_m": count_params(model) / 1e6,
        "amp": t.amp, "augment": cfg.data.augment, "scheduler": t.scheduler,
    }
    _append_row(out / "results_per_seed.csv", row)
    (out / "checkpoints" / run / "last.pt").unlink(missing_ok=True)  # best.pt is kept
    logger.info("seed=%d %-9s test_acc=%.4f macro_f1=%.4f best_epoch=%d/%d (%.1f min)",
                seed, model_name, m["acc"], m["macro_f1"], res.best_epoch, res.epochs_run,
                res.train_seconds / 60)
    return row


def run_experiment(cfg: ExperimentConfig, logger: Logger) -> pd.DataFrame:
    """Train all seeds × models, write ``results.md`` and figures to ``cfg.output_dir``."""
    out = Path(cfg.output_dir)
    for sub in ("preds", "history", "figures", "checkpoints"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    with open(out / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
    device = get_device()
    env = {"torch": torch.__version__, "device": str(device),
           "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}
    (out / "environment.json").write_text(json.dumps(env, indent=2))
    logger.info("Environment: %s", env)

    per_seed_path = out / "results_per_seed.csv"
    done: set[tuple[int, str]] = set()
    if per_seed_path.exists():
        prev = pd.read_csv(per_seed_path)
        done = set(zip(prev["seed"].astype(int), prev["model"], strict=True))
        logger.info("Resuming: %d (seed, model) runs already finished", len(done))

    for seed in cfg.seeds:
        for model_name in cfg.models:
            if (seed, model_name) not in done:
                run_one(cfg, model_name, seed, out, device, logger)

    per_seed = pd.read_csv(per_seed_path)
    per_seed = per_seed[per_seed["seed"].isin(cfg.seeds) & per_seed["model"].isin(cfg.models)]
    agg = aggregate(per_seed)
    agg.to_csv(out / "results.csv")
    baseline = BASELINE if BASELINE in agg.index else None
    header = (f"# Results: {cfg.name} ({cfg.data.dataset})\n\n"
              f"Official test set, identical for every seed and model; val = {cfg.data.val_size} images "
              f"held out from train with split_seed={cfg.data.split_seed}. Mean ± std over seeds "
              f"{cfg.seeds}. Weights are from the best-val-loss epoch.\n\n")
    (out / "results.md").write_text(header + to_markdown(agg, baseline) + "\n", encoding="utf-8")
    logger.info("\n%s", to_markdown(agg, baseline))
    make_figures(cfg, out, logger)
    return agg


def make_figures(cfg: ExperimentConfig, out: Path, logger: Logger) -> None:
    """Learning curves (first seed), confusion matrix + per-class report for every model."""
    from dlfund.data import CLASSES

    classes = CLASSES[cfg.data.dataset]
    seed = cfg.seeds[0]
    histories = {m: pd.read_csv(out / "history" / f"seed{seed}_{m}.csv") for m in cfg.models
                 if (out / "history" / f"seed{seed}_{m}.csv").exists()}
    plots.learning_curves(histories, out / "figures" / "learning_curves.png")
    report = [f"# Per-class analysis (seed {seed})\n"]
    for m in histories:
        z = np.load(out / "preds" / f"seed{seed}_{m}.npz")
        pred = z["logits"].argmax(1)
        plots.confusion(z["y"], pred, classes, out / "figures" / f"confusion_{m}.png",
                        f"{m} (seed {seed}), row-normalised")
        pc = per_class_accuracy(z["y"], pred, classes)
        report += [f"\n## {m}\n", "Per-class accuracy:\n", pc.round(4).to_frame().to_markdown(), "\n",
                   "Most frequent confusions:\n", top_confusions(z["y"], pred, classes, 8).round(4)
                   .to_markdown(index=False), "\n"]
    (out / "per_class.md").write_text("\n".join(report), encoding="utf-8")
    logger.info("Figures in %s", out / "figures")
