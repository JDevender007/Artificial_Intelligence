"""Logging, seeding, JSON and mean ± std helpers (no torch imports)."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def get_logger(name: str = "ragqa", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(level)
    return logger


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) torch, including CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(row: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=float) + "\n")


def aggregate(rows: list[dict[str, Any]], metrics: list[str], by: str = "model") -> pd.DataFrame:
    """Mean and sample std (ddof=1) of each metric per group, plus ``n_seeds``. Group order is kept."""
    df = pd.DataFrame(rows)
    out = []
    for key in dict.fromkeys(df[by]):
        g = df[df[by] == key]
        rec: dict[str, Any] = {by: key, "n_seeds": len(g)}
        for m in metrics:
            vals = g[m].astype(float)
            rec[f"{m}_mean"] = vals.mean()
            rec[f"{m}_std"] = vals.std(ddof=1) if len(g) > 1 else float("nan")
        out.append(rec)
    return pd.DataFrame(out)


def mean_std_markdown(agg: pd.DataFrame, metrics: list[str], by: str = "model", digits: int = 3) -> str:
    head = f"| {by} | " + " | ".join(metrics) + " | seeds |"
    lines = [head, "|" + "---|" * (len(metrics) + 2)]
    for _, r in agg.iterrows():
        cells = []
        for m in metrics:
            mu, sd = r[f"{m}_mean"], r[f"{m}_std"]
            cells.append(f"{mu:.{digits}f}" if pd.isna(sd) else f"{mu:.{digits}f} ± {sd:.{digits}f}")
        lines.append(f"| {r[by]} | " + " | ".join(cells) + f" | {int(r['n_seeds'])} |")
    return "\n".join(lines)
