"""Logging, seeding, image/label I/O helpers (no torch or ultralytics imports)."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_logger(name: str = "ppedet", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(level)
    return logger


def set_seed(seed: int) -> None:
    """Seed Python and NumPy (Ultralytics seeds torch itself from ``seed=``)."""
    random.seed(seed)
    np.random.seed(seed)


def list_images(folder: str | Path) -> list[Path]:
    """Sorted image files under ``folder`` (recursive)."""
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXTS)


def label_path(image: str | Path) -> Path:
    """YOLO convention: ``.../images/x.jpg`` -> ``.../labels/x.txt`` (last ``images`` segment)."""
    parts = list(Path(image).parts)
    idx = max(i for i, p in enumerate(parts) if p == "images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_yolo_labels(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(cls [N] int, xywhn [N, 4] float)``; an absent or empty file means no objects."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return np.zeros(0, dtype=int), np.zeros((0, 4))
    rows = [ln.split() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    arr = np.array([[float(v) for v in r[:5]] for r in rows], dtype=float).reshape(-1, 5)
    return arr[:, 0].astype(int), arr[:, 1:5]


def image_size(path: str | Path) -> tuple[int, int]:
    """``(width, height)`` without decoding pixels."""
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def write_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
