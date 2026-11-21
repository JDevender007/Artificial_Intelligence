"""Dataset location, the fixed train/val/test split and computed dataset statistics."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ppedet.boxes import area, size_bucket, yolo_to_xyxy
from ppedet.utils import get_logger, image_size, label_path, list_images, read_yolo_labels

log = get_logger(__name__)


def locate_dataset(data_yaml: str) -> dict[str, Any]:
    """Resolve (and download if needed) a dataset via Ultralytics; returns its info dict."""
    from ultralytics.data.utils import check_det_dataset  # lazy: heavy import

    return check_det_dataset(data_yaml)


def split_train_val(images: list[Path], val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    """Deterministic random split of the official train images into train/val."""
    if not images:
        raise ValueError("No training images found")
    order = np.random.default_rng(seed).permutation(len(images))
    n_val = max(1, int(round(val_fraction * len(images))))
    val = sorted(images[i] for i in order[:n_val])
    train = sorted(images[i] for i in order[n_val:])
    return train, val


def write_split(train_dir: str | Path, test_dir: str | Path, names: dict[int, str], out_dir: str | Path,
                val_fraction: float = 0.15, seed: int = 42) -> Path:
    """Write ``train.txt / val.txt / test.txt`` image lists and ``ppe_split.yaml``.

    The OFFICIAL val split becomes our TEST set; our val is carved from the official
    train split, so model selection (early stopping, best.pt) never sees test images.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train, val = split_train_val(list_images(train_dir), val_fraction, seed)
    test = list_images(test_dir)
    if set(map(str, train + val)) & set(map(str, test)):
        raise RuntimeError("train/val and test overlap")
    for name, files in (("train", train), ("val", val), ("test", test)):
        (out / f"{name}.txt").write_text("\n".join(str(p.resolve()) for p in files) + "\n", encoding="utf-8")
    cfg = {"path": str(out.resolve()), "train": "train.txt", "val": "val.txt", "test": "test.txt",
           "names": {int(k): v for k, v in names.items()}}
    y = out / "ppe_split.yaml"
    y.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    log.info("split: %d train / %d val / %d test -> %s", len(train), len(val), len(test), y)
    return y


def read_split(split_dir: str | Path, name: str) -> list[Path]:
    txt = Path(split_dir) / f"{name}.txt"
    return [Path(ln) for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()]


def split_stats(images: list[Path], names: dict[int, str]) -> dict[str, Any]:
    """Computed statistics for one split: per-class instances/images, boxes per image, size buckets."""
    inst, imgs, sizes = Counter(), Counter(), Counter()
    per_image = []
    for img in images:
        cls, xywhn = read_yolo_labels(label_path(img))
        per_image.append(len(cls))
        if len(cls):
            w, h = image_size(img)
            for c, b in zip(cls, size_bucket(area(yolo_to_xyxy(xywhn, w, h))), strict=True):
                inst[names[int(c)]] += 1
                sizes[f"{names[int(c)]}/{b}"] += 1
            for c in set(cls.tolist()):
                imgs[names[int(c)]] += 1
    bpi = np.array(per_image) if per_image else np.zeros(1)
    return {
        "images": len(images),
        "empty_images": int(sum(n == 0 for n in per_image)),
        "instances": int(sum(inst.values())),
        "boxes_per_image": {"mean": float(bpi.mean()), "median": float(np.median(bpi)), "max": int(bpi.max())},
        "per_class": {n: {"instances": inst[n], "images": imgs[n],
                          **{b: sizes[f"{n}/{b}"] for b in ("small", "medium", "large")}}
                      for n in names.values()},
    }


def stats_markdown(stats: dict[str, dict[str, Any]]) -> str:
    """Render ``{split: split_stats}`` as one markdown table per split (from computed numbers only)."""
    lines = []
    for split, s in stats.items():
        b = s["boxes_per_image"]
        lines += [f"### {split}: {s['images']} images, {s['instances']} boxes, {s['empty_images']} empty, "
                  f"boxes/image mean {b['mean']:.1f} (median {b['median']:.0f}, max {b['max']})", "",
                  "| class | instances | images | small | medium | large |", "|---|---|---|---|---|---|"]
        for n, c in s["per_class"].items():
            lines.append(f"| {n} | {c['instances']} | {c['images']} | {c['small']} | {c['medium']} | {c['large']} |")
        lines.append("")
    return "\n".join(lines)
