"""Thin wrapper around Ultralytics training / test evaluation / prediction (imported lazily)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ppedet.boxes import yolo_to_xyxy
from ppedet.config import ExperimentConfig, Variant
from ppedet.utils import get_logger, label_path, read_yolo_labels

log = get_logger(__name__)


def run_name(variant: Variant, seed: int) -> str:
    return f"{variant.name}_s{seed}"


def train_one(cfg: ExperimentConfig, variant: Variant, seed: int, data_yaml: str | Path) -> Path:
    """Train one (variant, seed); return the path of ``best.pt`` (selected on OUR val split).

    If ``last.pt`` exists from an interrupted run, training resumes from it; if resuming
    fails (e.g. the run had already finished), training restarts from scratch.
    """
    from ultralytics import YOLO

    project = Path(cfg.output_dir).resolve() / "runs"
    name = run_name(variant, seed)
    wdir = project / name / "weights"
    if (wdir / "last.pt").exists():
        try:
            log.info("resuming %s from last.pt", name)
            YOLO(str(wdir / "last.pt")).train(resume=True)
            return wdir / "best.pt"
        except Exception as e:  # noqa: BLE001 - Ultralytics raises plain AssertionError here
            if (wdir / "best.pt").exists() and "finished" in str(e).lower():
                return wdir / "best.pt"
            log.warning("resume failed for %s (%s); retraining", name, e)
    t = cfg.train
    args: dict[str, Any] = dict(
        data=str(data_yaml), epochs=t.epochs, imgsz=t.imgsz, batch=t.batch, patience=t.patience,
        workers=t.workers, fraction=t.fraction, optimizer=t.optimizer, seed=seed, deterministic=True,
        pretrained=variant.pretrained, project=str(project), name=name, exist_ok=True, plots=True,
        verbose=False,
    )
    args.update(variant.overrides)
    model = YOLO(variant.model)
    model.train(**args)
    return Path(model.trainer.best)


def test_one(cfg: ExperimentConfig, weights: str | Path, data_yaml: str | Path, name: str) -> dict[str, Any]:
    """Evaluate ``weights`` once on the TEST split; return scalar metrics + per-class mAP50-95."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    m = model.val(data=str(data_yaml), split="test", imgsz=cfg.train.imgsz, batch=cfg.train.batch,
                  project=str(Path(cfg.output_dir).resolve() / "test"), name=name, exist_ok=True,
                  plots=True, verbose=False)
    names = m.names
    idx = [int(c) for c in m.box.ap_class_index]
    params = sum(p.numel() for p in model.model.parameters())
    return {
        "map50_95": float(m.box.map), "map50": float(m.box.map50), "map75": float(m.box.map75),
        "precision": float(m.box.mp), "recall": float(m.box.mr),
        "ms_per_img": float(m.speed.get("inference", np.nan)),
        "params_m": params / 1e6,
        "per_class_map50_95": {names[c]: float(m.box.maps[c]) for c in idx},
    }


def predict_records(weights: str | Path, images: list[Path], conf: float, imgsz: int) -> list[dict[str, Any]]:
    """Predict on ``images`` and pair each with its ground truth (pixels, original image size)."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out = []
    t0 = time.perf_counter()
    for r in model.predict(source=[str(p) for p in images], conf=conf, imgsz=imgsz, stream=True, verbose=False):
        h, w = r.orig_shape
        cls, xywhn = read_yolo_labels(label_path(r.path))
        b = r.boxes
        out.append({
            "image": str(r.path), "w": int(w), "h": int(h),
            "gt_boxes": yolo_to_xyxy(xywhn, w, h).tolist(), "gt_cls": cls.tolist(),
            "pred_boxes": b.xyxy.cpu().numpy().tolist(),
            "pred_cls": b.cls.cpu().numpy().astype(int).tolist(),
            "pred_conf": b.conf.cpu().numpy().tolist(),
        })
    log.info("predicted %d images in %.1fs", len(out), time.perf_counter() - t0)
    return out
