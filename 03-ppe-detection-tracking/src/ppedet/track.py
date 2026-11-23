"""Tracker comparison (ByteTrack vs BoT-SORT) on synthetic camera-motion sequences."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ppedet.boxes import yolo_to_xyxy
from ppedet.mot import MOTAccumulator
from ppedet.sequences import make_sequence, render
from ppedet.utils import get_logger, label_path, read_yolo_labels

log = get_logger(__name__)


def pick_images(images: list[Path], n: int, min_objects: int, seed: int) -> list[Path]:
    """Choose ``n`` test images with at least ``min_objects`` GT boxes (deterministic)."""
    ok = [p for p in images if len(read_yolo_labels(label_path(p))[0]) >= min_objects]
    rng = np.random.default_rng(seed)
    return sorted(ok[i] for i in rng.permutation(len(ok))[:n])


def sum_accumulators(accs: list[MOTAccumulator]) -> MOTAccumulator:
    """Pool counts of several accumulators (e.g. one per class) into one."""
    tot = MOTAccumulator()
    for a in accs:
        for f in ("fp", "fn", "idsw", "matches", "gt"):
            setattr(tot, f, getattr(tot, f) + getattr(a, f))
        tot.iou_sum += a.iou_sum
        tot.frames = max(tot.frames, a.frames)
    return tot


def evaluate_trackers(weights: str | Path, images: list[Path], trackers: list[str], names: dict[int, str],
                      n_sequences: int, n_frames: int, min_objects: int, seed: int, conf: float, imgsz: int
                      ) -> list[dict[str, Any]]:
    """CLEAR-MOT per tracker, for the Person class and for all classes (class-aware matching)."""
    import cv2
    from ultralytics import YOLO

    chosen = pick_images(images, n_sequences, min_objects, seed)
    log.info("tracking on %d sequences x %d frames", len(chosen), n_frames)
    person = next(k for k, v in names.items() if v == "Person")
    rows = []
    for tracker in trackers:
        accs = {c: MOTAccumulator() for c in names}
        for k, img_path in enumerate(chosen):
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            cls, xywhn = read_yolo_labels(label_path(img_path))
            frames = make_sequence(yolo_to_xyxy(xywhn, w, h), cls, w, h, n_frames,
                                   np.random.default_rng(seed + k), out_size=(w, h))
            model = YOLO(str(weights))  # fresh model = fresh tracker state per sequence
            for a in accs.values():
                a.reset_sequence()
            for fr in frames:
                res = model.track(render(img, fr.window, (w, h)), persist=True, tracker=tracker,
                                  conf=conf, imgsz=imgsz, verbose=False)[0]
                b = res.boxes
                if b is None or b.id is None:
                    tb, tc, ti = np.zeros((0, 4)), np.zeros(0, int), np.zeros(0, int)
                else:
                    tb = b.xyxy.cpu().numpy()
                    tc = b.cls.cpu().numpy().astype(int)
                    ti = b.id.cpu().numpy().astype(int)
                for c, a in accs.items():
                    g = fr.gt_cls == c
                    a.update(fr.gt_ids[g], fr.gt_boxes[g], ti[tc == c], tb[tc == c])
        rows.append({"tracker": tracker, "scope": "Person", **accs[person].result().to_dict()})
        rows.append({"tracker": tracker, "scope": "all classes",
                     **sum_accumulators(list(accs.values())).result().to_dict()})
    return rows


def tracking_markdown(rows: list[dict[str, Any]]) -> str:
    L = ["| tracker | scope | MOTA | MOTP (IoU) | IDSW | FP | FN | GT boxes |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['tracker']} | {r['scope']} | {r['mota']:.3f} | {r['motp']:.3f} | {r['idsw']} | "
                 f"{r['fp']} | {r['fn']} | {r['gt']} |")
    return "\n".join(L) + "\n"
