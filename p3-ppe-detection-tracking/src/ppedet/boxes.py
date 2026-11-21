"""Box geometry and detection matching (pure numpy; boxes are ``xyxy`` in pixels)."""

from __future__ import annotations

import numpy as np

# COCO-style size buckets on box area in ORIGINAL-image pixels.
SIZE_BUCKETS = (("small", 0.0, 32.0**2), ("medium", 32.0**2, 96.0**2), ("large", 96.0**2, np.inf))


def yolo_to_xyxy(xywhn: np.ndarray, w: int, h: int) -> np.ndarray:
    """Normalised YOLO ``cx, cy, w, h`` -> pixel ``x1, y1, x2, y2``."""
    xywhn = np.asarray(xywhn, dtype=float).reshape(-1, 4)
    cx, cy, bw, bh = xywhn[:, 0] * w, xywhn[:, 1] * h, xywhn[:, 2] * w, xywhn[:, 3] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)


def area(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def _intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    return wh[..., 0] * wh[..., 1]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU, shape ``[len(a), len(b)]``."""
    inter = _intersection(a, b)
    union = area(a)[:, None] + area(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def ioa_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over the area of ``a`` (how much of each ``a`` box lies inside each ``b``)."""
    inter = _intersection(a, b)
    return np.where(area(a)[:, None] > 0, inter / np.maximum(area(a)[:, None], 1e-12), 0.0)


def size_bucket(areas: np.ndarray) -> np.ndarray:
    """Map box areas to ``small`` / ``medium`` / ``large``."""
    areas = np.asarray(areas, dtype=float)
    out = np.empty(areas.shape, dtype=object)
    for name, lo, hi in SIZE_BUCKETS:
        out[(areas >= lo) & (areas < hi)] = name
    return out


def match_detections(gt_boxes: np.ndarray, gt_cls: np.ndarray, pred_boxes: np.ndarray,
                     pred_cls: np.ndarray, pred_conf: np.ndarray, iou_thr: float = 0.5
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy, class-aware matching in descending confidence (the VOC/COCO rule).

    Returns ``(gt_matched [G] bool, pred_tp [P] bool, pred_error [P] str)`` where
    ``pred_error`` labels every false positive (TIDE-style, simplified):

    - ``"tp"``          matched a same-class GT at IoU >= ``iou_thr``
    - ``"duplicate"``   same-class GT at IoU >= thr, but it was already taken
    - ``"class"``       a GT of ANOTHER class at IoU >= thr (classification error)
    - ``"localization"`` same-class GT at 0.1 <= IoU < thr (box off)
    - ``"background"``  nothing nearby (hallucination)
    """
    gt_boxes = np.asarray(gt_boxes, float).reshape(-1, 4)
    pred_boxes = np.asarray(pred_boxes, float).reshape(-1, 4)
    gt_cls, pred_cls = np.asarray(gt_cls).reshape(-1), np.asarray(pred_cls).reshape(-1)
    G, P = len(gt_boxes), len(pred_boxes)
    gt_matched = np.zeros(G, dtype=bool)
    pred_tp = np.zeros(P, dtype=bool)
    pred_error = np.full(P, "background", dtype=object)
    if P == 0:
        return gt_matched, pred_tp, pred_error
    ious = iou_matrix(pred_boxes, gt_boxes) if G else np.zeros((P, 0))
    for p in np.argsort(-np.asarray(pred_conf, float).reshape(-1), kind="stable"):
        if G == 0:
            continue
        same = gt_cls == pred_cls[p]
        cand = np.where(same & ~gt_matched, ious[p], -1.0)
        best = int(np.argmax(cand))
        if cand[best] >= iou_thr:
            gt_matched[best] = pred_tp[p] = True
            pred_error[p] = "tp"
        elif np.any(same & (ious[p] >= iou_thr)):
            pred_error[p] = "duplicate"
        elif np.any(~same & (ious[p] >= iou_thr)):
            pred_error[p] = "class"
        elif np.any(same & (ious[p] >= 0.1)):
            pred_error[p] = "localization"
    return gt_matched, pred_tp, pred_error
