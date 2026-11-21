"""Synthetic camera-motion sequences built from still test images.

The dataset has no videos with identity labels, so tracker quality is measured on
sequences made by sliding and zooming a crop window smoothly across a test image
(a pan/zoom camera). Ground-truth boxes are transformed exactly and keep a stable
id per object, so CLEAR-MOT can be computed. Objects stay rigid and nobody moves:
this tests camera motion, missed detections and re-association, not occlusion
between walking people. That limitation is stated in the README.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    window: tuple[int, int, int, int]  # crop x1, y1, x2, y2 in the source image
    gt_boxes: np.ndarray  # [N, 4] xyxy in the OUTPUT frame (out_w x out_h)
    gt_cls: np.ndarray  # [N]
    gt_ids: np.ndarray  # [N]


def camera_windows(w: int, h: int, n_frames: int, rng: np.random.Generator,
                   min_scale: float = 0.6, max_scale: float = 0.9) -> list[tuple[int, int, int, int]]:
    """Crop windows (same aspect as the image) that move/zoom linearly between two random poses."""
    def pose() -> tuple[float, float, float]:
        s = rng.uniform(min_scale, max_scale)
        return s, rng.uniform(0, 1 - s), rng.uniform(0, 1 - s)

    (s0, x0, y0), (s1, x1, y1) = pose(), pose()
    out = []
    for t in np.linspace(0.0, 1.0, n_frames):
        s, fx, fy = s0 + (s1 - s0) * t, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        cx1, cy1 = int(round(fx * w)), int(round(fy * h))
        out.append((cx1, cy1, min(w, cx1 + int(round(s * w))), min(h, cy1 + int(round(s * h)))))
    return out


def transform_boxes(boxes: np.ndarray, window: tuple[int, int, int, int], out_w: int, out_h: int,
                    min_visible: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Map source-image boxes into a resized crop; return ``(boxes_out, keep_mask)``.

    Boxes whose visible fraction inside the crop is below ``min_visible`` are dropped
    (a tracker should not be penalised for objects that are mostly out of frame).
    """
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    x1, y1, x2, y2 = window
    clipped = np.stack([np.clip(boxes[:, 0], x1, x2), np.clip(boxes[:, 1], y1, y2),
                        np.clip(boxes[:, 2], x1, x2), np.clip(boxes[:, 3], y1, y2)], axis=1)
    full = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    vis = np.clip(clipped[:, 2] - clipped[:, 0], 0, None) * np.clip(clipped[:, 3] - clipped[:, 1], 0, None)
    keep = (full > 0) & (vis >= min_visible * np.maximum(full, 1e-12))
    sx, sy = out_w / max(x2 - x1, 1), out_h / max(y2 - y1, 1)
    out = (clipped - np.array([x1, y1, x1, y1])) * np.array([sx, sy, sx, sy])
    return out[keep], keep


def make_sequence(gt_boxes: np.ndarray, gt_cls: np.ndarray, w: int, h: int, n_frames: int,
                  rng: np.random.Generator, out_size: tuple[int, int] | None = None) -> list[Frame]:
    """Build one sequence's frames (windows + transformed GT with stable ids)."""
    out_w, out_h = out_size or (w, h)
    ids = np.arange(len(gt_boxes))
    frames = []
    for win in camera_windows(w, h, n_frames, rng):
        b, keep = transform_boxes(gt_boxes, win, out_w, out_h)
        frames.append(Frame(win, b, np.asarray(gt_cls)[keep], ids[keep]))
    return frames


def render(image: np.ndarray, window: tuple[int, int, int, int], out_size: tuple[int, int]) -> np.ndarray:
    """Crop ``image`` (HxWxC) to ``window`` and resize to ``out_size`` (w, h). OpenCV imported lazily."""
    import cv2

    x1, y1, x2, y2 = window
    return cv2.resize(image[y1:y2, x1:x2], out_size, interpolation=cv2.INTER_LINEAR)
