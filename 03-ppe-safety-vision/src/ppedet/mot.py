"""CLEAR-MOT metrics (MOTA, IDSW, FP, FN, MOTP) implemented from the definition.

Per frame: GT-hypothesis pairs that were matched in the previous frame are kept
while their IoU stays >= ``iou_thr`` (Bernardin & Stiefelhagen, 2008); the rest are
matched with the Hungarian algorithm on ``1 - IoU``. An identity switch is counted
when a GT id is matched to a different track id than the last time it was matched.

    MOTA = 1 - (FN + FP + IDSW) / total_GT
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from ppedet.boxes import iou_matrix


@dataclass
class MOTResult:
    mota: float
    motp: float  # mean IoU of matched pairs (higher is better)
    idsw: int
    fp: int
    fn: int
    matches: int
    gt: int
    frames: int

    def to_dict(self) -> dict:
        return asdict(self)


class MOTAccumulator:
    """Accumulates CLEAR-MOT counts over the frames of one or more sequences."""

    def __init__(self, iou_thr: float = 0.5) -> None:
        self.iou_thr = iou_thr
        self.fp = self.fn = self.idsw = self.matches = self.gt = self.frames = 0
        self.iou_sum = 0.0
        self.reset_sequence()

    def reset_sequence(self) -> None:
        """Forget identity history (call between independent sequences)."""
        self._prev: dict[int, int] = {}  # gt_id -> track_id matched in the previous frame
        self._last: dict[int, int] = {}  # gt_id -> track_id matched most recently

    def update(self, gt_ids: np.ndarray, gt_boxes: np.ndarray, trk_ids: np.ndarray, trk_boxes: np.ndarray) -> None:
        gt_ids = np.asarray(gt_ids, dtype=int).reshape(-1)
        trk_ids = np.asarray(trk_ids, dtype=int).reshape(-1)
        gt_boxes = np.asarray(gt_boxes, float).reshape(-1, 4)
        trk_boxes = np.asarray(trk_boxes, float).reshape(-1, 4)
        self.frames += 1
        self.gt += len(gt_ids)
        ious = iou_matrix(gt_boxes, trk_boxes) if len(gt_ids) and len(trk_ids) else np.zeros((len(gt_ids), len(trk_ids)))
        pairs: list[tuple[int, int]] = []
        free_g, free_t = set(range(len(gt_ids))), set(range(len(trk_ids)))
        # 1) keep last frame's correspondences if still valid
        t_index = {int(t): j for j, t in enumerate(trk_ids)}
        for i, g in enumerate(gt_ids):
            t = self._prev.get(int(g))
            j = t_index.get(t) if t is not None else None
            if j is not None and j in free_t and ious[i, j] >= self.iou_thr:
                pairs.append((i, j))
                free_g.discard(i)
                free_t.discard(j)
        # 2) Hungarian on the remainder
        fg, ft = sorted(free_g), sorted(free_t)
        if fg and ft:
            sub = ious[np.ix_(fg, ft)]
            rows, cols = linear_sum_assignment(1.0 - sub)
            for r, c in zip(rows, cols, strict=True):
                if sub[r, c] >= self.iou_thr:
                    pairs.append((fg[r], ft[c]))
        new_prev: dict[int, int] = {}
        for i, j in pairs:
            g, t = int(gt_ids[i]), int(trk_ids[j])
            if g in self._last and self._last[g] != t:
                self.idsw += 1
            self._last[g] = t
            new_prev[g] = t
            self.iou_sum += float(ious[i, j])
        self._prev = new_prev
        self.matches += len(pairs)
        self.fn += len(gt_ids) - len(pairs)
        self.fp += len(trk_ids) - len(pairs)

    def result(self) -> MOTResult:
        mota = 1.0 - (self.fn + self.fp + self.idsw) / self.gt if self.gt else float("nan")
        motp = self.iou_sum / self.matches if self.matches else float("nan")
        return MOTResult(mota, motp, self.idsw, self.fp, self.fn, self.matches, self.gt, self.frames)
