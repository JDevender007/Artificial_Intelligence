"""Per-person PPE compliance from detections (rule-based, deliberately transparent).

For every ``Person`` box, each tracked item (helmet, vest) gets one status:

- ``"violation"``: a ``no_<item>`` box lies mostly inside the person (IoA >= ``ioa_thr``)
- ``"ok"``: an ``<item>`` box lies mostly inside the person and in the expected
  vertical band (helmet in the top 35% of the person box, vest in 15-75%)
- ``"unknown"``: neither, e.g. occluded, cropped or missed by the detector

A violation outranks an ok (safety-first). A person is ``compliant`` only if every
item is ``ok``; ``non_compliant`` if any item is a violation; else ``unknown``. The
rule never guesses: "unknown" is a valid output and is reported, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ppedet.boxes import ioa_matrix

# Expected vertical band of the item's centre, as a fraction of the person box height.
ITEM_BANDS = {"helmet": (0.0, 0.35), "vest": (0.15, 0.75)}


@dataclass
class PersonStatus:
    person_index: int
    items: dict[str, str]  # item -> ok | violation | unknown
    overall: str  # compliant | non_compliant | unknown


def _centre_in_band(item: np.ndarray, person: np.ndarray, band: tuple[float, float]) -> bool:
    cy = (item[1] + item[3]) / 2
    h = max(person[3] - person[1], 1e-6)
    rel = (cy - person[1]) / h
    return band[0] <= rel <= band[1]


def assess(boxes: np.ndarray, cls: np.ndarray, names: dict[int, str], items: tuple[str, ...] = ("helmet", "vest"),
           ioa_thr: float = 0.5) -> list[PersonStatus]:
    """Assess every person in one image. ``names`` maps class id -> class name."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    labels = np.array([names[int(c)] for c in np.asarray(cls).reshape(-1)], dtype=object)
    persons = np.where(labels == "Person")[0]
    out: list[PersonStatus] = []
    for k, pi in enumerate(persons):
        p = boxes[pi]
        status: dict[str, str] = {}
        for item in items:
            st = "unknown"
            neg = np.where(labels == f"no_{item}")[0]
            if len(neg) and (ioa_matrix(boxes[neg], p[None])[:, 0] >= ioa_thr).any():
                st = "violation"
            else:
                pos = np.where(labels == item)[0]
                inside = ioa_matrix(boxes[pos], p[None])[:, 0] >= ioa_thr if len(pos) else np.array([], bool)
                if any(_centre_in_band(boxes[j], p, ITEM_BANDS[item]) for j in pos[inside]):
                    st = "ok"
            status[item] = st
        vals = set(status.values())
        overall = "non_compliant" if "violation" in vals else ("compliant" if vals == {"ok"} else "unknown")
        out.append(PersonStatus(k, status, overall))
    return out
