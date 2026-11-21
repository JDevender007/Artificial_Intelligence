"""Error analysis on saved predictions and mean ± std aggregation (pure numpy/pandas)."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd

from ppedet.boxes import area, iou_matrix, match_detections, size_bucket
from ppedet.compliance import assess

# A record = one test image:
#   {"image": str, "gt_boxes": [[x1,y1,x2,y2]...], "gt_cls": [...],
#    "pred_boxes": [...], "pred_cls": [...], "pred_conf": [...]}   (pixels, original image)


def _arr(r: dict, k: str, shape4: bool = False) -> np.ndarray:
    a = np.asarray(r.get(k, []), dtype=float)
    return a.reshape(-1, 4) if shape4 else a.reshape(-1)


def error_analysis(records: list[dict[str, Any]], names: dict[int, str], iou_thr: float = 0.5) -> dict[str, Any]:
    """Recall per class and size bucket, FP breakdown per class, and top class confusions."""
    tp_cls, gt_n = Counter(), Counter()
    tp_size, gt_size = Counter(), Counter()
    fp_types: dict[str, Counter] = defaultdict(Counter)
    confusions = Counter()
    for r in records:
        gb, gc = _arr(r, "gt_boxes", True), _arr(r, "gt_cls").astype(int)
        pb, pc, pf = _arr(r, "pred_boxes", True), _arr(r, "pred_cls").astype(int), _arr(r, "pred_conf")
        matched, _, err = match_detections(gb, gc, pb, pc, pf, iou_thr)
        for c, m, s in zip(gc, matched, size_bucket(area(gb)), strict=True):
            gt_n[names[c]] += 1
            gt_size[s] += 1
            tp_cls[names[c]] += int(m)
            tp_size[s] += int(m)
        ious = iou_matrix(pb, gb) if len(gb) and len(pb) else None
        for j, (c, e) in enumerate(zip(pc, err, strict=True)):
            if e == "tp":
                continue
            fp_types[names[c]][e] += 1
            if e == "class" and ious is not None:
                confusions[(names[int(gc[int(np.argmax(ious[j]))])], names[c])] += 1
    per_class = {n: {"gt": gt_n[n], "recall": tp_cls[n] / gt_n[n] if gt_n[n] else float("nan"),
                     **{k: fp_types[n][k] for k in ("background", "localization", "class", "duplicate")}}
                 for n in names.values()}
    by_size = {s: {"gt": gt_size[s], "recall": tp_size[s] / gt_size[s] if gt_size[s] else float("nan")}
               for s in ("small", "medium", "large")}
    top_conf = [{"true": t, "predicted": p, "count": n} for (t, p), n in confusions.most_common(10)]
    return {"per_class": per_class, "by_size": by_size, "confusions": top_conf}


def compliance_agreement(records: list[dict[str, Any]], names: dict[int, str], iou_thr: float = 0.5) -> dict[str, Any]:
    """Compare the compliance rule applied to GT boxes vs to predicted boxes, per GT person.

    Each GT person is matched to the predicted Person box with the highest IoU (>= thr);
    an unmatched GT person counts as ``missed``. Returns a GT-status x predicted-status
    count table and the agreement rate over all GT persons.
    """
    table: Counter = Counter()
    person_ids = [k for k, v in names.items() if v == "Person"]
    if not person_ids:
        raise ValueError("names has no 'Person' class")
    pid = person_ids[0]
    for r in records:
        gb, gc = _arr(r, "gt_boxes", True), _arr(r, "gt_cls").astype(int)
        pb, pc = _arr(r, "pred_boxes", True), _arr(r, "pred_cls").astype(int)
        g_status, p_status = assess(gb, gc, names), assess(pb, pc, names)
        g_people, p_people = gb[gc == pid], pb[pc == pid]
        ious = iou_matrix(g_people, p_people) if len(g_people) and len(p_people) else None
        used: set[int] = set()
        for i, gs in enumerate(g_status):
            pred = "missed"
            if ious is not None:
                cand = np.where([j not in used for j in range(len(p_people))], ious[i], -1.0)
                j = int(np.argmax(cand))
                if cand[j] >= iou_thr:
                    used.add(j)
                    pred = p_status[j].overall
            table[(gs.overall, pred)] += 1
    total = sum(table.values())
    agree = sum(n for (g, p), n in table.items() if g == p)
    return {"persons": total, "agreement": agree / total if total else float("nan"),
            "table": [{"gt": g, "pred": p, "count": n} for (g, p), n in sorted(table.items())]}


def aggregate(rows: list[dict[str, Any]], metrics: list[str], by: str = "variant") -> pd.DataFrame:
    """Mean and std (ddof=1) per group, plus the number of seeds."""
    df = pd.DataFrame(rows)
    g = df.groupby(by, sort=False)[metrics]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std"))
    out["n_seeds"] = df.groupby(by, sort=False).size()
    return out.reset_index()


def mean_std_markdown(agg: pd.DataFrame, metrics: list[str], by: str = "variant", digits: int = 3) -> str:
    """Render an :func:`aggregate` frame as ``mean ± std`` markdown."""
    head = [by, *metrics, "seeds"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in agg.iterrows():
        cells = [str(r[by])]
        for m in metrics:
            mu, sd = r[f"{m}_mean"], r[f"{m}_std"]
            cells.append(f"{mu:.{digits}f} ± {sd:.{digits}f}" if pd.notna(sd) else f"{mu:.{digits}f}")
        cells.append(str(int(r["n_seeds"])))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def error_markdown(ea: dict[str, Any], comp: dict[str, Any] | None = None) -> str:
    """Markdown report for :func:`error_analysis` (+ optional :func:`compliance_agreement`)."""
    L = ["### Recall and false positives per class", "",
         "| class | GT | recall | FP background | FP localization | FP class | FP duplicate |",
         "|---|---|---|---|---|---|---|"]
    for n, c in ea["per_class"].items():
        L.append(f"| {n} | {c['gt']} | {c['recall']:.3f} | {c['background']} | {c['localization']} | "
                 f"{c['class']} | {c['duplicate']} |")
    L += ["", "### Recall by box size (COCO buckets, original pixels)", "", "| size | GT | recall |", "|---|---|---|"]
    L += [f"| {s} | {v['gt']} | {v['recall']:.3f} |" for s, v in ea["by_size"].items()]
    L += ["", "### Most frequent class confusions (FP labelled with the wrong class)", "",
          "| true | predicted | count |", "|---|---|---|"]
    L += [f"| {c['true']} | {c['predicted']} | {c['count']} |" for c in ea["confusions"]] or ["| - | - | 0 |"]
    if comp:
        L += ["", f"### Compliance rule: GT vs predictions ({comp['persons']} GT persons, "
              f"agreement {comp['agreement']:.3f})", "", "| GT status | predicted status | count |", "|---|---|---|"]
        L += [f"| {t['gt']} | {t['pred']} | {t['count']} |" for t in comp["table"]]
    return "\n".join(L) + "\n"
