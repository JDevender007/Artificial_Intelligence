from __future__ import annotations

import pytest

from ppedet.evaluate import aggregate, compliance_agreement, error_analysis, error_markdown, mean_std_markdown

PERSON, HELMET, VEST = [100, 100, 200, 400], [130, 100, 170, 140], [110, 170, 190, 300]


def _rec(gt, pred):
    return {"gt_boxes": [b for b, _ in gt], "gt_cls": [c for _, c in gt],
            "pred_boxes": [b for b, _ in pred], "pred_cls": [c for _, c in pred],
            "pred_conf": [0.9] * len(pred)}


def test_error_analysis_counts(names) -> None:
    recs = [_rec([(PERSON, 5), (HELMET, 0)], [(PERSON, 5), (HELMET, 6), ([500, 500, 520, 520], 2)])]
    ea = error_analysis(recs, names)
    assert ea["per_class"]["Person"]["recall"] == 1.0
    assert ea["per_class"]["helmet"]["recall"] == 0.0
    assert ea["per_class"]["no_helmet"]["class"] == 1
    assert ea["per_class"]["vest"]["background"] == 1
    assert ea["confusions"][0] == {"true": "helmet", "predicted": "no_helmet", "count": 1}
    assert "| helmet | no_helmet | 1 |" in error_markdown(ea)


def test_compliance_agreement(names) -> None:
    gt = [(PERSON, 5), (HELMET, 0), (VEST, 2)]
    recs = [_rec(gt, gt),                          # agree: compliant
            _rec(gt, [(PERSON, 5), (HELMET, 0)]),  # vest missed -> unknown
            _rec(gt, [(HELMET, 0)])]               # person missed
    c = compliance_agreement(recs, names)
    assert c["persons"] == 3 and c["agreement"] == pytest.approx(1 / 3)
    pairs = {(t["gt"], t["pred"]): t["count"] for t in c["table"]}
    assert pairs == {("compliant", "compliant"): 1, ("compliant", "unknown"): 1, ("compliant", "missed"): 1}


def test_aggregate_mean_std() -> None:
    rows = [{"variant": "a", "seed": s, "m": v} for s, v in enumerate([0.1, 0.2, 0.3])]
    rows.append({"variant": "b", "seed": 0, "m": 0.5})
    agg = aggregate(rows, ["m"])
    a = agg[agg.variant == "a"].iloc[0]
    assert a.m_mean == pytest.approx(0.2) and a.m_std == pytest.approx(0.1) and a.n_seeds == 3
    md = mean_std_markdown(agg, ["m"])
    assert "| a | 0.200 ± 0.100 | 3 |" in md and "| b | 0.500 | 1 |" in md
