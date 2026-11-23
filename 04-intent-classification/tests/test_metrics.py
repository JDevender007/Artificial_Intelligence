from __future__ import annotations

import numpy as np
import pytest

from textclf.metrics import (
    by_length,
    classification_metrics,
    ece,
    length_bucket_names,
    paired_bootstrap,
    per_class,
    top_confusions,
)

Y = np.array([0, 0, 1, 1, 2, 2])
P = np.array([0, 1, 1, 1, 2, 0])


def test_classification_metrics_by_hand() -> None:
    m = classification_metrics(Y, P, 3)
    # F1: class0 p=1/2 r=1/2 -> .5 ; class1 p=2/3 r=1 -> .8 ; class2 p=1 r=1/2 -> 2/3
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


def test_absent_class_counts_as_zero_f1() -> None:
    assert classification_metrics(np.array([0, 0]), np.array([0, 0]), 2)["macro_f1"] == pytest.approx(0.5)


def test_ece_perfect_and_overconfident() -> None:
    y = np.array([0, 1])
    assert ece(np.array([[1.0, 0.0], [0.0, 1.0]]), y) == pytest.approx(0.0)
    assert ece(np.array([[0.9, 0.1], [0.9, 0.1]]), y) == pytest.approx(0.4)  # conf .9, acc .5


def test_per_class_and_confusions() -> None:
    pc = per_class(Y, P, ["a", "b", "c"])
    assert pc[1]["recall"] == 1.0 and pc[2]["support"] == 2
    assert top_confusions(Y, P, ["a", "b", "c"]) == [{"true": "a", "predicted": "b", "count": 1},
                                                      {"true": "c", "predicted": "a", "count": 1}]


def test_length_buckets() -> None:
    assert length_bucket_names([5, 10]) == ["1-5", "6-10", ">10"]
    wc = np.array([3, 5, 6, 10, 11, 40])
    rows = by_length(wc, Y, P, [5, 10], 3)
    assert [r["n"] for r in rows] == [2, 2, 2]
    assert rows[0]["accuracy"] == pytest.approx(0.5) and rows[1]["accuracy"] == 1.0


def test_empty_bucket_is_nan() -> None:
    rows = by_length(np.array([1, 2]), np.array([0, 1]), np.array([0, 1]), [5, 10], 2)
    assert rows[1]["n"] == 0 and np.isnan(rows[1]["accuracy"])


def test_paired_bootstrap_detects_clear_gain() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    good, bad = y.copy(), y.copy()
    bad[:120] = (bad[:120] + 1) % 3
    r = paired_bootstrap(y, bad, good, 3, n=200, seed=0)
    assert r["diff"] > 0 and r["ci_low"] > 0 and r["p_b_not_better"] == 0.0
    same = paired_bootstrap(y, good, good, 3, n=50)
    assert same["diff"] == 0 and same["ci_low"] == 0 == same["ci_high"]
