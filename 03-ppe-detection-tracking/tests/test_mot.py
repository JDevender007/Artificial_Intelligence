from __future__ import annotations

import numpy as np

from ppedet.mot import MOTAccumulator
from ppedet.track import sum_accumulators

A, B = [0, 0, 10, 10], [50, 0, 60, 10]


def test_perfect_tracker() -> None:
    acc = MOTAccumulator()
    for t in range(5):
        boxes = np.array([[A[0] + t, 0, A[2] + t, 10], B], float)
        acc.update([1, 2], boxes, [7, 8], boxes)
    r = acc.result()
    assert r.mota == 1.0 and r.idsw == 0 and r.fp == 0 and r.fn == 0 and r.motp == 1.0


def test_id_swap_counted() -> None:
    acc = MOTAccumulator()
    acc.update([1, 2], [A, B], [7, 8], [A, B])
    acc.update([1, 2], [A, B], [8, 7], [A, B])  # tracker swapped the ids
    r = acc.result()
    assert r.idsw == 2 and r.mota == 1 - 2 / 4


def test_misses_and_false_positives() -> None:
    acc = MOTAccumulator()
    acc.update([1], [A], [], np.zeros((0, 4)))     # FN
    acc.update([], np.zeros((0, 4)), [9], [B])     # FP
    acc.update([1], [A], [9], [[30, 30, 40, 40]])  # FN + FP (no overlap)
    r = acc.result()
    assert (r.fn, r.fp, r.gt) == (2, 2, 2) and r.mota == 1 - 4 / 2


def test_reacquired_with_new_id_is_switch() -> None:
    acc = MOTAccumulator()
    acc.update([1], [A], [7], [A])
    acc.update([1], [A], [], np.zeros((0, 4)))
    acc.update([1], [A], [9], [A])
    assert acc.result().idsw == 1


def test_reset_sequence_forgets_ids() -> None:
    acc = MOTAccumulator()
    acc.update([1], [A], [7], [A])
    acc.reset_sequence()
    acc.update([1], [A], [3], [A])
    assert acc.result().idsw == 0


def test_sum_accumulators_pools_counts() -> None:
    a, b = MOTAccumulator(), MOTAccumulator()
    a.update([1], [A], [7], [A])
    b.update([1], [B], [], np.zeros((0, 4)))
    r = sum_accumulators([a, b]).result()
    assert (r.gt, r.matches, r.fn, r.fp) == (2, 1, 1, 0) and r.mota == 0.5
