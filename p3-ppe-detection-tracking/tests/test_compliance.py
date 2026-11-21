from __future__ import annotations

import numpy as np

from ppedet.compliance import assess

PERSON = [100, 100, 200, 400]  # 300 px tall


def _run(extra: list[tuple[list[int], int]], names):
    boxes = np.array([PERSON] + [b for b, _ in extra], float)
    cls = np.array([5] + [c for _, c in extra])
    return assess(boxes, cls, names)


def test_compliant_person(names) -> None:
    st = _run([([130, 100, 170, 140], 0), ([110, 170, 190, 300], 2)], names)
    assert st[0].items == {"helmet": "ok", "vest": "ok"} and st[0].overall == "compliant"


def test_violation_outranks_ok(names) -> None:
    st = _run([([130, 100, 170, 140], 0), ([130, 100, 170, 140], 6), ([110, 170, 190, 300], 2)], names)
    assert st[0].items["helmet"] == "violation" and st[0].overall == "non_compliant"


def test_helmet_in_wrong_place_is_unknown(names) -> None:
    # a "helmet" at the person's feet (e.g. carried) is not being worn
    st = _run([([130, 360, 170, 400], 0), ([110, 170, 190, 300], 2)], names)
    assert st[0].items["helmet"] == "unknown" and st[0].overall == "unknown"


def test_item_outside_person_ignored(names) -> None:
    st = _run([([500, 100, 540, 140], 0)], names)
    assert st[0].items["helmet"] == "unknown"


def test_no_person_no_output(names) -> None:
    assert assess(np.array([[0, 0, 10, 10]]), np.array([0]), names) == []
