from __future__ import annotations

import numpy as np

from ppedet.boxes import ioa_matrix, iou_matrix, match_detections, size_bucket, yolo_to_xyxy


def test_yolo_to_xyxy() -> None:
    np.testing.assert_allclose(yolo_to_xyxy([[0.5, 0.5, 0.2, 0.4]], 100, 50), [[40, 15, 60, 35]])


def test_iou_known_values() -> None:
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 50 / 150, 0.0]])


def test_ioa_is_asymmetric() -> None:
    small, big = np.array([[2, 2, 4, 4]]), np.array([[0, 0, 10, 10]])
    assert ioa_matrix(small, big)[0, 0] == 1.0
    assert ioa_matrix(big, small)[0, 0] == 0.04


def test_size_buckets() -> None:
    assert list(size_bucket([10.0, 32.0**2, 96.0**2 + 1])) == ["small", "medium", "large"]


def test_match_types() -> None:
    gt = np.array([[0, 0, 10, 10], [50, 50, 60, 60]])
    gc = np.array([0, 1])
    pred = np.array([[0, 0, 10, 10],         # tp (class 0)
                     [0, 0, 10, 10],         # duplicate (class 0, GT already taken)
                     [50, 50, 60, 60],       # class error (says 0, GT is 1)
                     [0, 0, 10, 30],         # localization (class 0, IoU 1/3)
                     [200, 200, 210, 210]])  # background
    pc = np.array([0, 0, 0, 0, 0])
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    matched, tp, err = match_detections(gt, gc, pred, pc, conf)
    assert list(matched) == [True, False]
    assert list(tp) == [True, False, False, False, False]
    assert list(err) == ["tp", "duplicate", "class", "localization", "background"]


def test_match_highest_confidence_wins() -> None:
    gt, gc = np.array([[0, 0, 10, 10]]), np.array([0])
    pred = np.array([[0, 0, 10, 10], [0, 0, 10, 11]])
    _, tp, _ = match_detections(gt, gc, pred, np.array([0, 0]), np.array([0.3, 0.9]))
    assert list(tp) == [False, True]


def test_match_empty_inputs() -> None:
    m, _, err = match_detections(np.zeros((0, 4)), [], [[0, 0, 1, 1]], [0], [0.5])
    assert len(m) == 0 and list(err) == ["background"]
    m, tp, _ = match_detections([[0, 0, 1, 1]], [0], np.zeros((0, 4)), [], [])
    assert list(m) == [False] and len(tp) == 0
