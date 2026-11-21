from __future__ import annotations

import numpy as np

from ppedet.sequences import camera_windows, make_sequence, render, transform_boxes


def test_windows_inside_image_and_smooth() -> None:
    wins = camera_windows(640, 480, 10, np.random.default_rng(0))
    assert len(wins) == 10
    for x1, y1, x2, y2 in wins:
        assert 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480
    assert np.abs(np.diff(np.array(wins), axis=0)).max() < 100  # no jumps between frames


def test_identity_window_keeps_boxes() -> None:
    b = np.array([[10, 10, 50, 50]], float)
    out, keep = transform_boxes(b, (0, 0, 100, 100), 100, 100)
    np.testing.assert_allclose(out, b)
    assert keep.all()


def test_crop_and_scale() -> None:
    out, _ = transform_boxes(np.array([[50, 50, 70, 70]]), (50, 50, 150, 150), 200, 200)
    np.testing.assert_allclose(out, [[0, 0, 40, 40]])


def test_mostly_hidden_box_dropped() -> None:
    _, keep = transform_boxes(np.array([[0, 0, 100, 10], [0, 0, 10, 10]]), (80, 0, 200, 100), 120, 100)
    assert list(keep) == [False, False]
    _, keep = transform_boxes(np.array([[70, 0, 100, 10]]), (80, 0, 200, 100), 120, 100)
    assert list(keep) == [True]


def test_sequence_ids_are_stable() -> None:
    boxes = np.array([[100, 100, 150, 150], [300, 200, 360, 280]], float)
    frames = make_sequence(boxes, np.array([5, 0]), 640, 480, 6, np.random.default_rng(1))
    assert len(frames) == 6
    for f in frames:
        assert set(f.gt_ids.tolist()) <= {0, 1}
        assert len(f.gt_ids) == len(f.gt_boxes) == len(f.gt_cls)


def test_render_shape() -> None:
    img = np.zeros((480, 640, 3), np.uint8)
    assert render(img, (10, 10, 330, 250), (640, 480)).shape == (480, 640, 3)
