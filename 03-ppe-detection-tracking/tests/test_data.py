from __future__ import annotations

from pathlib import Path

import yaml
from PIL import Image

from ppedet.data import read_split, split_stats, split_train_val, stats_markdown, write_split
from ppedet.utils import label_path, read_yolo_labels


def _fake_dataset(root: Path, n_train: int = 20, n_val: int = 5) -> None:
    for split, n in (("train", n_train), ("val", n_val)):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (640, 480)).save(root / "images" / split / f"{split}{i}.jpg")
            lines = ["5 0.5 0.5 0.4 0.8", "0 0.5 0.15 0.03 0.04"] if i % 4 else []
            (root / "labels" / split / f"{split}{i}.txt").write_text("\n".join(lines))


def test_label_path_and_reader(tmp_path: Path) -> None:
    _fake_dataset(tmp_path)
    img = tmp_path / "images" / "train" / "train1.jpg"
    assert label_path(img) == tmp_path / "labels" / "train" / "train1.txt"
    cls, xywh = read_yolo_labels(label_path(img))
    assert list(cls) == [5, 0] and xywh.shape == (2, 4)
    assert len(read_yolo_labels(tmp_path / "missing.txt")[0]) == 0


def test_split_is_deterministic_and_disjoint() -> None:
    imgs = [Path(f"/x/images/{i}.jpg") for i in range(100)]
    a = split_train_val(imgs, 0.15, 42)
    assert a == split_train_val(imgs, 0.15, 42)
    assert len(a[1]) == 15 and not set(a[0]) & set(a[1])
    assert split_train_val(imgs, 0.15, 7) != a


def test_write_split_and_stats(tmp_path: Path, names) -> None:
    _fake_dataset(tmp_path / "ds")
    y = write_split(tmp_path / "ds/images/train", tmp_path / "ds/images/val", names, tmp_path / "split", 0.2, 0)
    cfg = yaml.safe_load(y.read_text())
    assert cfg["test"] == "test.txt" and cfg["names"][5] == "Person"
    train, val, test = (read_split(tmp_path / "split", s) for s in ("train", "val", "test"))
    assert (len(train), len(val), len(test)) == (16, 4, 5)
    s = split_stats(test, names)
    # test images 0..4: images 1, 2, 3 have a person + a helmet; 0 and 4 are empty
    assert s["images"] == 5 and s["empty_images"] == 2 and s["instances"] == 6
    assert s["per_class"]["Person"]["instances"] == 3 and s["per_class"]["helmet"]["small"] == 3
    assert "| Person | 3 | 3 |" in stats_markdown({"test": s})
