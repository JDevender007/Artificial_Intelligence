import numpy as np
import pandas as pd
import pytest

from dlfund.config import load_config
from dlfund.data import build_transforms, eval_loader_workers, get_datasets, split_indices
from dlfund.evaluate import aggregate, classification_metrics, per_class_accuracy, top_confusions


def test_split_is_deterministic_and_disjoint():
    a_tr, a_va = split_indices(1000, 100, 42)
    b_tr, b_va = split_indices(1000, 100, 42)
    assert a_tr == b_tr and a_va == b_va
    assert not set(a_tr) & set(a_va)
    assert len(a_tr) + len(a_va) == 1000
    assert split_indices(1000, 100, 7)[1] != a_va


def test_split_rejects_bad_val_size():
    with pytest.raises(ValueError):
        split_indices(10, 10, 0)


def test_fake_dataset_shapes_and_limit():
    cfg = load_config(None, ["data.dataset=fake", "data.limit_train=50"]).data
    tr, va, te, info = get_datasets(cfg, "cnn")
    assert len(tr) == 50 and len(va) == 100 and len(te) == 200
    x, y = tr[0]
    assert tuple(x.shape) == (3, 32, 32) and 0 <= y < 10
    assert info.n_classes == 10


def test_resnet_transform_upsamples_to_3_channels():
    tf = build_transforms("mnist", "resnet18", augment=False, train=False, resnet_img_size=64)
    from PIL import Image
    img = Image.fromarray(np.zeros((28, 28), dtype=np.uint8))
    assert tuple(tf(img).shape) == (3, 64, 64)


def test_val_transform_has_no_augmentation():
    tf = build_transforms("cifar10", "cnn", augment=True, train=False)
    names = [type(t).__name__ for t in tf.transforms]
    assert "RandomCrop" not in names and "RandomHorizontalFlip" not in names


def test_config_overrides_and_validation():
    cfg = load_config(None, ["train.epochs=3", "model_overrides={resnet18: {lr: 0.001}}"])
    assert cfg.train.epochs == 3
    assert cfg.train_for("resnet18").lr == 0.001 and cfg.train_for("cnn").lr == cfg.train.lr
    with pytest.raises(KeyError):
        load_config(None, ["train.epoch=3"])
    with pytest.raises(ValueError):
        load_config(None, ["models=[vit]"])


def test_classification_metrics():
    logits = np.array([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
    y = np.array([0, 1, 1, 1])
    m = classification_metrics(logits, y)
    assert m["acc"] == 0.75
    ce = -np.log(np.exp(2) / (np.exp(2) + 1))
    assert m["loss"] == pytest.approx((3 * ce + (2 + ce)) / 4)


def test_per_class_and_confusions():
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 1, 1, 1, 2, 0])
    pc = per_class_accuracy(y, pred, ["a", "b", "c"])
    assert list(pc) == [0.5, 1.0, 0.5]
    tc = top_confusions(y, pred, ["a", "b", "c"])
    assert set(map(tuple, tc[["true", "predicted"]].values)) == {("a", "b"), ("c", "a")}


def test_aggregate_mean_std():
    df = pd.DataFrame({"model": ["m", "m", "m"], "test_acc": [0.8, 0.9, 1.0]})
    agg = aggregate(df)
    assert agg.loc["m", "test_acc_mean"] == pytest.approx(0.9)
    assert agg.loc["m", "test_acc_std"] == pytest.approx(0.1)
    assert agg.loc["m", "n_seeds"] == 3


def test_eval_loaders_skip_workers_on_windows():
    assert eval_loader_workers(4, platform="nt") == 0
    assert eval_loader_workers(4, platform="posix") == 4
