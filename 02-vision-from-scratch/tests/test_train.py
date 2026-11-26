import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dlfund.config import TrainConfig
from dlfund.models import build_model
from dlfund.train import EarlyStopping, build_optimizer, build_scheduler, fit, predict
from dlfund.utils import set_seed


def _separable_loaders(n: int = 512, seed: int = 0) -> tuple[DataLoader, DataLoader]:
    """Two Gaussian blobs in a 1x8x8 'image': trivially learnable."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 2, (n,), generator=g)
    x = torch.randn(n, 1, 8, 8, generator=g) * 0.5 + (y.float() * 2 - 1).view(-1, 1, 1, 1)
    ds = TensorDataset(x, y)
    return (DataLoader(ds, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed)),
            DataLoader(ds, batch_size=128))


def _cfg(**kw) -> TrainConfig:
    base = {"epochs": 5, "lr": 0.05, "amp": False, "early_stop_patience": 0, "scheduler": "onecycle"}
    return TrainConfig(**{**base, **kw})


def test_early_stopping_counts_and_resets():
    es = EarlyStopping(patience=2)
    assert es.step(1.0) and not es.should_stop
    assert not es.step(1.1) and not es.should_stop
    assert not es.step(1.2) and es.should_stop
    es = EarlyStopping(patience=2)
    es.step(1.0)
    es.step(1.1)
    assert es.step(0.5) and es.bad_epochs == 0


def test_early_stopping_disabled_with_zero_patience():
    es = EarlyStopping(patience=0)
    for v in [1, 2, 3, 4]:
        es.step(v)
    assert not es.should_stop


def test_no_weight_decay_on_bias_and_norm():
    model = build_model("mlp", 1, 8, 2)
    opt = build_optimizer(model, _cfg(weight_decay=0.1))
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.ndim > 1 for p in decay["params"])
    assert all(p.ndim == 1 for p in no_decay["params"])


@pytest.mark.parametrize("steps", [1, 3, 4, 5, 8, 400])
def test_onecycle_survives_short_runs(steps):
    model = build_model("linear", 1, 8, 2)
    t = _cfg(epochs=1)
    opt = build_optimizer(model, t)
    sched = build_scheduler(opt, t, steps)
    for _ in range(steps):
        opt.step()
        if sched is not None:
            sched.step()


def test_fit_learns_separable_data(tmp_path):
    set_seed(0)
    tr, va = _separable_loaders()
    model = build_model("mlp", 1, 8, 2)
    res = fit(model, tr, va, _cfg(), torch.device("cpu"), tmp_path)
    assert res.epochs_run == 5
    assert res.history[-1]["val_acc"] > 0.95
    assert res.history[-1]["train_loss"] < res.history[0]["train_loss"]
    assert (tmp_path / "best.pt").exists()


def test_fit_resumes_from_checkpoint(tmp_path):
    set_seed(0)
    tr, va = _separable_loaders()
    model = build_model("linear", 1, 8, 2)
    fit(model, tr, va, _cfg(epochs=2, scheduler="none"), torch.device("cpu"), tmp_path)
    # A longer run in the same checkpoint dir must continue at epoch 3, not restart.
    model2 = build_model("linear", 1, 8, 2)
    res = fit(model2, tr, va, _cfg(epochs=4, scheduler="none"), torch.device("cpu"), tmp_path)
    assert [h["epoch"] for h in res.history] == [1, 2, 3, 4]


def test_same_seed_same_result(tmp_path):
    accs = []
    for i in range(2):
        set_seed(123, deterministic=True)
        tr, va = _separable_loaders(seed=123)
        model = build_model("cnn", 1, 8, 2)
        fit(model, tr, va, _cfg(epochs=1), torch.device("cpu"), tmp_path / str(i))
        _, logits, _ = predict(model, va, torch.device("cpu"), amp=False)
        accs.append(logits)
    np.testing.assert_allclose(accs[0], accs[1], rtol=1e-5, atol=1e-5)
