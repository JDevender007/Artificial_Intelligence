"""End-to-end: the full experiment runner on FakeData (no download, CPU, ~seconds)."""

import pandas as pd

from dlfund.config import load_config
from dlfund.experiment import run_experiment
from dlfund.utils import get_logger


def test_run_experiment_end_to_end_and_resume(tmp_path):
    cfg = load_config(None, [
        f"output_dir={tmp_path}", "seeds=[0,1]", "models=[linear,cnn]", "tensorboard=false",
        "data.dataset=fake", "data.num_workers=0", "data.limit_train=128",
        "train.epochs=1", "train.batch_size=32", "train.amp=false",
    ])
    logger = get_logger("test", None)
    agg = run_experiment(cfg, logger)
    assert set(agg.index) == {"linear", "cnn"}
    assert (agg["n_seeds"] == 2).all()
    for f in ["results.md", "results.csv", "per_class.md", "figures/learning_curves.png",
              "figures/confusion_cnn.png", "preds/seed1_cnn.npz", "history/seed0_linear.csv"]:
        assert (tmp_path / f).exists(), f
    assert not (tmp_path / "checkpoints/seed0_cnn/last.pt").exists()
    assert (tmp_path / "checkpoints/seed0_cnn/best.pt").exists()

    # Rerunning the same command trains nothing new.
    run_experiment(cfg, logger)
    assert len(pd.read_csv(tmp_path / "results_per_seed.csv")) == 4
