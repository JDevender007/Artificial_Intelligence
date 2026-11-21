from __future__ import annotations

from pathlib import Path

import pytest

from movierec.config import load_config
from movierec.data import build_dataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def smoke_cfg(tmp_path_factory: pytest.TempPathFactory):
    d = tmp_path_factory.mktemp("p6")
    return load_config(ROOT / "configs" / "smoke.yaml", [f"output_dir={d / 'out'}", f"data.raw_dir={d / 'raw'}"])


@pytest.fixture(scope="session")
def ds(smoke_cfg):
    return build_dataset(smoke_cfg.data)


@pytest.fixture(scope="session")
def smoke_run(smoke_cfg):
    from movierec.experiment import run_experiment

    return run_experiment(smoke_cfg)
