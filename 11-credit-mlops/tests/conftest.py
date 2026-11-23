import copy
import json
from pathlib import Path

import pytest
import yaml

from credops import pipeline
from credops.data import split
from credops.synthetic import synthetic_frame

ROOT = Path(__file__).resolve().parents[1]


def make_params(base: Path, mlflow: bool = False) -> dict:
    p = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    p = copy.deepcopy(p)
    p["data"].update(raw_dir=str(base / "raw"), processed_dir=str(base / "processed"))
    p["paths"].update(model_dir=str(base / "models"), report_dir=str(base / "reports"))
    p["train"]["seeds"] = [0, 1]
    p["train"]["params"]["lgbm"].update(n_estimators=30, n_jobs=1)
    p["drift"]["evidently"] = False
    p["mlflow"].update(enabled=mlflow, tracking_uri=f"sqlite:///{(base / 'mlflow.db').as_posix()}")
    return p


def write_processed(p: dict, n: int = 3000) -> None:
    df = synthetic_frame(n, seed=0)
    tr, te = split(df, 0.2, 42)
    out = Path(p["data"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    tr.to_parquet(out / "train.parquet", index=False)
    te.to_parquet(out / "test.parquet", index=False)
    (out / "data_card.json").write_text(json.dumps({"rows": n, "synthetic": True}), encoding="utf-8")


@pytest.fixture(scope="session")
def trained(tmp_path_factory) -> dict:
    """A synthetic project trained end to end (MLflow off): params with every path inside a temp dir."""
    base = tmp_path_factory.mktemp("proj")
    p = make_params(base)
    write_processed(p)
    p["_summary"] = pipeline.train(p)
    return p
