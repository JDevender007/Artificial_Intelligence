import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from credops import pipeline
from credops.api import create_app
from credops.monitor import check
from credops.schema import MODEL_INPUTS
from credops.simulate import scenario

from .conftest import make_params, write_processed


def test_train_exports_and_reports(trained):
    mdir, rdir = Path(trained["paths"]["model_dir"]), Path(trained["paths"]["report_dir"])
    meta = json.loads((mdir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["features"] == MODEL_INPUTS and 0 < meta["threshold"] < 1 and meta["kind"] in ("logreg", "lgbm")
    assert (mdir / "model.joblib").exists() and (mdir / "reference_scores.npy").exists()
    runs = pd.read_csv(rdir / "train_runs.csv")
    assert len(runs) == 4 and set(runs.model) == {"logreg", "lgbm"}
    assert "mean ± std over 2 seeds" in (rdir / "results.md").read_text(encoding="utf-8")
    assert trained["_summary"]["champion"] == meta["kind"]


def test_drift_stage(trained):
    out = pipeline.drift(trained)
    assert not out["none"]["drift"]["alert"]
    assert out["recession"]["default_rate"] > out["none"]["default_rate"]
    assert "| limit_cut |" in (Path(trained["paths"]["report_dir"]) / "drift.md").read_text(encoding="utf-8")


ROW = {c: 0 for c in MODEL_INPUTS} | {"LIMIT_BAL": 50_000, "BILL_AMT1": 20_000, "PAY_AMT1": 1_000}


@pytest.fixture
def client(trained, tmp_path):
    log = tmp_path / "pred.jsonl"
    with TestClient(create_app(trained["paths"]["model_dir"], log)) as c:
        c.log_path = log
        yield c


def test_api_endpoints(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json()["status"] == "ready"
    r = client.post("/predict", json=ROW).json()
    assert 0 <= r["probability"] <= 1 and r["flag"] == (r["probability"] >= r["threshold"])
    b = client.post("/predict/batch", json={"rows": [ROW, ROW | {"PAY_1": 3}]}).json()
    assert len(b["probabilities"]) == 2 and b["probabilities"][0] == pytest.approx(r["probability"])
    assert client.get("/model-info").json()["features"] == MODEL_INPUTS
    metrics = client.get("/metrics").text
    assert 'predict_requests_total{endpoint="predict"} 1.0' in metrics and "predict_rows_total 3.0" in metrics
    logged = [json.loads(line) for line in client.log_path.read_text(encoding="utf-8").splitlines()]
    assert len(logged) == 3 and "SEX" not in logged[0] and logged[0]["model_version"] == r["model_version"]


@pytest.mark.parametrize("bad", [ROW | {"SEX": 1}, ROW | {"PAY_1": 12}, ROW | {"LIMIT_BAL": 0},
                                 {k: v for k, v in ROW.items() if k != "PAY_AMT6"}])
def test_api_rejects_invalid(client, bad):
    assert client.post("/predict", json=bad).status_code == 422


def test_batch_limits(client):
    assert client.post("/predict/batch", json={"rows": []}).status_code == 422
    assert client.post("/predict/batch", json={"rows": [ROW] * 1001}).status_code == 422


def test_monitor_on_logged_traffic(trained, tmp_path):
    test = pd.read_parquet(Path(trained["data"]["processed_dir"]) / "test.parquet")
    log = tmp_path / "pred.jsonl"
    with TestClient(create_app(trained["paths"]["model_dir"], log)) as c:
        c.post("/predict/batch", json={"rows": test[MODEL_INPUTS].head(100).to_dict("records")})
    assert check(log, trained, min_rows=500)["status"] == "insufficient_data"
    with TestClient(create_app(trained["paths"]["model_dir"], log)) as c:
        cut = scenario(test, "limit_cut")[MODEL_INPUTS]
        for i in range(0, 600, 200):
            c.post("/predict/batch", json={"rows": cut.iloc[i:i + 200].to_dict("records")})
    out = check(log, trained, min_rows=500)
    assert out["rows"] == 700 and out["top"][0]["column"] == "LIMIT_BAL" and "score_psi" in out


def test_mlflow_registry(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    p = make_params(tmp_path, mlflow=True)
    p["train"]["seeds"] = [0]
    p["gate"].update(min_pr_auc_margin_std=-1e9, max_ece=1.0)  # force the GBDT champion: skops must trust LightGBM
    write_processed(p, n=1500)
    assert pipeline.train(p)["champion"] == "lgbm"
    mlflow.set_tracking_uri(p["mlflow"]["tracking_uri"])
    mv = mlflow.MlflowClient().get_model_version_by_alias(p["mlflow"]["registered_model"], "champion")
    assert str(mv.version) == "1"
