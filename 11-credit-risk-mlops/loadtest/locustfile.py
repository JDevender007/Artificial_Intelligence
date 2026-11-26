"""Locust load test for the API.

    locust -f loadtest/locustfile.py --host http://127.0.0.1:8000 --headless -u 50 -r 10 -t 60s \
        --csv reports/locust/run --html reports/locust/run.html

Payloads are held-out test rows when ``data/processed/test.parquet`` exists, otherwise synthetic rows.
Traffic mix: 90% single predictions, 9% batches of 50, 1% model-info.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
from locust import HttpUser, between, task

from credops.schema import MODEL_INPUTS
from credops.synthetic import synthetic_frame

_TEST = Path(__file__).resolve().parents[1] / "data" / "processed" / "test.parquet"
ROWS = (pd.read_parquet(_TEST) if _TEST.exists() else synthetic_frame(2000, seed=1))[MODEL_INPUTS]
ROWS = [{k: int(v) for k, v in r.items()} for r in ROWS.to_dict("records")]


class Client(HttpUser):
    wait_time = between(0.05, 0.2)

    @task(90)
    def predict(self) -> None:
        self.client.post("/predict", json=random.choice(ROWS))

    @task(9)
    def batch(self) -> None:
        self.client.post("/predict/batch", json={"rows": random.sample(ROWS, 50)}, name="/predict/batch[50]")

    @task(1)
    def info(self) -> None:
        self.client.get("/model-info")
