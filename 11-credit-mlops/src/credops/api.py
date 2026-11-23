"""FastAPI service for the exported credit-default model.

    uvicorn credops.api:app --host 0.0.0.0 --port 8000          # MODEL_DIR, PREDICTION_LOG env vars

Endpoints: ``/health`` (liveness), ``/ready`` (model loaded), ``/model-info``, ``/predict``, ``/predict/batch``,
``/metrics`` (Prometheus). Each prediction is appended to a JSONL log (model inputs, probability, latency) that the
drift job reads; protected attributes are not part of the schema, so they are never received or logged.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest

from credops.schema import MODEL_INPUTS, Applicant, Batch, BatchPrediction, Prediction

log = logging.getLogger("credops.api")


class PredictionLog:
    """Thread-safe JSONL append log. ``None`` path disables logging."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, records: list[dict[str, Any]]) -> None:
        if not self.path:
            return
        text = "".join(json.dumps(r) + "\n" for r in records)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(text)


class Service:
    """Model, threshold, metadata and metrics for one app instance."""

    def __init__(self, model_dir: str | Path, log_path: str | Path | None) -> None:
        model_dir = Path(model_dir)
        self.model = joblib.load(model_dir / "model.joblib")
        self.meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        if self.meta["features"] != MODEL_INPUTS:
            raise RuntimeError("model features do not match the API schema")
        self.threshold = float(self.meta["threshold"])
        self.version = self.meta["version"]
        self.log = PredictionLog(log_path)
        self.registry = CollectorRegistry()
        self.requests = Counter("predict_requests_total", "Prediction requests", ["endpoint"], registry=self.registry)
        self.rows = Counter("predict_rows_total", "Rows scored", registry=self.registry)
        self.flags = Counter("predict_flags_total", "Rows flagged for review", registry=self.registry)
        self.latency = Histogram("predict_latency_seconds", "Model latency per request", ["endpoint"],
                                 buckets=(.001, .0025, .005, .01, .025, .05, .1, .25, .5, 1), registry=self.registry)
        self.prob = Histogram("predict_probability", "Predicted probabilities",
                              buckets=tuple(i / 10 for i in range(1, 10)), registry=self.registry)

    def score(self, rows: list[Applicant], endpoint: str) -> tuple[str, list[float], list[bool]]:
        rid = uuid.uuid4().hex
        t0 = time.perf_counter()
        X = pd.DataFrame([r.model_dump() for r in rows], columns=MODEL_INPUTS)
        probs = self.model.predict_proba(X)[:, 1].tolist()
        dt = time.perf_counter() - t0
        flags = [p >= self.threshold for p in probs]
        self.requests.labels(endpoint).inc()
        self.latency.labels(endpoint).observe(dt)
        self.rows.inc(len(rows))
        self.flags.inc(sum(flags))
        for p in probs:
            self.prob.observe(p)
        ts = time.time()
        self.log.write([{"ts": ts, "request_id": rid, "model_version": self.version, "latency_s": dt,
                         "probability": p, **r.model_dump()} for r, p in zip(rows, probs, strict=True)])
        return rid, probs, flags


def create_app(model_dir: str | Path | None = None, log_path: str | Path | None = "unset") -> FastAPI:
    """App factory. Defaults come from ``MODEL_DIR`` (``models``) and ``PREDICTION_LOG`` (``logs/predictions.jsonl``;
    empty string disables the log)."""
    model_dir = model_dir or os.environ.get("MODEL_DIR", "models")
    if log_path == "unset":
        log_path = os.environ.get("PREDICTION_LOG", "logs/predictions.jsonl") or None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.svc = Service(model_dir, log_path)
        log.info("loaded model %s (threshold %.4f)", app.state.svc.version, app.state.svc.threshold)
        yield

    app = FastAPI(title="Credit default risk API", version="1.0.0", lifespan=lifespan,
                  description="Calibrated P(default next month) for a human-reviewed referral workflow.")

    def svc(request: Request) -> Service:
        s = getattr(request.app.state, "svc", None)
        if s is None:
            raise HTTPException(503, "model not loaded")
        return s

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request) -> dict[str, str]:
        return {"status": "ready", "model_version": svc(request).version}

    @app.get("/model-info")
    def model_info(request: Request) -> dict[str, Any]:
        m = svc(request).meta
        return {k: m[k] for k in ("version", "kind", "threshold", "test_metrics", "features", "trained_at")}

    @app.post("/predict", response_model=Prediction)
    def predict(row: Applicant, request: Request) -> Prediction:
        s = svc(request)
        rid, probs, flags = s.score([row], "predict")
        return Prediction(request_id=rid, probability=probs[0], flag=flags[0], threshold=s.threshold,
                          model_version=s.version)

    @app.post("/predict/batch", response_model=BatchPrediction)
    def predict_batch(batch: Batch, request: Request) -> BatchPrediction:
        s = svc(request)
        rid, probs, flags = s.score(batch.rows, "batch")
        return BatchPrediction(request_id=rid, probabilities=probs, flags=flags, threshold=s.threshold,
                               model_version=s.version)

    @app.get("/metrics")
    def prom(request: Request) -> Response:
        return Response(generate_latest(svc(request).registry), media_type=CONTENT_TYPE_LATEST)

    return app


def __getattr__(name: str) -> Any:
    """Lazy ``app`` for ``uvicorn credops.api:app`` so importing the module does not need a model on disk."""
    if name == "app":
        return create_app()
    raise AttributeError(name)
