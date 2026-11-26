from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from movierec.config import TowerConfig
from movierec.models import BPRRecommender, Popularity, TwoTower, TwoTowerRecommender, known_users
from movierec.utils import read_jsonl


def test_popularity_ranks_by_train_counts(ds):
    s = Popularity(ds).score_fn("test")(np.array([0, 1]))
    assert s.shape == (2, ds.n_items)
    assert np.argmax(s[0]) == np.argmax(ds.item_counts("train"))


def test_bpr_cold_users_fall_back_to_popularity(ds, smoke_cfg):
    rec = BPRRecommender(ds, smoke_cfg.bpr, smoke_cfg.train, torch.device("cpu"))
    cold = np.flatnonzero(~known_users(ds))
    assert len(cold)
    s = rec.score_fn("test")(cold[:3])
    assert np.allclose(s, Popularity(ds).scores)


def test_two_tower_vectors_normalised_and_loss_finite(ds, smoke_cfg):
    torch.manual_seed(0)
    rec = TwoTowerRecommender(ds, smoke_cfg.tower, smoke_cfg.train, torch.device("cpu"))
    loss = next(iter(rec._batches()))
    assert torch.isfinite(loss)
    iv = rec.item_vectors()
    assert iv.shape == (ds.n_items, smoke_cfg.tower.dim)
    assert torch.allclose(iv.norm(dim=-1), torch.ones(ds.n_items), atol=1e-5)


def test_features_change_cold_user_embedding():
    cfg = TowerConfig(dim=8, emb=4, hidden=8)
    m = TwoTower(5, 6, cfg, np.full((6, 6), 18), np.zeros(6, dtype=np.int64)).eval()
    uid, hist = torch.tensor([5, 5]), torch.full((2, 3), -1)
    v = m.user_vec(uid, torch.tensor([[0, 1, 2], [1, 3, 4]]), hist)
    assert not torch.allclose(v[0], v[1])


def test_smoke_experiment_writes_reports_and_resumes(smoke_run, smoke_cfg):
    out = smoke_run
    rows = read_jsonl(out / "results.jsonl")
    assert len(rows) == len(smoke_cfg.models) * len(smoke_cfg.seeds)
    for f in ("results.md", "segments.md", "bootstrap.md", "error_analysis.md", "index.md", "data_summary.md"):
        assert (out / f).exists(), f
    for r in rows:
        assert 0 <= r["recall@10"] <= 1 and 0 <= r["ndcg@10"] <= 1
    pop = [r for r in rows if r["model"] == "popularity"]
    assert pop[0]["recall@10"] == pytest.approx(pop[1]["recall@10"])  # deterministic
    from movierec.experiment import run_experiment

    run_experiment(smoke_cfg)  # every (model, seed) is skipped
    assert len(read_jsonl(out / "results.jsonl")) == len(rows)
    meta = json.loads((out / "serve" / "meta.json").read_text(encoding="utf-8"))
    assert len(meta["item_raw"]) == np.load(out / "serve" / "item_vecs.npy").shape[0]


def test_service_endpoints(smoke_run):
    from fastapi.testclient import TestClient

    from movierec.service import create_app

    client = TestClient(create_app(smoke_run / "serve"))
    assert client.get("/health").json()["status"] == "ok"
    meta = json.loads((smoke_run / "serve" / "meta.json").read_text(encoding="utf-8"))
    uid = meta["user_raw"][0]
    r = client.get(f"/recommend/{uid}?k=5")
    assert r.status_code == 200 and len(r.json()["items"]) == 5
    seen = {meta["item_raw"][i] for i in meta["seen"].get("0", [])}
    assert not seen & {it["movie_id"] for it in r.json()["items"]}
    assert client.get("/recommend/99999999").status_code == 404
    liked = meta["item_raw"][:3]
    body = {"gender": "F", "age": 25, "occupation": 3, "liked_movie_ids": liked, "k": 4}
    r = client.post("/recommend/cold", json=body)
    assert r.status_code == 200 and len(r.json()["items"]) == 4
    assert not set(liked) & {it["movie_id"] for it in r.json()["items"]}
    assert client.post("/recommend/cold", json={"gender": "F", "age": 26, "occupation": 3}).status_code == 422
